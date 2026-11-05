"""Prequentielle Codelänge der ganzen Additionstabelle für einen Arm (primärer MDL-Endpunkt).

Pro Seed: feste Zufallsreihenfolge aller p*p Paare, Blöcke 64, 128, ... bis p*p.
Block 1 uniform codiert; für jeden weiteren Block wird auf dem Präfix trainiert
(Warmstart aus dem vorigen Block, `steps_per_block` Schritte, Full-Batch) und dann
der Block mit dem Modell codiert, BEVOR es ihn sieht. Self-Switch (v2): pro Block wählt
der Codierer einen Trainingsstand (nach 0, 30, 100, 300, 1.000, 3.000 Schritten auf dem
Präfix) und eine Temperatur T (oder uniform) und zahlt dafür log2(Anzahl Optionen) Bit
(zweiteiliger Code, ehrlich mitgezählt; „Early Stopping als Codewahl“, Blier & Ollivier).
Weitertrainiert wird immer vom Endstand. Gezählt wird im Kern
(`core.codelength.prequential_codelength`).

  uv run python stages/s0a_grokking/prequential.py --arm wd --seeds 100 101 102

Stufe 0b (Transfer): `--op sub --init-from runs/s0a/<A-Lauf>` startet von den Gewichten eines
auf Aufgabe A trainierten Laufs (gleiche Seeds). Kontrollen: `--init-control scaled`
(Zufallsinit, je Tensor auf die Norm der A-Gewichte skaliert) und `permuted` (A-Gewichte je
Tensor zufällig permutiert: gleiche Werteverteilung, Struktur zerstört). Nur für Arme none/wd.

0a+ B1 („erst lernen, dann komprimieren“): `--post wd|mdl|bitnet|prune --post-steps K` hängt je
Block an den Endstand der Phase 1 (Arm none/wd) eine Nachtrainings-Phase mit K Schritten an.
Sie ist ein Zweig: Der nächste Block geht vom Stand der Phase 1 aus. Ihre Trainingsstände
(p30, p100, …, pK) kommen als zusätzliche Self-Switch-Kandidaten hinzu; die Wahl wird wie immer
bezahlt (alle Post-Varianten zahlen dieselben Bits). `--post none` (Standard) = Verhalten wie vorher.

0a+ B1-Zyklus (Wach-Schlaf, Entscheidung Jonas 2026-11-04): `--cycles C` (nur mit `--restart` und
`--post …`) trainiert je Block C Zyklen: Wach = `steps_per_block` Schritte mit dem Arm (wd), Schlaf =
`post_steps` Schritte mit dem Kompressionsverfahren `--post` auf den eigenen Gewichten; die nächste
Wachphase lernt vom komprimierten Stand (deterministische Gewichte des Schlafs) weiter. Optimizer-
Zustand wird an jedem Phasenwechsel zurückgesetzt, auch in der Kontrolle `--post wd`. Kandidaten:
c0, c30, c100, c300 (erste Wachphase), Ende jeder Wach- und Schlafphase (w1, s1, w2, s2, …).
`--cycles 0` (Standard) = Verhalten wie vorher.
"""

from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass
from typing import Literal

import mlx.core as mx
import numpy as np
import tyro
from mlx.utils import tree_map

from kompression.core.codelength import doubling_blocks, prequential_codelength
from kompression.model_mlx import forward, init_params, label_logprobs
from kompression.objectives import make_arm
from kompression.runlog import Run
from kompression.train import TrainConfig, build_fns, stack_trees


@dataclass(frozen=True)
class Args:
    arm: Literal["none", "wd", "mdl", "bitnet", "subspace"] = "wd"
    seeds: tuple[int, ...] = (100, 101, 102)
    p: int = 97
    first_block: int = 64
    steps_per_block: int = 3000
    weight_decay: float = 1.0
    lr: float = 1e-3
    mdl_beta: float = 0.1
    penalty_n_min: int = 0  # mdl: Strafterm durch max(Präfixgröße, penalty_n_min) teilen
    beta_warmup: int = 2000  # mdl: beta linear über die ersten Schritte hochfahren (wie im Training)
    restart: bool = False  # je Block von der Initialisierung neu trainieren (Blier & Ollivier) statt Warmstart
    subspace_rank: int = 64
    op: Literal["add", "sub", "mul"] = "add"
    init_from: str = ""
    init_control: Literal["none", "scaled", "permuted"] = "none"
    name: str = ""
    # free: ohne Regularisierung; mdl_norm: Norm-Zwilling (mdl-Schlaf intern, dann nur die
    # Wachgewichte je Tensor auf die Norm des mdl-Ergebnisses skalieren; nur mit --cycles)
    post: Literal["none", "wd", "free", "mdl", "mdl_norm", "bitnet", "prune"] = "none"
    post_steps: int = 1000
    post_weight_decay: float = 1.0  # für post wd und prune (Nachtraining mit Weight Decay)
    post_mdl_beta: float = 0.1
    post_beta_warmup: int = 0
    prune_frac: float = 0.9
    cycles: int = 0  # > 0: Wach-Schlaf-Zyklen je Block (siehe Docstring)
    post_mdl_lr_mult: float = 1.0  # Schlaf/Post mdl: Lernrate von rho und Prior × mult
    post_mdl_init_log_sigma: float = -6.0
    post_mdl_snap: bool = False  # Zyklus: nach mdl-Schlaf Gewichte auf Prior-Zentren setzen
    post_mdl_wd: float = 0.0  # Schlaf mdl/mdl_norm: zusätzlicher L2 nur auf die Gewichte (Hybrid)
    post_mdl_pi0: float = 0.0  # Schlaf mdl: Start-Logit der Null-Komponente des Priors
    post_mdl_s0: float = 0.05  # Schlaf mdl: Startbreite der Null-Komponente des Priors
    # Zyklus, selbst gewählter Schlaf (E30): Kandidaten "mdl:<beta>" oder "wd:<lambda>"; jeder
    # Kandidat schläft auf dem Präfix ohne die letzten Anteil post_select_holdout, jeder Seed
    # behält den Kandidaten mit der kürzesten Codelänge (Temperatur 1) auf diesem zurückgehaltenen
    # Teil. Wach lernt weiter auf dem ganzen Präfix. Leer (Standard) = Verhalten wie vorher.
    post_select: tuple[str, ...] = ()
    post_select_holdout: float = 0.125


def _tnorm(x: mx.array) -> mx.array:
    """Frobenius-Norm je Seed (Achse 0), in die Form von x broadcastbar."""
    n = mx.sqrt((x * x).reshape(x.shape[0], -1).sum(axis=1))
    return n.reshape((x.shape[0],) + (1,) * (x.ndim - 1))


def load_inits(run_dir: str, seeds, random_inits, control: str):
    """A-Gewichte je Seed laden (gestapelt in params*.npz); optional Kontrolle anwenden."""
    import json
    from pathlib import Path

    from mlx.utils import tree_flatten, tree_unflatten

    d = Path(run_dir)
    run_seeds = json.loads((d / "config.json").read_text())
    run_seeds = (run_seeds.get("train") or run_seeds.get("args"))["seeds"]
    npz = d / ("params.npz" if (d / "params.npz").exists() else "params_final.npz")
    flat = dict(np.load(npz))
    out = []
    for s, rnd in zip(seeds, random_inits, strict=True):
        i = run_seeds.index(s)
        a_w = {k: v[i] for k, v in flat.items()}
        res = []
        rng = np.random.default_rng(10_000 + s)
        for k, r in tree_flatten(rnd):
            w = a_w[k]
            if control == "scaled":
                r = np.array(r)
                w = r * (np.linalg.norm(w) / (np.linalg.norm(r) + 1e-12))
            elif control == "permuted":
                w = rng.permutation(w.ravel()).reshape(w.shape)
            res.append((k, mx.array(np.asarray(w, dtype=np.float32))))
        out.append(tree_unflatten(res))
    return out


def main(a: Args) -> None:
    cfg = TrainConfig(p=a.p, op=a.op, seeds=a.seeds, steps=a.steps_per_block, lr=a.lr,
                      weight_decay=a.weight_decay if a.arm == "wd" else 0.0, arm=a.arm,
                      mdl_beta=a.mdl_beta, subspace_rank=a.subspace_rank, tags=(a.arm, "preq"))
    run = Run("s0a_preq", a.name or a.arm, {"args": dataclasses.asdict(a)})
    print(f"Lauf: {run.dir}", flush=True)
    task, mcfg = cfg.task(), cfg.model()
    arm = make_arm(a.arm, subspace_rank=a.subspace_rank)
    x_all, y_all = task.all_pairs()
    n = len(y_all)
    perms = [np.random.default_rng(s).permutation(n) for s in a.seeds]
    X = mx.array(np.stack([x_all[pm] for pm in perms]))
    Y = mx.array(np.stack([y_all[pm] for pm in perms]))

    inits = [init_params(mcfg, s) for s in a.seeds]
    if a.init_from:
        if a.arm not in ("none", "wd"):
            raise ValueError("Transfer-Init nur für Arme none/wd")
        inits = load_inits(a.init_from, a.seeds, inits, a.init_control)
    pairs = [arm.init(p0, s) for p0, s in zip(inits, a.seeds, strict=True)]
    tr = stack_trees([q[0] for q in pairs])
    const = stack_trees([q[1] for q in pairs])
    m = tree_map(mx.zeros_like, tr)
    v = tree_map(mx.zeros_like, tr)
    t = 0

    temps = (1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 16.0, 32.0)
    checkpoints = tuple(c for c in (0, 30, 100, 300, 1000, 3000, 10000) if c <= a.steps_per_block)
    if a.steps_per_block not in checkpoints:
        checkpoints = (*checkpoints, a.steps_per_block)
    labels = [f"c{c}" for c in checkpoints]
    post_ckpts: tuple[int, ...] = ()
    if a.post != "none":
        if a.arm not in ("none", "wd"):
            raise ValueError("Nachtraining (--post) nur nach Phase-1-Arm none/wd")
        post_ckpts = tuple(c for c in (30, 100, 300, 1000, 3000) if c < a.post_steps)
        post_ckpts = (*post_ckpts, a.post_steps)
        labels += [f"p{c}" for c in post_ckpts]
        post_arm = {"free": "none", "mdl_norm": "mdl"}.get(a.post, a.post)
        if a.post == "mdl_norm" and not a.cycles:
            raise ValueError("--post mdl_norm nur mit --cycles")
        arm2 = make_arm(post_arm, prune_frac=a.prune_frac,
                        mdl_init_log_sigma=a.post_mdl_init_log_sigma,
                        mdl_pi0_logit=a.post_mdl_pi0, mdl_s0=a.post_mdl_s0)
        cfg2 = dataclasses.replace(
            cfg, arm=post_arm, mdl_beta=a.post_mdl_beta, mdl_aux_lr_mult=a.post_mdl_lr_mult,
            mdl_mu_weight_decay=a.post_mdl_wd,
            weight_decay=a.post_weight_decay if a.post in ("wd", "prune") else 0.0)
    early = tuple(c for c in (30, 100, 300) if c < a.steps_per_block)
    if a.cycles:
        if a.post == "none" or not a.restart:
            raise ValueError("--cycles braucht --post <Verfahren> und --restart")
        labels = ["c0", *[f"c{c}" for c in early]]
        for c in range(1, a.cycles + 1):
            labels += [f"w{c}", f"s{c}"]
    switch_bits = math.log2(len(labels) * len(temps) + 1)  # + uniform

    def make_vlogp(arm_):
        def block_logp(tr_, const_, x, y):
            p_ = arm_.deterministic(tr_, const_)
            logits = forward(p_, x, mcfg)
            return mx.stack([label_logprobs(logits / tt, y, cfg.use_stablemax) for tt in temps])

        return mx.vmap(block_logp, in_axes=(0, 0, 0, 0))

    vlogp = make_vlogp(arm)
    if a.post != "none":
        vlogp2 = make_vlogp(arm2)
    vdet_phase1 = mx.vmap(arm.deterministic, in_axes=(0, 0))
    if a.post != "none":
        vdet2 = mx.vmap(arm2.deterministic, in_axes=(0, 0))

    sel_counts: dict[str, int] = {}
    if a.post_select:
        if not a.cycles:
            raise ValueError("--post-select nur mit --cycles")
        sel = []
        for c_ in a.post_select:
            kind, val = c_.split(":")
            if kind == "mdl":
                arm_k = make_arm("mdl", mdl_init_log_sigma=a.post_mdl_init_log_sigma,
                                 mdl_pi0_logit=a.post_mdl_pi0, mdl_s0=a.post_mdl_s0)
                cfg_k = dataclasses.replace(cfg, arm="mdl", mdl_beta=float(val), weight_decay=0.0,
                                            mdl_aux_lr_mult=a.post_mdl_lr_mult)
            elif kind == "wd":
                arm_k = make_arm("wd")
                cfg_k = dataclasses.replace(cfg, arm="wd", weight_decay=float(val))
            else:
                raise ValueError(f"unbekannter Schlaf-Kandidat: {c_}")
            sel.append((c_, arm_k, cfg_k, float(val) if kind == "mdl" else 0.0,
                        mx.vmap(arm_k.deterministic, in_axes=(0, 0))))
            sel_counts[c_] = 0

    def select_sleep(w, xs, ys, start):
        """Jeder Kandidat schläft vom selben Wachstand; je Seed gewinnt die kürzeste Holdout-Codelänge."""
        h = max(1, int(round(start * a.post_select_holdout)))
        xt, yt, xh, yh = xs[:, :start - h], ys[:, :start - h], xs[:, start - h:], ys[:, start - h:]
        dummy = mx.zeros((1,))
        c1 = stack_trees([q[1] for q in pairs])
        ws, scores = [], []
        for _, arm_k, cfg_k, beta_k, vdet_k in sel:
            vstep_k, _, _ = build_fns(cfg_k, arm_k, n_train=max(start - h, a.penalty_n_min))
            pk = [arm_k.init(tree_map(lambda x, i=i: x[i], w), s) for i, s in enumerate(a.seeds)]
            trk, ck = stack_trees([q[0] for q in pk]), stack_trees([q[1] for q in pk])
            mk, vk = tree_map(mx.zeros_like, trk), tree_map(mx.zeros_like, trk)
            for t2 in range(1, a.post_steps + 1):
                noise = arm_k.noise_like(trk) if arm_k.needs_noise else dummy
                b2 = beta_k * (min(1.0, t2 / a.post_beta_warmup) if a.post_beta_warmup else 1.0)
                trk, mk, vk, _, _ = vstep_k(trk, ck, mk, vk, mx.array(float(t2)), mx.array(b2),
                                            noise, xt, yt)
                if t2 % 50 == 0:
                    mx.eval(trk, mk, vk)
            wk = vdet_k(trk, ck)
            lp_h = vlogp(wk, c1, xh, yh)[:, 0, :].sum(axis=1)  # [S], Temperatur 1
            mx.eval(wk, lp_h)
            ws.append(wk)
            scores.append(-np.array(lp_h, dtype=np.float64))
        best = np.argmin(np.stack(scores), axis=0)  # [S]
        for b in best:
            sel_counts[sel[int(b)][0]] += 1
        idx = mx.array(best)

        def pick(*cs):
            out_ = cs[0]
            for k_ in range(1, len(cs)):
                mask = (idx == k_).reshape((-1,) + (1,) * (cs[k_].ndim - 1))
                out_ = mx.where(mask, cs[k_], out_)
            return out_

        return tree_map(pick, *ws)

    def cycle_block(start, end):
        """Wach-Schlaf-Zyklen auf dem Präfix; liefert die Kandidaten-Logprobs für den Block."""
        vstep_w, _, _ = build_fns(cfg, arm, n_train=max(start, a.penalty_n_min))
        vstep_s, _, _ = build_fns(cfg2, arm2, n_train=max(start, a.penalty_n_min))
        xs, ys, xb, yb = X[:, :start], Y[:, :start], X[:, start:end], Y[:, start:end]
        dummy = mx.zeros((1,))

        def ev(vl, tr_, const_):
            lp_c = vl(tr_, const_, xb, yb)
            mx.eval(lp_c)
            return np.array(lp_c, dtype=np.float64)

        mx.random.seed(start)  # mdl-Rauschen reproduzierbar und über Bedingungen gepaart
        w = stack_trees([q[0] for q in pairs])
        c1 = stack_trees([q[1] for q in pairs])
        out = [ev(vlogp, w, c1)]
        for c in range(a.cycles):
            m_, v_ = tree_map(mx.zeros_like, w), tree_map(mx.zeros_like, w)
            for t_ in range(1, a.steps_per_block + 1):  # Wach
                w, m_, v_, _, _ = vstep_w(w, c1, m_, v_, mx.array(float(t_)), mx.array(0.0),
                                          dummy, xs, ys)
                if t_ % 50 == 0:
                    mx.eval(w, m_, v_)
                if c == 0 and t_ in early:
                    out.append(ev(vlogp, w, c1))
            out.append(ev(vlogp, w, c1))
            if a.post_select:
                w = select_sleep(w, xs, ys, start)
                out.append(ev(vlogp, w, c1))
                continue
            pairs2 = [arm2.init(tree_map(lambda x, i=i: x[i], w), s) for i, s in enumerate(a.seeds)]
            tr2 = stack_trees([q[0] for q in pairs2])
            const2 = stack_trees([q[1] for q in pairs2])
            m2, v2 = tree_map(mx.zeros_like, tr2), tree_map(mx.zeros_like, tr2)
            for t2 in range(1, a.post_steps + 1):  # Schlaf: eigene Gewichte komprimieren
                noise = arm2.noise_like(tr2) if arm2.needs_noise else dummy
                beta2 = a.post_mdl_beta * (min(1.0, t2 / a.post_beta_warmup)
                                           if a.post_beta_warmup else 1.0)
                tr2, m2, v2, _, _ = vstep_s(tr2, const2, m2, v2, mx.array(float(t2)),
                                            mx.array(beta2), noise, xs, ys)
                if t2 % 50 == 0:
                    mx.eval(tr2, m2, v2)
            if a.post == "mdl_norm":  # Norm-Zwilling: nur die Normen des mdl-Ergebnisses übernehmen
                wm = vdet2(tr2, const2)
                w = tree_map(lambda wa, wb: wa * (_tnorm(wb) / (_tnorm(wa) + 1e-12)), w, wm)
                out.append(ev(vlogp, w, c1))
            elif a.post == "mdl" and a.post_mdl_snap:  # Kompression in die Gewichte übernehmen
                w = mx.vmap(arm2.snap)(tr2)
                out.append(ev(vlogp, w, c1))
            else:
                out.append(ev(vlogp2, tr2, const2))
                w = vdet2(tr2, const2)  # nächste Wachphase startet vom komprimierten Stand
        return out

    bounds = doubling_blocks(n, a.first_block)
    per_seed_logp: dict[tuple[int, int], np.ndarray] = {}
    t0 = time.time()
    for start, end in zip(bounds, bounds[1:], strict=False):
        if a.cycles:
            cands = cycle_block(start, end)
        else:
            vstep, _, _ = build_fns(cfg, arm, n_train=max(start, a.penalty_n_min))
            noise_dummy = mx.zeros((1,))
            if a.restart:  # frischer Start je Block: gleiche Init, Optimizer und Warmup zurückgesetzt
                tr = stack_trees([q[0] for q in pairs])
                m = tree_map(mx.zeros_like, tr)
                v = tree_map(mx.zeros_like, tr)
                t = 0
            cands = []  # je Checkpoint: [S, len(temps), block]
            for step_in_block in range(a.steps_per_block + 1):
                if step_in_block in checkpoints:
                    lp_c = vlogp(tr, const, X[:, start:end], Y[:, start:end])
                    mx.eval(lp_c)
                    cands.append(np.array(lp_c, dtype=np.float64))
                if step_in_block == a.steps_per_block:
                    break
                t += 1
                noise = arm.noise_like(tr) if arm.needs_noise else noise_dummy
                beta_t = a.mdl_beta * min(1.0, t / a.beta_warmup) if a.beta_warmup else a.mdl_beta
                tr, m, v, _, _ = vstep(tr, const, m, v, mx.array(float(t)), mx.array(beta_t),
                                       noise, X[:, :start], Y[:, :start])
                if t % 50 == 0:  # Graph regelmäßig auswerten (sonst Metal-Pufferlimit, v. a. bei mdl)
                    mx.eval(tr, m, v)
            if a.post != "none":  # Zweig: Nachtraining vom Endstand der Phase 1
                vstep2, _, _ = build_fns(cfg2, arm2, n_train=max(start, a.penalty_n_min))
                p1 = vdet_phase1(tr, const)
                pairs2 = [arm2.init(tree_map(lambda x, i=i: x[i], p1), s) for i, s in enumerate(a.seeds)]
                tr2 = stack_trees([q[0] for q in pairs2])
                const2 = stack_trees([q[1] for q in pairs2])
                m2 = tree_map(mx.zeros_like, tr2)
                v2 = tree_map(mx.zeros_like, tr2)
                for t2 in range(1, a.post_steps + 1):
                    noise = arm2.noise_like(tr2) if arm2.needs_noise else noise_dummy
                    if a.post_beta_warmup:
                        beta2 = a.post_mdl_beta * min(1.0, t2 / a.post_beta_warmup)
                    else:
                        beta2 = a.post_mdl_beta
                    tr2, m2, v2, _, _ = vstep2(tr2, const2, m2, v2, mx.array(float(t2)),
                                               mx.array(beta2), noise, X[:, :start], Y[:, :start])
                    if t2 % 50 == 0:
                        mx.eval(tr2, m2, v2)
                    if t2 in post_ckpts:
                        lp_c = vlogp2(tr2, const2, X[:, start:end], Y[:, start:end])
                        mx.eval(lp_c)
                        cands.append(np.array(lp_c, dtype=np.float64))
        lp = np.concatenate(cands, axis=1)  # [S, len(labels)*len(temps), block]
        uni = np.full(lp.shape[-1], -math.log(task.p))
        chosen, bits = [], []
        for i in range(len(a.seeds)):
            options = [*lp[i], uni]
            costs = [-o.sum() for o in options]
            k = int(np.argmin(costs))
            # Wahl der Temperatur wird als Teil des Codes bezahlt (switch_bits)
            per_seed_logp[(i, start)] = options[k] - switch_bits * math.log(2) / len(uni)
            chosen.append("uniform" if k == lp.shape[1] else
                          f"{labels[k // len(temps)]}/T{temps[k % len(temps)]:g}")
            bits.append(costs[k] / math.log(2) + switch_bits)
        bits = np.array(bits)
        run.log(block_start=start, block_end=end, bits_per_seed=bits.tolist(),
                bits_per_label=(bits / (end - start)).tolist(), temperature=chosen,
                elapsed_s=time.time() - t0)
        print(f"Block [{start},{end}) Bits/Label je Seed: {np.round(bits / (end - start), 3)} T={chosen}",
              *([f"Schlafwahl bisher: {sel_counts}"] if a.post_select else []), flush=True)

    results = {}
    for i, s in enumerate(a.seeds):
        r = prequential_codelength(n, task.p, bounds, lambda st, en, i=i: per_seed_logp[(i, st)])
        results[s] = {"total_bits": r["total_bits"], "uniform_bits": r["uniform_bits"],
                      "per_block_bits": r["per_block_bits"]}
    run.finish({"prequential": results, "bounds": bounds, "seconds": time.time() - t0,
                **({"post_select_counts": sel_counts} if a.post_select else {})})
    print({s: round(v["total_bits"]) for s, v in results.items()}, "uniform",
          round(results[a.seeds[0]]["uniform_bits"]), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
