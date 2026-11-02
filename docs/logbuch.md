# Logbuch

Neueste Einträge oben. Kurz: Datum, was, Ergebnis, Verweis auf `runs/…`.

## 2026-11-02

- Kalibrierung `wd` (λ = 1, StableMax, 5 Seeds × 20k, `runs/s0a/20261102-121039_kalib_wd`): alle Seeds erreichen Test ≥ 0,99 erstmals bei Schritt 2.200–2.400, **brechen danach wiederholt ein** (3–6 Einbrüche unter 0,9) und sind erst ab 16.700–19.600 dauerhaft ≥ 0,99. Endnorm 62–76. Folgerung fürs Protokoll: Endpunkt „erster Treffer“ ist instabil; zusätzlich „stabil gegrokt“ (≥ 0,99 bis Laufende bzw. ≥ 2.000 Schritte am Stück) und Anteil der Zeit ≥ 0,99. Budget 20k zu knapp für stabilen Endpunkt → 40k.
- Kalibrierung `subspace` (Rang 16/4/64) und `bitnet` gestartet (`scripts/kalib_struct.sh`).

- Pilot-Bench Qwen2.5-Coder-1.5B-Instruct-4bit, ohne Bibliothek, 6 Familien × 4 Aufgaben × 4 Samples (`runs/s1/*bench_prefill`): Pass-Rate gcd 0,94 · count_bits 0,88 · word_sorting 0,64 · letter_counting 0,63 · prime_factorization 0,50 · base_conversion 0,00 (Stichprobe danach 2/8). Verifizierte Programme kosten 1–18 Bit unter dem Modell. 16 Samples dauern 9–26 s bei geteilter GPU. Engineering: Ohne Code-Prefill antwortete das Modell oft direkt (z. B. „6“); Prefill ```` ```python ```` behebt das (count_bits 0 → 0,88).
- Beobachtung für das Design: Weil das Modell die Programme selbst gesampelt hat, sind sie unter ihm sehr wahrscheinlich (wenige Bit). L(y|x,B) ist damit klein gegenüber L(B); der Kompressionsfortschritt muss vor allem aus kürzeren/häufiger richtigen Lösungen auf schwereren Aufgaben kommen. Prüfen: Bits über alle n Samples (Erwartungswert) statt nur der besten Lösung.

- Kalibrierung `none` (StableMax, keine Regularisierung, 5 Seeds × 20k, `runs/s0a/20261102-115946_kalib_none`): **kein Seed groked**; Train 100 %, Test 1–3 %, Gewichtsnorm ≈ 510–530. Widerspricht dem Bericht von Prieto et al. für unsere Einstellungen (lr 1e-3, Full-Batch, 40 % Train); Ursache offen (dort teils längeres Training/andere Optimierer). Kalibrierung, kein Befund.
- Kalibrierung `mdl` über β ∈ {0,001; 0,01; 0,1; 1} gestartet (Seeds 100–102, `scripts/kalib_mdl.sh`). Anfangs-KL ≈ 2,2 Mbit (≈ 5,6 Bit/Gewicht bei σ₀ = e⁻⁶).

- Bauphase/Messphase-Regel von Jonas freigegeben (AGENTS.md).
- apple/container installiert; Sandbox-Backend `container`: VM ohne Netz, 4 CPUs, 2 GB, Start 0,7 s; Netzsperre per Test geprüft (E17 umgesetzt).

- Repo angelegt, vier Recherchen ausgewertet (`docs/recherche.md`), Entscheidungen E1–E26 (`docs/entscheidungen.md`).
- Kern und Training gebaut, 16 Tests grün.
- Befund Engineering: Metal-GPU ist nicht bitgenau deterministisch (Abweichung ca. 6e-8); Tests prüfen mit Toleranz.
- Befund Engineering: `mx.vmap` über 10 Seeds ≈ 10 × Zeit eines Seeds (4,7 vs. 47,7 Schritte/s). Kein Parallelgewinn bei diesem Modell.
- Kalibrierungslauf `none` / `wd` gestartet.
