# Recherche-Notizen (2026-11-02)

Kurzfassung der vier Recherchen. Werte aus Community-Benchmarks sind vor Nutzung selbst nachzumessen.

## Architektur
- MoE lohnt bei kleinen Modellen nur mit viel mehr Daten; Befunde widersprüchlich. [Joint MoE Scaling Laws](https://arxiv.org/html/2502.05172v2), [Li et al.](https://arxiv.org/html/2506.12119v1)
- MoE unter RL braucht Routing Replay; Problem nicht gelöst. [GSPO](https://arxiv.org/pdf/2507.18071), [R3](https://arxiv.org/abs/2510.11370v1), [RSPO](https://aclanthology.org/2026.acl-long.1165.pdf)
- Modelle: [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (ca. 80 tok/s, M4 Pro, Community), [Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base)
- Absolute Zero erst ab ca. 3B belegt; [Replikation](https://www.lesswrong.com/posts/HcKpxEHX5b4JcRbPz/reproducing-absolute-zero) brauchte Format-SFT.
- Wachsende MoE: [BTX](https://arxiv.org/abs/2403.07816), [DynamicMoE](https://arxiv.org/html/2511.18987v1); kein MDL-Kriterium publiziert.

## MDL-Ziel und Messung
- [Complexity Dynamics of Grokking](https://arxiv.org/html/2412.09810v2): Komplexität per Low-Rank + Quantisierung + bzip2 unter Loss-Toleranz.
- [Shaw et al. 2026](https://arxiv.org/html/2509.22445v3): variationale Codelänge mit GMM-Prior; Optimierer finden die kurze Lösung nicht.
- [Lan et al. 2024](https://arxiv.org/abs/2402.10013): MDL nicht differenzierbar, genetischer Algorithmus.
- Prequential: [Blier & Ollivier](https://ar5iv.labs.arxiv.org/html/1802.07044), [Voita & Titov](https://ar5iv.labs.arxiv.org/html/2003.12298), [Bornschein et al.](https://arxiv.org/abs/2210.07931)
- Konfunde: [StableMax](https://arxiv.org/abs/2501.04697), [Omnigrok](https://arxiv.org/abs/2210.01117)

## Self-Play und Bibliothekslernen
- Keine Arbeit nutzt Kompression als Proposer-Belohnung. Am nächsten: [Signed Compression Progress on a Sealed Audit](https://arxiv.org/abs/2606.11417), [SOAR](https://arxiv.org/abs/2601.18778), [Skill Self-Play](https://arxiv.org/abs/2607.22529).
- Form der Proposer-Belohnung wirkt oft wenig: [2510.27072](https://arxiv.org/html/2510.27072)
- Kritik Bibliothekslernen: [Library Learning Doesn't](https://arxiv.org/abs/2410.20274), [2504.03048](https://arxiv.org/abs/2504.03048), [2507.22069](https://arxiv.org/abs/2507.22069)
- Werkzeuge: [Absolute Zero](https://github.com/LeapLabTHU/Absolute-Zero-Reasoner), [Stitch](https://github.com/mlb2251/stitch), [ReGAL](https://github.com/esteng/regal_program_learning), [Reasoning Gym](https://github.com/open-thought/reasoning-gym), [apple/container](https://github.com/apple/container)

## Engineering-Stack
- Versionen (PyPI, Oktober 2026): mlx 0.32.4, mlx-lm 0.32.2, mlx-lm-lora 5.10.2, torch 2.15.0.
- [MLX compile](https://ml-explore.github.io/mlx/build/html/usage/compile.html), [Function transforms](https://ml-explore.github.io/mlx/build/html/usage/function_transforms.html), [PyTorch MPS notes](https://docs.pytorch.org/docs/2.15/notes/mps.html)
