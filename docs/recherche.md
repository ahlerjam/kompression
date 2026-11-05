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

## Neuheitsprüfung (2026-11-04, drei Agenten; Kernquellen selbst geprüft: 2606.11417, 2510.01265, 2506.11058, 2009.07624)

**Stufe 1 (LLM, Wach-Schlaf, Bibliothek, Proposer-Belohnung Kompressionsfortschritt):**
- Korrektur der Aussage oben: Kompressionsfortschritt als Belohnung für Aufgabenwahl GIBT es: [Mittal & Gupta 2026, Signed Compression Progress on a Sealed Audit](https://arxiv.org/abs/2606.11417) (Belohnung = Abnahme eines versiegelten Audit-Verlusts, mit Vorzeichen; ARC-TGI-Generatoren; kein LLM-Programm-/Bibliothekssystem). Gefährlichste Vorarbeit.
- MDL-Kriterium für Code-Bibliotheken mit LM: [Kovačič, …, Ellis 2025, Librarian](https://arxiv.org/abs/2506.11058) (MDL korreliert am besten mit guten Refaktorierungen; Formel nur im Volltext).
- Wach-Schlaf + L(B)+L(Programme|B): DreamCoder, LILO, Stitch. Proposer/Solver mit Learnability: Absolute Zero. Eingefrorenes LLM + Curriculum + Skill-Bibliothek: [Voyager](https://arxiv.org/abs/2305.16291). Lernfortschritt-Lehrer: SOAR, MAGELLAN, [SEC](https://arxiv.org/abs/2505.14970).
- Nach Suche neu: Kompressionsfortschritt einer Programmbibliothek (Bit unter eingefrorenem LLM) als Auswahl-Belohnung für selbst erzeugte Aufgaben, im kontrollierten Vergleich mit Learnability bei identischem Mechanismus und Budget. Vorbehalt: 2026er Arbeiten unvollständig indexiert.

**Gewichtsseite (0a, 0b, 0a+, Stufe 2):**
- 0a: Optimierungsschwäche variationaler Codes bekannt ([Blier & Ollivier 2018](https://arxiv.org/abs/1802.07044), [Shaw et al. 2509.22445](https://arxiv.org/abs/2509.22445)).
- 0b: Transfer/Vergessen über prequentielle Codelänge: [Zhang et al. 2020, Measuring Information Transfer](https://arxiv.org/abs/2009.07624) (Details im Volltext prüfen). Warmstart-Transfer zwischen Grokking-Aufgaben: [Jia 2609.18078](https://arxiv.org/abs/2609.18078) (nur Abstract, prüfen). Struktur-/Normkontrollen und Herkunftsvergleich MDL/WD/BitNet nicht gefunden. Erklärung mul: Subtraktion teilt den Fourier-Algorithmus der Addition, Multiplikation nicht ([2402.16726](https://arxiv.org/abs/2402.16726)); Norm/Lazy-Regime ([Omnigrok](https://arxiv.org/abs/2210.01117), [Kumar et al.](https://arxiv.org/abs/2310.06110)); Warmstart-Schaden und Shrink & Perturb ([Ash & Adams](https://arxiv.org/abs/1910.08475)).
- 0a+: Zyklisches Komprimieren/Weitertrainieren bekannt: [DSD](https://arxiv.org/abs/1607.04381), [AC/DC](https://arxiv.org/abs/2106.12379), [Cyclic Sparse Training](https://arxiv.org/abs/2406.02773) (Zyklus hilft teils auch ohne Pruning → Pflichtkontrolle: gleicher Zyklus ohne Kompression, bei uns `--post wd`/`free` mit gleichen Optimizer-Resets); Pruning beschleunigt Grokking ([Grokking Tickets](https://arxiv.org/abs/2310.19470)). Variationales MDL als Schlafphase, prequentiell gemessen: nicht gefunden.
- Stufe 2: Kern besetzt durch [RLP, Hatamizadeh et al., ICLR 2026](https://arxiv.org/abs/2510.01265) (RL-Belohnung = Informationsgewinn für Vorhersage künftiger Tokens, Qwen3-1.7B, Nemotron-12B; nicht als Kompression formuliert) und [L2T](https://arxiv.org/abs/2505.10425). Neu allenfalls: Belohnung = zweiteilige Codelänge inkl. eigener Gewichte, prequentiell gemessen (schmaler Unterschied).
- Positionspapier: [Herrmann & Schmidhuber 2026](https://arxiv.org/abs/2605.14831).

**Faire Formulierung (TRC):** Bausteine bekannt (DreamCoder-Wach-Schlaf, MDL-Bibliotheken, Kompressionsfortschritt als Curriculum-Belohnung, Informationsgewinn-RL); Beitrag = kontrollierte, vorregistrierte Tests in Bit (Struktur-/Normkontrollen, Vergleich mit Learnability bei identischem Mechanismus).

## Nachtrag 2026-11-05: Neuheit Zeitplan-Effekt (Prüfstand E30) und Richtungssuche

Befund (Prüfstand, kein Hauptlauf): Schlaf „erst stark schrumpfen (L2 λ 10), dann schwach“ schlägt dauerhaft starkes L2 auf p = 47/97 in allen Seeds; mdl statt schwachem L2 in den späten Phasen bringt nach fairer Abstimmung keinen Vorteil (Logbuch 2026-11-05).

Neuheit: **mäßig.** Bekannt: stärkerer Weight Decay beschleunigt Grokking ([Nanda et al. 2023](https://arxiv.org/pdf/2301.05217), [Lyu et al. 2023](https://arxiv.org/pdf/2311.18817)); Weight-Decay-Pläne allgemein: optimaler WD geht bei langem Training gegen 0 ([Xie et al., Stable Weight Decay](https://arxiv.org/abs/2011.11152)), [Scheduled Weight Decay, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/040d3b6af368bf71f952c18da5713b48-Abstract.html), [Cyclical Weight Decay](https://arxiv.org/abs/2202.08835); WD nach dem Auswendiglernen erhöhen beschleunigt Grokking auf (a+b) mod 97 ([Thermodynamic Weight Decay 2026](https://arxiv.org/html/2607.20552), ohne Kontrolle mit festem Zuschlag); Theorie der WD-Zeitskala ([2607.23967](https://arxiv.org/abs/2607.23967), [2606.13753](https://arxiv.org/html/2606.13753)). Nicht gefunden: „stark, dann schwach“ im Wach-Schlaf-Zyklus, gemessen als prequentielle Codelänge; Weight-Decay-Pläne für prequentielles MDL sind laut [Bornschein et al. 2023](https://arxiv.org/abs/2210.07931) offen. → Kleiner, regularisierungsseitiger Beitrag, kein Kompressionsbefund.

Wo Kompression strukturell gegen Standard gewinnen kann (Richtungssuche):
- MDL-Ziel macht die perfekt verallgemeinernde Lösung zum Optimum, wo L1/L2/Early Stopping es nicht tun; Gradientenabstieg findet sie aber oft nicht (Suchproblem) ([Lan et al. 2022](https://arxiv.org/abs/2111.00600), [Lan et al. ACL 2024](https://arxiv.org/abs/2402.10013), [Abudy et al. 2025](https://arxiv.org/abs/2505.13398)).
- [CompressARC (Liao & Gu 2025)](https://arxiv.org/abs/2512.06104): 76 k Parameter, ohne Vortraining, nur MDL je Rätsel, ~20 % ARC-AGI-1.
- MDL für Transformer: asymptotisch optimale Beschreibungslängen-Ziele ([Shaw et al., ICLR 2026](https://arxiv.org/pdf/2509.22445)); Längen-Generalisierung in Arithmetik bisher über Positionscodierung/Looping gelöst ([Fan et al. ICLR 2025](https://mlanthology.org/iclr/2025/fan2025iclr-looped)), nicht über MDL (keine Arbeit gefunden).

## Recherche E31 (2026-11-05): Regeln aus wenigen Beispielen

Vier Teilrecherchen (über 150 Arbeiten, Schwerpunkt 2025–2026, arXiv-API zeitweise gedrosselt; Kennzeichnung Volltext/Abstract in den Dateien): `docs/recherche_e31/arc_kleine_netze.md`, `laengen_verallgemeinerung.md`, `programmsuche.md`, `theorie_dateneffizienz.md`. Zusammenfassung mit Plan im Claude Doc „Regeln aus wenigen Beispielen – Recherche und Plan“ (https://claude.ai/artifact/BKeCAFK7ikeJrn7WETH5rU). Kern: MDL-Ziel gelöst, Suche offen (Shaw et al. 2509.22445, Abudy et al. 2505.13398); Extrapolation mit wenigen reinen Ein-/Ausgabepaaren ohne vorgeführte Zwischenschritte nicht gefunden; empfohlener erster Strang K1 + K2.
