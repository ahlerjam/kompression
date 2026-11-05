# Recherche E31: Längen-Verallgemeinerung und algorithmische Extrapolation (2023–2026)

Stand: 2026-11-05. Quellen: arXiv-API und arXiv-Suche (Abstracts) sowie die HTML-Volltexte der wichtigsten Arbeiten über WebFetch.
Kennzeichnung: **[V]** = Zahl aus dem geöffneten Volltext, **[A]** = nur aus dem Abstract, **(?)** = unsicher oder nicht gefunden.
WebFetch fasst lange Seiten zusammen und liest höchstens 100.000 Zeichen. Einzelzahlen vor der Übernahme in ein Protokoll im PDF gegenprüfen.

---

## 1. Die wichtigsten Arbeiten

Spalte „Daten“ = Zahl der Trainingsbeispiele. Bei „online“ erzeugt der Generator unbegrenzt neue Beispiele, eine feste Datenmenge gibt es dann nicht.

| # | Arbeit | Autoren | Datum | Kernidee | Konkrete Zahlen | Bezug zu E31 |
|---|---|---|---|---|---|---|
| 1 | [Length Generalization in Arithmetic Transformers](https://arxiv.org/abs/2306.15400) | Jelassi, d'Ascoli, Domingo-Enrich et al. | 2023-06 | Relative Positionen für Addition; „Priming“ mit wenigen langen Beispielen für Multiplikation | [V] **N_train = 5.000** (Addition). 5 → 15 Stellen: 97,2 % (Base, RPE), bei 20 Stellen nur 21 %. APE: 0 %. Multiplikation 5×3 → 35×3: **50 lange Priming-Beispiele (1 %)** ergeben fast 100 %, unter 25 klappt es nicht. Bedarf an Priming-Beispielen wächst logarithmisch: 30 bei 10³, 70 bei 10⁴, 100 bei 10⁵ Trainingsbeispielen. Modelle 6–10 Schichten, d = 512–1024 | **Einzige ältere Arbeit mit kleiner, fester Datenmenge.** Zeigt: wenige lange Beispiele sind sehr wertvoll |
| 2 | [Teaching Arithmetic to Small Transformers](https://arxiv.org/abs/2307.03381) | Lee, Sreenivasan, Lee, Lee, Papailiopoulos | 2023-07 | Datenformat (umgekehrt, Scratchpad) bestimmt die Stichprobeneffizienz | [V] NanoGPT 10,6 M. 3-stellige Addition: plain nie 100 % (85–88 % bei 10k), reverse ab ca. 2,5k Beispielen, vereinfachter Scratchpad ca. 2k, detaillierter Scratchpad **ca. 1k**. Keine Längen-Verallgemeinerung über trainierte Stellenzahlen hinaus | Referenzwerte für Daten-Effizienz **innerhalb** der Verteilung |
| 3 | [What Algorithms can Transformers Learn? (RASP-L)](https://arxiv.org/abs/2310.16028) | Zhou, Bradley, Littwin et al. | 2023-10 | RASP-Vermutung: Transformer verallgemeinern in der Länge, wenn ein kurzes RASP-L-Programm die Aufgabe für alle Längen löst | [A] Verbesserungen bei Parität und Addition. Zahlen nicht geöffnet (?) | Theoretischer Kompass: welche Regeln „leicht“ sind |
| 4 | [Transformers Can Achieve Length Generalization But Not Robustly](https://arxiv.org/abs/2402.09371) | Zhou, Alon, Chen et al. | 2024-02 | FIRE-Positionen, umgekehrtes Format und Index-Hinweise für Addition | [V] 40 → 100 Stellen: >98 % (2,5×). 30 → 45 (1,5×), 20 → 25, 10 → keine Verallgemeinerung. Daten: **30 M** (1–40 Stellen) bis 120 M (1–10). 25 M Parameter, 50k Schritte. Starke Streuung über Seeds und Datenreihenfolge | Warnung: **weniger Trainingslänge, weniger Extrapolation**; Seeds streuen stark |
| 5 | [Transformers Can Do Arithmetic with the Right Embeddings (Abacus)](https://arxiv.org/abs/2405.17399) | McLeish, Bansal, Stein et al. | 2024-05 | Gleiche Positions-Einbettung je Stellenwert, mit zufälligem Offset; dazu Looping und Input Injection | [V] Addition 20 → 100/120 Stellen: **99,1 %** OOD (geloopt 8×2, 64 M; 16×1 standard 122 M). Daten **ca. 20 M Beispiele**, 1 GPU (RTX A4000) für 24 h (ca. 8 EFLOP). Multiplikation bis 15 Stellen: fast perfekt nur in der Verteilung, 8 GPUs × 24 h. Sortieren 10 → 30 Elemente: **4,48 % exakt** | Stand der Technik bei Addition, aber datenhungrig. Sortieren scheitert |
| 6 | [Position Coupling](https://arxiv.org/abs/2405.20671) | Cho, Cha, Kim et al. | 2024-05 | Gleiche Positions-ID für Ziffern gleicher Bedeutung (Aufgabenstruktur fest eingebaut) | [V] Addition 1–30 → 200 Stellen: **>95 %** (6,67×), 1-Schicht-Modell reicht. N×2-Multiplikation braucht ≥ 2 Schichten. Minesweeper-2D: Breite 5–9 → 12: >98 %, → 14: ca. 90 %. Datenmenge nicht gefunden (?) | Struktur per Hand vorgegeben, also kein Lernen der Regel |
| 7 | [Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count](https://arxiv.org/abs/2410.15787) | Cho, Cha, Cho et al. | 2024-10 | Scratchpad plus mehrstufiges Position Coupling | [V] Mehr-Operanden-Addition: 10 Stellen × 10 Operanden → 30 × 30 mit ≥ 90 % (Median). Multiplikation 10×10 → 20×15 mit ≥ 78,55 %. 63 M Parameter, **500k Beispiele**. Datenmenge von 20k bis 10M ohne deutlichen Effekt | Multiplikation: bestes Transformer-Ergebnis ohne Selbsttraining, aber nur ca. 2× |
| 8 | [Universal Length Generalization with Turing Programs](https://arxiv.org/abs/2407.03310) | Hou, Brandfonbrener, Kakade et al. | 2024-07 | CoT als Turingband-Kopie mit kleinen Änderungen, Hard-ALiBi | [V] Addition 50 → 100: 98 %. Multiplikation n×1 und n×3, 50 → 100: 97,1 % bzw. 96,8 %. SGD-Simulation 50 → 80: >95 %. TM-Simulation 2–3× (nur Ein-Schritt). 150 M Parameter, 200k–400k Schritte | Allgemeines Rezept, aber großes Modell und lange CoT |
| 9 | [Looped Transformers for Length Generalization](https://arxiv.org/abs/2409.15647) | Fan, Du, Ramchandran, Lee | 2024-09 (ICLR 2025) | Geloopter Block, Schrittzahl wächst mit der Länge (n-RASP-L), adaptiver Stopp | [V] Parität 1–20 → 40+: fast perfekt. Kopieren, Addition und weitere: Training bis 19, Test 30–35 ca. 1,0. Binäre Multiplikation 1–11 → 16. GPT-2-Block d = 256, NoPE. **Online-Daten, 100k Schritte × Batch 64 (ca. 6,4 M Beispiele)** | Standard-Kontrolle für E31 (steht schon im Plan). Braucht Schrittzahl als Überwachung |
| 10 | [How Far Can Transformers Reason? Globality Barrier, Inductive Scratchpad](https://arxiv.org/abs/2406.06467) | Abbe, Bengio, Lotfi et al. | 2024-06 | „Globalitätsgrad“ bestimmt die Lernbarkeit; induktiver Scratchpad hilft | [A] bis 6× Längen-Verallgemeinerung bei einigen Arithmetik-Aufgaben | Theorie, warum lange Syllogismus-Ketten scheitern |
| 11 | [Discrete Neural Algorithmic Reasoning (DNAR)](https://arxiv.org/abs/2402.11628) | Rodionov, Prokhorenkova | 2024-02 (ICML 2025) | GNN mit diskreten Zuständen, harter Attention, Hinweis-Überwachung | [V] BFS, DFS, Prim, Dijkstra, MIS, Exzentrizität: **100 % auf 16 bis 1600 Knoten**. Training ≤ 16 Knoten. **Richtige Übergänge schon aus Graphen mit 4–5 Knoten lernbar.** 1000 Schritte × 32 Graphen. Ohne Hinweise: BFS 64 Knoten Graph-Ebene 0 % | **Sehr nah an E31**: diskret, perfekte Extrapolation, kleine Fälle reichen. Braucht aber Zwischenschritte (Hints) |
| 12 | [A Generalist Neural Algorithmic Learner (CLRS, Triplet-GMPNN)](https://arxiv.org/abs/2209.11142) | Ibarz, Kurin, Papamakarios et al. | 2022-09 | Ein GNN für 30 CLRS-Algorithmen | [V] Train n ≤ 16, Test n = 64: Mittel **74,1 %** micro-F1. Strings 49 %, Suche 59 %, Sortieren 60 %. 11/30 Aufgaben über 90 %. Ca. 320k Trajektorien je Aufgabe | Referenzlinie für den NAR-Strang (vor 2023, aber Basis) |
| 13 | [End-to-end Algorithm Synthesis / Deep Thinking](https://arxiv.org/abs/2202.05826) | Bansal, Schwarzschild, Borgnia et al. | 2022-02 | Rekurrentes CNN mit Recall und Progressive Loss: mehr Iterationen beim Test | [V] Präfixsummen 32 → 512 Bit: 97,1 %. Labyrinthe 9×9 → 59×59: 97,3 %, → 201×201: 74 % (Fehler meist 1 Pixel) | Klassiker „Easy-to-hard“; Basis des Easy-to-hard-Benchmarks |
| 14 | [Self-Improving Transformers](https://arxiv.org/abs/2502.01612) | Lee, Cai, Schwarzschild, Lee, Papailiopoulos | 2025-02 | Modell erzeugt Lösungen für etwas längere Fälle, filtert per Mehrheitsvotum oder Länge, trainiert darauf | [V] 14 M Parameter (6 Schichten, d = 384). Start: **2 M Beispiele** (Addition), 5 M (Multiplikation, Labyrinth), dann **50k pro Runde**. Reverse-Addition 16 → 100+ Stellen. Forward-Addition → 70 (>98 %). Multiplikation 5×5 → 10×10 in 19 Runden. Labyrinth 9 → 30 Hops. Ohne Filter: 6×6-Multiplikation nur 13,7 % | Entspricht K3 im Plan (Selbst-Curriculum) **und existiert bereits**; Startdaten aber Millionen |
| 15 | [Extrapolation by Association](https://arxiv.org/abs/2506.09251) | Cai, Lee, Schwarzschild, Oymak, Papailiopoulos | 2025-06 | Längen-Verallgemeinerung überträgt sich von einer verwandten Hilfsaufgabe mit längeren Beispielen | [A] Arithmetik, Strings, Labyrinth. Zahlen nicht geöffnet (?) | Relevant für „neue Regel aus wenig Daten, wenn verwandte Regel lang gesehen“ |
| 16 | [The Imitation Game: TAIL](https://arxiv.org/abs/2507.13332) | Hua, Zhang, Lyu et al. | 2025-07 | CoT, die eine Turingmaschine nachahmt (atomare Schritte, expliziter Speicherzugriff) | [A] Qwen2.5-7B, 18 Aufgaben aus 8 Algorithmenklassen, besser als DeepSeek-R1. Zahlen (?) | Groß, nicht Mac-tauglich zum Training |
| 17 | [Exact Learning of Arithmetic with Differentiable Agents (DFST)](https://arxiv.org/abs/2511.22751) | Papazov, D'Angelo, Flammarion | 2025-11 | Differenzierbare endliche Transduktoren, trainiert auf Aktionsfolgen eines Experten-Agenten auf einem Gitter | [V] **Trainingsmengen: binäre Addition 20 Beispiele, dezimale Addition 225, binäre Multiplikation 750, dezimale Multiplikation 10.000.** Training ≤ 3 Stellen (Addition) bzw. ≤ 5 (Multiplikation). Fehlerfrei getestet bis ≥ 3.850 / 2.450 / 600 / 180 Stellen (Grenze war der Speicher). 1k–162k Parameter. 0,5–3 M Iterationen | **Stärkster Befund zu „wenige Beispiele + Länge“**. Braucht aber Experten-Trajektorien (sehr dichte Überwachung) |
| 18 | [Transformers Provably Learn CoT with Length Generalization](https://arxiv.org/abs/2511.07378) | (nicht geöffnet) | 2025-11 | Beweis zur Längen-Verallgemeinerung mit CoT | nur Titel gesehen (?) | Theorie |
| 19 | [To Infinity and Beyond: Tool-Use for SSMs](https://arxiv.org/abs/2510.14826) | Malach, Saremi, Williamson et al. | 2025-10 | SSMs können lange Ausgaben nicht, mit Werkzeugzugriff (externes Gedächtnis) aber beliebige Länge | [A] Arithmetik, Reasoning, Code. Zahlen (?) | Externer Speicher als Lösung |
| 20 | [Length Generalization Bounds for Transformers](https://arxiv.org/abs/2603.02238) | Yang, Bergsträßer, Zetzsche, Chiang et al. | 2026-02 | Keine berechenbaren Schranken für C-RASP mit 2 Schichten; für positives C-RASP exponentielle Schranke | [A] Theorie | Wie viel Trainingslänge reicht beweisbar? |
| 21 | [Barriers to Universal Reasoning With Transformers](https://arxiv.org/abs/2604.25800) | Kraus, Sarrof, Yao, Koller et al. | 2026-04 | Mit festem Alphabet und Standard-PE bleibt CoT bei TC⁰; „Wegweiser-Tokens“ und Kodierung nur der Änderungen erlauben längen-verallgemeinernde TM-Simulation | [A] Theorie und empirische Hinweise | Erklärt, warum Kopieren und „letztes Vorkommen“ scheitern |
| 22 | [On the Mirage of Long-Range Dependency (Multiplikation)](https://arxiv.org/abs/2603.29069) | Wei | 2026-03 | Multiplikation als 2D-Gitter, jeder Schritt lokal (3×3), dann ein neuronaler zellulärer Automat | [A] **321 Parameter**, perfekte Längen-Verallgemeinerung bis **683×** Trainingsbereich. Transformer (6.625 Parameter), RoPE und Mamba scheitern mit gleicher Darstellung. Datenmenge (?) (HTML nicht verfügbar) | **Darstellung schlägt Architektur.** Kleines Modell, mac-tauglich |
| 23 | [Length Generalization with Log-Depth Recurrent Units](https://arxiv.org/abs/2605.26035) | Pert, Alrajeh, Russo | 2026-05 | Assoziativ-verzerrter Operator per paralleler Reduktion | [A] 21 reguläre Sprachen: 100 % OOD bei 18, ≥ 99,9 % bei 3 | Stand für reguläre Sprachen |
| 24 | [Stabilizing Extrapolation in Looped Transformers via Learned Stochastic Stopping](https://arxiv.org/abs/2606.29983) | Kuo, Chayti, Reizinger, Brendel et al. | 2026-06 | Hohe OOD-Streuung kommt von der Scheinkorrelation Länge ↔ Loop-Zahl; zufällige Loop-Zahl hilft | [A] Binäre Addition, Dyck-1, Unique Set, Kopieren | Wichtig für faire Looped-Kontrolle |
| 25 | [The Automaton Underneath (Householder-RNN)](https://arxiv.org/abs/2609.18966) | Howe | 2026-07 | Additiver Eingabepfad b_t ist ein „parasitärer Attraktor“. Ohne ihn lernt das lineare RNN den exakten Automaten | [V] **4.000 Trainingssequenzen** der Länge 32, 40 Epochen (ca. 320 Schritte), d = 112. Test bis 512 (16×): Parität, S₄, A₅, S₅ jeweils **1,00** (ohne b), mit b 0,20–0,50. **202 Läufe in 53 min auf einer RTX 3080**, vorregistriert | **Direkt nachrechenbar auf dem Mac**; kleine Datenmenge plus Längen-Verallgemeinerung |
| 26 | [When Does Recurrence Become an Algorithm?](https://arxiv.org/abs/2607.20594) | Zhang, Hu, Peng, Xie | 2026-07 | Geloopte Transformer lernen eine „Frontier“ (v ≈ n_train/T_train Positionen pro Loop). Weight-Tying wählt den seriellen Algorithmus | [V] 1,6–25 M Parameter. S₅ scheitert an der Ordnung 120 (> 40k Schritte für n = 2), ein Operator-zuerst-Curriculum löst es in 10/10 Seeds (≥ 0,92 bei 2×). A₅ von Grund auf: 8/22 Seeds. Easy-to-hard-Benchmark (10k Beispiele, 32 Bit): 1,00 bis 128 Bit, 0 bei 256/512. 8× RTX 3090, Minuten bis 1,5 h pro Lauf | Klare Rezepte und Grenzen für Looping |
| 27 | [Universal Transformers for Circuit Computations](https://arxiv.org/abs/2608.31067) | Ito, Puri, Campbell, Ram | 2026-08 | UT mit Tiefen-PE und harter Attention reduziert pro Iteration eine Schaltkreis-Ebene | [V] **280 Parameter** (Boolesch), 924 (modulare Arithmetik), 4.416 (ListOps). Training Tiefe 1–2, Test Tiefe 3–30: **100 %**. 9–10/10 Seeds. Eingaben voll geklammert. Datenmenge (?) | Kleinst-Modell, exakte Regel. Addition mit Übertrag ausdrücklich nicht abgedeckt |
| 28 | [From Shortcut Learning to Discrete Neural Insertion Sort](https://arxiv.org/abs/2609.31114) | Mylonas, Spyropoulos | 2026-09 | CLRS-Baseline lernt eine Abkürzung statt Insertion Sort; diskrete Variante mit Kettengraph | [V] Training Länge 16 → Test 64 und 128: **100 % sortiert**. Ohne Überwachung des globalen Schleifenzustands scheitert es schon bei Länge 16. Online, 32 Sequenzen je Schritt | Sortieren lösbar, aber nur mit viel eingebautem Vorwissen |
| 29 | [Beyond the Training Horizon: Looped Transformers](https://arxiv.org/abs/2609.33144) | Liang, Jin, Pan | 2026-09 | Mechanistik zweier Loop-Trainingsschemata; beide brechen in großer Tiefe ein | [A] Polynom-Iteration, endliche Zustandskomposition, Wissensgraph-Traversierung | Grenzen des Loopings |
| 30 | [Algebraic Decomposition Theory for Transformer LG](https://arxiv.org/abs/2608.13433) | Yang, Veseli, Barloy, Cadilhac et al. | 2026-08 | Vollständige Charakterisierung, bei welchen regulären Sprachen Transformer in der Länge verallgemeinern (C-RASP); Entscheidungsalgorithmus | [A] Theorie plus Experimente | Vorhersage: welche Sprachaufgaben der Standard schafft |
| 31 | [Learning to Extrapolate in Depth with Looped Transformers](https://arxiv.org/abs/2610.14456) | Yang, Huang, Wang, Mirzasoleiman | 2026-10 | Graph-Zusammenhang: endliches Extrapolationsfenster der Loop-Tiefe, mit Schranken | [A] Theorie, Experimente bestätigen die Fenster | Graphen: Looping hat beweisbare Grenzen |
| 32 | [Transformers Learn Two Competing Sorting Algorithms](https://arxiv.org/abs/2610.26974) | Gatmiry, Henry, Su, Abbe et al. | 2026-10 | 2-Schicht-Transformer finden je nach Initialisierung eine von zwei Lösungen; nur die zweistufige verallgemeinert | [V] Training K = 16, Test bis N = 256/512: zweistufige Lösung fast perfekt. Nur ca. **26 % der Seeds** finden sie (Basis); per Init-Skala steuerbar (0 % bis 100 %). Online, 60k Schritte × 4096. Bei K = 4 verallgemeinert kein Seed | **Seed-Lotterie** ist zentral: Gate mit ≥ 5 Seeds ist berechtigt |
| 33 | [PAR: Progress-Aligned Recurrence](https://arxiv.org/abs/2610.27628) | Wang, Hu, Dong, Lin et al. | 2026-10 | Iteration k soll den Zustand nach k Operationen treffen, dazu Norm-Anker | [V] Qwen2.5-0.5B: Swap-S₅ ≤ 32 → 512 Züge: 95,5 % (RLT 55,7 %, CoT 0,8 %). S₅ 16 → 128: 99,3 %. Graph-Erreichbarkeit nur 2,6×. **ca. 400k Beispiele**, 2× RTX 4090 | Zwischenzustände überwachen hilft stark |
| 34 | [Putting Transformer Theories to the Test](https://arxiv.org/abs/2610.32104) | Altabaa, Schnabel, Tomlinson | 2026-10 | 7 Theorien gegen Experimente zu Stichprobenkomplexität und Längen-Verallgemeinerung | [V] N = 256 bis 65.536 Beispiele. Parität scheitert sogar in der Verteilung (49,7–59 % bei n = 8, N = 65k). Binäre Addition ohne LG. 2-Hop nur n = 8, 3-Hop nie. Erreichbarkeit scheitert schon bei n = 8. Keine Theorie erklärt alles | **Einzige Arbeit, die Stichprobenkomplexität und LG systematisch zusammen misst** |
| 35 | [Quantitative Generalization Bounds … A Few Long Strings can Beat Many Short Ones](https://arxiv.org/abs/2610.35715) | Henry, Rizvi-Martel, Hahn | 2026-10 | Gemeinsame Schranke für Stichproben- und Längen-Verallgemeinerung | [V] Längere Trainingsstrings können **exponentiell effizienter** sein als mehr Stichproben (Bedarf ~ \|Σ\|^C bei K = C, Faktor ≥ \|Σ\|^C/2 weniger bei K ≈ \|Σ\|·C). Experimente bestätigen es qualitativ (Test bei 5× Länge, Schwelle 75 %) | **Kerntheorie für E31**: Trainingslänge statt Beispielzahl |
| 36 | [A Fine-Grained Hierarchy and Benchmark for State Tracking (StateBench)](https://arxiv.org/abs/2611.03094) | Petty, Merrill, Sabharwal | 2026-11 | 4 Stufen: r-trivial, aperiodisch, auflösbar, allgemein endlich | [V] 275 M-Modelle, Training 8–256 → Test bis 64k. RoPE ca. 10×, NoPE bricht sofort ein. GDN generalisiert fast perfekt auf den ersten drei Stufen. Frontier-LLMs auf Zufallsniveau bei „auflösbar“ ab ca. 10k–64k Tokens. ca. 1× Chinchilla-Daten | Benchmark-Hierarchie für Zustandsverfolgung |
| 37 | [Attending to Memory Writes Reduces Seed Variability](https://arxiv.org/abs/2611.04902) | Goo, Yun, Jung | 2026-11 | „Bounded delta attention“ über Update-Vorschläge im rekurrenten Speicher | [V] Modulo-5-Präfix, Training 1–40, Test 80: 70,8 ± 8,3 % gegen 69,2 ± 22,4 % (SD −63 %). Bei 160/320 deutlich schlechter. 1,2–1,4 M Parameter, online 2,56 M Sequenzen | Klein und mac-tauglich, aber schwacher Effekt (nur Streuung) |
| 38 | [A Positional Realizability Theory of LG in Single-Head Attention](https://arxiv.org/abs/2610.41218) | Han, Chai, Wang | 2026-10 | Vorab berechenbares α_L sagt LG für reine Positions-Köpfe voraus | [A] 2.625 Messungen, α_L trennt exakt. Ausdrücklich **nicht** für inhaltsabhängige Aufgaben wie Addition oder Sortieren | Theorie, begrenzter Nutzen |
| 39 | [Tiny Recursive Message Passing](https://arxiv.org/abs/2610.18649) | Salvatori, Forasassi, Toumi, de Felice | 2026-10 | TRM-Tricks (Deep Supervision, trunkierte Rekursion) auf GNNs | [A] **ca. 16k Parameter**, ca. 99 % Sudoku-Extreme, ca. 10 min H100 bzw. ca. 9 h CPU. Zero-shot auf größere Instanzen (N-Damen, SAT). Datenmenge (?) | Klein, auf CPU/Mac machbar |

Ergänzend gesehen, aber nicht vertieft (Abstract):
- [2610.07676 Exact-Solution Volume](https://arxiv.org/abs/2610.07676): Lösungsvolumen FIRST Θ(1), MAJORITY Θ(1/(n log n)), INDEX Θ(1/n³), PARITY 0. Erklärt, warum Parität mit Gradientenabstieg kaum zu finden ist.
- [2609.08851 LG via Compression](https://arxiv.org/abs/2609.08851): exponentiell bessere Stichprobenschranke für C-RASP+, polynomiell mit komprimierten Strings. Passt zum alten Kompressions-Strang.
- [2610.04518 LG Needs Proper Regularization](https://arxiv.org/abs/2610.04518): Weight Decay kann Extrapolation schaden, Dropout an der richtigen Stelle hilft.
- [2602.08857 Decompiling Transformers to RASP](https://arxiv.org/abs/2602.08857): Längen-generalisierende Transformer implementieren tatsächlich kurze RASP-Programme.
- [2602.07599 Rational Transductors](https://arxiv.org/abs/2602.07599): WFA-Rekurrenz plus Attention für Parität und S₅.
- [2610.24135 Inside the Loop](https://arxiv.org/abs/2610.24135): Ein geloopter 4-Schicht-Block × 8 löst Permutations-Traversierung in jedem Seed, der ungebundene gleich tiefe Stack nie.
- [2610.19768 Learning to Compute ≠ Retain](https://arxiv.org/abs/2610.19768): Training bei Loop 32, bei Loop 64 nur noch 22 % der Antworten richtig. Zufällige Überwachungsposition ergibt 99,8 %.
- [2608.03425 Complex State Propagator](https://arxiv.org/abs/2608.03425): Mod-3, 16 → 64.
- [2602.18508 Parallelizable NTM](https://arxiv.org/abs/2602.18508): P-NTM löst alle Aufgaben des Benchmarks perfekt auf ungesehenen Längen.
- [2602.13106 Which Algorithms Can GNNs Learn?](https://arxiv.org/abs/2602.13106): [V] hinreichende Bedingungen dafür, dass ein MPNN einen Algorithmus aus kleinen Instanzen lernt. **Bellman-Ford: Trainingsmenge K+1 Pfad-Instanzen** für K Schichten, Test bis 1024 Knoten ohne Fehleranstieg.

---

## 2. Stand der Technik je Aufgabe

### Addition (Dezimal, zwei Operanden)
- Stand der Technik: Position Coupling, 1–30 → 200 Stellen, >95 % [V]. Abacus plus Looping, 20 → 100–120 Stellen, 99,1 % [V]. Self-Improving, 16 → 100+ Stellen [V]. FIRE, 40 → 100, >98 %, aber nur 2,5× und seed-fragil [V].
- Datenmengen: 2 M bis 120 M Beispiele (Abacus ca. 20 M; FIRE 30 M; Self-Improving 2 M plus 50k je Runde).
- Mit wenigen Daten: DFST lernt aus **225 Beispielen** dezimale Addition fehlerfrei bis ≥ 2.450 Stellen [V], mit Experten-Trajektorien. Jelassi: 5.000 Beispiele, 5 → 15 Stellen 97 % [V]. Lee: ca. 1k–2,5k Beispiele, aber nur innerhalb der Länge [V].
- Mehr Operanden: 10×10 → 30×30, ≥ 90 % [V].

### Multiplikation
- Transformer ohne CoT: Abacus fast perfekt nur in der Verteilung, bis 15 Stellen [V]. Cho et al. 10×10 → 20×15 mit 78,6 % [V]. Bei 4×4 entscheidet das Format: Trennzeichen zwischen Ziffern heben die Genauigkeit von 1 % auf 89 % [A, 2610.06605].
- Mit CoT oder Struktur: Turing Programs n×1/n×3, 50 → 100 Stellen, ca. 97 % [V]. Self-Improving 5×5 → 10×10 [V]. Jelassi: 5×3 → 35×3 mit 50 Priming-Beispielen [V].
- Kleine Spezialmodelle: NCA auf 2D-Gitter, 321 Parameter, 683× [A]. DFST, binär aus 750 Beispielen ≥ 600 Stellen, dezimal aus 10k Beispielen ≥ 180 Stellen [V].
- Weiterhin offen: **N×N mit beiden Längen wachsend, ohne handgebaute Darstellung und ohne CoT**. Die Self-Improving-Autoren schreiben, reine Multiplikation ohne Zwischenschritte verallgemeinere schlecht [V].

### Sortieren
- Abacus: 10 → 30 Elemente, nur 4,48 % exakt [V].
- 2-Schicht-Transformer, NoPE, K = 16 → 256/512: fast perfekt, aber nur ca. 26 % der Seeds [V].
- CLRS-GNN: Sortieren OOD (n = 64) ca. 60 % micro-F1 [V]. Diskrete Insertion Sort, 16 → 128: 100 %, aber mit viel Vorwissen und Zusatz-Überwachung [V].

### Parität und modulares Zählen
- Looped Transformer 20 → 40+: fast perfekt [V]. Householder-RNN ohne additiven Pfad: 32 → 512, 1,00, aus **4.000 Sequenzen** [V].
- Standard-Transformer: lernt Parität oft nicht einmal in der Verteilung (≤ 59 % bei n = 8, 65k Beispiele) [V]. Theorie: Lösungsvolumen 0 [A], Sensitivität isoliert Lösungen [A, 2402.09963].
- Mod-5-Präfix mit rekurrentem Speicher, 40 → 80: nur ca. 70 % [V].

### Kopieren
- Looped 19 → 35 ca. 1,0 [V]. Self-Improving: Kopieren/Umkehren 10 → über 120 nach ca. 100 Runden [V]. Theorie: wiederholtes Kopieren ist eine Kernhürde für CoT-LG [A, 2604.25800].

### Formale Sprachen und Zustandsverfolgung
- Reguläre Sprachen: MLP-LDRU 100 % OOD auf 18/21 [A]. Für Transformer ist entscheidbar, welche reguläre Sprache in der Länge verallgemeinert (C-RASP) [A].
- Gruppen: S₅ ist mit PAR 16× [V], mit Householder-RNN 16× [V] und mit Operator-zuerst-Curriculum 2× [V] lösbar. GDN und Hybride bewältigen aperiodisch und auflösbar fast perfekt. Frontier-LLMs scheitern bei „auflösbar“ ab langen Kontexten [V].
- Ausdrücke mit Verschachtelungstiefe (Boolesch, ListOps, modular): UT-Schaltkreis 1–2 → 30, 100 %, mit 280–4.416 Parametern [V].
- Dyck mit unbeschränkter Tiefe: Standard-Transformer verallgemeinert praktisch nie (1 von 135 Konfigurationen und Seeds) [V].

### Graphen
- DNAR (BFS, DFS, Dijkstra, Prim, MIS) 16 → 1600 Knoten: 100 %, Übergänge aus 4–5-Knoten-Graphen lernbar [V]. Ohne Hints scheitert es.
- Bellman-Ford: K+1 Instanzen reichen beweisbar, Test bis 1024 Knoten [V].
- Transformer: Erreichbarkeit scheitert schon bei n = 8 in der Verteilung [V]. PAR schafft auf Erreichbarkeit nur 2,6× [V]. Für Graph-Zusammenhang hat geloopte Tiefe ein endliches Fenster [A].
- Labyrinthe: Deep Thinking 9×9 → 59×59 97 % [V]. Self-Improving 9 → 30 Hops [V].

---

## 3. Offene Lücken laut Autoren

1. **Daten-Effizienz wird kaum gemessen.** Fast alle Verfahren nutzen Online-Generatoren mit Millionen Beispielen: [Abacus](https://arxiv.org/abs/2405.17399) ca. 20 M, [FIRE](https://arxiv.org/abs/2402.09371) 30–120 M, [Looped](https://arxiv.org/abs/2409.15647) online, [Self-Improving](https://arxiv.org/abs/2502.01612) 2–5 M Startdaten. Die Self-Improving-Autoren nennen keine Mindestmenge [V].
   - Die einzige systematische Messung von Stichprobenkomplexität und LG zusammen ist [Altabaa et al. 2026](https://arxiv.org/abs/2610.32104). Die Autoren verlangen von künftiger Theorie „an account of learnability and inductive biases“ [A].
2. **Trainingslänge gegen Stichprobenzahl.** [Henry/Rizvi-Martel/Hahn](https://arxiv.org/abs/2610.35715) zeigen den exponentiellen Vorteil von Länge, aber ihre Schranken sind „asymptotic rates, not concrete numerical guarantees“, analysieren idealisierte Lerner statt Gradientenabstieg und lassen Positionskodierungen aus [V]. Bei [FIRE](https://arxiv.org/abs/2402.09371) verallgemeinert ein Training auf 1–10 Stellen gar nicht [V].
3. **Seed-Abhängigkeit.**
   - FIRE: große Varianz über Seeds und Datenreihenfolge [V].
   - Sortieren: nur 26 % der Seeds finden die richtige Lösung. Offen ist, ob man das früh im Training erkennt [V, 2610.26974].
   - PAR: SD 33–43 Punkte auf größeren Modellen [V].
   - Looped Transformer: „brittle, high OOD variance“ [A, 2606.29983].
4. **Überwachung der Zwischenschritte als versteckte Kosten.**
   - DNAR: ohne Hints keine perfekte Validierung, DFS besonders schwer. Curriculum bleibt künftige Arbeit [V].
   - Insertion Sort: ohne Zusatz-Überwachung scheitert es schon bei Länge 16. Ob schwächere Überwachung oder automatisch gefundene Zustände reichen, ist offen [V].
   - DFST braucht Experten-Trajektorien [V].
   - Looped braucht die Schrittzahl als Ground Truth [V].
   - PAR: ohne Zwischenlabels nur per Präfix-Selbstdestillation (85 % bei Tiefe 256) [V].
5. **Struktur per Hand.** Position Coupling und das Mehr-Operanden-Coupling gelten nur für Aufgaben mit expliziter Token-Struktur. Ein Ziel ist „extending … to tasks with implicit or unknown structure“, die Designentscheidungen hängen „heavily on intuition“ [V].
6. **Multiplikation.** Für die Self-Improving-Autoren ist Multiplikation ohne Zwischenschritte nicht selbstverbessernd lernbar [V]. Abacus berichtet nur Ergebnisse in der Verteilung [V]. Bei Tiny Transformers bricht Multiplikation auf ungesehenen 4-stelligen Operanden auf 0 % ein [A, 2610.09003].
7. **Looping-Stabilität.** Mehr Loops können richtige Antworten zerstören („overthinking“). Ein mit Loop 32 trainiertes Modell behält bei Loop 64 nur 22 % [A, 2610.19768]. Die Frontier bricht zwischen 128 und 256 Bit [V, 2607.20594]. In großer Tiefe werden beide Loop-Mechanismen unzuverlässig [A, 2609.33144].
8. **Theorie.** Die Bedingung, wann ein Gradientenverfahren die längen-verallgemeinernde Lösung tatsächlich findet, ist offen. Sie wird jeweils als offenes Problem genannt:
   - 2602.13106: „no proof gradient descent reaches such solutions“ [V].
   - 2610.35715: idealisierte Lerner [V].
   - 2610.41218: „general deterministic case … open conjecture“ [A].

---

## 4. Was ist auf dem Mac mit MLX nachrechenbar?

Die Rechenangaben stammen aus den Papieren. Die Mac-Schätzungen sind meine Einschätzung, nicht gemessen.

| Arbeit | Größe und Aufwand laut Papier | Mac M4 Pro (Einschätzung) |
|---|---|---|
| Householder-RNN (2609.18966) | 4k Sequenzen, ca. 320 Schritte, d = 112. 202 Läufe in 53 min auf einer RTX 3080 | **sehr gut** (Minuten pro Lauf). Code und Artefakte laut Autor vorhanden |
| UT-Schaltkreis (2608.31067) | 280–4.416 Parameter. Test-Länge durch Speicher begrenzt (bis 41k Tokens) | **gut**. Testlängen eventuell kleiner wählen |
| NCA-Multiplikation (2603.29069) | 321 Parameter | **sehr gut** (Datenmenge unbekannt) |
| DFST-Arithmetik (2511.22751) | 1k–162k Parameter, 0,5–3 M Iterationen bei Batch 32. Code auf GitHub | **gut**. Viele Iterationen, aber winziges Modell, Stunden statt Tage (?) |
| Looped Transformer (2409.15647) | d = 256, 1–4 Schichten im Block, 100k Schritte × 64 | **machbar**. Ca. Stunden pro Aufgabe. Ist schon als Kontrolle geplant |
| Sortieren, 2 Schichten (2610.26974) | 60k Schritte × Batch 4096, kleine Breite (32–80) | **machbar**. Batch 4096 kostet, ca. Stunden (?) |
| Looping-Frontier (2607.20594) | 1,6–25 M Parameter, Minuten bis 1,5 h auf einer RTX 3090 | **machbar** für die Skala s/m |
| Jelassi (2306.15400) | 5k Beispiele, 6 Schichten, d = 512 | **gut**. Passt genau zu E31-Familie A |
| Lee, Teaching Arithmetic (2307.03381) | NanoGPT 10,6 M, 1k–10k Beispiele | **gut** |
| Self-Improving (2502.01612) | 14 M, 2 M Start plus 50k × bis 100 Runden | **knapp**. Mit kleineren Startdaten wäre es genau die offene E31-Frage |
| DNAR (2402.11628) | 1000 Schritte × 32 Graphen, 1 A100 | **machbar** (GNN in MLX selbst bauen, sonst PyTorch-CPU/MPS) |
| Abacus (2405.17399) | 20 M Beispiele, 1 GPU-Tag (8 EFLOP), 12–122 M | **knapp bis schwer**. Ein M4 Pro ist grob einige TFLOP/s, also eher mehrere Tage (?) |
| FIRE (2402.09371) | 25 M, 30 M+ Beispiele, 50k Schritte × 128 × 2048 Tokens | **schwer** |
| Turing Programs (2407.03310) | 150 M, 200k–400k Schritte | **zu groß** |
| PAR, TAIL, StateBench | 0,5–7 B bzw. 275 M mit Chinchilla-Daten | **nicht sinnvoll** |

Fazit: Alle Arbeiten, die ausdrücklich kleine Daten **und** Längen-Verallgemeinerung zeigen (Householder-RNN, DFST, UT-Schaltkreis, NCA, DNAR, Jelassi), sind auf dem Mac nachrechenbar.

---

## 5. Ehrliche Einschätzung: Wo ist das Feld voll, wo ist Platz?

**Voll (nicht neu erfinden):**
- **Positionskodierungen für Addition** (Abacus, Position Coupling, FIRE, Random Float Sampling, PRISM …). 20 → 100–200 Stellen sind gelöst.
- **Looped Transformer und Universal Transformer.** Allein seit September 2026 zeigt die arXiv-Abfrage um die 50 Arbeiten, darunter Halting, Stochastic Stopping, InfiLoop, PAR, Frontier-Theorie und Mechanistik.
- **Selbst-Curriculum, Easy-to-hard** (Self-Improving Transformers, Extrapolation by Association, Task-Centric Theory 2602.10014). Plan-Kandidat K3 deckt sich weitgehend mit Lee et al. 2025. Neu wäre er nur mit sehr wenigen Startbeispielen.
- **Theorie zu C-RASP und RASP-L** (Schranken, Entscheidbarkeit für reguläre Sprachen). Sehr aktive Theoriegruppe (Yang, Chiang, Zetzsche, Hahn).
- **Zustandsverfolgung S₅ und Parität** mit rekurrenten oder linearen RNNs (GDN, DeltaProduct, Householder, PD-SSM, CSP, Rational Transductors).
- **Diskrete NAR auf Graphen mit Hints** (DNAR, Discrete Insertion Sort).

**Platz für Neues (nach dem, was ich gefunden habe):**
1. **Wenige Beispiele (10–1.000) plus Längen-Verallgemeinerung, ohne Zwischenschritt-Überwachung.**
   - Die starken Ergebnisse mit wenig Daten (DFST 20–10.000 Beispiele, DNAR mit 4–5-Knoten-Graphen, Householder 4.000) brauchen alle dichte Überwachung: Trajektorien, Hints oder Ausgabe pro Position. Oder sie brauchen eine handgebaute Architektur.
   - Mit nur Eingabe→Ausgabe-Paaren und < 1.000 Beispielen habe ich keine Arbeit gefunden, die Addition oder Sortieren 4× oder mehr extrapoliert. Unsicher: Die Suche war nicht erschöpfend.
2. **Messung der Datenkurve.** Kaum eine LG-Arbeit berichtet, wie die OOD-Genauigkeit von N = 10, 100, 1.000 abhängt. Altabaa et al. messen N ≥ 256 und kein N < 256. Henry et al. sagen theoretisch „wenige lange Strings schlagen viele kurze“ vorher, ohne praktische Zahlen für Gradientenabstieg. Eine saubere Kurve für N × Trainingslänge, über Seeds und auf mehreren Aufgabenfamilien, wäre ein eigenständiger Beitrag und passt zur Messkultur des Repos.
3. **Finden statt Einbauen der Struktur.**
   - Position Coupling, Abacus, NCA-Gitter und UT-Tiefen-PE kodieren die Regel-Struktur von Hand. Die Autoren nennen „implicit or unknown structure“ als offen.
   - Bezug zu den Plan-Kandidaten:
     - K1 (Programmsuche): sucht die Struktur und baut sie nicht ein.
     - K6 (MDL-Gewichtssuche): ebenso.
     - K2 (diskrete Zustände): müsste ohne Hints auskommen, um neu zu sein.
4. **Seed-Lotterie gezielt lösen.** Bei Sortieren 26 %, bei A₅ 8/22, und FIRE ist fragil. Die richtige Lösung existiert, Gradientenabstieg findet sie nur manchmal, wie im Plan beschrieben („Engpass ist die Suche“). Mit wenigen Beispielen gibt es viele Lösungen, die die Daten erklären; ein Auswahlkriterium (MDL, Einfachheit, Robustheit) wäre neu, wenn es ohne OOD-Daten die verallgemeinernde Lösung findet. Teilweise vorhanden: 2610.26974 steuert per Initialisierung, 2609.18966 entfernt einen Pfad.
5. **Multiplikation N×N ohne CoT und ohne handgebaute Darstellung.** Bleibt laut Autoren offen. Es ist aber ein hartes Ziel und für ein „90 % gegen 10 %“-Gate riskant.

**Risiken:**
- Die Konkurrenz ist sehr schnell (2026: mehrere Arbeiten pro Woche zu Looping und LG).
- Eine Neuheitsprüfung vor jedem Kandidaten ist Pflicht. Besonders zu prüfen sind 2511.22751 (DFST), 2610.35715 (lange Strings) und 2602.13106 (kleine Trainingsmengen für GNNs). Diese drei kommen dem E31-Ziel am nächsten.

---

### Quellenlage und Unsicherheiten
- Bei folgenden Arbeiten sind nur Abstracts gelesen: 2310.16028, 2406.06467, 2506.09251, 2507.13332, 2510.14826, 2511.07378, 2603.02238, 2603.29069, 2604.25800, 2605.26035, 2606.29983, 2608.13433, 2609.33144, 2610.14456, 2610.41218, 2610.18649. Die Zahlen dort stammen nur aus dem Abstract.
- Neural GPU (Kaiser & Sutskever 2015): Das Abstract nennt Training bis 20 Bit und Test „much longer“ ohne Fehler. Die oft zitierten 2.000 Bit habe ich nicht in einer geöffneten Quelle bestätigt.
- Die Zahl der Trainingsbeispiele fehlt bei: Position Coupling, Turing Programs (nur Schritte), UT-Schaltkreis, NCA-Multiplikation und Tiny Recursive Message Passing.
- Die Abfragen an die arXiv-API wurden zeitweise gedrosselt (HTTP 429), deshalb lief ein Teil über die arXiv-Websuche. Die Abdeckung von 2025 ist dünner als die von 2026.
