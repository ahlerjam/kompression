# ARC-AGI und abstraktes Schließen mit kleinen Netzen (Stand 2026-11-05)

Recherche für das Ziel: *„Ein kleines Netz lernt aus sehr wenigen Beispielen eine Regel und wendet sie richtig auf größere oder neue Fälle an.“*

**Wie die Angaben entstanden:** Alle Zahlen stammen aus Quellen, die ich in dieser Recherche geöffnet habe. Meist war das die arXiv-Abstract-Seite, bei einigen Arbeiten die HTML-Volltextseite (markiert mit „HTML“). Wo ich nur den Abstract kenne, fehlen oft Parameterzahl und Rechenaufwand. Das steht dann als „n. a.“ (nicht angegeben bzw. nicht geprüft) in der Tabelle. Eigene Schätzungen sind als **[Schätzung]** markiert, unsichere Angaben als **[unsicher]**.

Wichtige Begriffe:
- **pass@2**: Eine Aufgabe zählt als gelöst, wenn einer von 2 Versuchen exakt stimmt. Das ist die ARC-Standardmetrik.
- **Public Eval vs. Semi-private/Private**: Auf den öffentlichen Evaluationsaufgaben liegen die Ergebnisse systematisch höher. Bei HRM waren es 41 % öffentlich gegenüber 32 % semi-privat ([ARC Prize HRM-Analyse](https://arcprize.org/blog/hrm-analysis)).
- **„Transduktives“ Training**: Die TRM/HRM-Familie trainiert auf den Demonstrationspaaren der Evaluationsaufgaben mit, nicht auf deren Test-Outputs. Sie braucht dafür eine Puzzle-ID-Einbettung. Faktisch ist das Test-Time-Training. Die Zahlen sind deshalb nicht direkt mit „ein Modell, unbekannte Aufgabe, ohne Training“ vergleichbar.

---

## 1. Die wichtigsten Arbeiten

| # | Arbeit | Autoren | Datum | Kernidee (1 Satz) | Konkrete Zahlen | Bezug zu unserem Ziel |
|---|---|---|---|---|---|---|
| 1 | [Less is More: Recursive Reasoning with Tiny Networks (TRM)](https://arxiv.org/abs/2510.04871) (HTML) | A. Jolicoeur-Martineau | 2025-10 | Ein einziges 2-Layer-Netz verfeinert rekursiv einen Latent-Zustand z und eine Antwort y, mit Deep Supervision. | 7M Param. (Attention-Variante); ARC-AGI-1 44,6 %, ARC-AGI-2 7,8 %; Sudoku-Extreme 87,4 % (5M, MLP), Maze-Hard 85,3 %. ARC: 1000 Augmentierungen pro Beispiel, ca. 3 Tage auf 4×H100. Sudoku: 1 L40S, < 36 h. n=6, T=3, bis 16 Supervision-Schritte. | Referenzarchitektur für „kleines Netz + Rekursion“. Gewinner des ARC Prize 2025 Paper Award (1. Platz). |
| 2 | [HRM-Analyse: The Hidden Drivers of HRM](https://arcprize.org/blog/hrm-analysis) | ARC Prize Foundation | 2025 | Ablation von HRM (27M) durch die Benchmark-Betreiber. | Verifiziert: ARC-AGI-1 32 % semi-privat (öffentlich behauptet 41 %), ARC-AGI-2 2 %. Die Hierarchie bringt kaum etwas: Ein gleich großer Transformer liegt ca. 5 Punkte dahinter. Der äußere Refinement-Loop ist entscheidend: 1→2 Loops +13 Punkte, 1→8 Loops etwa doppelte Leistung. Training nur auf den 400 Eval-Aufgaben ergibt immer noch 31 %. 300 statt 1000 Augmentierungen reichen fast. | Zentrale Ablation: Die Leistung kommt aus Refinement-Loop und Training auf den Testaufgaben, nicht aus der Hierarchie. |
| 3 | [TRM on ARC-AGI-1: Inductive Biases, Identity Conditioning, and Test-Time Compute](https://arxiv.org/abs/2512.11847) | Roye-Azar et al. | 2025-12 | Ablation des veröffentlichten TRM-Checkpoints. | Voting über 1000 Samples bringt ca. +11 Punkte Pass@1 gegenüber einem einzelnen Durchlauf. Mit falscher oder leerer Puzzle-ID ist die Genauigkeit **0**. Der größte Teil der Genauigkeit ist schon nach dem ersten Rekursionsschritt erreicht („shallow effective recursion“). | Kritik: Die TRM-Leistung ist eher Effizienz + Task-Konditionierung + Test-Time-Compute als „tiefes Denken“. |
| 4 | [Test-time Adaptation of Tiny Recursive Models](https://arxiv.org/abs/2511.02886) | R. McGovern | 2025-11 | TRM auf öffentlichen Aufgaben vortrainieren, dann in der Kaggle-Laufzeit voll feinjustieren. | Vortraining: 1280 Aufgaben, 700k+ Schritte, 48 h auf 4×H100. Test-Time: 12.500 Gradientenschritte. ARC-AGI-2: ca. 10 % öffentlich, 6,67 % semi-privat. | Zeigt, wie TRM in ein echtes Test-Time-Budget passt. |
| 5 | [ARC-AGI Without Pretraining (CompressARC)](https://arxiv.org/abs/2512.06104) | I. Liao, A. Gu | 2025-12 | Pro Aufgabe ein Netz von Null an trainieren, das die Beschreibungslänge (MDL) des Puzzles minimiert. Es gibt keinerlei Trainingsdaten außer der einen Aufgabe. | 76k Param.; ARC-AGI-1 20 % (Eval). Laut ARC-Prize-Seite ca. 4 % auf ARC-AGI-2 **[nur Sekundärquelle]**. Rechenzeit pro Aufgabe steht nicht im Abstract. | Am nächsten an „aus sehr wenigen Beispielen“ im strengen Sinn. 3. Platz beim Paper Award 2025. |
| 6 | [ARC Is a Vision Problem! (VARC)](https://arxiv.org/abs/2511.14761) (HTML) | K. Hu, …, J. Andreas, K. He | 2025-11 | Ein ViT behandelt ARC als Bild-zu-Bild-Aufgabe auf einer „Canvas“, mit Test-Time-Training pro Aufgabe. | 18M ViT: ARC-1 54,5 %, ARC-2 8,3 %; Ensemble mit U-Net (73M): 60,4 % bzw. 11,1 %. Ohne RE-ARC-Daten 31,5 %. Ohne Offline-Training (nur TTT) 26,4 %. Offline-Training 8×H100 / 4,8 h; TTT ca. 70 s pro Aufgabe auf 1×H100. Ablationen: 2D-RoPE +3,5, Scale-Augmentierung +6,2, Translation +2,9, visuelle Priors insgesamt +27,7. | Saubere Ablationen, wie viel von Daten (RE-ARC), Priors, TTT und Multi-View kommt. |
| 7 | [LoopViT: Scaling Visual ARC with Looped Transformers](https://arxiv.org/abs/2602.02156) | W.-J. Shu et al. | 2026-02 | VARC mit gewichtsgeteiltem Loop-Block (Conv + Attention) und entropiebasiertem Dynamic Exit. | 18M: ARC-AGI-1 65,8 % (Abstract). Die Größenangaben zu kleineren Varianten unterscheiden sich zwischen Blog (5,9M → 57,2 %) und Sekundärquellen (3,8M → 60,1 %) **[unsicher]**. | Rekursion + visuelle Priors zusammen. Iterationen statt Breite als Skalierungsachse. |
| 8 | [Universal Reasoning Model (URM)](https://arxiv.org/abs/2512.14693) | Z. Gao et al. | 2025-12 | Universal Transformer mit Short-Convolution und abgeschnittenem Backprop. Die Gewinne kommen laut Autoren aus „recurrent inductive bias and strong nonlinear components“. | ARC-AGI-1 53,8 % **pass@1**, ARC-AGI-2 16,0 % pass@1. Parameterzahl nicht im Abstract; laut CHARM-Paper ca. 13,7M Backbone. | Die Ablation spricht für einfache Rekurrenz statt komplizierter Architektur. |
| 9 | [What Limits Recursive Reasoning Models](https://arxiv.org/abs/2609.39967) | Shakhvalieva, …, Oseledets et al. | 2026-09 | Rezept für stabile Rekursion: mittlerer Gradient-Horizont, große Batches, kontrollierte Zustandsupdates. Keine Hierarchie nötig. | 13,6M Param.; ARC-AGI-1 59,5 % pass@2; Sudoku 98,41 %; Arithmetik-OOD 71,2 % (gegenüber 36,2 % bester Baseline). Rechenaufwand nicht im Abstract. | Direkt relevant: Optimierung und Stabilität der Rekurrenz sind der Engpass, nicht die Architektur. |
| 10 | [Structured Sparse Memory for Recurrent Reasoning (CHARM)](https://arxiv.org/abs/2609.33270) (HTML) | Z. Zhao et al. (Argonne), NeurIPS 2026 | 2026-09 | Kleines rekurrentes Backbone (URM, ca. 14M) plus kompakte, kompositionelle Task-Memory-Einbettung (CoSE) plus synthetische Daten plus Checkpoint-Aggregation. | **ARC-AGI-1 84,0 % pass@2, ARC-AGI-2 46,7 % pass@2 (Public Eval)**; 44M bzw. ca. 54–57M Parameter inkl. Task-Memory. Ohne synthetische Daten: ARC-1 64,9 %, ARC-2 18,6 %. Mit Re-ARC/Re-ARC2: 80,7 % bzw. 37,2 %. Bis 1,6M bzw. 2,38M Trainingsschritte. Nutzt Demo-Paare der Public-Eval-Aufgaben. | **Aktueller Höchststand kleiner Modelle** (laut Selbstangabe). Zeigt, dass synthetische Daten etwa 16–19 Punkte ausmachen. |
| 11 | [Symbol-Equivariant Recurrent Reasoning Models (SE-RRM)](https://arxiv.org/abs/2603.02193) (HTML) | Freinschlag, …, Klambauer | 2026-03 | Permutationsäquivarianz über Symbole/Farben in die Architektur einbauen statt Farb-Augmentierung. | 2M Param.; ARC-AGI-1 45,3 % (TRM 44,6 %), ARC-AGI-2 7,1 % (TRM 7,8 %), mit **8 statt 1000 Augmentierungen**. Sudoku: auf 9×9 trainiert, 4×4 95,5 % gelöst. 16×16 und 25×25: 0 % ganz gelöst (52 % bzw. 31 % Zellgenauigkeit). TRM/HRM können gar nicht auf andere Größen übertragen. | **Sehr nah an unserem Ziel** („größere Fälle“): Die Größen-Extrapolation bleibt ungelöst. |
| 12 | [Scaling to Tens of Thousands of Test-Time Iterations (InfiLoop)](https://arxiv.org/abs/2610.11570) | P. Li, …, S. Liu | 2026-10 | Loop-eigene Residualverbindung gegen „latent overthinking“: Mehr Iterationen zerstören dann nicht mehr korrekte Zwischenergebnisse. | 7M; Sudoku-Extreme 97,9 %; ARC-AGI-2 13,6 % pass@2; Sudoku verbessert sich bis > 20.000 effektive Schritte. | Test-Time-Iterationen als Hebel für schwerere Fälle. |
| 13 | [Probabilistic TRM (PTRM)](https://arxiv.org/abs/2605.19943) | Sghaier, Parviz, Jolicoeur-Martineau | 2026-05 | Gaußsches Rauschen pro Rekursionsschritt, parallele Läufe, Auswahl über den Q-Head. | 7M; Sudoku-Extreme 87,4 → 98,75 %; Pencil Puzzle Bench 62,6 → 91,2 % (Frontier-LLMs 55,1 %). | Test-Time-Suche ohne Neutraining. |
| 14 | [Resample the Answer (GAR)](https://arxiv.org/abs/2610.16427) | M. Illouz | 2026-10 | Fehlgeschlagene TRM-Läufe tragen den Fehler im Antwortzustand y. Deshalb nur y neu samplen, wenn der Halting-Head ablehnt. | Sudoku-Extreme 85,9 → 99,4 %; Pencil Puzzle 68,8 → 85,6 %; max. 64 Restarts; schlägt PTRM, auch wenn PTRM 42–197× mehr Compute bekommt. | Billige Test-Time-Compute-Strategie, mit wenig Rechenaufwand nachprüfbar. |
| 15 | [Latent Reasoning in TRMs is Secretly a Policy Improvement Operator](https://arxiv.org/abs/2511.16886) | Asadulaev et al. | 2025-11 (v5 2026-05) | Rekursion als Policy Improvement deuten und „dead compute“ entfernen. | 18× weniger Forward-Passes bei gleicher Leistung. | Erklärt, wann Rekursion hilft. |
| 16 | [Tiny Recursive Reasoning with Mamba-2 Attention Hybrid](https://arxiv.org/abs/2602.12078) | Wang, Reid | 2026-02 | Transformer-Blöcke in TRM durch Mamba-2-Hybrid ersetzen. | ca. 6,8M; ARC-AGI-1 pass@2 45,88 % gegenüber 43,88 %, pass@100 +4,75. | Der Operator ist austauschbar, der Effekt ist klein. |
| 17 | [Tiny Autoregressive Recursive Models](https://arxiv.org/abs/2603.08082) | Rauba, Fanconi, van der Schaar | 2026-03 | TRM-Refinement auf autoregressive Next-Token-Prediction übertragen. | „no reliable performance gains from the full Autoregressive TRM“ | Negativ-Ergebnis. Nicht wiederholen. |
| 18 | [Nested Shared-Operator Recursion on ARC-AGI-1 (HRM/TRM/URM)](https://arxiv.org/abs/2610.26666) | Neri et al. | 2026-10 | Zusätzlicher verschachtelter Zustand bei gleicher Parameterzahl. | Unter einem Single-GPU-Budget ist exaktes Lösen „near zero for all configurations“; kein konsistenter Vorteil. | **Warnung:** Mit kleinem Budget lässt sich die TRM-Familie auf ARC kaum reproduzieren. |
| 19 | [Pretraining and Test-Time Training Budgets Matter for ARC (VARC vs TRM)](https://arxiv.org/abs/2610.13660) | Deliège, Van Droogenbroeck | 2026-10 | Kontrollierter Vergleich über 5 Vortrainings- × 4 TTT-Budgets. | Mehr TTT-Sample-Präsentationen helfen beiden Modellen. Mehr Task-Varianten helfen nur TRM. VARC bleibt ohne Vortraining konkurrenzfähig, TRM braucht Vortraining viel stärker (Zahlen nicht im Abstract). | Methodik: Vergleiche nur bei gleichem Budget sinnvoll. |
| 20 | [Implicit Rule Induction with Test-Time Task Embeddings (Embed-TTT)](https://arxiv.org/abs/2609.21181) | Deliège, Beger, Van Droogenbroeck, M. Mitchell | 2026-09 | Zuerst nur die Task-Einbettung trainieren (< 0,01 % der Parameter), dann das Backbone. So lässt sich prüfen, ob die Regel tatsächlich gelernt wird. | Getestet auf ARC-1, ConceptARC, Mini-ARC und zwei Datensätzen mit bekannten Regeln. Ergebnis: „rule-wise interpolation, but not extrapolation“. | **Kernlücke unseres Ziels**, von den Autoren selbst benannt. |
| 21 | [Compositional-ARC](https://arxiv.org/abs/2504.01445) | Mondorf, …, Plank (ICLR 2026) | 2025-04 (v3 2026-02) | Meta-Learning for Compositionality: Bekannte Transformationen werden neu kombiniert. | 5,7M Encoder-Decoder; schlägt o3-mini, GPT-4o und Gemini 2.0 Flash und liegt gleichauf mit dem 8B-TTT-Gewinner 2024 (Prozentzahlen nicht im Abstract). | Systematische Generalisierung mit kleinem Netz: direkt passend. |
| 22 | [When Does Recurrence Become an Algorithm?](https://arxiv.org/abs/2607.20594) | T. Zhang et al. | 2026-07 | Gewichtsgeteilte Loop-Transformer auf Gruppen-Wortproblemen (A5, S5) mit Längenextrapolation. | Gesetz: Gelöste Positionen pro Loop skalieren mit n_train/T_train (Exponent 0,98 ± 0,04, R² 0,99). Ungebundene Modelle extrapolieren am schlechtesten. Eine Operator-first-Curriculum entfernt die S5-„Wand“. | **Sehr relevant** für „größere Fälle“: kontrollierte Theorie zur Extrapolation. |
| 23 | [Searching Latent Program Spaces (LPN)](https://arxiv.org/abs/2411.08706) | Macfarlane, Bonnet (NeurIPS 2025 Spotlight) | 2024-11 | Latenter Programmraum, in dem zur Testzeit per Gradient gesucht wird. | Test-Time-Suche verdoppelt die OOD-Leistung (Abstract; keine ARC-Prozentzahlen dort). | Alternative zu Puzzle-IDs: Die Regel wird als Latent inferiert. |
| 24 | [Reasoning with Neural Cellular Automata](https://arxiv.org/abs/2609.36126) | Etcheverry, …, Niklasson (Google) | 2026-09 | Rein lokale, asynchrone NCAs für Mazes, Sudoku und ARC-AGI-1. | OOD-Generalisierung auf größere Grids und längere Rollouts (keine Zahlen im Abstract). | Lokalität als Prior für Größen-Generalisierung. |
| 25 | [ARC-NCA](https://arxiv.org/abs/2505.08778) | Guichard et al. | 2025-05 | NCA/EngramNCA pro Aufgabe trainiert. | Proof of Concept, „comparable to … ChatGPT 4.5“ (keine Zahlen im Abstract). | Vorläufer von #24. |
| 26 | [Vector Symbolic Algebras for ARC](https://arxiv.org/abs/2511.08747) | Joffe, Eliasmith | 2025-11 | Neurosymbolische, objektzentrierte Programmsynthese mit VSA. | v1: ARC-1-Train 10,8 %, ARC-1-Eval 3,0 %; Sort-of-ARC 94,5 %; 1D-ARC 83,1 % (laut Suchergebnis; spätere Versionen 3,5 % bzw. Ensemble 6,3 % Eval **[unsicher, nicht geöffnet]**). | Sehr sample-effizient, aber niedrige ARC-Werte. |
| 27 | [The Surprising Effectiveness of Test-Time Training for Few-Shot Learning](https://arxiv.org/abs/2411.07279) | Akyürek et al. | 2024-11 | LoRA-TTT pro Aufgabe auf augmentierten Demos. | 8B: 53,0 % ARC-1 öffentlich; mit Programmsynthese-Ensemble 61,9 %. | Ursprung des TTT-Paradigmas (großes Modell). |
| 28 | [Product of Experts with LLMs (ARChitects)](https://arxiv.org/abs/2505.07859) | Franzen, Disselhoff, Hartmann (ICML 2025) | 2025-05 | Augmentierungen beim Training, Generieren und Bewerten; LLM als Generator und Scorer. | ARC-1 71,6 %; ca. 2 Cent pro Aufgabe (RTX 4090). | Augmentierung als „Perspektiven“ ist Standard. |
| 29 | [SOAR: Self-Improving LMs for Evolutionary Program Synthesis](https://arxiv.org/abs/2507.14172) | Pourcel, Colas, Oudeyer (ICML 2025) | 2025-07 | Evolutionäre Programmsuche + Hindsight-Finetuning des LLM auf eigene Suchergebnisse. | 52 % ARC-1 Public Test (Modellgrößen nicht im Abstract). | Programmsynthese-Linie, 2. Platz Paper Award 2025. |
| 30 | [ARC Prize 2025 Technical Report](https://arxiv.org/abs/2601.10904) + [Results & Analysis](https://arcprize.org/blog/arc-prize-2025-results-analysis) | Chollet, Knoop, Kamradt, Landers | 2026-01 | Überblick: „Refinement loop“ als zentrales Thema 2025. | Kaggle ARC-AGI-2 privat: NVARC 24,03 %, ARChitects 16,53 %, MindsAI 12,64 %. 1455 Teams. Frontier: Opus 4.5 37,6 % ($2,20 pro Aufgabe), Gemini 3 Pro + Poetiq 54 % (ca. $30 pro Aufgabe). | Einordnung; Liste der Paper Awards (Abschnitt 2). |
| 31 | [Rules, Traces and Routing Statistics … Open Models on ARC-AGI-2](https://arxiv.org/abs/2611.06827) | Nguyen, Trinh | 2026-11 | Welche Daten helfen offenen Modellen unter Kaggle-Bedingungen (4×L4, 12 h)? | NVARC-Qwen3-4B: 28,96 → 33,06 % (semi-privat) durch 1920 echte Aufgaben. **Kein Checkpoint mit synthetischen Daten (bis 95.993 Aufgaben) schlägt das Modell auf echten Aufgaben.** Qwen3.8-27B 44,03 %; MoE (125B/6B aktiv) 56,67 %. | Zeigt, wo Kaggle 2026 steht (LLMs mit Milliarden Parametern), und dass synthetische Daten nicht automatisch helfen. |
| 32 | [ARC-AGI-3](https://arxiv.org/abs/2603.24621) | ARC Prize Foundation | 2026-03 | Interaktive, anweisungslose Spiele; Bewertung nach Aktionseffizienz relativ zu Menschen. | Menschen 100 %, Frontier-Modelle < 1 % (März 2026). | Neues Format; s. Abschnitt 2. |
| 33 | [Perception, Not Planning, Governs ARC-AGI-3](https://arxiv.org/abs/2611.06793) | H. S. Banwait | 2026-11 | Parameterfreie Wahrnehmungs-Engine (Objekte, Geometrie) vor dem LLM. | Hidden Kaggle: 16,53 in 9 h auf 1 RTX PRO 6000 (13,7× Milestone-1-Gewinner); 143/183 öffentliche Level. | Wahrnehmung/Repräsentation ist der Engpass, nicht die Planung. |

---

## 2. Stand der Technik (kleine Modelle, wenige Beispiele), mit Zahlen

### 2.1 ARC-AGI-1 (öffentliche Evaluation, pass@2, wenn nicht anders angegeben)

| System | Parameter | ARC-AGI-1 | Quelle |
|---|---|---|---|
| CompressARC (kein Vortraining, nur die eine Aufgabe) | 76k | 20 % | [2512.06104](https://arxiv.org/abs/2512.06104) |
| HRM | 27M | 40,3 % (verifiziert semi-privat 32 %) | [TRM-Paper](https://arxiv.org/abs/2510.04871), [ARC Prize](https://arcprize.org/blog/hrm-analysis) |
| TRM | 7M | 44,6 % | [2510.04871](https://arxiv.org/abs/2510.04871) |
| SE-RRM (8 Augmentierungen) | 2M | 45,3 % | [2603.02193](https://arxiv.org/abs/2603.02193) |
| URM | ca. 14M | 53,8 % (pass@1) | [2512.14693](https://arxiv.org/abs/2512.14693) |
| VARC (ViT) | 18M | 54,5 % | [2511.14761](https://arxiv.org/abs/2511.14761) |
| „What Limits RRMs“ | 13,6M | 59,5 % | [2609.39967](https://arxiv.org/abs/2609.39967) |
| VARC-Ensemble | 73M | 60,4 % (Menschen-Durchschnitt laut Paper 60,2 %) | [2511.14761](https://arxiv.org/abs/2511.14761) |
| LoopViT | 18M | 65,8 % | [2602.02156](https://arxiv.org/abs/2602.02156) |
| **CHARM** | 14M Backbone / 44M gesamt | **84,0 %** (pass@1 78,6 %) | [2609.33270](https://arxiv.org/abs/2609.33270) |

Zum Vergleich mit LLMs: ARChitects 71,6 % ([2505.07859](https://arxiv.org/abs/2505.07859)), Akyürek 8B TTT 53 % bzw. 61,9 % mit Ensemble ([2411.07279](https://arxiv.org/abs/2411.07279)).

### 2.2 ARC-AGI-2

- **Kleine Modelle (Public Eval):** TRM 7,8 %, HRM 5,0 % ([TRM-Paper](https://arxiv.org/abs/2510.04871)); VARC 8,3 % / Ensemble 11,1 %; InfiLoop 13,6 % ([2610.11570](https://arxiv.org/abs/2610.11570)); URM 16,0 % pass@1; **CHARM 46,7 % pass@2** ([2609.33270](https://arxiv.org/abs/2609.33270), selbst berichtet, Public Eval, nutzt die Demo-Paare der Eval-Aufgaben im Training).
- **Kaggle 2025 (privat):** NVARC 24,03 %, ein Qwen3-4B-Ensemble mit TRM-Komponenten und synthetischen Daten ([ARC Prize 2025 Results](https://arcprize.org/blog/arc-prize-2025-results-analysis)).
- **Kaggle 2026:** Laut einer Sekundärquelle lag Tufa Labs am 9.10.2026 auf dem öffentlichen Kaggle-Board bei 88,06 %, Rabbithole bei 80,56 % ([toolnavs](https://toolnavs.com/article/2447-arc-agi-2-high-score-refresh-tufa-labs-tops-the-board-at-8806-past-the-85-bonus)). **[unsicher: keine Primärquelle geöffnet; vorläufiger Wert auf ca. der Hälfte der Testdaten; Methode unbekannt]**. Der Wettbewerb ist beendet, die Ergebnisse kommen am 4.12.2026 ([arcprize.org](https://arcprize.org/competitions/2026/arc-agi-2)). Das zeigt, wohin die Spitze 2026 geht: Ein Paper zu Kaggle-tauglichen Ansätzen kommt mit offenen LLMs von 27B bzw. einem MoE mit 125B gesamt / 6B aktiv auf 44–57 % semi-privat ([2611.06827](https://arxiv.org/abs/2611.06827)). **Für die Kaggle-Spitze 2026 sind „kleine Netze“ (≤ 50M) nach meinem Befund nicht mehr konkurrenzfähig. Die Spitze bilden lokal laufende offene LLMs mit Milliarden Parametern.**

### 2.3 ARC-AGI-3 (seit März 2026)

- Menschen 100 %, Frontier-LLMs < 1 % beim Start ([2603.24621](https://arxiv.org/abs/2603.24621)). Laut Sekundärquellen: Gemini 3.1 Pro 0,37 %, GPT-5.4 0,26 %, Opus 4.6 0,25 % **[Sekundärquellen]**.
- Preview-Wettbewerb 2025: StochasticGoose (CNN sagt voraus, welche Aktionen den Zustand ändern, plus einfaches RL) 12,58 % ([arcprize.org Preview](https://arcprize.org/competitions/arc-agi-3-preview-agents), nur über Suchergebnis). Kleine, nicht-LLM-basierte Ansätze waren dort vorne.
- 2026 Milestone 1: Alle drei Ersten nutzen lokal laufende LLMs/VLMs mit 27–31B Parametern (Qwen 3.6 27B, Gemma-4-31B) ([arcprize.org Milestone 1](https://arcprize.org/blog/arc-prize-2026-milestone-1)).
- Bestes von mir gefundenes Paper: 16,53 auf dem Hidden-Kaggle-Set mit parameterfreier Wahrnehmungs-Engine + LLM ([2611.06793](https://arxiv.org/abs/2611.06793)).

### 2.4 Was die Ablationen über die Quellen der Leistung sagen

- **Augmentierung / Voting:** +11 Punkte Pass@1 durch Voting über 1000 Samples bei TRM ([2512.11847](https://arxiv.org/abs/2512.11847)). Multi-View-Inferenz bei VARC: pass@1 von 35,9 auf 49,8 %. 30 statt 1000 Augmentierungen liegen nur ca. 4 % unter dem Maximum (HRM, [ARC Prize](https://arcprize.org/blog/hrm-analysis)). Äquivarianz in der Architektur ersetzt 1000 durch 8 Augmentierungen ohne Verlust ([SE-RRM](https://arxiv.org/abs/2603.02193)).
- **Rekursion:** Der äußere Refinement-Loop ist essenziell (HRM: 1→8 Loops ca. ×2). Die innere „tiefe“ Rekursion ist dagegen flach: Das meiste ist nach dem 1. Schritt erreicht ([2512.11847](https://arxiv.org/abs/2512.11847)). Die Hierarchie (HRM) bringt nichts gegenüber einem gleich großen rekurrenten Transformer ([ARC Prize](https://arcprize.org/blog/hrm-analysis), [2609.22197](https://arxiv.org/abs/2609.22197), [2609.39967](https://arxiv.org/abs/2609.39967)).
- **Training auf den Testaufgaben (Puzzle-ID):** Mit falscher ID ist das Ergebnis 0 ([2512.11847](https://arxiv.org/abs/2512.11847)). Nur auf den Eval-Aufgaben trainiert erreicht HRM immer noch 31 % statt 41 % ([ARC Prize](https://arcprize.org/blog/hrm-analysis)).
- **Synthetische Daten:** VARC ohne RE-ARC 31,5 % statt 54 %. CHARM ohne synthetische Daten 64,9 % statt 80,7 % (ARC-1) bzw. 18,6 % statt 37,2 % (ARC-2). Aber: Für ein Qwen3-4B unter Kaggle-Bedingungen half kein synthetischer Datensatz ([2611.06827](https://arxiv.org/abs/2611.06827)).
- **Task-Memory-Kapazität:** Die Task-Einbettungen sind bei bisherigen Systemen oft mehr als 30× größer als das Backbone ([CHARM](https://arxiv.org/abs/2609.33270)). Ein großer Teil der „7M-Parameter“-Kapazität steckt also versteckt in den Puzzle-Embeddings.

---

## 3. Offene Fragen, die die Autoren selbst als Lücke nennen

1. **Warum hilft Rekursion?** „the question of why recursion helps so much compared to using a larger and deeper network remains to be explained“ und „Scaling laws are needed to parametrize these networks optimally“ ([TRM, 2510.04871](https://arxiv.org/abs/2510.04871)).
2. **Generative statt überwachte Rekursion:** „recursive reasoning models such as HRM and TRM are supervised learning methods rather than generative models“. Es wäre „interesting to extend TRM to generative tasks“ ([TRM](https://arxiv.org/abs/2510.04871)).
3. **Extrapolation auf neue Regeln:** Embed-TTT lernt „rule-wise interpolation, but not extrapolation“. Die Autoren fordern eine klarere Trennung von Regel-Induktion und Regel-Ausführung ([2609.21181](https://arxiv.org/abs/2609.21181)).
4. **Größen-Extrapolation:** SE-RRM: „the trained SE-RRM model does not solve any of the larger Sudokus“ (16×16, 25×25). Außerdem: „we only used SE-RRMs with 2M parameters“ ([2603.02193](https://arxiv.org/abs/2603.02193)).
5. **Wissen vs. Schließen trennen:** „We still need new ideas, like how to separate knowledge and reasoning, among others“. Dazu eine Warnung vor „knowledge overfitting“ auf ARC-AGI-1/2 ([ARC Prize 2025 Analysis](https://arcprize.org/blog/arc-prize-2025-results-analysis)).
6. **Overfitting beim Skalieren:** VARC: „Going beyond this regime can lead to overfitting in our current setting.“ Und: „still makes mistakes on some tasks that are simple for humans“ ([2511.14761](https://arxiv.org/abs/2511.14761)).
7. **Vergleichbarkeit:** Ergebnisse hängen stark von Vortrainings- und TTT-Budget ab und sollten nur bei vergleichbaren Budgets verglichen werden ([2610.13660](https://arxiv.org/abs/2610.13660)). CHARM selbst: „These accuracies are not directly comparable to our final results“ und Ablationen „limit[ed] … to a single run“ ([2609.33270](https://arxiv.org/abs/2609.33270)).
8. **Reproduzierbarkeit mit kleinem Budget:** Unter Single-GPU-Budget bleibt exaktes Lösen bei HRM/TRM/URM „near zero for all configurations“. Die Hypothese bleibt offen, ein Confirmatory Study wird vorgeschlagen ([2610.26666](https://arxiv.org/abs/2610.26666)).
9. **Welche Hyperparameter stabilisieren die Rekursion, und gilt das allgemein?** Die Aussagen gelten ausdrücklich nur für „the recursive architectures studied here“ ([2609.39967](https://arxiv.org/abs/2609.39967)).
10. **Schwierigkeitswände bei Längenextrapolation:** Bei der Gruppenordnung (S5) gibt es eine harte Grenze. Ob Operator-first-Curricula das allgemein lösen, ist nur an Gruppen-Wortproblemen gezeigt ([2607.20594](https://arxiv.org/abs/2607.20594)).

---

## 4. Was auf einem MacBook M4 Pro (48 GB, MLX) realistisch nachrechenbar ist

Bekannter Rechenaufwand aus den Quellen:

| Arbeit | Rechenaufwand laut Quelle | Mac realistisch? **[Schätzung]** |
|---|---|---|
| TRM auf ARC | ca. 3 Tage × 4 H100 (ca. 290 H100-Stunden) | **Nein.** Bei einem angenommenen Faktor von 20–50× langsamer als eine H100 wären das Monate bis über ein Jahr. |
| TRM-Vortraining (McGovern) | 48 h × 4 H100 | Nein. |
| TRM auf Sudoku-Extreme | 1 L40S, < 36 h | Grenzwertig: etwa Wochen. Mit kleinerem Datensatz und weniger Epochen als Pilot machbar. |
| TRM auf Maze-Hard | 4 L40S, < 24 h | Nein bzw. nur stark reduziert. |
| VARC Offline-Training | 8 H100 × 4,8 h (ca. 38 H100-Stunden) | Grenzwertig bis nein (Wochen). |
| VARC TTT pro Aufgabe | ca. 70 s auf 1 H100 | **Ja**, wenn vortrainierte Gewichte vorhanden sind (ca. Minuten bis 1 h pro Aufgabe). Für 400 Aufgaben nur als Stichprobe. |
| CompressARC | pro Aufgabe; Zeit im Abstract nicht angegeben | **Wahrscheinlich ja** pro Einzelaufgabe (76k Parameter). Der Gesamtlauf über 400 Aufgaben muss erst gemessen werden **[unsicher]**. |
| GAR / PTRM (Test-Time auf fertigem TRM) | max. 64 Restarts pro Puzzle, kein Training | **Ja**, mit veröffentlichten Checkpoints (Inferenz eines 7M-Modells). |
| SE-RRM (2M) Sudoku | 10.000 Epochen, Batch 272 (Zeit n. a.) | Kleiner Pilot ja; volle Replikation unklar. |
| Loop-Transformer auf Gruppen-Wortproblemen (2607.20594) | n. a., synthetische Aufgaben | **Ja, gut geeignet**: kleine kontrollierte Aufgaben mit eigener Datenerzeugung. |
| Compositional-ARC (5,7M) | n. a. | Wahrscheinlich ja (kleines Modell, synthetische 2D-Transformationen) **[unsicher]**. |
| Alles mit ≥ 4B-LLMs (NVARC, Kaggle 2026, ARC-AGI-3-Agenten) | 4×L4 / 12 h bzw. 27B+-Modelle | Inferenz eines 4B-8B-Modells in MLX geht. TTT pro Aufgabe mit LoRA ist langsam, ganze Wettbewerbsläufe sind unrealistisch. |

Die Faktoren 20–50× sind eine grobe eigene Annahme (FP16/BF16-Durchsatz M4 Pro gegenüber H100, ohne Messung). Sie sollten vor jeder Planung mit einem kleinen Testlauf gemessen werden.

**Folgerung:** Auf dem Mac lohnen sich **kontrollierte, kleine synthetische Aufgabenfamilien** (Regel bekannt, Größe variierbar) und **Test-Time-Experimente auf fertigen Checkpoints**. Der volle ARC-AGI-Leaderboard-Wettlauf lohnt sich nicht.

---

## 5. Ehrliche Einschätzung: Wo ist das Feld voll, wo ist noch Platz?

**Voll bzw. für ein kleines Team kaum zu gewinnen:**
- **ARC-AGI-1/2-Score-Jagd mit TRM-artigen Modellen.** Nach TRM (Okt. 2025) kamen in einem Jahr mindestens 20 Nachfolger: URM, LoopViT, InfiLoop, CHARM, SE-RRM, PTRM, GAR, Mamba-TRM, „What limits RRMs“ und weitere. CHARM steht bei 84 % / 46,7 %. Jeder weitere Punkt kostet Hunderte GPU-Stunden.
- **Kaggle ARC-AGI-2/3:** Dort dominieren 2026 lokal laufende LLMs mit 4–125B Parametern ([2611.06827](https://arxiv.org/abs/2611.06827), [Milestone 1](https://arcprize.org/blog/arc-prize-2026-milestone-1)). ARC-AGI-2 läuft 2026 zum letzten Mal als Kaggle-Wettbewerb.
- **Architektur-Kosmetik an TRM** (Hierarchie, Verschachtelung, Operator-Tausch, autoregressive Varianten): mehrfach als wirkungslos oder klein gezeigt ([HRM-Analyse](https://arcprize.org/blog/hrm-analysis), [2610.26666](https://arxiv.org/abs/2610.26666), [2603.08082](https://arxiv.org/abs/2603.08082), [2602.12078](https://arxiv.org/abs/2602.12078)).
- **Test-Time-Compute-Tricks für TRM auf Sudoku** (Rauschen, Resampling): schon da und fast gesättigt (98,75 % bzw. 99,4 %).

**Noch Platz (passt zu unserem Ziel und zum Mac):**
1. **Echte Größen- und Regel-Extrapolation mit kleinen Netzen.** Genau das bezeichnen die Autoren als ungelöst: SE-RRM löst kein 16×16-Sudoku, Embed-TTT „interpolation, not extrapolation“. Kontrollierte Aufgabenfamilien mit bekannter Regel und variierbarer Größe (vgl. [2607.20594](https://arxiv.org/abs/2607.20594), [Compositional-ARC](https://arxiv.org/abs/2504.01445), [NCA](https://arxiv.org/abs/2609.36126)) sind billig und noch dünn besetzt. Zu prüfen wäre: Welche Priors bringen Extrapolation (Äquivarianz, Lokalität, gewichtsgeteilte Loops, Halting nach T* = n/v)?
2. **Ohne Puzzle-ID, also echte Few-Shot-Induktion:** Die TRM-Familie scheitert ohne ID auf 0 %. Ein kleines Netz, das die Regel *aus den Demos* inferiert (in-context oder per Latent-Suche wie [LPN](https://arxiv.org/abs/2411.08706)) und auf größere Fälle anwendet, wäre der eigentlich interessante Fall. Es ist weniger erforscht als der transduktive Weg. **[Einschätzung]**
3. **Trennung von Regel-Induktion und Regel-Ausführung** (gefordert in [2609.21181](https://arxiv.org/abs/2609.21181) und von ARC Prize „separate knowledge and reasoning“). Messbar auf Datensätzen mit bekannter Regel.
4. **Saubere, budget-kontrollierte Ablationen** mit mehreren Seeds sind in diesem Feld rar. Sie werden ausdrücklich verlangt ([2610.13660](https://arxiv.org/abs/2610.13660), [2610.26666](https://arxiv.org/abs/2610.26666)). Das passt zu einem kleinen Budget, aber nur auf kleinen Aufgaben, nicht auf vollem ARC.
5. **MDL/Kompression pro Aufgabe (CompressARC-Linie):** Für 2026 habe ich keinen direkten Nachfolger gefunden (Websuche ohne Treffer). Pro Aufgabe ist das billig. Das ist eine dünn besetzte Nische **[unsicher, ob sie wirklich leer ist]**.

**Risiko:** Das Feld bewegt sich sehr schnell. Allein im Oktober/November 2026 erschienen 5+ relevante arXiv-Papiere. Vor jeder Projektentscheidung sollte noch einmal gezielt gesucht werden (arXiv-Suche „tiny recursive“, „looped transformer extrapolation“, „ARC“).

---

### Nicht geprüft / Lücken dieser Recherche
- Die Methode von Tufa Labs auf Kaggle 2026 und die Endergebnisse (kommen am 4.12.2026).
- Volltexte von CompressARC, URM, LoopViT, „What Limits RRMs“ (nur Abstracts gelesen); damit fehlen Rechenzeiten.
- E. Pang („Efficient Evolutionary Program Synthesis“), J. Berman, ArcMemo, MindsAI-Paper: nur über die [ARC-Prize-Award-Liste](https://arcprize.org/blog/arc-prize-2025-results-analysis) erfasst, nicht geöffnet. Alle sind LLM-basiert und für „kleine Netze“ weniger relevant.
- Die arXiv-API war zeitweise gedrosselt (HTTP 429). Einige Suchen liefen deshalb nur über die Websuche.
