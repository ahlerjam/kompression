# Architektur- und Engineering-Entscheidungen

Stand: 2026-11-02 · Grundlage: Konzept „Kompressionsgetriebene Selbstverbesserung v2“ und vier Recherchen (Quellen in `recherche.md`).
Status: E1–E27 am 2026-11-02 freigegeben.

| # | Entscheidung | Kurzbegründung |
|---|---|---|
| E1 | Stufen 0a/0b: eigener dichter 1–2-Layer-Transformer in MLX | MoE bringt bei der Größe nichts und verschleiert die MDL-Messung |
| E2 | Stufe 1: eingefrorenes MoE-Modell nur für Inferenz, Qwen3.6-35B-A3B 4-Bit (Fallback Qwen3-Coder-30B-A3B) | ca. 80 tok/s und ca. 20 GB auf M4 Pro; RL-Probleme von MoE spielen ohne Training keine Rolle |
| E3 | Stufe 2: **dichtes** Basismodell (Qwen3.5-0.8B/2B-Base, Fallback Qwen2.5-Coder-0.5B/1.5B) | MoE unter GRPO braucht Routing Replay, und der Router driftet; das würde das Kompressionssignal verrauschen |
| E4 | MoE-Wachstum („Experte, wenn er mehr Bits erklärt als er kostet“) erst ab Stufe 4, per Upcycling/BTX | Das Kriterium ist neu und unbelegt, also selbst ein Experiment, kein Fundament |
| E5 | Framework: MLX lokal, PyTorch ab Stufe 2 in der Cloud | MLX ist auf Apple Silicon führend (vmap, compile, mlx-lm); PyTorch-MPS kann kein float64, JAX-Metal ruht seit 2024 |
| E6 | Messcode in reinem numpy (float64), frameworkfrei, im Kern | Identische Messung auf Mac und in der Cloud; Paritätstest MLX↔PyTorch auf den Logits |
| E7 | Primäres MDL-Ziel 0a: variationale Codelänge in Bit mit lernbarem GMM-Prior (Soft-Weight-Sharing) | differenzierbar, direkt in Bit, entspricht dem L(W)-Term; Stand nach Shaw et al. 2026 |
| E8 | L(W) wird **unabhängig vom Trainingsziel** nach festem Rezept gemessen: Low-Rank + Quantisierung unter Loss-Toleranz, dann Entropiecodierung; lzma als zweite Schranke | Kein Arm bewertet sich mit seinem eigenen Maß |
| E9 | StableMax-Arm **ohne** Regularisierung ist Pflicht | Mit StableMax groken Modelle schon ohne Regularisierung (Prieto et al. 2025) |
| E10 | Norm-Kontrolle gleicht die **Trajektorie** der Gewichtsnorm an, nicht nur den Endwert | Die Norm erklärt Grokking zum großen Teil (Omnigrok) |
| E11 | Statistik: ≥ 10 Seeds, gepaart; Endpunkte und Tests (Wilcoxon, Bootstrap-KI, Holm) **vor** dem Lauf festgelegt | Bei 10 Paaren sind nur große Effekte nachweisbar; keine nachträgliche Endpunktwahl |
| E12 | Stufe 1: Belohnung wirkt über **Auswahl** (Best-of-N/Bandit), weil der Proposer eingefroren ist; die Learnability-Baseline bekommt exakt denselben Mechanismus | Sonst vergleicht man Mechanismen statt Belohnungen |
| E13 | Probe-Set versiegelt, Panels rotieren, Fortschritt mit Vorzeichen und ohne Clipping | Goodhart-Resistenz nach 2606.11417 |
| E14 | Bibliothek: Wiederverwendung nur auf zurückgehaltenen Aufgaben zählen; Kontrollen „leere Bibliothek“ und „Self-Consistency“ bei gleichem Compute | „Library Learning Doesn't“: Gewinne kamen bisher meist aus mehr Compute |
| E15 | Compute-Budget pro Bedingung steht im Kern und wird erzwungen | Faire Baseline |
| E16 | Wiederverwenden statt neu bauen: mlx-grokking, Reasoning Gym, Stitch (`stitch_bindings`), Aufgabenformate aus Absolute Zero | Bewährt und lizenzrechtlich offen (ReGAL-Lizenz noch prüfen) |
| E17 | Sandbox für Codeausführung ab Stufe 1: `apple/container` (Linux-VM), darin Subprozess mit Timeout und Ressourcenlimits | Der Executor von Absolute Zero gilt laut Autoren als unsicher |
| E18 | Tracking local-first: pro Lauf `config.json`, `metrics.jsonl`, `meta.json` (Git-SHA, Diff, `uv.lock`-Hash, Versionen, Seed); W&B offline erst für die Cloud | robust, offline, portabel |
| E19 | Configs als frozen dataclasses mit CLI-Override (tyro), kein Hydra vorerst | einfach genug für die Sweep-Größe |
| E20 | Python 3.12 über `uv`, alle Versionen gesperrt (`uv.lock` im Repo) | Reproduzierbarkeit |
| E21 | Destillation nur als Kaltstart in Stufe 2 (Format und Grundlagen aus dem Stufe-1-Modell); Destillationsdaten zählen in L_gesamt; Kontrolle ohne Destillation | Ohne Kaltstart blieb die Proposer-Belohnung in Absolute-Zero-Nachbauten lange bei null; mit Kontrolle bleibt „Selbstverbesserung“ sauber |
| E22 | Hypernetze erst in Stufe 4, optional früher als eigener Arm mit L(W) = Länge des Rezepts | schwer zu optimieren, würde die Antwort in Stufe 0 verschleiern |
| E23 | Fortschritts-Vorhersager ab Stufe 1: ein kleines Modell schätzt den Kompressionsfortschritt einer Aufgabe, nur die besten werden echt gemessen | größter Compute-Hebel; Vorbild MAGELLAN, Herrmann & Schmidhuber 2026 |
| E24 | Auditor im Kern ab Stufe 1: liest Logs, sucht Reward Hacking, triviale Funktionen, Vielfalts-Kollaps; vom System nicht veränderbar | Lehre aus der Darwin Gödel Machine |
| E25 | „Welten mit bekannter Lösung“: Aufgaben aus einer versteckten, selbst geschriebenen Bibliothek erzeugen; messen, ob das System sie wiederfindet | eindeutiger als jeder Benchmark, keine Kontamination |
| E26 | BitNet (ternäre Gewichte, QAT) als zusätzlicher Arm in 0a, nicht als Hauptarchitektur | L(W) exakt messbar; als Hauptarchitektur erst bei Milliarden Parametern sinnvoll |
| E27 | L(W) als Minimum aus „W absolut“ und „Seed (32 Bit) + Differenz zu W0“ (Rezept `uq-entropy-v2`); dazu Arm `subspace`: W = W0 + A·Z·Bᵀ mit festen Zufallsbasen aus dem Seed, nur Z trainiert | Idee von Jonas: Die Initialisierung ist aus dem Seed reproduzierbar und darf nichts kosten; der Unterraum macht die Beschreibungslänge exakt steuerbar (verwandt: Li et al. 2018, VeRA, SeedLM) |
| E28 | Nach 0b zuerst Folgeexperiment 0a+ (Switch-Code-Nachrechnung ohne GPU, dann β wachsend mit Datenmenge auf Seeds 20–29, Protokoll vor Hauptlauf eingefroren), **danach** TRC-Antrag mit allen Ergebnissen; Repo nach 0b öffentlich; Stufe 1 ruht weiter (Jonas, 2026-11-04) | 0a+ adressiert die Schwäche „früh verlieren“ und die Frage, wann altes Wissen hilft; ein Ergebnis stärkt den Antrag (nur eine Einreichung) |
| E29 | 0a+ B als **Wach-Schlaf-Zyklus** (Jonas, 2026-11-04): je Block abwechselnd lernen (wd) und eigene Gewichte komprimieren (mdl), Weiterlernen vom komprimierten Stand; verglichen mit nur lernen sowie Pruning- und Quantisierungs-Schlaf bei gleichem Compute. Ersetzt „β wachsend“ und reines Nachtraining | Nachrechnung (Logbuch 10:43) zeigte β(n) ≈ Switch über β-Gitter (vorab negativ); Nachtraining ohne Rückkopplung trifft die Idee des Konzepts nicht |
| E30 | **Strategie „erst filtern, dann messen“** (Jonas, 2026-11-04 nach 0a+: „fair bleiben und alles verbessern, was geht“): (1) Hauptlauf nur, wenn unser Verfahren auf dem Prüfstand (eigene Seeds 200+, ≥ 5 Seeds, ≥ 2 Aufgabengrößen) **jede** geplante Kontrolle schlägt, auch die stärkste (zurzeit wd λ 10 im Schlaf); (2) erst Verfahren verbessern (Hybrid mdl + L2 nur auf Gewichte, Prior mit schwerer Null-Komponente, Aufteilung Wach/Schlaf, später selbst gesteuerter Schlaf), dann messen; (3) nur Hypothesen, die der Prüfstand stützt; (4) Hauptläufe mit 20 Seeds; (5) Messung weniger abhängig von einem einzigen Grok-Block; (6) Fragen, bei denen Kompression laut Theorie gewinnen sollte (Aufgabenfolge, Transfer, zweiteiliger Code). Kontrollen und Negativbefunde bleiben vollständig berichtet | 0a+ H2 scheiterte, obwohl der Prüfstand das schon andeutete; Ideen vor dem Hauptlauf filtern statt Ergebnisse danach. Garantie gibt es nicht; gescheiterte Prüfstand-Ideen stehen im Logbuch |
| E31 | **Neues Ziel** (Jonas, 2026-11-05 08:00): Kompressions-Strang beenden; neues Ziel: *ein kleines Netz lernt aus sehr wenigen Beispielen eine Regel und wendet sie richtig auf größere oder neue Fälle an*. Vorgehen nach `docs/plan_e31.md`: eigener Prüfstand (Addition-Länge, Listenfunktionen, formale Sprachen), fair abgestimmte Standard-Kontrollen, Ideen-Woche mit Kandidaten K1–K6 (je höchstens 1 Prüftag), Gate ≥ 30 Prozentpunkte exakte Trefferquote im Testbereich auf ≥ 5 Seeds und ≥ 2 Familien; Messkultur aus E30 bleibt | Prüfstand E30: mdl schlägt fair abgestimmtes L2 nicht (Zeitplan-Effekt statt Kompression); auf Lernen in derselben Verteilung ist Standard schon gut, dort ist kein großer Sprung zu holen. Große Effekte nur, wo Standard qualitativ versagt |

## Was das Konzept-Doc ändern sollte

1. In Stufe 0a einen fünften Arm ergänzen: StableMax ohne Regularisierung (E9). Die Norm-Kontrolle als Trajektorien-Kontrolle definieren (E10).
2. Das MDL-Ziel benennen (E7) und L(W) als getrennte Messung definieren (E8).
3. In Stufe 1 klarstellen, dass die Belohnung über Auswahl wirkt (E12), und das Probe-Set als versiegelt mit rotierenden Panels definieren (E13).
4. Die Kontrollen „leere Bibliothek“ und „Self-Consistency“ in Stufe 1 aufnehmen (E14).
5. Unter „Abgrenzung“ ergänzen: 2606.11417 (Signed Compression Progress), SOAR, Skill Self-Play, „Library Learning Doesn't“.
6. Stufe 2 sollte ausdrücklich dicht bleiben (E3); MoE erst in Stufe 4 (E4).

## Querschnitt: Schnittstellen

Jedes Teil hinter einer schmalen Schnittstelle, damit Mac ↔ Cloud und Modell A ↔ Modell B tauschbar sind.

| Teil | Schnittstelle | Kern? |
|---|---|---|
| Welten | `generate(seed) → Aufgabe`, `verify(Aufgabe, Lösung) → ok` | ja |
| Modell | `generate()`, `logprobs()`; MLX lokal, PyTorch/vLLM in der Cloud | nein |
| Bibliothek | Funktionen + Versionen, L(B) in Bit | nein |
| Messer | Bits, prequential, L_gesamt | ja |
| Vorhersager | schätzt Kompressionsfortschritt pro Aufgabe | nein |
| Auditor | liest Logs, meldet Auffälligkeiten | ja |
| Planer | Wach-/Schlafzyklus, Compute-Budget | ja |
| Archiv | Population von Bibliotheks- und Verfahrensvarianten | nein |
