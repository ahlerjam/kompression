# Protokoll 0a+ B1-Zyklus: Wach-Schlaf-Kompression (EINGEFROREN, Version 2, 2026-11-04 15:20)

Stand: 2026-11-04 12:25. Freigabe der Variante B1 durch Jonas (2026-11-04). Eingefroren am 2026-11-04 15:20 nach Kalibrierung (Seeds 100–102) und Testdurchlauf; Änderungen nur als neue datierte Version. Hauptlauf Seeds 20–29 (`scripts/b1_0a_plus.sh 1` und `2`).

## Frage

Lernt ein Modell neue Daten besser (kürzere prequentielle Codelänge), wenn sich Lernen (Wach) und Komprimieren der eigenen Gewichte (Schlaf) im Zyklus abwechseln und jede Wachphase vom komprimierten Stand weiterlernt? Und ist dabei unsere Kompression (variationale Codelänge mit GMM-Prior) besser als Standard-Kompression im Schlaf?

Version 2 ersetzt Version 1 (reines Nachtraining als Zweig, ohne Rückkopplung). Grund (Jonas, 2026-11-04): Die Idee des Konzepts ist der Wach-Schlaf-Zyklus, in dem sich das Modell selbst komprimiert und weiterlernt, nicht „erst trainieren, am Ende komprimieren“. Version-1-Kalibrierung wurde nach wenigen Minuten abgebrochen (`runs/abgebrochen/*kalB1*`).

**Einordnung (ehrlich):** Unser Schlafverfahren ist Soft Weight-Sharing (Ullrich, Meeds & Welling 2017). Abwechselndes Komprimieren und Weiterlernen gibt es als iteratives Pruning (Han et al. 2015; Lottery Ticket, Frankle & Carbin 2019) und Dense-Sparse-Dense (Han et al. 2017). Neu ist hier die Frage, ob Schlafphasen mit einem Codelängen-Ziel das *Lernen neuer Daten* (prequentiell, in Bit) verbessern, und der Vergleich der Schlafverfahren unter gleichem Compute. Grenze: Das Kompressionsverfahren ist fest vorgegeben; dass das System selbst wählt, *wie* es sich komprimiert, ist Stufe 1/2.

## Aufbau

- Aufgabe, Modell, Optimierer wie Protokoll 0a (Addition mod 97, ganze Tabelle, Blöcke ab 64, Neustart je Block, Self-Switch über Trainingsstände × Temperaturen, bezahlt).
- Je Block 4 Zyklen: Wach = 750 Schritte `wd` λ = 1, Schlaf = 250 Schritte Verfahren X auf den eigenen Gewichten; die nächste Wachphase startet von den deterministischen Gewichten nach dem Schlaf. Optimizer-Zustand wird an jedem Phasenwechsel zurückgesetzt (in allen Bedingungen). Gesamt 4.000 Schritte je Block, für alle Bedingungen gleich.
- Kandidaten für den Self-Switch: c0, c30, c100, c300, w1, s1, … w4, s4 (gleiche Anzahl, gleiche Wahl-Bits für alle).
- Schlafverfahren X:
  - `wd` (Kontrolle: nur lernen, gleiche Schritte und gleiche Optimizer-Resets)
  - `mdl` (unser Verfahren), β aus Kalibrierung, Warmup 50 Schritte je Schlaf
  - `bitnet` (ternäres quantisierungsbewusstes Training, Standard)
  - `prune` (Magnitude-Pruning je Matrix, Maske je Schlaf neu, Nachtraining unter wd; Wachphase darf nachwachsen), Anteil aus Kalibrierung
  - `free` (Kontrolle: Schlaf ohne jede Regularisierung; trennt „mdl hilft“ von „im Schlaf fehlt Weight Decay“)
  - `wd` mit λ = 10 (Norm-Kontrolle: starkes Schrumpfen ohne Codelängen-Ziel; trennt Kompression von reinem Norm-Effekt)
- Code: `prequential.py --cycles 4 --post X` (neue Option; `--cycles 0` bitgenau wie vorher geprüft), Start `scripts/b1_0a_plus.sh`.

## Kalibrierung (Seeds 100–102, kein Befund)

mdl β ∈ {0,1; 0,3} (Einstellungen wie Arm `mdl` in 0a: log σ₀ = −6, gleiche Lernrate für alle Parameter), Schlaf `free`, `wd` λ = 10, prune 0,9, `mdl_norm` β 0,3, bitnet, wd.

Ergebnis (Gesamt-Codelänge je Seed 100 / 101 / 102 in bit, Mittel in kbit; Läufe `runs/s0a_preq/*_kalZ_*`):

| Schlaf | Seeds 100 / 101 / 102 | Mittel |
|---|---|---|
| `wd` (nur lernen) | 26.152 / 25.004 / 26.284 | 25,8 |
| `mdl` β 0,1 | 14.415 / 14.754 / 14.433 | **14,5** |
| `mdl` β 0,3 | 17.850 / 26.418 / 15.250 | 19,8 |
| `wd` λ 10 | 16.777 / 17.505 / 18.373 | 17,6 |
| `mdl_norm` β 0,3 | 16.057 / 24.586 / 25.030 | 21,9 |
| `free` | 26.010 / 26.464 / 26.361 | 26,3 |
| `prune` 0,9 | 27.269 / 28.104 / 27.716 | 27,7 |
| `bitnet` | 20.969 / 27.279 / 20.092 | 22,8 |

Festlegung: **β = 0,1** (kleinste mittlere Codelänge); der Norm-Zwilling im Hauptlauf nutzt dasselbe β. Pruning-Anteil 0,9 (nur ein Wert kalibriert; prune ist im Hauptlauf beschreibend). Alle sieben Bedingungen gehen in den Hauptlauf. Der Unterschied mdl gegen wd entsteht fast ganz im Block [2048, 4096) (ob das Netz dort grokt). Kalibrierungen sind keine Befunde.

Testdurchlauf der Hauptlauf-Befehle (p = 23, Seeds 200–201, `runs/testB1.log`, Läufe `testB1_*`): alle sieben Bedingungen rechnen ohne NaN, die Konfigurationen stimmen, Schlaf-Kandidaten werden im Self-Switch gewählt.

Hinweis aus der Code-Prüfung (Logbuch 12:25): Mit lr 1e-3 bewegen sich log σ und Prior in 250 Schritten kaum; der mdl-Schlaf wirkt dann vor allem über den Zug des Priors auf die Mittelwerte (große Gewichte werden nach [−0,3; 0,3] gezogen). Eine „reparierte“ Variante (Lernrate ×30 für σ/Prior, log σ₀ = −3, Snapping auf Prior-Zentren) ist als Option vorhanden (`--post-mdl-lr-mult`, `--post-mdl-init-log-sigma`, `--post-mdl-snap`), war im Probelauf (p = 23) aber kaum besser als wd.

## Hypothesen (Hauptlauf, Seeds 20–29, gepaart; geschärft nach Prüfstand und Kalibrierung)

Bedingungen im Hauptlauf (Laufnamen `B1_*`): `mdl` (β = 0,1), `wd`, `wd_l10`, `mdlnorm` (Norm-Zwilling von mdl, β = 0,1), `bitnet`, `prune` (0,9), `free`.

| | Hypothese | Endpunkt |
|---|---|---|
| H1 (primär) | Zyklen mit `mdl`-Schlaf codieren kürzer als nur lernen (`wd`) | prequentielle Codelänge |
| H2 | `mdl`-Schlaf kürzer als `wd` λ = 10 im Schlaf (Norm-Kontrolle: schrumpfen ohne Codelängen-Ziel) | prequentielle Codelänge |
| H3 | `mdl`-Schlaf kürzer als sein Norm-Zwilling `mdl_norm` (nach jedem Schlaf werden die Wachgewichte nur auf die Frobenius-Norm des mdl-Ergebnisses skaliert) | prequentielle Codelänge |
| H4 (sekundär) | `mdl`-Schlaf präziser bei viel Daten als nur lernen | Bits der Blöcke [4096, 8192) und [8192, 9409) |

- Wilcoxon gepaart (10 Seeds), Bootstrap-95-%-KI, Holm über H1–H4. Auswertung: `stages/s0a_grokking/analyse_b1.py` (Test `tests/test_analyse_b1.py`).
- Deutung: Nur H1 → Schlaf hilft, aber womöglich bloß über die Norm. H1 + H2 + H3 → mdl-Schlaf hilft über die Norm hinaus (echte Kompression). Nach dem Prüfstand (p = 23) wird ein großer Teil des Gewinns als Norm-Effekt erwartet.
- Beschreibend (ohne Test-Anspruch): `bitnet`, `prune`, `free`, `wd_l10`, `mdlnorm` jeweils gegen `wd`.
- Grenzen: eine Aufgabe, eine Modellgröße, ein Wachverfahren, feste Zyklenzahl; β auf anderen Seeds kalibriert; Prior und Unsicherheiten von `mdl` werden je Schlaf neu initialisiert. Der Gewinn kann fast ganz aus einem einzigen Block stammen (ob das Netz bei 2.048 Beispielen schon grokt); das ist Teil des Ergebnisses, nicht Störung.
