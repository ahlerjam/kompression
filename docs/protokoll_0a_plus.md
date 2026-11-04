# Protokoll 0a+ B1-Zyklus: Wach-Schlaf-Kompression (ENTWURF, Version 2)

Stand: 2026-11-04 12:25. Freigabe der Variante B1 durch Jonas (2026-11-04). Wird nach der Kalibrierung (Seeds 100–102) eingefroren; erst dann Hauptlauf (Seeds 20–29).

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

mdl β ∈ {0,1; 0,3} (Einstellungen wie Arm `mdl` in 0a: log σ₀ = −6, gleiche Lernrate für alle Parameter), Schlaf `free`, `wd` λ = 10, prune 0,9, bitnet, wd. β wird für den Hauptlauf festgelegt (kleinste mittlere Codelänge); welche Kontrollen in den Hauptlauf gehen, wird vor dem Einfrieren festgelegt. [[Ergebnis eintragen]]

Hinweis aus der Code-Prüfung (Logbuch 12:25): Mit lr 1e-3 bewegen sich log σ und Prior in 250 Schritten kaum; der mdl-Schlaf wirkt dann vor allem über den Zug des Priors auf die Mittelwerte (große Gewichte werden nach [−0,3; 0,3] gezogen). Eine „reparierte“ Variante (Lernrate ×30 für σ/Prior, log σ₀ = −3, Snapping auf Prior-Zentren) ist als Option vorhanden (`--post-mdl-lr-mult`, `--post-mdl-init-log-sigma`, `--post-mdl-snap`), war im Probelauf (p = 23) aber kaum besser als wd.

## Hypothesen (Hauptlauf, Seeds 20–29, gepaart)

| | Hypothese | Endpunkt |
|---|---|---|
| H1 (primär) | Zyklen mit `mdl`-Schlaf codieren kürzer als nur lernen (`wd`) | prequentielle Codelänge |
| H2 | `mdl`-Schlaf kürzer als `bitnet`-Schlaf | prequentielle Codelänge |
| H3 | `mdl`-Schlaf kürzer als `prune`-Schlaf | prequentielle Codelänge |
| H4 (sekundär) | `mdl`-Schlaf präziser bei viel Daten als nur lernen | Bits der letzten beiden Blöcke |

- Wilcoxon gepaart (10 Seeds), Bootstrap-95-%-KI, Holm über H1–H4. Beschreibend zusätzlich: bitnet und prune gegen wd weiter.
- Grenzen: eine Aufgabe, eine Modellgröße, ein Wachverfahren, feste Zyklenzahl; Wahl von β und Pruning-Anteil auf anderen Seeds kalibriert; Prior und Unsicherheiten von `mdl` werden je Schlaf neu initialisiert.
