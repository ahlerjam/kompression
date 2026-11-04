# STATUS

**Phase: MESSPHASE für Stufe 0b** seit 2026-11-03 22:30 (Protokoll 0b Version 2 eingefroren). Stufe 0a abgeschlossen (Protokoll 0a bleibt eingefroren, Kern gesperrt).

**Plan ab 2026-11-04 (Jonas, E28):** 0b Multiplikation fertig → auswerten → Repo öffentlich (nach Geheimnis-Prüfung) → **0a+** (A: Switch-Code-Nachrechnung ohne GPU; B: β wachsend mit Datenmenge, Seeds 20–29, ca. 1 Tag) → TRC-Antrag mit allen Ergebnissen → Jonas reicht ein. Stufe 1 ruht weiter.

**Ergebnis 0b Multiplikation (11:45): NEGATIV.** Start aus Additions-Gewichten codiert Multiplikation 13,6 kbit länger als Zufall (27,0 vs. 40,6 kbit, Holm p = 0,008, Gegenrichtung); Struktur-, Norm- und mdl-Hypothesen nicht bestätigt (p = 1,0). Alle A-Starts gleich schlecht wie normgleiche Kontrollen → Norm-Effekt, kein Strukturtransfer. Gate 0b (Subtraktion) bleibt bestanden. Messphase 0b beendet. Details `docs/ergebnis_0b.md`.

**Repo öffentlich (12:00, Option a von Jonas):** https://github.com/ahlerjam/kompression

**0a+ A (explorativ, 11:05):** Switch-Code bringt in 0a vor allem Wechsel zwischen λ-Werten (26,5 → 18,4 kbit); mdl zusätzlich nur ~0,2 kbit, bitnet ~1,1 kbit. Kompression gewinnt auch mit Switching nicht gegen wd. Logbuch 11:05.

**Läuft (seit 11:52): 0a+ B1-Kalibrierung** (Jonas: B1, gegen Standardverfahren). tmux `b1-1` (wd weiter, mdl β 0,03, prune 0,9) und `b1-2` (bitnet, mdl β 0,1, prune 0,5), Seeds 100–102, Logs `runs/kalB1_*.log`. Danach Protokoll `docs/protokoll_0a_plus.md` einfrieren, Hauptlauf Seeds 20–29 (~6 h), dann TRC-Antrag.

**Reihenfolge streng nacheinander (Entscheidung Jonas, 2026-11-03 05:50):** erst 0a fertig messen und auswerten, dann Hauptlauf 0b, dann auf Basis beider Ergebnisse über Stufe 1 entscheiden. **Stufe 1 ruht komplett** (keine Läufe, kein Weiterbau). Alle GPU-Ressourcen gehen an das aktuelle Gate (zwei Stränge `H-A`/`H-B`; ein dritter sprengte den Speicher, siehe Logbuch 06:49); Bauarbeit nur für 0b (Vorbereitung, damit 0b direkt nach 0a starten kann).

## Ergebnis 0b Subtraktion für Jonas (2026-11-04 04:57) — Details: `docs/ergebnis_0b.md`

**Gate 0b bestanden.** Alle vier Transfer-Hypothesen signifikant (Holm p = 0,0078, alle 10 Seeds gleiche Richtung):
- Additions-Gewichte (wd) sparen beim Lernen der Subtraktion 57 % der Bits gegenüber Zufall (12,5 vs. 29,3 kbit).
- Übertragen wird Struktur: permutierte oder normgleiche Kontrollen sind sogar schlechter als Zufall (54,4 kbit).
- **Komprimierte Lösungen übertragen besser:** Start aus mdl-Gewichten 7,4 kbit vs. wd 12,5 kbit (H4). bitnet beschreibend 6,9 kbit.
- Zusammen mit 0a: mdl verliert auf der eigenen Aufgabe, findet aber ~9× kürzere Gewichte, die besser übertragen.

**Läuft noch:** Multiplikation (zweites Ziel), ca. 7 h. Danach TRC-Antrag fertig und Frage an Jonas zu 0a+/Stufe 1.

## Ergebnis 0a für Jonas (2026-11-03 22:25) — Details: `docs/ergebnis_0a.md`

**Ehrlich: Die beiden Hauptthesen sind in diesem Versuch widerlegt.**
- H1: Das MDL-Netz codiert die Daten **nicht** kürzer als Weight Decay, sondern länger (32,5 vs. 26,5 kbit; signifikant).
- H4: bitnet ist minimal länger als Weight Decay (27,1 vs. 26,5 kbit; signifikant, aber klein).

**Was bestätigt ist:**
- MDL findet ~9× kürzere Gewichte als wd (143 kbit vs. 1,29 Mbit) und ~5,7× kürzere als die normgleiche Kontrolle → es ist nicht nur die Norm.
- MDL groked stabiler als die normgleiche Kontrolle.
- Bei viel Daten codiert bitnet die Tabelle praktisch exakt (137 vs. 8.599 Bit in den letzten Blöcken); mdl tendenziell auch präziser (nicht signifikant).

**Deutung:** Kompression gewinnt spät, verliert früh, und die Summe misst vor allem „früh“. Genau das kann das Folgeexperiment 0a+ (Switch-Code, β wachsend mit Datenmenge) angehen — Entscheidung nach 0b mit Jonas.

**Läuft jetzt:** Hauptlauf 0b (Transfer in Bit), tmux `0b-1`/`0b-2`, Protokoll 0b Version 2 (Neustart je Block). Dauer grob 1 Tag.

**Ziel danach:** Bewerbung bei Google TPU Research Cloud (nur eine Einreichung pro Person → erst mit Ergebnissen aus 0a/0b). Claude meldet sich bei Jonas, sobald es so weit ist, mit fertigem Antragstext.

## Morgen-Zusammenfassung für Jonas (Stand 2026-11-03, 00:35)

**Was über Nacht geklärt wurde**

1. **Messinstrument (prequentielle Codelänge) funktioniert jetzt fair** — nach vier Korrekturen: Temperatur-Wahl, Self-Switch, Schritte je Block als globale Wahl, und vor allem **Neustart je Block** (der Warmstart ließ MDL auf kleinen Datenmengen irreversibel kollabieren).
2. **Stufe 0b: Transfer ist in Bit messbar.** Additions-Wissen spart beim Lernen der Subtraktion 7,6–26,4 kbit; die Kontrolle mit vermischten Gewichten spart nichts → übertragen wird Struktur.
3. **Stufe 0a, ehrliches Zwischenergebnis (Kalibrierung, 3 gepaarte Seeds):**

| Arm | Grokking (Training) | L(W) | prequential (Neustart, S = 3.000) |
|---|---|---|---|
| `mdl` β = 0,1 | stabil | **110–122 kbit** (kürzeste) | 29,5–29,6 kbit; letzter Block **0,13 Bit/Label** |
| `wd` | instabil | 0,93–1,42 Mbit | **26,7–27,6 kbit**; letzter Block 0,29 Bit/Label |
| `bitnet` | stabil | 674 kbit | **26,7–26,8 kbit**; letzter Block **0,01 Bit/Label** (praktisch exakt) |
| `none` | nein | 1,9–2,3 Mbit | 62,1 kbit (= Raten) |

   MDL findet die mit Abstand kürzesten Gewichte und ist auf großen Datenmengen doppelt so präzise — **lernt aber später** (ab ≈ 4.096 statt ≈ 2.048 Beispielen) und verliert deshalb in der Gesamtsumme. Die primäre Hypothese H1 („mdl codiert die Daten kürzer als wd“) würde nach dieser Kalibrierung eher scheitern.

**Entscheidung (2026-11-03, von Jonas an Claude delegiert, datenbasiert):** Option B — jeder Arm bekommt ein bezahltes Hyperparameter-Gitter (mdl β ∈ {0,03; 0,1; 0,3}, wd λ ∈ {0,3; 1; 3}), Neustart je Block, primär H1 (mdl < wd) und H4 (bitnet < wd) mit Holm; „Präzision bei viel Daten“ als sekundärer Endpunkt. **Hauptlauf 0a gestartet** 2026-11-03 02:37 (tmux `H-A`, `H-B`). Tempo bei geteilter GPU: `H_mdl` ca. 4.000 Schritte/h → Teil A eher 2–2,5 Tage; Ende voraussichtlich 05.11.

**Speicher (04:20):** Stufe-1-Kampagnen pausiert, weil der Swap fast voll war; Hauptlauf hat Vorrang. Fortsetzung nach dem Hauptlauf.

**Stufe 1:** ruht bis nach 0b (siehe oben). Stand beim Anhalten: Kampagnen gebaut; eine Bibliothek wurde einmal angenommen (Learnability-Kampagne), sonst abgelehnt.

Zuletzt aktualisiert: 2026-11-02 (Loop-Durchgang 1)

## Wo wir stehen

- **0a Kalibrierung abgeschlossen bis auf prequential `mdl`** (Seeds 100–102 bzw. 0–4, kein Befund):

| Arm | groked | stabil | L(W) | prequential (ganze Tabelle, uniform 62,1 kbit) |
|---|---|---|---|---|
| `none` | nein | – | 1,9–2,3 Mbit | 62,1 kbit (= uniform) |
| `wd` λ = 1 | ja | instabil, stabil ab 2,4k–17,1k | 0,93–1,42 Mbit | 44,1–45,2 kbit |
| `bitnet` | ja | ja, ab 2,6k–10,8k | 674 kbit | 55,7–56,2 kbit |
| `mdl` β = 0,1 | ja | ja, ab 7,4k–12,4k | 110–122 kbit | läuft |
| `mdl` β ≤ 0,01 | nein | – | 0,9–1,2 Mbit | – |
| `mdl` β = 1 | lernt nicht | – | 7 kbit | – |
| `subspace` r64 | nein (auswendig) | – | 0,5 Mbit | 62,1 kbit (= uniform) |

- **0b gebaut**, Kalibrierung läuft (Transfer add → sub; Zufall, A, A-Norm-skaliert, A-permutiert).
- **Stufe-1-Pilot:** Ziel korrigiert (ungelöst = 1.000 Bit Strafe, Korrektheit vor Kompression). 35B-A3B als Pilotmodell gewählt. Bibliotheken wurden bisher abgelehnt (kleinere Modelle wurden abgelenkt).

## Läuft gerade (Stand 20:01)

| tmux | Lauf |
|---|---|
| `s0a-preq` | prequential v2: bitnet, dann none |
| `s0a-preq-mdl` | wartet; danach mdl mit Strafterm-Normierung ≥ 3.764 (S = 3.000, 1.000) |
| `s0b` | 0b v2: A-Norm-skaliert, A-permutiert (S = 3.000) |
| `s0b-steps` | wartet; danach 0b mit S = 30/100/300 (Zufall, A, A-permutiert) |
| `s1-pilot` | Pilot v3: 35B-A3B, schweres Profil, verifizierte Wiederverwendung |
| `rc` | Remote Control „mac-ai“ |

## Nächste Schritte (Loop)

1. Prequential-Instrument fair machen: mdl mit Normierung prüfen; 0b mit kurzen S prüfen, ob Transfer messbar ist. Erst dann Protokoll 0a (Teil B) finalisieren und Freigabe erneut anfragen.
2. Pilot v3 auswerten; danach Bandit-Proposer A/B mit Kontrollen.

## Wartet auf Jonas

- (Info) **0b-Kalibrierung zeigt messbaren, strukturellen Transfer** (7,6–26,4 kbit, permutierte Kontrolle ohne Effekt). Protokoll 0b v1 liegt als Entwurf vor; Hauptlauf 0b braucht die A-Gewichte aus Hauptlauf 0a.

- **(ZURÜCKGESTELLT: Protokoll wird überarbeitet, siehe Logbuch — prequentieller Endpunkt benachteiligt MDL)** FREIGABE HAUPTLAUF 0a (Wechsel in die Messphase). Protokoll: `docs/protokoll_0a.md` (Version 1). Kurz: Seeds 10–19, Arme `mdl` (β = 0,1), `wd`, `wd_normproj`, `bitnet`, `none`; 40k Schritte; primärer Endpunkt prequentielle Codelänge (H1: mdl < wd); Wilcoxon + Holm; Dauer ca. 1–1,5 Tage GPU. Start mit `scripts/hauptlauf_0a.sh`, Auswertung `stages/s0a_grokking/analyse.py`. Antwort im Chat: „Hauptlauf 0a frei“.
- Optional: Cloud-GPU für Stufe 1 (Kosten grob 50–200 €).

## Bekannte Befunde Engineering

- Metal ist nicht bitgenau deterministisch (ca. 6e-8).
- GPU ist mit einem Lauf fast ausgelastet; zwei parallele Läufe bringen nur ca. 15 % mehr Durchsatz.
- Echtes MDL (β = 1) lernt bei 0a praktisch nicht: Die Daten tragen nur ca. 25 kbit, das Modell hat ca. 400k Gewichte.
