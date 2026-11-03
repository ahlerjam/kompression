# STATUS

**Phase: BAUPHASE** (Regeln in AGENTS.md)

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

**Deine Entscheidung für den Hauptlauf 0a** (eine der Optionen, oder eigene):
- **A) Protokoll so lassen** (H1 primär) und ehrlich messen — ein Negativbefund ist auch ein Ergebnis.
- **B) β als bezahlte globale Wahl** (z. B. β ∈ {0,01; 0,03; 0,1}, wie S): MDL darf seinen Kompressionsdruck an die Datenmenge anpassen; fairer im Sinne der MDL-Theorie, aber mehr Rechenzeit.
- **C) Primären Endpunkt auf „Präzision bei viel Daten“ (letzte Blöcke) legen** — trifft die These besser, wäre aber eine Festlegung *nach* Sicht auf Kalibrierungsdaten (muss so benannt werden).

Meine Empfehlung: **B**, mit A-Auswertung als Sekundärergebnis. Antwort einfach im Chat.

**Stufe 1:** Kampagnen (Kompression vs. Learnability vs. Zufall) laufen; erste Bibliothek wurde in einer Learnability-Kampagne angenommen. Ergebnisse folgen.

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
