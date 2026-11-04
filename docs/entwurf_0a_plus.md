# Entwurf Folgeexperiment 0a+ (ENTWURF, nicht freigegeben)

Stand: 2026-11-04 07:00. Nur Diskussionsgrundlage für die Entscheidung mit Jonas (0a+ vs. Stufe 1). Kein Lauf ohne Freigabe.

## Ausgangslage (aus `docs/ergebnis_0a.md`, `docs/ergebnis_0b.md`)

- 0a: `mdl` verliert die prequentielle Gesamtsumme gegen `wd` (32,5 vs. 26,5 kbit), weil es auf kleinen Präfixen später lernt; auf viel Daten codiert es präziser (H5 mdl nicht signifikant, bitnet signifikant). L(W) von `mdl` ~9× kürzer.
- 0b (Subtraktion): `mdl`-Gewichte übertragen besser als `wd`-Gewichte (7,4 vs. 12,5 kbit).
- Offene Frage: Ist der Nachteil von Kompression früh ein Artefakt fester Druckstärke (β konstant) und eines Codes, der nicht zwischen Modellen wechseln darf?

## Fragen und Varianten (Vorschlag)

| | Variante | Hypothese | Aufwand (Laptop, Schätzung) |
|---|---|---|---|
| A | **Switch-Code** über Arme (van Erven, Grünwald & de Rooij 2007): je Block Wahl zwischen `wd`- und `mdl`-Vorhersage, bezahlt | Switch(wd, mdl) < wd; reine Auswertung vorhandener Blockwerte möglich, aber nur als explorative Nachrechnung (Protokoll 0a eingefroren) | Minuten (keine GPU) |
| B | **β wachsend mit Datenmenge** (z. B. β ∝ n/N oder β nur ab Präfix ≥ 2.048) | `mdl`(β(n)) < `wd` prequential | 2 Gitterpunkte × 10 Seeds, ca. 5 h |
| C | **Skalierung** p = 97/197/397 | Vorteil von `mdl`/`bitnet` in späten Blöcken wächst mit Tabellengröße | p = 397: 157k Paare, auf dem Laptop nicht machbar → TRC |

## Vorschlag zur Reihenfolge

1. A sofort als explorative Nachrechnung (klar als „nicht vorab registriert“ markiert), weil sie keine GPU braucht und zeigt, wie groß der mögliche Gewinn ist.
2. B als neues, vorab eingefrorenes Protokoll mit neuen Seeds (20–29), Kalibrierung auf Seeds 100–102.
3. C als Kern des TRC-Antrags.

## Offene Punkte für Jonas

- 0a+ zuerst oder Stufe 1 (Sprachmodelle)? Empfehlung Claude: 0a+ A und B zuerst (1 Tag), weil sie die Lücke „früh vs. spät“ direkt adressieren und den TRC-Antrag stärken; Stufe 1 danach.
- Freigabe für neues Protokoll und Hauptlauf nötig.
