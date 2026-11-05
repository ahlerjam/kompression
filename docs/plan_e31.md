# Plan E31: Wenige Beispiele, richtige Verallgemeinerung

Stand: 2026-11-05 08:00. Entscheidung Jonas (E31): Kompressions-Strang beenden, neues Ziel.

## Ziel

**Ein kleines Netz lernt aus sehr wenigen Beispielen eine Regel und wendet sie richtig auf größere oder neue Fälle an.**

Das verbindet zwei Stellen, an denen der Standard (Transformer + Gradientenabstieg + übliche Regularisierung) qualitativ versagt:
1. Daten-Effizienz: Menschen lernen eine Regel aus 3–10 Beispielen; Netze brauchen Tausende.
2. Verallgemeinerung über das Gesehene hinaus: Netze lernen oft ein Muster statt des Algorithmus und scheitern an längeren oder neuen Eingaben.

Erfolg heißt ein großer, eindeutiger Unterschied (z. B. 90 % gegen 10 % exakte Trefferquote), nicht ein paar Prozent.

## Was wir mitnehmen

- Messkultur: vorregistrierte Protokolle, gepaarte Seeds, Holm, Kontrollen, Negativbefunde berichtet; Prüfstand vor Hauptlauf (E30).
- Lehre aus 0a–0a+: Gegen die **fair abgestimmte** stärkste Standardvariante vergleichen, sonst täuscht ein Effekt (dreimal passiert: 0b mul Norm-Effekt, 0a+ H2, E30 Zeitplan-Effekt).
- Lehre aus der Literatur (`docs/recherche.md`, Nachtrag 2026-11-05): Für formale Sprachen ist die perfekt verallgemeinernde Lösung unter MDL das Optimum, Gradientenabstieg findet sie aber nicht. Der Engpass ist die **Suche**, nicht das Ziel.
- GPU-Streuung: MLX nicht bitgenau reproduzierbar → Schwellen-Effekte mit mehreren Seeds und Wiederholungen absichern.

## Prüfstand „Regeln“ (eigener Benchmark, klein, schnell, objektiv)

Aufgabenfamilien, jeweils mit **Trainingsbereich** (kleine Fälle, wenige Beispiele) und **Testbereich** (größer oder neu):

| Familie | Training | Test | Warum |
|---|---|---|---|
| A Mehrstellige Addition | ≤ 5 Stellen, N ∈ {30, 100, 1000} Beispiele | 10, 20, 40 Stellen | klassischer Fall von Längen-Versagen |
| B Listenfunktionen | k ∈ {3, 5, 10} Beispiele je Regel (umdrehen, sortieren, jedes zweite, verschieben, zählen …), Listen ≤ 8 | Listen 16–64, neue Regeln derselben Art | Regel aus wenigen Beispielen (ähnlich ARC, aber messbar) |
| C Formale Sprachen | aⁿbⁿ, Dyck-1/2, n ≤ 10 | n bis 100 | MDL-Literatur liefert Referenzen |

Metrik: **exakte Trefferquote** im Testbereich (ganze Antwort richtig), dazu Codelänge im Trainingsbereich (beschreibend). Eigene Seeds 300+ für den Prüfstand.

Standard-Kontrollen (fair abgestimmt, gleiche Daten, Rechenzeit berichtet): Transformer mit Weight-Decay-Raster und Dropout, relative Positionen (RoPE), geloopter Transformer mit adaptiver Tiefe (Stand der Technik für Längen-Verallgemeinerung), Testzeit-Training. Optional als Referenz: ein kleines offenes LLM mit Beispielen im Prompt (lokal, ohne Kosten).

## Ideen-Woche: Kandidaten

Jeder Kandidat: kurze Neuheitsprüfung → höchstens 1 Prüftag (2 bei neuer Infrastruktur) → bestanden oder verworfen, beides ins Logbuch.

| # | Kandidat | Warum er den Standard schlagen könnte |
|---|---|---|
| K1 | **Programmsuche mit kürzester Beschreibung**: kleinste Programme in einer einfachen Sprache (DSL), die alle Beispiele erklären; Suche aufgezählt und/oder von einem LLM vorgeschlagen, geprüft an den Beispielen | Ein gefundenes Programm verallgemeinert auf jede Länge; MDL über Programme statt Gewichte umgeht das Suchproblem des Gradientenabstiegs |
| K2 | **Netz mit diskreten Zuständen und Rechenschritten** (z. B. geloopt, Zustände am Ende auf diskrete Werte gezwungen) | Diskrete, schrittweise Rechnung kann einen Algorithmus exakt abbilden statt ihn anzunähern |
| K3 | **Selbst-Curriculum**: Das Netz löst etwas längere Fälle, prüft sich über Invarianten (z. B. Kommutativität, Probe durch Umkehrung) und trainiert auf eigenen geprüften Lösungen | Verlängert den Trainingsbereich ohne neue Daten |
| K4 | **Testzeit-Training mit Symmetrien**: je Aufgabe kurz nachtrainieren, Beispiele per Symmetrie vervielfacht | Bei ARC erfolgreich; Frage: auch für Länge? |
| K5 | **LLM schlägt vor, Prüfer entscheidet, Bibliothek wächst** (DreamCoder-Art, unser früherer Stufe-1-Gedanke): gefundene Teilprogramme werden wiederverwendet | Mit jeder gelösten Regel wird die nächste billiger |
| K6 | **Gewichtssuche nach MDL ohne Gradienten** (evolutionär/diskret) für kleine exakte Netze | Laut Literatur ist die MDL-Lösung das Optimum, nur der Gradient findet sie nicht |

Reihenfolge: zuerst Prüfstand + Standard-Kontrollen (zeigt, wie schlecht Standard bei wenigen Beispielen wirklich ist; ist er schon gut, Prüfstand nachschärfen), dann K1 und K3 (größte erwartete Wirkung, billig), dann K2, K4, K6, K5.

## Gate (E31)

Ein Kandidat kommt in einen Hauptlauf nur, wenn er auf dem Prüfstand
- die **beste fair abgestimmte** Standard-Kontrolle um **≥ 30 Prozentpunkte** exakte Trefferquote im Testbereich schlägt (oder ≥ 90 % erreicht, wo Standard < 50 % bleibt),
- auf **≥ 5 Seeds** und **≥ 2 Aufgabenfamilien**,
- mit gleichem Datenbudget; Rechenzeit wird berichtet (mehr als 10× Standard wird markiert).

Dann: Neuheitsprüfung gründlich, Protokoll einfrieren, Testläufe, Hauptlauf mit 20 Seeds, Auswertung wie bisher.

## Regeln (unverändert)

Höchstens zwei MLX-Prozesse; Geld nur mit Freigabe von Jonas (auch API-Kosten für LLM-Vorschläge in K1/K5: zuerst lokales Modell); keine Ergebnisse ohne Lauf in `runs/`; Abgebrochenes nach `runs/abgebrochen/`; `make test` grün; alter Code und eingefrorene Protokolle bleiben unverändert, neuer Code in `stages/s2_regeln/`; STATUS und Logbuch aktuell, thematisch committen und pushen; TRC-Antrag erst nach Entscheidung von Jonas.
