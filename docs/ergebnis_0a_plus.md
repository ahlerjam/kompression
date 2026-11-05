# Ergebnis 0a+ B1-Zyklus (Wach-Schlaf-Kompression)

Stand: 2026-11-04 22:30. Hauptlauf nach `docs/protokoll_0a_plus.md` Version 2 (eingefroren 2026-11-04 15:20), Seeds 20–29, gepaart, alle Läufe `git_dirty: false`. Auswertung `uv run python stages/s0a_grokking/analyse_b1.py` (Rohausgabe unten unverändert, Kopie `runs/analyse_b1.txt`).

Aufbau: Addition mod 97, prequentiell (Neustart je Block). Je Block 4 Zyklen aus Wach (750 Schritte wd λ 1) und Schlaf (250 Schritte Verfahren X auf den eigenen Gewichten); die nächste Wachphase lernt vom Stand nach dem Schlaf weiter.

## Kurz

**H1, H3, H4 bestätigt, H2 nicht.**

| | Hypothese | Ergebnis | Holm-p |
|---|---|---|---|
| H1 | mdl-Schlaf < nur lernen (wd) | **ja**: 18,3 vs. 25,5 kbit (KI95 der Differenz [−10,8; −2,8] kbit), 9 von 10 Seeds | 0,041 |
| H2 | mdl-Schlaf < wd λ 10 im Schlaf (Norm-Kontrolle) | **nein**: 18,3 vs. 17,9 kbit (KI95 [−2,3; +3,4] kbit), 7 von 10 Seeds | 1,0 |
| H3 | mdl-Schlaf < Norm-Zwilling | **ja**: 18,3 vs. 24,8 kbit (KI95 [−10,1; −2,2] kbit), 9 von 10 Seeds | 0,041 |
| H4 | letzte Blöcke: mdl-Schlaf < nur lernen | **ja**: 787 vs. 1.112 bit | 0,008 |

Beschreibend (ohne Test-Anspruch, Mittel kbit): wd λ 10 17,9; mdl 18,3; bitnet 24,0; Norm-Zwilling 24,8; wd 25,5; free 26,4; prune 29,5.

## Woher der Unterschied kommt (Blockzerlegung, Mittel bit)

| Block | wd | mdl | wd λ10 | mdl_norm | bitnet | prune | free |
|---|---|---|---|---|---|---|---|
| 64–1024 (Summe) | 13.061 | 13.010 | 12.888 | 13.005 | 13.104 | 12.816 | 13.090 |
| **2048** | **10.933** | **4.087** | **1.487** | 10.398 | 9.735 | 11.690 | 12.312 |
| 4096 | 849 | 597 | 2.463 | 751 | 548 | 3.888 | 432 |
| 8192 | 263 | 190 | 619 | 252 | 211 | 697 | 137 |

- Fast der ganze Unterschied entsteht im Block [2048, 4096): Grokt das Netz mit 2.048 Beispielen (≈ 300–1.700 bit) oder nicht (≈ 11.000–13.000 bit)?
- Mit wd-Schlaf grokt dort 1 von 10 Seeds, mit Norm-Zwilling 1, bitnet 1, prune und free 0, **mdl 7 von 10**, **wd λ 10 alle 10**.
- Danach dreht es sich: wd λ 10 codiert die späten Blöcke schlecht (3.083 bit in den letzten beiden gegenüber 787 mit mdl). Starkes Schrumpfen hilft beim frühen Groken, kostet aber Präzision bei viel Daten; mdl ist dort präzise wie die anderen.

## Deutung

1. **Schlafphasen helfen beim Weiterlernen, aber nur mit Schrumpfen.** Ein Schlaf ohne Regularisierung (free) oder mit dem normalen wd bringt nichts; mdl- und starker wd-Schlaf sparen ~7 kbit (28 %).
2. **Es ist nicht bloß die globale Norm.** Der Norm-Zwilling (Wachgewichte nur auf die Norm des mdl-Ergebnisses skaliert) bringt fast nichts (H3). Es kommt darauf an, *welche* Gewichte schrumpfen.
3. **Unser Kompressionsziel ist aber nicht besser als einfaches L2-Schrumpfen** (H2 negativ). wd λ 10 erreicht in der Summe dasselbe, verlässlicher (10/10 Seeds), aber auf Kosten der späten Präzision. Da L2 selbst eine Codelänge unter einem Gauß-Prior ist, lautet der vorsichtige Schluss: Kompression im Schlaf hilft; ein reichhaltiger GMM-Prior hat hier keinen Gesamtvorteil gegenüber dem einfachsten Prior, ist aber spät präziser (beschreibend).
4. Pruning (90 %) schadet; bitnet hilft wenig.

Grenzen: eine Aufgabe, ein Modell, ein Zyklusschema; der Gesamteffekt hängt an einer Schwelle (groken bei 2.048 Beispielen) und ist daher seed-empfindlich (mdl: 3 Seeds ohne frühes Groken); β und λ 10 nicht gemeinsam optimiert.

## Rohausgabe

```
wd       n = 10  Mittel kbit:   25.53  letzte Blöcke Mittel bit:    1112  (20261104-172933_B1_wd)
mdl      n = 10  Mittel kbit:   18.31  letzte Blöcke Mittel bit:     787  (20261104-151733_B1_mdl)
wd_l10   n = 10  Mittel kbit:   17.88  letzte Blöcke Mittel bit:    3083  (20261104-151733_B1_wd_l10)
mdlnorm  n = 10  Mittel kbit:   24.83  letzte Blöcke Mittel bit:    1003  (20261104-170124_B1_mdlnorm)
bitnet   n = 10  Mittel kbit:   24.02  letzte Blöcke Mittel bit:     759  (20261104-192208_B1_bitnet)
prune    n = 10  Mittel kbit:   29.51  letzte Blöcke Mittel bit:    4586  (20261104-192515_B1_prune)
free     n = 10  Mittel kbit:   26.39  letzte Blöcke Mittel bit:     570  (20261104-212212_B1_free)

# Hypothesen (Holm über H1–H4)
{"kontrast": "H1 mdl-Schlaf < nur lernen (wd)", "n": 10, "mittel_a": 18306.69427290885, "mittel_b": 25529.352826809336, "diff_ki95": [-10824.935529232545, -2831.406205933767], "p": 0.013671875, "richtung_wie_erwartet": true, "p_holm": 0.041015625}
{"kontrast": "H2 mdl-Schlaf < wd λ10-Schlaf (Norm-Kontrolle)", "n": 10, "mittel_a": 18306.69427290885, "mittel_b": 17879.9420275665, "diff_ki95": [-2251.9664279023045, 3358.1098617833272], "p": 1.0, "richtung_wie_erwartet": false, "p_holm": 1.0}
{"kontrast": "H3 mdl-Schlaf < Norm-Zwilling (mdl_norm)", "n": 10, "mittel_a": 18306.69427290885, "mittel_b": 24828.465498921258, "diff_ki95": [-10086.279196761549, -2162.38915624495], "p": 0.013671875, "richtung_wie_erwartet": true, "p_holm": 0.041015625}
{"kontrast": "H4 letzte Blöcke: mdl-Schlaf < nur lernen", "n": 10, "mittel_a": 787.1549608740521, "mittel_b": 1112.047496189131, "diff_ki95": [-430.4277480561, -238.46828026294477], "p": 0.001953125, "richtung_wie_erwartet": true, "p_holm": 0.0078125}

# Beschreibend (ohne Test-Anspruch)
{"kontrast": "beschreibend: bitnet gegen wd", "n": 10, "mittel_a": 24020.303370612906, "mittel_b": 25529.352826809336, "diff_ki95": [-2594.2965278995525, -598.7862351042379], "p": 0.013671875, "richtung_wie_erwartet": true}
{"kontrast": "beschreibend: prune gegen wd", "n": 10, "mittel_a": 29514.069507708435, "mittel_b": 25529.352826809336, "diff_ki95": [1234.5011163677484, 7304.113893794192], "p": 0.013671875, "richtung_wie_erwartet": false}
{"kontrast": "beschreibend: free gegen wd", "n": 10, "mittel_a": 26394.411844613584, "mittel_b": 25529.352826809336, "diff_ki95": [-264.10228881328896, 2782.242523137772], "p": 0.921875, "richtung_wie_erwartet": false}
{"kontrast": "beschreibend: wd_l10 gegen wd", "n": 10, "mittel_a": 17879.9420275665, "mittel_b": 25529.352826809336, "diff_ki95": [-9228.896952066587, -5422.50052577012], "p": 0.00390625, "richtung_wie_erwartet": true}
{"kontrast": "beschreibend: mdlnorm gegen wd", "n": 10, "mittel_a": 24828.465498921258, "mittel_b": 25529.352826809336, "diff_ki95": [-1151.5865364666615, -250.396231288834], "p": 0.02734375, "richtung_wie_erwartet": true}
```
