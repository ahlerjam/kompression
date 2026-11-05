# Zwischenbericht an Jonas

Stand: 2026-11-04 22:35. Grundlage: `docs/ergebnis_0a.md`, `docs/ergebnis_0b.md`, `docs/ergebnis_0a_plus.md` (nach Hauptlauf), `docs/recherche.md` (Neuheitsprüfung). Alle Zahlen aus `runs/`, Seeds 10–19 bzw. 20–29, gepaart, Wilcoxon + Holm, Protokolle vor den Hauptläufen eingefroren.

## 1. Was wir gemessen haben

| Stufe | Frage | Ergebnis |
|---|---|---|
| 0a | Codiert ein Netz mit Kompressionsdruck (mdl, bitnet) die Additionstabelle kürzer als Weight Decay? | **Nein.** mdl 32,5 vs. wd 26,5 kbit (Holm p = 0,004), bitnet 27,1 kbit. Aber: mdl findet ~9× kürzere Gewichte (143 vs. 1.288 kbit), bitnet codiert die letzten Blöcke fast exakt. Kompression verliert früh, gewinnt spät. |
| 0b sub | Überträgt eine komprimierte Additions-Lösung besser auf Subtraktion? | **Ja.** Zufall 29,3, wd-Start 12,5, **mdl-Start 7,4 kbit**; alle vier Hypothesen in allen 10 Seeds (Holm p = 0,008). Permutierte und normgleiche Kontrollen schlechter als Zufall → Struktur wird übertragen. |
| 0b mul | … auf Multiplikation (andere Struktur)? | **Nein, negativer Transfer.** Additions-Starts 40,6 vs. Zufall 27,0 kbit; normgleiche Kontrollen gleich schlecht → Norm-Effekt, kein Strukturtransfer. |
| 0a+ A | Hilft ein Switch-Code (Modellwechsel an Blockgrenzen) der Kompression? (explorativ) | Gewinn kommt fast nur aus dem Wechsel zwischen λ-Werten (26,5 → 18,4 kbit); mdl bringt zusätzlich ~0,2 kbit. |
| 0a+ B1 | Wach-Schlaf-Zyklus: Hilft Komprimieren der eigenen Gewichte im Schlaf beim Weiterlernen? Mehr als bloßes Schrumpfen? | **Teils.** Schlaf mit mdl spart gegenüber nur lernen 28 % (18,3 vs. 25,5 kbit, Holm p = 0,04) und ist besser als sein Norm-Zwilling (24,8 kbit), also nicht nur globale Norm. **Aber einfaches starkes L2-Schrumpfen im Schlaf (wd λ 10) schafft dasselbe** (17,9 kbit, H2 negativ). mdl ist spät präziser, wd λ 10 grokt verlässlicher. Details `docs/ergebnis_0a_plus.md`. |

## 2. Neuheit (ehrlich)

- Bausteine sind bekannt: Wach-Schlaf (DreamCoder), Soft Weight-Sharing (unser mdl), zyklisches Komprimieren/Weitertrainieren (DSD, AC/DC, Cyclic Sparse Training), Kompressionsfortschritt als Curriculum-Belohnung (Mittal & Gupta 2026), Informationsgewinn als RL-Belohnung (RLP, ICLR 2026), Transfer prequentiell gemessen (Zhang et al. 2020).
- Unser Beitrag ist methodisch: vorregistrierte Messung in Bit mit Struktur- und Normkontrollen. Die Kontrollen sind der wertvollste Teil: Sie zeigen zweimal (0b mul, 0a+), dass ein scheinbarer Kompressionseffekt ein Norm-Effekt ist.

## 3. Einschätzung

- **Paper:** Ein Workshop-Paper (z. B. ICLR/NeurIPS-Workshop zu Compression, MDL oder Science of Deep Learning) ist realistisch: „Compression → transfer, measured in bits, with norm controls“. Für eine Hauptkonferenz fehlen Breite (mehr Aufgaben, Modellgrößen) und ein klarer positiver Kerneffekt.
- **Skalierung:** Wir haben **keinen Beleg**, dass der Ansatz besser skaliert als übliches Training. Alles ist ein 2-Schicht-Transformer auf einer Aufgabenfamilie. Die Hinweise sind gemischt: Kompression hilft beim Transfer bei geteilter Struktur und spät in den Daten, schadet früh und bei fremder Struktur, und ein großer Teil scheinbarer Gewinne ist Norm.
- **Wach-Schlaf (0a+):** Die Idee „zwischendurch die eigenen Gewichte komprimieren, dann weiterlernen“ funktioniert messbar (−28 % Bits), aber der Vorteil kommt aus dem gezielten Schrumpfen, nicht aus unserem speziellen Kompressionsziel: L2 mit großem λ ist genauso gut. Das ist ein ehrliches, publizierbares Ergebnis, aber kein Durchbruch.
- **Wo es heiß sein könnte:** Zwei robuste positive Befunde: (a) komprimierte Lösungen übertragen besser auf eine verwandte Aufgabe (0b sub), (b) Schlafphasen mit Schrumpfen beschleunigen das Weiterlernen (0a+). Beides zusammen ist genau die These „Kompression → schnelleres Weiterlernen“. Ob sich das über eine **Aufgabenfolge** aufsummiert (jede neue Aufgabe billiger), ist messbar und der nächste sinnvolle Test.

## 4. Empfehlung

1. **Aufgabenfolge-Gesamtcode (lokal, ~2–3 Tage):** Ein Modell lernt nacheinander mehrere Aufgaben (z. B. +, −, x+y², x−y, …); Endpunkt = Summe der prequentiellen Codelängen über die ganze Folge. Vergleich: nur lernen, Wach-Schlaf mit mdl, Wach-Schlaf mit wd λ 10 (die starke Kontrolle aus 0a+), Norm-Zwilling. Das testet die Kernthese direkt und nutzt unseren einzigen positiven Befund.
2. **Kleine LM-Skalierungskurve (lokal, ~1 Woche):** Dasselbe Messprinzip auf einem kleinen Sprachmodell (z. B. 3–4 Größen bis ~50 M Parameter, Text-Stream prequentiell). Erst das sagt etwas über Skalierung.
3. **Dann TRC-Antrag** mit diesen Ergebnissen und der Skalierungsfrage als Ziel (TPUs für größere Modelle). Text liegt als Entwurf in `docs/trc_antrag.md`; Einreichung erst nach deiner Entscheidung.
4. Stufe 1 (LLM-Proposer) bleibt ruhen, bis 1 oder 2 einen positiven Effekt zeigt.

**Deine Entscheidung:** Weg wie empfohlen, oder anders gewichten (z. B. direkt TRC, oder Paper-Entwurf zuerst)?
