# Protokoll Stufe 0a (ENTWURF, noch nicht eingefroren)

**Frage:** Hilft ein MDL-Ziel beim Generalisieren mehr als Weight Decay, wenn man Gewichtsnorm und StableMax kontrolliert?

## Aufbau

- Aufgabe: (a + b) mod 97, Eingabe `[a, op, b, =]`, Trainingsanteil 0,4, Split pro Seed fest.
- Modell: dichter Transformer, 2 Layer, d = 128, 4 Köpfe, MLP 512, float32.
- Optimierer: AdamW, lr = 1e-3, β = (0,9; 0,98), Full-Batch.
- StableMax in allen Armen.
- Seeds: 0–9 (10), gepaart über alle Arme.
- Schrittbudget: _nach Kalibrierung festlegen_ (vorläufig 20.000).

## Arme

| Arm | Beschreibung | Status |
|---|---|---|
| `none` | keine Regularisierung | gebaut |
| `wd` | Weight Decay, λ aus Kalibrierung (Seeds 100–102, nicht 0–9) | gebaut |
| `mdl` | variationale Codelänge, GMM-Prior (K = 16, eine Komponente fix bei 0) | offen |
| `wd_normproj` | Weight Decay, Normtrajektorie auf `mdl` projiziert | offen |
| `bitnet` | ternäre Gewichte (QAT), L(W) exakt | offen |
| `mdl_normproj` | optional: MDL, Normtrajektorie auf `wd` projiziert | offen |

## Endpunkte (primär)

1. Grok-Schritt: erster Schritt mit Test-Accuracy ≥ 0,99; Nicht-Grokking = zensiert beim Budget.
2. L(W) in Bit nach `core/weight_bits.py`, Rezept `uq-entropy-v1`, Toleranz ε = _festlegen_.
3. Finale Test-Accuracy.

## Hypothese und Test

- H1: `mdl` < `wd_normproj` in L(W) und Grok-Schritt.
- Test: Wilcoxon gepaart, Bootstrap-KI der Differenz, Holm über alle Kontraste gegen `mdl`.
- Gate: H1 signifikant (Holm-adjustiert p < 0,05) **oder** sauber dokumentierter Negativbefund.
