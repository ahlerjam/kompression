# STATUS

**Phase: BAUPHASE** (Regeln in AGENTS.md)

Zuletzt aktualisiert: 2026-11-02 (Loop-Durchgang 1)

## Wo wir stehen

- **0a Kalibrierung (Seeds 100–102 bzw. 0–4, kein Befund):** `none` groked nicht; `wd` groked instabil; `bitnet` groked schnell und stabil; `mdl` β = 0,001/0,01 groked nicht; `subspace` r16 lernt nicht, r64 lernt nur auswendig.
- **Messung:** L(W) Rezept v4 (absolut, Seed+Differenz, Low-Rank, strukturell). L(W) taugt in 0a nicht als MDL-Maßstab (alle Netze ≫ 25 kbit Daten) → primärer Endpunkt: prequentielle Codelänge (mit bezahlter Temperatur-Kalibrierung).
- **Pilot Stufe 1:** Wach-Schlaf-Zyklus läuft end-to-end; Bibliotheken werden vorgeschlagen und genutzt, aber abgelehnt, weil das 1,5B-Modell mit Bibliothek weniger löst.

## Läuft gerade

| tmux | Lauf |
|---|---|
| `s0a-mdl` | Kalibrierung `mdl` β = 0,1, dann 1 |
| `s0a-preq` | Prequential je Arm: wd → none → subspace → mdl → bitnet |
| `rc` | Remote Control „mac-ai“ |

## Nächste Schritte (Loop)

1. Prequential-Ergebnisse auswerten; Protokoll 0a mit prequentieller Codelänge als primärem Endpunkt fertigstellen → Freigabe Hauptlauf anfragen.
2. Pilot: Rauschen des Probe-Werts messen; Qwen3.5-4B testen; nur relevante Bibliotheksfunktionen zeigen (Kosten mitzählen); danach Bandit-Proposer A/B.
3. 0b bauen (Transfer per prequentieller Codelänge, Init aus A vs. Zufall vs. Norm-Kontrolle).

## Wartet auf Jonas

- **Richtungsentscheidung 0a (Info, kein Stopp):** L(W) ist in 0a kein tauglicher MDL-Maßstab (alle Netze ≫ 25 kbit Daten, siehe Logbuch). Ich baue die prequentielle Codelänge als primären Endpunkt ein (gehört ohnehin zu 0b). Einspruch bitte in STATUS oder im Chat.

- **Hauptlauf 0a starten** (Wechsel in die Messphase), sobald Kalibrierungen ausgewertet und Protokoll ausgefüllt sind.
- Optional: Cloud-GPU für Stufe 1 (Kosten grob 50–200 €).

## Bekannte Befunde Engineering

- Metal ist nicht bitgenau deterministisch (ca. 6e-8).
- GPU ist mit einem Lauf fast ausgelastet; zwei parallele Läufe bringen nur ca. 15 % mehr Durchsatz.
- Echtes MDL (β = 1) lernt bei 0a praktisch nicht: Die Daten tragen nur ca. 25 kbit, das Modell hat ca. 400k Gewichte.
