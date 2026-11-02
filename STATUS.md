# STATUS

Zuletzt aktualisiert: 2026-11-02

## Wo wir stehen

Stufe 0a, Bauplan-Schritte 1–5 erledigt: Umgebung, Kern (Bits, prequential, L(W), Statistik),
MLX-Modell mit StableMax, PyTorch-Spiegel, vmap-Training, Run-Log, 16 Tests grün.

## Läuft gerade

- `tmux` Sitzung `s0a-kalib`: Kalibrierung Arm `none` und `wd`, je 5 Seeds × 20.000 Schritte
  (Start 2026-11-02 11:59, Dauer ca. 70 min). Zweck: Grok-Zeitpunkte sehen, um Schrittbudget und
  λ für das Protokoll festzulegen. **Kein Hauptlauf**; Ergebnisse nicht als Befund zitieren.

## Nächste Schritte

1. Kalibrierung auswerten → Schrittbudget und λ in `docs/protokoll_0a.md` eintragen.
2. `objectives.py`: variationales MDL mit GMM-Prior (Arm `mdl`), Norm-Projektion (Arm `wd_normproj`), BitNet-QAT (Arm `bitnet`).
3. L(W)-Messung nach festem Rezept in den Lauf einhängen (alle N Schritte).
4. Protokoll 0a einfrieren, Hauptlauf 5–6 Arme × 10 Seeds (ca. eine Nacht).
5. Auswertung mit `core/stats.py`, Gate 0a entscheiden.

## Offene Punkte

- Durchsatz: vmap bringt kaum Gewinn (ca. 47 Schritte/s pro Seed). Optimierung (Mehrschritt-compile, kleinere Evaluationsfrequenz) prüfen.
- Rechner-Betrieb: Remote Control als Dienst, Weiterlaufen bei geschlossenem Deckel (sudo nötig).
