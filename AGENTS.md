# AGENTS.md: Einstieg für Agenten und Menschen

Dieses Repo prüft empirisch, ob **Kompressionsfortschritt als Lernsignal** zu besserer
Generalisierung und rekursiver Selbstverbesserung führt (Schmidhubers These, mit
heutigen Sprachmodellen). Sprache im Repo: Deutsch; Code-Bezeichner: Englisch.

## Zuerst lesen (in dieser Reihenfolge)

1. `STATUS.md`: Wo stehen wir, was läuft gerade, was ist der nächste Schritt.
2. `docs/entscheidungen.md`: Getroffene Architektur-Entscheidungen (E1, E2, …). Nicht stillschweigend davon abweichen.
3. `docs/bauplan.md`: Schritte der aktuellen Stufe mit Fertig-Kriterien.
4. `docs/protokoll_0a.md`: Vorab registriertes Protokoll der laufenden Stufe.
5. Konzept (extern, Claude Doc): https://claude.ai/artifact/Rtv6Vber1es4JKp8vCcAAU

## Befehle

| Zweck | Befehl |
|---|---|
| Umgebung aufsetzen | `make setup` |
| Tests (schnell, < 30 s) | `make test` |
| Lint/Format | `make lint` / `make fmt` |
| Kern-Integrität prüfen | `make core-check` |
| Smoke-Lauf 0a | `make smoke` |
| Lauf 0a | `uv run python stages/s0a_grokking/run.py --arm wd --seeds 0 1 2` |
| Laufende Jobs | `tmux ls`, anhängen mit `tmux attach -t <name>` |

Alles läuft über `uv run`; nie `pip install` direkt. Neue Pakete: `uv add <paket>`.

## Struktur

```
src/kompression/core/   UNANTASTBARER KERN: Messung (Bits), L(W)-Rezept, Statistik.
                        Reines numpy, float64. Geschützt durch MANIFEST.sha256.
src/kompression/        Lernbarer/experimenteller Code: Daten, Modell (MLX + Torch-Spiegel), Training, Run-Log.
stages/s0a_grokking/    Einstiegsskripte pro Stufe.
tests/                  pytest. Jede Änderung hält `make test` grün.
scripts/                Hilfsskripte (core_manifest.py).
runs/                   Ergebnisse, nicht in Git. Ein Ordner pro Lauf.
docs/                   Entscheidungen, Bauplan, Protokolle, Recherche, Logbuch.
```

## Harte Regeln

1. **Den Kern (`src/kompression/core/`) nie ohne ausdrückliche menschliche Freigabe ändern.**
   Der Test `test_core_manifest` schlägt sonst fehl. `make core-manifest` (Neuschreiben der
   Prüfsummen) nur nach Freigabe ausführen. Grund: Ein System, das seine eigene Messung
   ändern kann, kann „kürzer“ umdefinieren (Konzept, Abschnitt „Unantastbarer Kern“).
2. **Protokolle sind eingefroren, sobald ein Hauptlauf startet.** Arme, Endpunkte, Seeds und
   Tests stehen in `docs/protokoll_*.md`. Änderungen danach nur als neue, datierte Version mit Begründung.
3. **Jeder Lauf über `kompression.runlog.Run`.** So entstehen `config.json`, `meta.json`
   (Git-SHA, Diff, Versionen), `metrics.jsonl` und `results.json`. Hauptläufe nur mit sauberem
   Git-Stand (`git_dirty: false`).
4. **Gepaartes Design:** Für einen Seed sind Daten-Split und Initialisierung in allen Armen identisch.
5. **Gleiches Compute für alle Arme** einer Vergleichsfrage.
6. **Keine Ergebnisse behaupten, die nicht in `runs/` liegen.** Negative Befunde werden
   genauso dokumentiert wie positive.
7. Lange Läufe in `tmux` starten (`tmux new -d -s <name> "caffeinate -i …"`), nie im Vordergrund einer Agentensitzung.
8. Nach jeder Arbeitssitzung `STATUS.md` und `docs/logbuch.md` aktualisieren.

## Technische Hinweise

- Rechner: MacBook Pro M4 Pro, 48 GB, MLX auf der GPU. Unter ca. 40 GB aktivem Speicher bleiben.
- Metal summiert nicht bitgenau: Wiederholungen stimmen bis ca. 1e-6 überein, nicht exakt.
- `mx.vmap` über Seeds funktioniert, bringt bei diesem Modell aber kaum Durchsatz (ca. 47 Schritte/s pro Seed).
- Das MLX-Modell ist rein funktional (Parameter als dict); `model_torch.py` ist sein Spiegel.
  Der Paritätstest muss nach jeder Modelländerung grün bleiben.
- Commits: kleine, thematische Commits mit deutscher Beschreibung.
