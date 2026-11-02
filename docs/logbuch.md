# Logbuch

Neueste Einträge oben. Kurz: Datum, was, Ergebnis, Verweis auf `runs/…`.

## 2026-11-02

- Repo angelegt, vier Recherchen ausgewertet (`docs/recherche.md`), Entscheidungen E1–E26 (`docs/entscheidungen.md`).
- Kern und Training gebaut, 16 Tests grün.
- Befund Engineering: Metal-GPU ist nicht bitgenau deterministisch (Abweichung ca. 6e-8); Tests prüfen mit Toleranz.
- Befund Engineering: `mx.vmap` über 10 Seeds ≈ 10 × Zeit eines Seeds (4,7 vs. 47,7 Schritte/s). Kein Parallelgewinn bei diesem Modell.
- Kalibrierungslauf `none` / `wd` gestartet.
