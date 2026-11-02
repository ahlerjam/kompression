# Logbuch

Neueste Einträge oben. Kurz: Datum, was, Ergebnis, Verweis auf `runs/…`.

## 2026-11-02

- Kalibrierung `none` (StableMax, keine Regularisierung, 5 Seeds × 20k, `runs/s0a/20261102-115946_kalib_none`): **kein Seed groked**; Train 100 %, Test 1–3 %, Gewichtsnorm ≈ 510–530. Widerspricht dem Bericht von Prieto et al. für unsere Einstellungen (lr 1e-3, Full-Batch, 40 % Train); Ursache offen (dort teils längeres Training/andere Optimierer). Kalibrierung, kein Befund.
- Kalibrierung `mdl` über β ∈ {0,001; 0,01; 0,1; 1} gestartet (Seeds 100–102, `scripts/kalib_mdl.sh`). Anfangs-KL ≈ 2,2 Mbit (≈ 5,6 Bit/Gewicht bei σ₀ = e⁻⁶).

- Bauphase/Messphase-Regel von Jonas freigegeben (AGENTS.md).
- apple/container installiert; Sandbox-Backend `container`: VM ohne Netz, 4 CPUs, 2 GB, Start 0,7 s; Netzsperre per Test geprüft (E17 umgesetzt).

- Repo angelegt, vier Recherchen ausgewertet (`docs/recherche.md`), Entscheidungen E1–E26 (`docs/entscheidungen.md`).
- Kern und Training gebaut, 16 Tests grün.
- Befund Engineering: Metal-GPU ist nicht bitgenau deterministisch (Abweichung ca. 6e-8); Tests prüfen mit Toleranz.
- Befund Engineering: `mx.vmap` über 10 Seeds ≈ 10 × Zeit eines Seeds (4,7 vs. 47,7 Schritte/s). Kein Parallelgewinn bei diesem Modell.
- Kalibrierungslauf `none` / `wd` gestartet.
