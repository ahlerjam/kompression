# Logbuch

Neueste Einträge oben. Kurz: Datum, was, Ergebnis, Verweis auf `runs/…`.

## 2026-11-02

- Kalibrierung `mdl` β = 0,01 (`runs/s0a/20261102-135612_kalib_mdl_b0.01`): kein Grokking (Test 2–9 %), KL ≈ 0,66 Mbit, L(W) ≈ 0,90 Mbit.
- Prequential ohne Kalibrierung: frühe Blöcke 6,4–10 Bit/Label (> uniform 6,6), weil das Modell auf kleinem Präfix übermütig auswendig lernt. Behoben durch Temperatur-Wahl pro Block aus festem Gitter inkl. uniform, bezahlt mit log2(9) ≈ 3,2 Bit je Block (zweiteiliger Code). Smoke p = 23: 2.100 Bit gegenüber uniform 2.393. Kalibrierung `preq_*` neu gestartet.

- Kalibrierung `subspace` Rang 64 (`runs/s0a/20261102-135741_kalib_subspace_r64`, L(W) v3): Train 100 %, **Test ≈ 1 % (reines Auswendiglernen)**, L(W) = **500–574 kbit** (strukturell: Seed + Z), kürzeste Beschreibung aller Arme.
- **Methodischer Befund zum Design von 0a:** Die 3.764 Trainings-Labels kosten uniform nur **24,8 kbit** (die ganze Tabelle 62,1 kbit). *Jedes* trainierte Netz liegt mit L(W) = 0,5–2,3 Mbit 20- bis 90-mal darüber. Kein Netz ist damit eine Kompression der Daten; L(W)-Vergleiche zwischen Armen sagen in diesem Regime nichts über MDL-Optimalität (der kürzeste Code ist eine Tabelle). Folge: Der Unterraum lernt auswendig *und* ist am kürzesten, `wd` generalisiert *und* ist länger. Der geeignete MDL-Messwert für 0a ist die **prequentielle Codelänge der Daten** (misst, wie gut der Lerner ungesehene Beispiele vorhersagt, und belohnt damit Generalisierung direkt), nicht L(W). Vorschlag: prequentielle Codelänge als primären Kompressions-Endpunkt in 0a und 0b; L(W) als sekundär.

- **Kernänderung (Bauphase):** Rezept `uq-entropy-v4` ergänzt eine Low-Rank-Beschreibung für alle Arme (SVD, Singulärwerte ≥ τ·σ_max, τ-Gitter 0–0,5; nach DeMoss et al.). Nachgerechnet: `none` 1,88–2,28 Mbit (unverändert, keine Low-Rank-Struktur), `wd` **0,93–1,42 Mbit** (gewinnt mit Low-Rank τ = 0,05–0,1), `mdl` β = 0,001 0,91–1,18 Mbit (unverändert, absolut). **Beobachtung (Kalibrierung, kein Befund):** Gegrokte Netze sind niedrigrangig; MDL komprimiert vergleichbar stark, aber nicht niedrigrangig, und generalisiert nicht. Hypothese für 0a: Es kommt auf die *Art* der Kompression an (strukturell vs. Entropie einzelner Gewichte).

- Kalibrierung `mdl` β = 0,001 fertig (`runs/s0a/20261102-125912_kalib_mdl_b0.001`): kein Grokking (Test 1,6–9,3 %), Train 100 %, KL ≈ 0,95–1,07 Mbit, L(W) nach Rezept v2 0,91–1,18 Mbit.
- L(W) nachträglich (`scripts/lw_posthoc.py`, Toleranz 0,05): `none` 1,88–2,28 Mbit (kein Grokking), `wd` 1,73–2,10 Mbit (groked).
- **Beobachtung (Kalibrierung, ungepaart, verschiedene Seeds — kein Befund):** MDL β = 0,001 erreicht die kürzeste Beschreibung, generalisiert aber nicht; `wd` generalisiert bei kaum kürzerer Beschreibung als `none`. Spricht gegen die naive Lesart „kürzer ⇒ generalisiert“, zumindest unter diesem Rezept. Alle L(W) liegen weit über der Datenmenge (≈ 25 kbit) → das Rezept ist eine lockere obere Schranke; Low-Rank-Vergröberung (DeMoss) als Rezept-Variante prüfen, bevor 0a eingefroren wird.

- **Kernänderung (Bauphase):** L(W)-Rezept `uq-entropy-v3`: Arme reichen strukturelle Beschreibungen ein (Unterraum: Seed + Z; BitNet: ternäre Indizes + Skalen); der Kern quantisiert, rekonstruiert über `decode` und prüft den Loss selbst. Grund: Unterraum-Lauf hatte nach v2 ≈ 4 Mbit, strukturell ≈ 0,1 Mbit. Smoke (p = 23): Unterraum 2,4 kbit statt 1,7 Mbit; BitNet 638 kbit statt 984 kbit. Kalibrierung `subspace` r64 mit v3 neu gestartet.

- Kalibrierung `subspace` Rang 16 (3 Seeds × 20k, `runs/s0a/20261102-131727_kalib_subspace_r16`): **lernt nicht** (Train 7,5–8,6 %, Test ≈ 1 %), Delta-Norm wächst auf 320–530. Rang 4 deshalb übersprungen, Rang 64 vorgezogen. Offen: Kapazität vs. Optimierung (Basis-Skalierung, lr). Hinweis: L(W) nach Rezept ≈ 3,8–4,1 Mbit, obwohl die strukturelle Beschreibung (Seed + 3.840 Zahlen à 32 Bit ≈ 123 kbit) viel kürzer wäre → strukturelle Beschreibungen müssen in den Kern.
- Kalibrierung `mdl` β = 0,001 (`runs/s0a/20261102-125912_kalib_mdl_b0.001`): bei 15k kein Grokking (Test 2–10 %), KL sinkt von 2,2 auf ≈ 1,0 Mbit.

- Kalibrierung `wd` (λ = 1, StableMax, 5 Seeds × 20k, `runs/s0a/20261102-121039_kalib_wd`): alle Seeds erreichen Test ≥ 0,99 erstmals bei Schritt 2.200–2.400, **brechen danach wiederholt ein** (3–6 Einbrüche unter 0,9) und sind erst ab 16.700–19.600 dauerhaft ≥ 0,99. Endnorm 62–76. Folgerung fürs Protokoll: Endpunkt „erster Treffer“ ist instabil; zusätzlich „stabil gegrokt“ (≥ 0,99 bis Laufende bzw. ≥ 2.000 Schritte am Stück) und Anteil der Zeit ≥ 0,99. Budget 20k zu knapp für stabilen Endpunkt → 40k.
- Kalibrierung `subspace` (Rang 16/4/64) und `bitnet` gestartet (`scripts/kalib_struct.sh`).

- Pilot-Bench Qwen2.5-Coder-1.5B-Instruct-4bit, ohne Bibliothek, 6 Familien × 4 Aufgaben × 4 Samples (`runs/s1/*bench_prefill`): Pass-Rate gcd 0,94 · count_bits 0,88 · word_sorting 0,64 · letter_counting 0,63 · prime_factorization 0,50 · base_conversion 0,00 (Stichprobe danach 2/8). Verifizierte Programme kosten 1–18 Bit unter dem Modell. 16 Samples dauern 9–26 s bei geteilter GPU. Engineering: Ohne Code-Prefill antwortete das Modell oft direkt (z. B. „6“); Prefill ```` ```python ```` behebt das (count_bits 0 → 0,88).
- Beobachtung für das Design: Weil das Modell die Programme selbst gesampelt hat, sind sie unter ihm sehr wahrscheinlich (wenige Bit). L(y|x,B) ist damit klein gegenüber L(B); der Kompressionsfortschritt muss vor allem aus kürzeren/häufiger richtigen Lösungen auf schwereren Aufgaben kommen. Prüfen: Bits über alle n Samples (Erwartungswert) statt nur der besten Lösung.

- Kalibrierung `none` (StableMax, keine Regularisierung, 5 Seeds × 20k, `runs/s0a/20261102-115946_kalib_none`): **kein Seed groked**; Train 100 %, Test 1–3 %, Gewichtsnorm ≈ 510–530. Widerspricht dem Bericht von Prieto et al. für unsere Einstellungen (lr 1e-3, Full-Batch, 40 % Train); Ursache offen (dort teils längeres Training/andere Optimierer). Kalibrierung, kein Befund.
- Kalibrierung `mdl` über β ∈ {0,001; 0,01; 0,1; 1} gestartet (Seeds 100–102, `scripts/kalib_mdl.sh`). Anfangs-KL ≈ 2,2 Mbit (≈ 5,6 Bit/Gewicht bei σ₀ = e⁻⁶).

- Bauphase/Messphase-Regel von Jonas freigegeben (AGENTS.md).
- apple/container installiert; Sandbox-Backend `container`: VM ohne Netz, 4 CPUs, 2 GB, Start 0,7 s; Netzsperre per Test geprüft (E17 umgesetzt).

- Repo angelegt, vier Recherchen ausgewertet (`docs/recherche.md`), Entscheidungen E1–E26 (`docs/entscheidungen.md`).
- Kern und Training gebaut, 16 Tests grün.
- Befund Engineering: Metal-GPU ist nicht bitgenau deterministisch (Abweichung ca. 6e-8); Tests prüfen mit Toleranz.
- Befund Engineering: `mx.vmap` über 10 Seeds ≈ 10 × Zeit eines Seeds (4,7 vs. 47,7 Schritte/s). Kein Parallelgewinn bei diesem Modell.
- Kalibrierungslauf `none` / `wd` gestartet.
