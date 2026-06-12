# Study: fit-workflow-diagnostics

**Umbrella:** `wimda-parity-gap` · Wave B · absorbs the `minos-error-analysis`
candidate (top-scored, 20). **Status:** study → implementing.

A basket of fit-quality and fit-workflow items sharing one engine seam
(`core/fitting/engine.py` + `gui/panels/fit_panel.py`): MINOS asymmetric errors,
the χ² quality band, sequential chain-seeding for scans, mid-fit abort, and a
persistent fit record. Individually small; together one coherent session.

## Scope (five items)

1. **MINOS asymmetric errors** — a *display-only* opt-in overlay. One shared
   minimisation helper (migrad + explicit HESSE + opt-in MINOS) drives all three
   minimiser sites; the asymmetric `+err`/`−err` show in the parameter table and
   result summaries for single, global, grouped **and** count-domain fits (α
   especially, with its α–amplitude correlation context). Works on the
   value-domain model-fit dialog for free, since it routes through `FitEngine.fit`.
2. **χ² quality band** — wire the *already-exact* `core/fitting/fit_quality.py`
   helper through `result_summary.py` and every fit surface: a coloured
   good/poor/overdone verdict chip beside χ²ᵣ with a teaching tooltip.
3. **Sequential chain-seeding** — a "chain from previous run" option for series
   fits (`fit_grouped_series` per-member loop + the Global-tab batch path),
   ordered by the series order key. The WiMDA `itPrevious` analogue.
4. **Mid-fit abort** — a cancellation contract mirroring the MaxEnt one: a core
   `cancel_callback` kwarg + a dedicated `CancelledError`, cooperative checks
   between member fits and (where safe) in-fit via the cost function; a Stop
   button replacing the disabled Fit button while running. No partial result on
   abort.
5. **Persistent fit record (reframed)** — *not* a background append-file. The
   `.asymp` project already stores the latest fit per `(dataset, representation)`
   (`FitSlot.result`) and per batch member (`FitSeries.results_by_run`) — the
   structured equivalents of WiMDA's overwrite `.fit`/`.bfit` snapshots. We
   **enrich** those persisted records with the additive `quality` +
   `uncertainties_asymmetric` keys (free from items 1–2) plus light provenance,
   add a Qt-free `core/fitting/fit_log.py` *formatter* (record → readable block)
   feeding the existing `LogPanel` and an on-demand "Export fit report", and
   surface provenance through existing surfaces. No new window, no schema break.

## Binding decisions (Ben, 2026-06-12)

These are settled and override the original brief where they differ.

### MINOS = display-only overlay

- Additive fields only: `FitResult.minos_errors: dict[str, tuple[lo, hi]] | None`
  and an `"uncertainties_asymmetric"` key in `fit_result_summary`. Shown in the
  results text and the parameter table.
- The meaning of `FitResult.uncertainties` / summary `"uncertainties"` does **not**
  change (symmetric HESSE). Downstream surfaces stay symmetric **on purpose**:
  trend error bars, GLE export, `composite_parameters` propagation, link-follower
  inheritance (`engine.py` ~191–209), and the promoted `alpha_error` (scalar — fed
  the symmetric HESSE σ). MINOS is a per-fit diagnostic overlay, not a new error
  model that propagates.
- Rationale: asymmetric intervals are not closed under the linear error algebra
  those surfaces use; mixing a MINOS lo/hi into quadrature or a trend bar would be
  a category error. See [comparison.md](comparison.md) §MINOS.

### χ² band

- Confidence **R fixed at 0.95** (WiMDA's `Rgoodfit` default), a module constant.
  Configurability is a cheap follow-on (the helper already takes `confidence`).
- The verdict travels as an additive `"quality"` key inside `fit_result_summary`;
  surfaces render the chip + teaching tooltip from it (W7).
- `fit_quality.assess_fit_quality` is already an **exact** reproduction of WiMDA's
  classification — `scipy.stats.chi2.cdf(χ², ν) ≡ Gammp(ν/2, χ²/2)`, same
  overdone/poor/good thresholds, same target band. This item is *wiring*, not new
  numerics. (Verified against `$WIMDA_SRC/src/Model.pas:1483–1508`.)

### Chain-seeding

- Distinct naming: **"chain from previous run"**. *"Warm-start"* is taken —
  `global_fit_wizard.warm_start_source` already names single-fit→global seeding.
  Both are disambiguated in docs.
- Chained seeds for grouped fits MUST pass through the normalised-polarisation
  seed contract (`grouped_time_domain.normalize_to_grouped_contract`:
  amplitudes→1, backgrounds→0) before use (W5).

### Abort (W6)

- Mirrors the MaxEnt contract exactly: a core `Callable[[], bool] cancel_callback`
  kwarg (same name) + a dedicated `CancelledError` in `asymmetry.core`.
- Minimum granularity = cooperative checks **between** member fits in series/global
  loops. In-fit abort via raising from the iminuit cost function is **adopted**
  (decision in [implementation-options.md](implementation-options.md) §Abort): it
  is safe because an aborted `Minuit` object is discarded entirely and no
  `FitResult` is recorded — matching WiMDA's discard-on-abort semantics
  (`$WIMDA_SRC/src/Fitucode.pas:267,561`).
- GUI: bool-flag `cancel()` + `cancelled` signal on the existing fit workers,
  wired like `_launch_maxent_worker`; Stop button enabled-while-busy like
  `maxent_panel`'s.

### Fit record (W14)

- The module is `core/fitting/fit_log.py` / `FitLog` — never "logbook" (that name
  is the run table, `core/data/logbook.py`).
- Reframed to leverage `.asymp` (Ben, this session): the durable artifact is the
  structured record already in the project, enriched additively. `FitLog` is a
  *formatter*, not a file-appending log. External output is an **on-demand**
  "Export fit report" only. In-app surfacing reuses existing panels.

### Trimmed to follow-ons (record, don't build)

- In-batch co-add / re-fit-coadded (needs run-arithmetic's kernel, parallel session).
- The `fgAll` → Poisson cost-factory unification.

## Reference seams

- WiMDA Object Pascal at `$WIMDA_SRC/src` (oracle reading only; GPL — behaviour
  documented, no code copied). MINOS has no WiMDA analogue; its references are the
  iminuit docs + [`docs/porting/candidates/minos-error-analysis`](../candidates/minos-error-analysis/README.md).
- Textbook: Blundell, De Renzi, Lancaster & Pratt, *Muon Spectroscopy: An
  Introduction* (OUP, 2022) — "the textbook" below — for the per-sample
  calibration framing and the statistics of fit assessment.

## Repository awareness

Four sibling Wave B sessions run in parallel (run-arithmetic, spectral-moments,
rrf, python-user-functions). `fit_panel.py` is ours alone (rrf's fit-layer work is
core-only by directive). Shared append-only files (`index.json`, `schema.py`
panel-state keys, toctrees, mainwindow hooks): minimal additive changes only.
