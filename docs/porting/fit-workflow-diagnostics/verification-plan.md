# Verification plan: fit-workflow-diagnostics

Maps each scope item to concrete checks. ✅ = automated test; 👁 = manual/GUI smoke.

## MINOS

- ✅ MINOS vs HESSE on the fixed-seed low-stat KT fit: the skewed parameter's
  `minos_errors` is asymmetric (`|lo| ≠ |hi|`) and differs from HESSE in the
  documented direction; a well-determined parameter has MINOS ≈ HESSE.
- ✅ MINOS runs through the **shared helper** from all three sites: single
  (`FitEngine.fit`), global (`FitEngine.global_fit`), and count-domain
  (`fit_fb_alpha`) all populate `FitResult.minos_errors` when `minos=True` and leave
  it `None` when `minos=False`.
- ✅ Grouped-fit MINOS populates asymmetric errors for free per-group/global params.
- ✅ Count-domain α: asymmetric interval present; **promote payload unchanged**
  (`alpha_error` is the scalar HESSE σ; `promote_alpha_to_grouping` untouched).
- ✅ MINOS-off is the default; `uncertainties` (HESSE) identical with MINOS on vs off
  (MINOS does not perturb the reported symmetric σ).
- ✅ Failed-scan fallback: a parameter whose MINOS scan is invalid keeps its HESSE σ
  and gains no asymmetric entry (no crash).
- 👁 Parameter table renders `value  +hi / −lo` when asymmetric errors are present;
  result summary shows MINOS rows; α shows its correlation context.

## χ² quality band

- ✅ Verdict-boundary fixtures: overdone / good / poor labels at R = 0.95 for crafted
  `(χ², ν)` triples; `band_low`/`band_high` match tabulated χ² quantiles.
- ✅ **Identical verdict from every surface** for the same fit: the `"quality"` dict
  in `fit_result_summary` is the single source; single, grouped, global, and
  model-dialog records carry the same verdict for the same `(χ², ν)`.
- ✅ dof sourcing: `FitResult.dof` populated at the core sites; summary fallback
  inference matches when `dof` is unset.
- ✅ Suppression guard: a surface fitting with non-real errors (if any) yields no
  verdict (`None`) rather than a misleading "good".
- 👁 Chip colour + teaching tooltip render under the post-#53 palette.

## Chain-seeding

- ✅ Synthetic T-scan through a transition: chained seeding converges with fewer
  failures / tighter χ²ᵣ spread than static seeding; ordered by the temperature key,
  not run id.
- ✅ Seed contract: the seed handed to member N+1 has amplitude≡1, background≡0
  regardless of member N's fitted amplitude/background (passes through
  `normalize_to_grouped_contract`).
- ✅ Failed member falls back to the static seed for the next member (no diverged
  seed propagation).
- ✅ Naming/non-collision: the new mode is distinct from
  `global_fit_wizard.warm_start_source`; both reachable, independently.
- 👁 (gated) EuO PSI T-scan confirms convergence through Tc.

## Abort

- ✅ Mid-series abort raises `FitCancelledError`, records **no** partial
  `FitSeries`/`FitSlot`; project state unchanged; a subsequent fit completes.
- ✅ In-fit abort during one long fit raises and yields no `FitResult`.
- ✅ `cancel_callback` defaults to `None`; existing fit tests pass unchanged (no
  behaviour change when not cancelling).
- 👁 Stop button replaces the disabled Fit button while busy; clicking it returns the
  panel to idle with nothing recorded.

## FitLog / persistence

- ✅ Enriched `fit_result_summary` carries `quality`, `uncertainties_asymmetric`,
  `model_name`, `fit_range`, `npar`, `ndof`, `provenance`.
- ✅ `.asymp` round-trips the enriched record on `FitSlot.result` and
  `FitSeries.results_by_run`; older projects without the keys still load (additive,
  no schema bump).
- ✅ `FitLog.format_record` produces a block with run/model/verdict and MINOS columns
  when present; deterministic given a fixed injected timestamp.
- 👁 "Export fit report" writes the readable text for the current dataset / all
  latest fits.

## Regression / harness

- ✅ Existing fit tests (`tests/test_*fit*`, grouped/global/count suites) unchanged
  and green.
- ✅ `python tools/harness.py validate` green (GUI tests `QT_QPA_PLATFORM=offscreen`).
- ✅ `python tools/harness.py docs` builds the user-guide pages clean (MINOS teaching
  page; chain-seeding vs warm-start disambiguation; "assessing a fit" cross-links per
  reconciliation N5).

## Divergences to confirm documented (in comparison.md)

- MINOS overlay-only; downstream surfaces stay symmetric by design.
- Chain-seeding re-normalises to the grouped contract (vs WiMDA's verbatim `p[]`
  carry-forward).
- Abort granularity finer than WiMDA's per-iteration poll (per cost evaluation) with
  identical discard-on-abort semantics.
- Fit record stored structurally in `.asymp` (vs WiMDA's overwrite text files), since
  both represent the same latest-snapshot artifact.
