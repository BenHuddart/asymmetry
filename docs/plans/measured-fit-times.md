# Measured fit times behind the wizards' slow tag

Status: implemented 2026-09-30 on `feat/measured-fit-times` (off `main` at
44a2a93), not pushed. It follows up
[model-family-picker](model-family-picker.md) decision D3 ("slow is the
registry's `expensive` tier for now"). Decisions from Ben on 2026-09-30; the
implementation details below were refined while reading the code.

## Problem

The model family picker tags a component **slow** when the registry says its
`ComputationalCost` is `EXPENSIVE`. **Leave out slow models** (`skip_slow`)
resolves to `ScopeQuery.max_cost = MODERATE` (`core/fitting/wizard_scope.py`).
The tier is a poor guide to what the user waits for:

1. **The tier is per-call cost, but fits multiply it.** A per-call benchmark
   (2026-09-30, 20k points) found dynamic Gaussian KT at about 0.1–0.3 ms per
   call, yet its wizard fits take about 20 s. F–μ–F + third F costs 17 ms per
   call. A single static number cannot rank these correctly.
2. **Nothing measures template fits.** `core/fitting/wizard_timing.py` times
   wizard *stages* (Stage 1, Stage 2, refinement) for headless callers. It
   does not time the individual template fits, which is where a model's cost
   shows.
3. **Nothing persists timings.** Every wizard run starts from nothing, so
   even a measured number would be lost at the end of the run.

## Settled design (decision log)

- **D1: measure where the fits happen.** Every candidate fit of both wizards
  goes through one function, `_execute_assessment_task`
  (`core/fitting/fit_wizard.py`). This covers the single-run wizard's Stage 1,
  Stage 2, null baselines and refinement, the global wizard's per-run analyses,
  and its completion cells. It runs in a spawn-pool worker, a thread, or
  inline. It times `_assess_candidate_template` with `time.perf_counter` and
  attaches `FitTiming(seconds, points)` to the returned `CandidateAssessment`
  as its new `timing` field. `points` is the fitted record's length, which is
  the analysed (possibly rebinned) record. The timing is plain data, so it
  crosses the process boundary with the assessment.
  - One sample is one template assessment: the whole seed ladder, gates and
    diagnostics. That is what the user waits for. Stage-1 rows are
    call-capped, and refinement rows re-fit with the full ladder. A
    recommendation keeps one row per template, whichever stage produced it.
    The median across templates (D2) and across runs (D3) absorbs the spread.
  - Pool workers run concurrently, so a sample includes the contention of
    the wizard's own pool. That is the time this computer takes to screen, so
    it is the time we want.
  - **Not persisted.** `_serialize_candidate_assessment` writes explicit
    keys, and `timing` is not one of them. A row restored from a cache
    therefore has `timing = None`. This is deliberate, not just the cheap
    option: a restored row's time belongs to a past run and must never be
    recorded again.
  - **The global wizard's screening table carries timings too.** Its per-run
    analyses are single-run recommendations. Its completion cells come from
    the same task function. `GlobalFitWizardScreeningTable.fitted_assessments`
    names the rows the call actually fitted: every row of the analyses it
    generated (`generated_run_numbers`), plus the completion cells. A kept
    cell is "carried through as the very same object" (the contract of
    `_assemble_completed_run`), so completion cells are the completed rows
    that are not, by identity, rows of a source analysis. A reused analysis
    is never recorded twice.
- **D2: attribute to components.** A template is a composite. Its
  seconds-per-1000-points goes to every non-background component it contains.
  One wizard run adds one sample per component: the median over that run's
  timed fits that contain it. For the global wizard, that means the median
  over the whole series' fresh fits, so one series cannot flood the rolling
  window.
  Invariant (in code): *a background component rides along in every template,
  so it is never attributed a time.* The Constant is therefore never timed. It
  falls back to its `cheap` tier and is never slow.
- **D3: the store** is `core/fitting/fit_time_store.py::FitTimeStore`, which
  is Qt-free.
  - It holds `{component: recent samples of seconds per 1000 points}`, keeps
    the last `SAMPLES_KEPT = 9` samples, and its estimate is their median.
  - File format (UTF-8 JSON):
    `{"version": 1, "seconds_per_kpoint": {"DynamicGaussianKT": [0.21, …]}}`.
  - `FitTimeStore.load(path)` reads a path the caller provides. An absent
    file is an empty store: a computer that has never timed a fit is a valid
    state, not an error. A malformed file raises `ValueError` naming the
    path, whether the fault is bad JSON, the wrong keys or version, or a
    sample that is not a finite non-negative number. `save(path)` writes it
    back, creating the folder.
  - **A corrupt store is discarded with a log line.** The GUI catches that
    `ValueError`, logs a warning, starts empty, and overwrites the file on the
    next save. The file is a per-machine cache that only speeds up a
    judgement, and the data it holds can be measured again. Refusing to open
    the wizard over it would put the cache above the analysis.
  - The GUI keeps **one store per process**:
    `gui/utils/fit_times.py::shared_fit_time_store()`, loaded once from
    `QStandardPaths.AppDataLocation/fit_times.json`. Both wizards share it, so
    one window's update is visible in the other and neither window overwrites
    the other's samples. `record_fit_times(assessments)` records a finished
    run and saves the file. It runs on the GUI thread when the result
    arrives. The file is a few hundred bytes, so the I/O is trivial.
  - Tests never touch the real file. An autouse fixture in `tests/conftest.py`
    points `fit_times_path` at the test's `tmp_path` and clears the cached
    store.
- **D4: the slow judgement lives in core.**
  - `FitTimeStore.estimates(datasets) -> FitTimeEstimates` is the one
    function that computes expected seconds per run: the median seconds per
    1000 points × the runs' typical screening length / 1000.
  - The typical length is the **median** over runs of
    `fit_wizard.screening_points(dataset)`. That is the run's length after the
    wizard's cost rebin (`n // max(1, n // _FIT_SAMPLE_BUDGET)`), the same
    point count the samples were normalised by. It deliberately ignores
    `analysis_rebin_factor`'s bandwidth protection, which needs the peak
    analysis. Bandwidth protection can only keep more points, so the estimate
    is a lower bound for a run with a fast line and exact for every other run.
    The raw point count would overstate a 10⁵-bin run tenfold.
  - With no runs there is nothing to scale to, and every component is
    untimed.
  - `FitTimeEstimates` (`core/fitting/wizard_scope.py`) holds the
    per-component seconds. `is_slow(name, definition)` returns whether the
    estimate exceeds `SLOW_SECONDS_PER_RUN = 5.0`. For a component never timed
    on this computer, it returns whether the component is in the registry's
    `EXPENSIVE` tier. `UNTIMED` is the empty judgement.
  - **One source of truth.** The slow set is judged per series, not per run.
    `ScopeQuery.max_cost`, `WizardScope.max_cost` and the scope module's
    `_COST_RANK` are deleted. `resolve_scope`, `resolve_scope_for_dataset(s)`
    and `describe_scope` take `fit_times: FitTimeEstimates = UNTIMED`. With
    `skip_slow`, a component is left out exactly when `fit_times.is_slow` says
    so, and `describe_scope`'s `ScopeComponent.slow` asks the same question.
    The wizard entry points thread the same value down to every resolution:
    `build_fit_wizard_recommendation` and the global wizard's public builders
    take `fit_times` with the same default. Their private helpers take it as a
    required keyword, so a forgotten forward fails loudly. Each window
    snapshots one `FitTimeEstimates` for the picker and for the worker, so
    the models the picker tags are the models the run leaves out.
  - **Cache signatures do not include the slow set.** A scope's signature
    records the question "leave out slow models", not today's answer. The
    store changes after every run, so putting the set into the signature would
    make cached analyses flap whenever an estimate crossed 5 s. The global
    wizard's alphabet is still filtered by the current judgement
    (`_scope_filtered_templates`).
  - **The CLI keeps the registry tier.** `core/workflow/screen.py` never sets
    `skip_slow` and exposes no flag for it, so the judgement cannot change
    what it screens. It resolves with the default `UNTIMED`. It does not read
    the per-machine file, so agent-driven screening stays reproducible across
    computers and in tests.
- **D5: the picker shows it.**
  - `ScopeComponent` gains `estimated_seconds: float | None`, and `slow`
    comes from the judgement.
  - The details panel's **Fitting cost** row read "Slow — its fits dominate
    screening time" or "Quick". It now reads:
    - "Slow — ≈ 12 s per run on this computer" or "Quick — ≈ 0.4 s per run
      on this computer" once timed;
    - "Slow (not yet timed on this computer)" or "Quick" until then.
  - Seconds print to two significant figures, or as a whole number from
    100 s.
  - The pill's **slow** tag and the footer's "K slow: …" line follow `slow`
    unchanged.
  - After a run records its timings, the window re-describes the picker, so
    new estimates show at once.

## Code map (verified 2026-09-30)

- `core/fitting/fit_wizard.py`: `CandidateAssessment` (`:286`) gains
  `timing`. `FitTiming` is defined beside it. `_execute_assessment_task`
  (`:2330`) times the fit. `screening_points` sits beside
  `analysis_rebin_factor` (`:5920`). `build_fit_wizard_recommendation`
  (`:2491`) takes `fit_times` and passes it to `resolve_scope_for_dataset`
  (`:2588`).
- `core/fitting/global_fit_wizard.py`: `GlobalFitWizardScreeningTable`
  (`:1828`) gains `fitted_assessments`. These take `fit_times`:
  `_preview_series_templates` (`:1233`), `_scope_filtered_templates`
  (`:1490`), `build_global_fit_wizard_candidate_portfolio` (`:1539`),
  `_run_single_run_analyses` (`:1710`), the phase-1 builder (`:1852`),
  `build_global_fit_wizard_screening_recommendation` (`:2211`),
  `build_global_fit_wizard_recommendation` (`:3012`) and the staged builder
  (`:3112`).
- `core/fitting/wizard_scope.py`: `FitTimeEstimates`, `UNTIMED`,
  `SLOW_SECONDS_PER_RUN`; `_component_exclusion_reason`, `resolve_scope*`,
  `ScopeComponent`, `describe_scope`.
- `core/fitting/fit_time_store.py` (new): `FitTimeStore`.
- `gui/utils/fit_times.py` (new): `fit_times_path`,
  `shared_fit_time_store`, `record_fit_times`.
- `gui/widgets/model_family_picker.py`: `_DetailsPanel.show_component`.
- `gui/windows/fit_wizard_window.py`: the picker's describer,
  `_create_worker_task` and `_populate_results`.
- `gui/windows/global_fit_wizard_window.py`: `_GlobalAnalysisResult` gains
  `fitted_assessments`. Also changed: `_run_global_fit_wizard_analysis`, the
  picker's describer, `_create_worker_task`, `_populate_results` and the
  Setup preview portfolio.
- Docs: `docs/reference/fit_wizard.rst` § "Choosing which models to screen",
  `docs/reference/global_fit_wizard.rst` where it mentions slow models, and
  CHANGELOG `[Unreleased]`.

## Phases

Each phase is one commit on `feat/measured-fit-times`.

1. **Plan.** This file and the `docs/PLANS.md` entry.
2. **Core capture.** `FitTiming`, `CandidateAssessment.timing`, timing in
   `_execute_assessment_task`, and `GlobalFitWizardScreeningTable.fitted_assessments`.
   Tests: a real tiny assessment carries a timing, restored rows do not, and
   the table's fitted rows exclude reused ones.
3. **Store and judgement.** `FitTimeStore`, `screening_points`,
   `FitTimeEstimates`, the scope resolution change and the `fit_times`
   threading. Tests inject timings; nothing asserts on wall time.
4. **Picker.** `ScopeComponent.estimated_seconds` and the details wording.
5. **Wiring.** `gui/utils/fit_times.py`, both windows and the conftest
   isolation. Tests use a `tmp_path` store.
6. **Docs.** The reference pages and CHANGELOG.

## Follow-ups (not this PR)

- The Stage-2 variant budget (`_stage2_variant_budget`) still reads the
  `EXPENSIVE` tier. It could read the same judgement once the estimates have
  settled.
- A long HIFI run whose fast line blocks rebinning is estimated low (D4). If
  that matters, the recommendation's `analysed_points` could feed a
  per-run-length correction.
