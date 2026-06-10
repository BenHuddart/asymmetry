# Simulate mode: implementation options and agreed plan

Decisions confirmed with Ben 2026-06-10 (checkpoints 1 and 3); the full
decision log is in [README.md](README.md). This document records the options
considered, the chosen design, and the step-by-step implementation plan.
Implementation can start cold from this file plus the sibling study docs.

## Options considered

### A. Sampling core location

1. **New Qt-free `core/simulate.py`** ← **chosen**. Matches the umbrella
   brief, keeps the core scriptable, gives the other portfolio projects a
   clean import for synthetic test data.
2. Extend `docs/screenshots/data/archetypes.py` in place — rejected: dev-only
   path, not part of the installed package, wrong layering.

### B. NeXus writing strategy

1. **Minimal standalone ISIS muon NeXus V1 writer** ← **chosen** (decision
   4). Writes exactly the datasets `NexusLoader._load_v1` consumes plus a
   `/run/simulation` provenance group. Works for templates from any loader
   (PSI `.bin`/`.mdu`, ROOT), no stale logs, shape freedom, exact t0.
2. Template-copy à la WiMDA (`shutil.copyfile` + h5py overwrite) — rejected:
   NeXus-only templates, shape lock, stale sample logs masquerading as
   simulation provenance. (Unlike WiMDA's `NXacc_rdwr`, h5py *could* reshape
   — but the copy approach's other costs stand.)
3. V2 writer — rejected for now: V1 is the simpler contract, loads through
   the same `NexusLoader`, and matches WiMDA's own output convention.

### C. Degrade sampling law

1. **Binomial thinning for f < 1; `Poisson(k·f)` for f > 1** ← **chosen**.
   Thinning a Poisson process is exactly Poisson — the statistically correct
   "shorter run". See comparison.md divergence 4 for the WiMDA contrast and
   the f > 1 over-dispersion caveat.
2. WiMDA-exact `Poisson(k·f)` everywhere — rejected: measurably
   over-dispersed for the headline f < 1 use case.

### D. GUI shape

1. **Modeless-launched modal `QDialog` from the File menu, reusing
   `FitFunctionBuilderDialog` for the model and the registry parameter
   metadata for the table** ← **chosen**; degrade as a small modal prompt
   from the Data Browser context menu.
2. A new `QMainWindow` tool window — rejected: no plot surface needed; the
   result lands in the existing browser/plot workflow.

## Chosen design

### `src/asymmetry/core/simulate.py` (new, Qt-free)

Public API (final signatures may grow keyword-only options, never lose
these):

```python
def simulate_run(
    template: Run,
    model,                      # CompositeModel | Callable[[ndarray], ndarray] (percent asymmetry)
    parameters: Mapping[str, float] | None = None,   # bound into model when CompositeModel
    *,
    total_events: float,        # expected envelope budget (events, not MEv)
    seed: int = 0,
    alpha: float | None = None,         # default: template grouping alpha
    background_per_bin: float = 0.0,    # expected flat counts/bin/detector
    run_number: int | None = None,      # default: allocator-friendly None → caller assigns
    title: str | None = None,
) -> Run
```

- Resolves the instrument template from `template`: detector count, bins,
  bin width, per-detector `t0_bin`, good-bin window, grouping dict
  (forward/backward groups, α, good_frames), field/temperature/instrument
  metadata.
- Forward model per detector d in group g (fractional a = model/100):
  `expected_d(i) = n0_d · exp(−t_i/τ_μ) · (1 ± a(t_i)) + b` for `t_i ≥ 0`,
  `expected_d(i) = b` for `t_i < 0`, with `t_i = (i − t0_bin_d)·Δt`,
  `τ_μ = MUON_LIFETIME_US` from `core/utils/constants.py`, `+` for the
  forward group and `−` for the backward group.
- Envelope normalisation (WiMDA-equivalent): `n0 = total_events/n_det ·
  Δt/τ_μ`; α split at group level `n0_F = 2·n0·α/(1+α)`,
  `n0_B = 2·n0/(1+α)`, divided equally among each group's member detectors.
- Single `np.random.default_rng(seed)`; one `rng.poisson(expected)` per
  detector array.
- Returns a `Run` whose grouping is a copy of the template's with
  `deadtime_correction: False` and `dead_time_us` zeroed, and whose
  metadata carries `"synthetic": True` plus a `"simulation"` dict:
  model expression, parameter values, seed, total_events,
  background_per_bin, alpha, template run number + source file.

```python
def simulate_run_from_group_signals(
    template: Run,
    group_signals: Mapping[int, Callable | ndarray],  # group id → fractional a_g(t)
    group_n0: Mapping[int, float] | None = None,      # group id → per-detector n0 weight
    *,
    total_events: float, seed: int = 0, background_per_bin: float = 0.0, ...
) -> Run
```

The scriptable per-group seam (decision 6): `simulate_run` is a thin wrapper
that maps the single model to `{forward_gid: +a, backward_gid: −a}` with the
α-split weights. Multi-group simulation (per-group phases etc.) is available
to scripts and other portfolio projects from day one; only the *dialog* is
F/B-scoped in v1.

```python
def degrade_run(run: Run, factor: float, *, seed: int = 0) -> Run
```

- f < 1: `rng.binomial(counts.astype(int64), factor)` per detector; f ≥ 1:
  `rng.poisson(counts · factor)` (WiMDA branch, documented caveat); f = 1 →
  identity copy with new provenance.
- Result is a new `Run` (decision 3): histograms replaced, grouping and
  metadata copied, `metadata["degraded"] = {"factor", "seed",
  "source_run_number"}`, run_label/title badged.

```python
def expected_counts(template, group_signals, ...) -> list[ndarray]
```

Expectation mode (no sampling) exposed for tests and the verification plan's
bin-by-bin checks.

Archetypes promotion (overlap only): move the body of
`_build_run_with_detector_asymmetries` and `_poisson_errors` into
`core/simulate.py` as `build_run_from_detector_asymmetries(...)` and
`poisson_asymmetry_errors(...)` with the muon lifetime taken from
`core/utils/constants.py` (archetypes' local `MUON_LIFETIME_US = 2.197`
rounds the canonical 2.1969811 — the promoted code uses the canonical
constant; screenshot regeneration tolerance is unaffected at 4 s.f.).
`docs/screenshots/data/archetypes.py` then imports both from core and keeps
zero local synthesis logic. `simulate_run_from_group_signals` and
`build_run_from_detector_asymmetries` share the same internal histogram
builder.

### `src/asymmetry/core/io/nexus_writer.py` (new)

```python
def write_nexus_v1(run: Run, path: str | Path) -> None
```

Emits (h5py, HDF5) exactly the `_load_v1` contract, verified against
`core/io/nexus.py:115–225` and `:685–760`:

| Path | Content |
|---|---|
| `/run/analysis` | `"muonTD"` (drives V1 layout detection) |
| `/run/IDF_version` | `1` |
| `/run/number`, `/run/title`, `/run/notes` | run number; synthetic-badged title; provenance one-liner |
| `/run/start_time`, `/run/stop_time` | ISO timestamps (creation time) |
| `/run/good_frames` | from template grouping (default 1) |
| `/run/instrument/name` | template instrument |
| `/run/instrument/detector/orientation` | template orientation when known |
| `/run/sample/temperature`, `magnetic_field`, `magnetic_field_state` | from run metadata |
| `/run/histogram_data_1/counts` | `[n_det, n_bins]` int |
| `/run/histogram_data_1/corrected_time` | bin centres `(i − t0_common)·Δt` μs |
| `/run/histogram_data_1/grouping` | per-detector group ids, forward → 1, backward → 2 |
| `/run/histogram_data_1/dead_time` | zeros `[n_det]` |
| `/run/histogram_data_1/time_zero` | per-detector **t0 bin indices** (V1 `time_zero_is_microseconds=False`; scalar collapses are accepted but per-detector preserves PSI-style staggered t0) |
| `/run/histogram_data_1/first_good_bin`, `last_good_bin` | good-bin window |
| `/run/simulation/*` | model expression, parameters (JSON string), seed, total_events, alpha, template identity — ignored by the loader's required path, surfaced via `nexus_fields` |

Writer notes: one period only (`histogram_data_1`); α is **not** invented
into the standard layout (ISIS V1 has no α field — comparison.md divergence
9); strings as fixed-length ASCII for h5py portability.

### GUI

`src/asymmetry/gui/windows/simulate_dialog.py` (new) — `QDialog`, conventions
from `grouping_dialog.py` / `run_info_dialog.py`:

- Template combo (loaded runs from the Data Browser; preselects current
  selection). Requires ≥ 1 loaded run with histograms (decision 1); the menu
  action is disabled otherwise.
- Model row: formula label + "Edit Model…" → `FitFunctionBuilderDialog`
  (`gui/panels/fit_function_builder.py`, domain "time").
- Parameter table (name, value, unit from `param_info`): seeded from the
  template run's entry in `FitPanel`'s per-run single-fit state when one
  exists (decision 5), else registry defaults.
- Events (MEv) double-spin (default: template's realised events when
  available, else 10); background counts/bin spin (default 0); "Fixed seed"
  checkbox + spin (default checked, 0).
- Buttons: Generate (core call → `DataBrowserPanel.add_dataset`, badged) and
  "Save as NeXus…" (enabled after Generate; `QFileDialog` → `write_nexus_v1`).

Data Browser: context-menu action "Degrade Statistics…" (single-run
selection) → small modal (factor double-spin 0.01–100 default 0.5, seed) →
`degrade_run` → `add_dataset`. Badging: synthetic/degraded runs get a
distinct label decoration (reuse the existing run_label convention, e.g.
"SIM 90001" / "1234 ÷2"; exact rendering follows existing browser display
code — keep minimal, polish pass comes later).

`mainwindow.py`: one menu action ("Generate Synthetic Run…", File menu after
"Open Data File(s)…") + its handler. Keep both this and the data_browser
touch additive — Wave A conflict rule.

## Ordered implementation plan

Each phase ends with full `python tools/harness.py validate` green (from the
project venv) and a local commit. No pushes.

**Phase 1 — sampling core.**
1. `core/simulate.py`: internal histogram builder + `expected_counts` +
   `simulate_run_from_group_signals` + `simulate_run` + provenance.
2. `degrade_run` (binomial/Poisson branches).
3. Promote `build_run_from_detector_asymmetries` / `poisson_asymmetry_errors`;
   rewire `docs/screenshots/data/archetypes.py` to import them
   (`tests/docs/test_archetypes.py` must stay green — screenshot
   determinism contract).
4. Tests: `tests/test_simulate.py` — verification-plan §1 (envelope, α
   split, signal forwarding in expectation mode, per-detector t0, pre-t0,
   determinism, provenance) and §4 (degrade mean, 1/√f error scaling,
   thinning law, provenance, source untouched).

**Phase 2 — NeXus round trip.**
5. `core/io/nexus_writer.py::write_nexus_v1` per the table above.
6. Tests: `tests/test_nexus_writer.py` — verification-plan §2 (file
   identity through `NexusLoader`, provenance survival via `nexus_fields`,
   per-detector t0 round trip, tmp_path only).
7. Refit-recovery test (single seed) + pull-distribution test (≥ 100 seeds,
   small histograms; runtime budget ≲ 20 s) — verification-plan §2–3.
8. Corpus round-trip test, skip-if-missing (nickel HDF5; EuO `.bin`
   template → writer → reload) — verification-plan §2.

**Phase 3 — GUI.**
9. `gui/windows/simulate_dialog.py` + File-menu hook in `mainwindow.py`.
10. Degrade action + prompt in `gui/panels/data_browser.py`.
11. Badging of synthetic/degraded entries in the browser (minimal).
12. Tests: `tests/test_simulate_dialog.py` (offscreen; seeding from fit
    state, generate→browser, save-as-NeXus to tmp_path) and degrade-action
    coverage in the data-browser test module — verification-plan §5.

**Phase 4 — docs + closeout.**
13. `docs/user_guide/simulation.rst`: pedagogical page in the established
    register (result-first physics prose, rendered math, "when to use
    this" diagnostic for simulate and degrade, the Poisson-of-expected-counts
    rationale, the f > 1 caveat, worked GUI + scripting examples); hook into
    `docs/user_guide/index.rst`; references list (textbook, APS style).
14. API docs entry for `core/simulate.py` / `nexus_writer.py` under
    `docs/api/`; `python tools/harness.py docs` clean.
15. Update this study's README/comparison with any implementation-time
    discoveries; flip `docs/porting/index.json` status to "implemented";
    final validate + commit.

## File-by-file touch list

| File | Action |
|---|---|
| `src/asymmetry/core/simulate.py` | **new** — sampling core, degrade, promoted archetypes helpers |
| `src/asymmetry/core/io/nexus_writer.py` | **new** — minimal V1 writer |
| `src/asymmetry/gui/windows/simulate_dialog.py` | **new** — dialog |
| `src/asymmetry/gui/mainwindow.py` | +1 File-menu action + handler (additive; Wave A shared file) |
| `src/asymmetry/gui/panels/data_browser.py` | + context-menu action, degrade prompt, badge decoration (additive; Wave A shared file) |
| `docs/screenshots/data/archetypes.py` | synthesis helpers replaced by core imports |
| `tests/test_simulate.py`, `tests/test_nexus_writer.py`, `tests/test_simulate_dialog.py` | **new** test modules |
| `tests/docs/test_archetypes.py` | only if promotion changes imports it checks |
| `docs/user_guide/simulation.rst`, `docs/user_guide/index.rst`, `docs/api/*` | user + API docs |
| `docs/porting/simulate-mode/*`, `docs/porting/index.json` | study updates, status flips |

Not touched: `core/project/schema.py` (decision 2 — NeXus-file persistence
needs no schema change), loaders, fitting engine, plot panels.

## As implemented (2026-06-10)

All four phases landed as planned, each with full validate green. Deviations
and discoveries beyond the plan:

- **Envelope normalisation refined**: the per-bin rate at t = 0 uses the
  exact telescoping form `n0 = N_d·(1 − exp(−Δt/τ_μ))` rather than WiMDA's
  first-order `N_d·Δt/τ_μ`, so the zero-signal window total equals
  `total_events·(1 − exp(−T/τ_μ))` to machine precision (tested at
  rtol 1e-12).
- **α as group weights, budget-normalised**: weights are normalised over the
  assigned detectors, so the event budget is independent of α and of group
  imbalance (WiMDA's per-histogram allocation drifts for unequal groups).
- **Error-model finding (verification §2)**: against a known truth, χ²ᵣ
  centres on E[(1−A²)/(1+A²)] < 1, not 1 — the shipped
  `compute_asymmetry` error formula propagates F ± αB as *independent*,
  over-estimating σ_A by (1+A²)/(1−A²) relative to exact Poisson
  propagation (4α²FB(F+B)/(F+αB)⁴). Exact at A = 0; ≈ 9 % in variance at
  A = 0.21. Affects all fits, not just synthetic data; the refit test
  documents and centres on the analytic expectation. Recorded as a
  follow-on investigation below.
- **Run-number allocation**: synthetic/degraded runs draw from a 90001+
  series reserved through `DataBrowserPanel.next_derived_run_number()`
  (the dialog accepts an allocator so browser-side degrades and dialog
  generates cannot collide).
- **Project-file note**: a synthetic run has `source_file = ""`, so saving
  a project containing one records a dataset that cannot be re-loaded
  (decision 2: persistence is via Save-as-NeXus). The user guide says so;
  a save-time warning is a possible refinement.
- `tests/test_simulate_dialog.py` added to the E402 per-file-ignores in
  `pyproject.toml` (the established GUI-test convention).

## Recorded follow-ons

- **Built-in ideal-instrument template** (no loaded run required) — the
  teaching case; needs a template registry decision first.
- **Multi-group dialog support**: per-group amplitude/phase table seeded
  from a grouped fit (`grouped_time_domain` nuisance params); the core seam
  already exists.
- **Double-pulse + two-period + count-mode simulation** — for
  `count-domain-fit-modes` to claim (`Simulate.pas` lines 76–79 mechanics
  documented in README/comparison).
- **Deadtime-distortion injection** (simulate the non-paralyzable count
  loss, write real deadtimes, exercise the correction end-to-end).
- **Archetype gallery** ("simulate the EuO T-scan") — candidate-era idea,
  still attractive for teaching once the dialog exists.
- **Pull-distribution diagnostic in the GUI** (re-simulate-and-refit from a
  fit result) — natural extension of the verification machinery.
- **Asymmetry error-formula investigation**: decide whether
  `compute_asymmetry` should move to exact Poisson propagation
  (1−A²)/(F+αB)-style instead of the independent-numerator/denominator
  (1+A²) form — a correctness question for *all* fits surfaced by the
  simulate verification suite; needs its own study (changes every fitted
  uncertainty slightly and must be cross-checked against Mantid's actual
  AsymmetryCalc behaviour).
- **Project save warning for unsaved synthetic runs** (see as-implemented
  notes).
