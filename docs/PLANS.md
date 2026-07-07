# Execution Plans

This file tracks repo-local plans that are useful for agents. Keep short tasks
inline here; create a dedicated plan file only when a project spans multiple
subsystems or days.

## Active

### Bayesian Experimental Design: Suggest Next Point (Trending)

Status: complete — Phases 1–2 (core + dialog GUI) shipped in PR #203,
Phase 3 (model discrimination, cost weighting, user docs) in PR #205; only
the study's explicitly deferred refinements remain

During a live experiment, after a trend model fit, suggest the next x
(temperature/field) and event count that most constrains the trend model's
parameters — Laplace expected-information-gain acquisition with c-/D-optimality,
a target-precision event-count rule (counts, not time; a user-supplied count
rate converts for display), and closed-form design oracles as tests. A
numerical prototype validated the acquisition math and showed the raw
rank-one post-σ is ~5–10× optimistic near critical points, so the counting
recommendation is calibrated by a Monte-Carlo refit pass
([studies/prototype-results.md](studies/prototype-results.md)).

Full study, method, decision log, and the phased plan (core → dialog GUI →
cost/discrimination → deferred refinements) live in
[studies/bed-next-point-suggestion.md](studies/bed-next-point-suggestion.md).

Acceptance criteria (v1 = Phases 1–2):

- `ParameterModelFitResult` exposes the fit covariance (serialised with the
  project; legacy projects tolerated).
- `core/fitting/experiment_design.py::suggest_next_point` +
  `calibrate_suggestion` are GUI-free and pass the oracle tests (line →
  interval extremes, Arrhenius → 1/T extreme, order parameter → just below
  T_c, ranking honesty + MC-calibrated post-σ vs direct refit,
  unreachable-goal detection).
- Model Fit dialog shows the utility curve + suggestion on demand, with an
  ill-conditioned-fit warning banner; `harness validate` green.

### Fit Function Builder Redesign (Option C)

Status: implemented on branch `feat/fit-builder-redesign`; awaiting live GUI
review before the PR opens

Replace the calculator-style expression dialogs with a two-panel builder:
searchable component library (left) + structured model-row list (right), with
fraction groups as visual containers and an "Edit as text" escape hatch.
Both `FitFunctionBuilderDialog` and `ParameterModelBuilderDialog` build on one
shared base in `gui/widgets/function_builder/`.

Core rework included: fraction groups expose n−1 free parameters named
`f_<Component>` (last term = remainder `max(0, 1−Σ)` computed in
`CompositeModel`); legacy `fraction_i` parameter values are migrated on
project load. Deep component search (name/alias/category/param/description,
ranked) lives GUI-free in `core/fitting/component_search.py`.

Acceptance criteria:

- Both builder dialogs use the shared library + row-list base; the old
  free-text-primary editor and text-selection Fractions gesture are gone.
- Fraction groups survive structural edits; the displayed model is never
  ambiguous (no hidden parentheses).
- Pre-rework `.asymp` projects load, display, and refit with migrated
  fraction values.
- `harness validate` passes; docs updated with GUI walkthrough screenshots.

### Harness Adoption

Status: in progress

- Maintain `AGENTS.md` as the short navigation map.
- Keep `docs/HARNESS.md`, `docs/QUALITY.md`, and this file indexed from
  `docs/INDEX.md`.
- Run `python tools/harness.py structural` for fast boundary validation.
- Run `python tools/harness.py validate` before larger handoffs when local
  dependencies are available.

Acceptance criteria:

- Agents can discover architecture, validation commands, and current quality
  risks from repo-local files.
- CI runs the same harness command used locally for structural, lint, and test
  validation.
- New repeated conventions are promoted into checks or docs.

## Deferred

### FFT Zero-Padding as a Display-Only Concern

Today zero padding is a transform parameter: the padded (sinc-interpolated,
correlated) samples ARE the stored spectrum, threaded through recipes,
staleness comparison, caching, fits, and moments. The statistically clean
architecture stores the spectrum at padding 1 (independent samples — what
fits, moments, and exports consume) and treats the smooth curve as a pure
rendering: the plot draws the sinc envelope (computed from the complex
spectrum at display time) with the independent samples as points on top.
This dissolves the correlated-samples problem instead of correcting for it
(the interim effective-sample-size correction — `error_oversampling`,
WiMDA's `dof := n div zpad - nv` made consistent — ships with the padding
work and covers the statistics until then). Non-trivial because the display
curve must be derived from the COMPLEX spectrum (interpolating a magnitude
display is wrong near its kinks), so the envelope has to come out of the
compute path, not the plot panel; recipes and the project schema carry the
padding field today and would need a migration story.

### GUI Journey Harness

Build a repeatable smoke workflow for the most important GUI paths:

- launch the app in offscreen mode
- load a small synthetic or fixture-backed dataset
- open grouping, fitting, Fourier, and project-save paths
- capture failures with enough context for an agent to reproduce locally

### Documentation Freshness Checks

Consider a structural check that verifies `docs/INDEX.md` references all
top-level developer docs and that public API docs build without warnings.

### Ruff Baseline Cleanup

Status: complete

`python tools/harness.py lint` now checks `src`, `tests`, and `tools`, and
`lint-all` remains as an alias for that full baseline.

### Loader Fixture Catalog

Create a small fixture index documenting which file-format assumptions are
covered by tests and which rely on synthetic data.
