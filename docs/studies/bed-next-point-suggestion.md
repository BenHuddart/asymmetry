# Study: Bayesian Experimental Design — "Suggest Next Point" for Parameter Trending

Status: study complete, implementation not started
Date: 2026-07-06
Slug: `bed-next-point-suggestion`

## 1. Motivation

During a live experiment a user accumulates runs at a series of x values
(temperature, field, ...), fits each run in the time domain, trends a fitted
parameter against x, and fits a trend model (order parameter, Arrhenius,
gap function, ...) to that series. The question they face at the instrument
is: **where should the next run go, and how long should it count, to
constrain the trend model the most?**

Bayesian experimental design (BED) answers this quantitatively. This study
defines the method, grounds it in the published literature on autonomous
experiment steering at scattering facilities, maps it onto the existing
trending machinery, and lays out a phased implementation plan.

This is not a port — no reference program (WiMDA, musrfit, Mantid) has this
feature — so it lives here rather than under `docs/porting/`.

## 2. Problem framing

There are two layers of fitting in the trending workflow:

1. **Per-run (time-domain) fit** — produces the observable y (e.g.
   relaxation rate λ) and its uncertainty σ_y at each x. These become the
   trend series `(x_i, y_i, σ_i)`.
2. **Trend model fit** — `fit_parameter_model` fits f(x; θ) to the series,
   producing θ̂ (e.g. T_c, β) with uncertainties.

The design variables are the next point's **x** and its **event count N**
(σ_y scales roughly as 1/√N; counts, not wall-clock time, set the
statistics — the user converts to time via their instrument's count rate). The design objective is information
about **θ** — the trend model's parameters — under two standard optimality
criteria:

- **c-optimality** (default): minimise the posterior variance of one
  user-chosen parameter of interest ("pin down T_c").
- **D-optimality** ("all parameters"): maximise the determinant of the
  Fisher information matrix, i.e. shrink the whole covariance ellipsoid.

## 3. Method

### 3.1 Acquisition functions (Laplace approximation)

After a converged trend fit we have θ̂, the parameter covariance matrix Σ
(computed by iminuit; see §5.1 for the exposure gap), and the model as an
evaluatable function. For a hypothetical new measurement at x with predicted
uncertainty σ_new(x), define the sensitivity vector

    g(x) = ∂f(x; θ)/∂θ |_θ̂        (numerical central differences)

Then, treating the posterior as Gaussian (Laplace approximation):

- **D-optimal / expected information gain** for one new datum:

      IG(x) = ½ log[ 1 + g(x)ᵀ Σ g(x) / σ_new²(x) ]

  This is the increase in log-determinant of the information matrix from a
  rank-one update — equivalently the mutual information between the new
  datum and θ under the Gaussian approximation.

- **c-optimal** for parameter k (Sherman–Morrison rank-one posterior
  update Σ' = Σ − Σ g gᵀ Σ / (σ_new² + gᵀ Σ g)):

      ΔVar_k(x) = (Σ g(x))_k² / ( σ_new²(x) + g(x)ᵀ Σ g(x) )

  Utility = ΔVar_k(x); predicted post-measurement variance = Σ_kk − ΔVar_k(x).

Both are evaluated on a candidate grid over a user-set achievable range
[x_min, x_max]; the deliverable is the **whole utility curve** plus its
argmax, not a bare number. Cost per candidate is one numerical gradient
(≈ 2·n_params model evaluations on a vector of x), so a few hundred
candidates cost milliseconds.

### 3.2 Counting statistics — target-precision stopping rule

The user states a goal, e.g. "σ(T_c) ≤ 0.1 K". With
σ_new²(x, N) = σ_emp²(x) · N_ref / N (N_ref = reference event count of the
existing runs, e.g. their median Mevents), the post variance at fixed x is
monotone in N, and the required N solves in closed form:

    Σ_kk − (Σg)_k² / (σ_emp²·N_ref/N + gᵀΣg)  =  target²

If the N → ∞ limit (Σ_kk − (Σg)_k²/(gᵀΣg)) still exceeds target², the goal
is **unreachable with a single added point** and the UI must say so rather
than suggest an absurd count. Because a candidate at an already-measured x
is perfectly valid, "collect more events where you are" competes on equal
footing with "move to a new x".

The recommendation is expressed in **events** (Mevents), the quantity that
actually sets the statistics; wall-clock time is beam- and
instrument-dependent, so the GUI offers an optional user-supplied count
rate purely to display the equivalent time alongside. The reported figure
is a **planning estimate, not a promise**: Laplace error bars for
nonlinear models are known to be miscalibrated (see §4), so it is
presented as order-of-magnitude.

### 3.3 Predicted noise σ_new(x)

v1 uses an **empirical model**: interpolate the observed per-point σ_i over
x (linear in x, clamped to end values outside the measured span), scaled by
√(N_ref/N). This captures heteroscedasticity for free — e.g. σ(λ) growing
near a transition — without any counting-statistics modelling. The existing
stabilisation floor for non-finite/zero errors (2% of median |y|, floor
1e-9, as used in trend assembly) applies. A forward model that predicts σ
from event counts and the current time-domain asymmetry model is Phase 4
territory (§7).

### 3.4 Guards: discovery vs refinement

The strongest empirical lesson from the literature (§4): model-driven
suggestion is **worse than a coarse uniform scan when the feature has not
been localised yet**, because with few points θ̂ and Σ are unreliable and
the acquisition confidently optimises the wrong model. v1 behaviour:
always show the utility curve, but display an explicit warning banner when
the fit is poorly conditioned, using signals already computed by
`fit_parameter_model`:

- `n_points` close to the number of free parameters,
- any entry in `params_at_bound`,
- large `reduced_chi_squared`,
- non-positive-definite / non-finite covariance.

The pitch to users is "you have a coarse scan and a trend fit — where
next?", not "let the algorithm drive from point one".

### 3.5 Known failure modes and their mitigations

- **Clustering**: a pure information criterion for an m-parameter model
  concentrates on ~m support points (order parameter → hammer just below
  T_c) and never tests whether the model form is wrong. Mitigations:
  show the full curve (user judgement stays in the loop); Phase 3 adds an
  explicit model-discrimination mode (§7.3) rather than silently blending
  exploration into the utility.
- **Local optimality**: IG is evaluated at θ̂. Early in a scan this can be
  badly off. The warning banner covers the acute case; a robustified
  version (average IG over a few θ draws from Σ) is a cheap Phase 4
  upgrade if the locally-optimal version proves twitchy.
- **Calibration**: Hessian-derived intervals under-cover for nonlinear
  models (measured 10–70% actual coverage at nominal 90% in the closest
  published analogue). Our own prototype (see `prototype-results.md`)
  quantified this for the μSR bread-and-butter case: for an
  order-parameter model with β &lt; 1, the rank-one prediction
  underestimated the Monte-Carlo-realised post-fit variance by ~5.6× at
  the suggested point, because ∂f/∂T_c diverges as T→T_c⁻ and the
  local-quadratic assumption fails exactly where the utility peaks.
  *Ranking* survives (good and useless candidates stay orders of
  magnitude apart), so the utility curve and argmax are trustworthy — but
  the predicted post-σ and events-to-target figures are not, near
  critical points. v1 therefore **calibrates the counting recommendation
  with a Monte-Carlo refit pass**: shortlist top candidates analytically,
  then run ~50 simulated add-point-and-refit trials at the shortlisted x
  (sub-second each, off-thread) and report the MC-calibrated post-σ and
  event count instead of the raw rank-one figures (plan task 1.4).

## 4. Literature grounding

- **TAS-AI** (arXiv:2604.23821, NIST; code: github.com/usnistgov/tasai) —
  autonomous triple-axis spectrometry decomposed into
  *detection → model inference → parameter refinement*. Its refinement
  acquisition is exactly the Laplace IG of §3.1 (with a cost denominator,
  `IG^γ/time`, γ ≈ 0.7), validated in-loop. Directly transferable
  findings: physics-driven planning loses to coarse scanning for blind
  discovery (Table 1 of the paper); Laplace interval under-coverage;
  posterior lock-in when refinement starves falsification regions — fixed
  by an explicit max-disagreement channel scored against **all** live
  competitor models, not just the runner-up; AIC weights (not BIC, whose
  p·ln n penalty drifts under sequential acquisition) for live model
  ranking; nuisance backgrounds frozen during discrimination.
- **Physics-informed Bayesian active learning for neutron diffraction**
  (arXiv:2108.08918) and **log-Gaussian-process autonomous TAS / ARIANE
  lineage** (arXiv:2105.07716) — the model-agnostic (GP) cousins; relevant
  if we ever want a "no trend model yet" exploration mode, which is out of
  scope here.
- Classical optimal design gives the closed-form test oracles used in §6:
  D-optimal designs for a straight line sit at the interval extremes; for
  Arrhenius, at the extremes of reachable 1/T; for an order-parameter
  curve, information about T_c concentrates just below T_c.

## 5. Codebase mapping

### 5.1 What exists

- `src/asymmetry/core/fitting/parameter_models.py` —
  `fit_parameter_model(x, y, yerr, model, parameters, ...)` (iminuit
  Migrad, multi-start) → `ParameterModelFitResult` with `parameters`,
  `uncertainties` (marginal), `chi_squared`, `reduced_chi_squared`,
  `n_points`, `params_at_bound`.
- Trend models are pure vectorised functions in the
  `PARAMETER_MODEL_COMPONENTS` registry, composable via
  `ParameterCompositeModel`, evaluable at arbitrary (x, θ) through
  `ParameterCompositeModel.function` / `sample_parameter_model`. No
  analytic gradients — numerical differentiation is the norm.
- Series assembly: `gui/panels/fit_parameters_panel.py` builds
  `(x, y, σ_y)` from per-run fit results (`r.values`, `r.errors`), with
  the 2%-median error floor.
- GUI: `gui/panels/model_fit_dialog.py::ModelFitDialog` with
  `TrendPreviewCanvas` (data + model curve + residuals), per-range model
  fits (`ModelFitRange`), error-mode selector, off-thread fit execution.

### 5.2 Gaps (prerequisites)

1. **Covariance is not exposed.** `ParameterModelFitResult` carries only
   marginal `uncertainties`; iminuit's `m.covariance` exists at fit time
   and is already surfaced for cross-group fits (`global_correlations`)
   but discarded for single-series fits (`_run_parameter_model_minuit`
   reads only `m.values`/`m.errors`). Fix: optional
   `covariance: NDArray | None` field (ordered like the free-parameter
   list) populated in `fit_parameter_model`. The serialisation seam is
   **not** `core/project/` — trend fit results are persisted by
   `fit_parameters_panel.py::_serialize_model_fits_mapping` /
   `_deserialize_model_fits` (a panel-state dict embedded in the project
   file), so `covariance` is added there. Note: that serializer currently
   also drops `params_at_bound`, which the §3.4 warning banner needs
   after a project reload — restore it in the same change.
2. **No predicted-σ machinery** for a hypothetical new point — new but
   small (§3.3).

### 5.3 Proposed core API (Phase 1)

New GUI-free module `src/asymmetry/core/fitting/experiment_design.py`:

```python
@dataclass(frozen=True)
class NextPointSuggestion:
    x_candidates: NDArray          # candidate grid
    utility: NDArray               # per-candidate utility (same length)
    best_x: float
    target: str | None             # parameter name (c-optimal) or None (D)
    sigma_new: NDArray             # predicted per-candidate σ at N = N_ref
    predicted_post_sigma: float | None    # for the target param at best_x
    events_factor_to_target: float | None # N/N_ref at best_x, None if no goal
    target_unreachable: bool       # single-point N→∞ limit above goal
    warnings: tuple[str, ...]      # ill-conditioning / extrapolation notes

def suggest_next_point(
    model, parameters, covariance,        # from the trend fit
    x_data, y_err,                        # measured series (for σ_new model)
    x_min, x_max, *,
    target: str | None = None,           # None => D-optimal
    sigma_goal: float | None = None,     # e.g. 0.1 (same units as target)
    n_candidates: int = 257,
) -> NextPointSuggestion
```

Design notes: pure numpy + model evaluation, no iminuit dependency at
suggestion time; candidates include the measured x values themselves (so
"re-measure" competes); extrapolated candidates (outside the measured span)
are flagged via `warnings`/mask so the GUI can style them; degenerate Σ
(non-finite, non-PD) degrades to a warning + empty suggestion, never an
exception.

Numerical guards required by the prototype findings
(`prototype-results.md`): gradients near model domain boundaries (e.g.
within a finite-difference step of the fitted T_c) are *finite but
unstable* — huge just below, exactly zero above — so use one-sided
differences (or deprioritise) when a central-difference step would
straddle a boundary; check `np.linalg.cond(Σ)` (warn above ~1e4 —
observed 1.1e5 with |corr(T_c, β)| ≈ 0.95 on a realistic 4-parameter
fit) and require the minimiser's valid/accurate flags before trusting Σ.

A companion `calibrate_suggestion(...)` runs the Monte-Carlo pass of
§3.5: for a shortlist of top candidates, simulate adding the point
(model + noise at the empirical σ), refit, and return the realised
post-variance distribution; the GUI reports these calibrated figures and
solves the event-count goal against them.

### 5.4 GUI surface (Phase 2)

In `ModelFitDialog`, after a successful fit: a "Suggest next point" section
with (a) target selector (model's free parameters + "all parameters"),
(b) achievable-range fields defaulting to the measured span, (c) optional
precision goal for the target parameter, (d) the utility curve overlaid on
`TrendPreviewCanvas` (secondary axis or shaded band) with a marker at
best_x, extrapolated region visually distinguished, and (e) a result line:
"Measure at x = 94.7 K, ~50 Mevents (≈2.5× your typical run) → σ(T_c) ≈
0.09 K (approximate)", with an optional count-rate field that additionally
renders the equivalent counting time (e.g. "≈ 4 h at 3.5 Mevents/h"). The
rate is display-only — it never enters the acquisition. Warning banner per §3.4. Suggestion runs on the GUI thread
(milliseconds) — no worker needed — but must be recomputed only on demand
(button), not on every fit, to keep the dialog snappy.

A persistent auto-refreshing suggestion in the trending panel (live
beamtime mode) is deliberately deferred until the dialog version has been
used in anger (§7.4).

## 6. Verification plan

Closed-form oracles pin the core function with tests (no reference
program needed):

1. **Straight line** — D-optimal argmax at an end of [x_min, x_max];
   c-optimal for the slope also at an end; c-optimal for the intercept
   near x = 0.
2. **Arrhenius** — c-optimal for E_a at an extreme of reachable 1/T.
3. **Order parameter** — c-optimal for T_c peaks just below the fitted
   T_c; utility ≈ 0 deep in the flat region above T_c.
4. **Rank-one honesty** — the update formula and a direct refit with the
   added point must agree on *ranking* (informative vs uninformative x
   separated by orders of magnitude), and the prototype-measured bias is
   pinned: near a critical point the raw rank-one post-variance is
   several-fold optimistic, and the MC-calibrated figure from
   `calibrate_suggestion` must track direct-refit reality within
   tolerance. A deliberately uninformative x predicts and realises ~no
   improvement.
5. **Event-count solve** — monotone in the goal; unreachable-goal path
   returns `target_unreachable=True` rather than a huge count.
6. **Degradation** — few-point / at-bound / non-PD-covariance fits produce
   warnings, never exceptions; empty series and single-point series are
   rejected cleanly.

A standalone numerical prototype implementing §3.1–3.2 against oracles
1–5 was run as part of this study; results are recorded in
`prototype-results.md` next to this file. The production tests re-derive
the same checks inside the test suite.

## 7. Phased implementation plan

Complexity ratings drive delegation: routine parts are delegated to
subagents with a precise spec and reviewed; high-complexity parts are done
directly by the orchestrating agent.

### Phase 1 — core (`experiment_design.py` + covariance exposure)

| # | Task | Complexity |
|---|------|------------|
| 1.1 | Add optional `covariance` to `ParameterModelFitResult`, populate from iminuit in `fit_parameter_model`, persist via `_serialize_model_fits_mapping`/`_deserialize_model_fits` (also restore the currently-dropped `params_at_bound`), tolerate legacy projects without either | routine |
| 1.2 | Implement `suggest_next_point` (acquisition kernel, σ_new interpolation, event-count solve, boundary-gradient and conditioning guards, warnings) | **high** — the math kernel and its numerical edge cases (near-boundary gradients, flat regions, non-PD/ill-conditioned Σ) |
| 1.3 | Oracle tests (§6) in `tests/core/test_experiment_design.py` | routine, from the spec above |
| 1.4 | `calibrate_suggestion` Monte-Carlo pass (simulate-add-refit over a candidate shortlist; needed because rank-one post-σ is ~5–10× optimistic near critical points — see `prototype-results.md`) + tests pinning that calibrated figures track direct-refit reality | moderate |

Acceptance: oracle tests green; `harness validate` green; no Qt imports in
core (structural check).

### Phase 2 — GUI (Model Fit dialog)

| # | Task | Complexity |
|---|------|------------|
| 2.1 | Suggestion section UI (target dropdown, range fields, goal field, button, result line, warning banner) | routine |
| 2.2 | Utility-curve overlay + best-x marker + extrapolation styling on `TrendPreviewCanvas` | moderate — must not disturb the existing preview/drag interactions |
| 2.3 | GUI tests beside existing dialog tests | routine |

Acceptance: suggestion renders and updates on demand for order-parameter
and Arrhenius fixtures; ill-conditioned fit shows the banner; focused GUI
test files green; `harness validate` green.

### Phase 3 — cost weighting and model discrimination

- 3.1 Optional movement/settling cost: `IG^γ / (t_count + t_move(x))`
  with a crude user-supplied directional model (e.g. cooling vs warming
  rates). Off by default.
- 3.2 Explicit A-vs-B discrimination mode: user selects two (or more)
  fitted candidate models for the same series; disagreement curve
  [f_A(x) − f_B(x)]² / 2σ²(x), scored leader-vs-**all** competitors; AIC
  weights for live ranking; nuisance/offset parameters frozen during
  discrimination. Requires a "candidate models per series" concept in the
  dialog — flagged so the current dialog redesign doesn't foreclose it.
- 3.3 User-facing docs: GUI walkthrough with screenshots
  (`docs/screenshots/capture.py` scenario), per the docs conventions.

### Phase 4 — deferred refinements

- Robustified acquisition (average IG over θ draws from Σ).
- Forward-model σ prediction from counting statistics and the current
  time-domain model (enables run planning beyond trending).
- Multi-trend aggregation: one run at x* informs *all* trended parameters;
  utility summed across active trend fits.
- Trending-panel live suggestion (auto-refresh as runs land).

## 8. Decision log

Settled with the maintainer (2026-07-06): c-optimality with user-picked
target parameter as default flavour (D-optimal available); counting
recommendation expressed in **events, not time** (user-supplied count rate
converts to time, display-only) via a target-precision stopping rule in v1; movement cost out of v1;
discrimination as an explicit separate mode, never blended; Model Fit
dialog first, panel later; user-set achievable range with flagged
extrapolation; warn-but-suggest on ill-conditioned fits; empirical
σ(x) + √t noise model.

Resolved by the prototype (2026-07-06): the counting recommendation
**does** need a calibration pass — the raw rank-one post-σ was ~5.6×
optimistic at the suggested point for a β = 0.35 order parameter, so v1
ships the Monte-Carlo calibration (task 1.4) rather than relying on an
"approximate" label alone.
