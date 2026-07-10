# Study: BED for Knight-Shift Angle Scans — Next Angle, Model & Assignment Discrimination

Status: study complete, implementation not started
Date: 2026-07-10
Slug: `bed-next-angle-knight-shift`

## 1. Motivation

The Knight-shift window fits N branches jointly against orientation angle
(`run_joint_fit` → `fit_assigned_angular_curves`, the classification-EM /
Hungarian-assignment fit). During a live rotation scan the user faces the
same question the trending BED already answers for temperature/field
scans: **which angle should the next run go to?** — with three distinct
flavours specific to the angle workflow:

1. **Refinement** — which θ most constrains the joint-fit curve
   parameters (K_iso/K_ax/θ0 per site)?
2. **Model discrimination** — is the scan consistent with an aligned
   rotation axis, or is there evidence of axis misalignment (a
   first-harmonic component)? Which angle best tests that?
3. **Assignment discrimination** — near branch crossings the EM fit can
   land on competing per-angle assignments (which physical site owns
   which measured component). Which angle best resolves the ambiguity?

This extends the shipped trending BED
([bed-next-point-suggestion.md](bed-next-point-suggestion.md), PRs #203 /
#205) to the Knight-shift window. It stays on the **local / Fisher
(Laplace) tier** of that study — the acquisition math, MC calibration,
and GUI patterns are reused, not reinvented.

### 1.1 MuRef survey (2026-07-10)

Before scoping, the sibling MuRef library was surveyed as a potential
source of ports:

- **No BED code exists in MuRef.** Its "Bayesian" machinery is inverse
  inference (muon-site posteriors), not experiment planning. Nothing to
  port.
- **No additional angular fit functions.** MuRef's single angular form,
  `K(φ) = c0 + c2c·cos 2φ + c2s·sin 2φ` (`muref/knight/analysis.py::
  fit_branch_sinusoid`), is the same 3-parameter second-harmonic family
  as both of our `ANGULAR_MODELS` (see §3.4).
- **One genuine model extension identified**: MuRef's forward model
  parameterises the rotation axis by Euler angles and notes that fitting
  them "self-calibrates sample misalignment" (MnSi Sec. III.A). The
  phenomenological signature of a tilted axis is **first-harmonic
  (cos θ / sin θ) leakage** into K(θ). MuRef itself defers the joint
  nonlinear solve; we adopt the phenomenological Fourier form instead
  (§3.4). Everything downstream of MuRef's sinusoid (tensor inversion,
  site recovery) requires the crystal structure and is out of scope.
- Recorded as a caveat, not a model: with the exact-ratio shift
  definition `K = (|B_loc| − B)/B` the sinusoid family is leading-order
  only — an O(K²) non-sinusoidal residual remains (MuRef
  `knight/analysis.py` module docstring). At µSR Knight-shift magnitudes
  (≲ 1%) this is negligible against measurement noise.

## 2. Problem framing

The angle scan differs from the scalar trending case in two structural
ways:

- **A new run yields a vector, not a scalar.** One run at θ produces one
  value per component — a datum for *every* curve simultaneously. The
  utility of a candidate angle is therefore aggregated across curves.
- **The components are unlabeled.** Which measured value belongs to which
  physical curve is itself inferred (the assignment). Near predicted
  crossings a new datum may attach to the wrong curve; and competing
  assignment hypotheses are themselves objects worth discriminating
  between.

A further structural fact that shapes the plan: **the two existing
angular models are the same function family.**
`KnightAnisotropy = K_iso + K_ax(3cos²(θ−θ0)−1)/2` expands to
`(K_iso + K_ax/4) + (3K_ax/4)cos 2(θ−θ0)`, i.e. exactly `AngularCos2`
reparameterised. Discrimination between them is vacuous — after refit
they predict identical curves. Model discrimination only becomes
meaningful once a model *outside* the family exists, which is why the
misalignment model (§3.4) is in scope and not deferred.

## 3. Method

### 3.1 Refinement — next angle for parameter precision

Reuses the Laplace rank-one acquisition of the trending study §3.1,
summed over curves. For curve m with fitted covariance Σ_m, sensitivity
g_m(θ) = ∂f_m/∂θ_params, and predicted new-point noise σ_m(θ):

    IG_total(θ) = Σ_m ½ log[ 1 + g_m(θ)ᵀ Σ_m g_m(θ) / σ_m²(θ) ]

(one run adds one datum to each curve; per-curve information gains add
because the curves' parameter sets are independent). c-optimality targets
one parameter of one curve ("pin down K_ax of site 2") and reuses
`suggest_next_point` on that curve directly — the sum collapses to the
existing single-series path.

**Assignment-risk flag**: the rank-one update assumes the new value
attaches to the right curve. Candidate angles where any two predicted
curves lie within k·σ_new (k ≈ 2) of each other are flagged in the
suggestion's `warnings`/mask so the GUI can style them — a new point at a
crossing risks feeding the wrong curve.

σ_new(θ) follows the trending study's empirical policy (§3.3 there):
per-curve interpolation of the realigned trace errors over θ, clamped
outside the measured span, with the 2%-median stabilisation floor.

**Periodicity**: candidates default to the measured span; the user may
extend the range up to the model period (180° for the pure
second-harmonic family, 360° once a first harmonic is present).
Candidates outside the measured span carry the standard extrapolation
flag. Wrapping the empirical σ model around the period is a deferred
refinement — the clamp is honest, just conservative.

### 3.2 Model discrimination — aligned vs misaligned

Reuses `suggest_discriminating_point`'s disagreement utility, summed over
curves with both hypotheses fitted to the same realigned data:

    U_disc(θ) = Σ_m [ f_m^lead(θ) − f_m^alt(θ) ]² / 2σ_m²(θ)

where lead = current model (e.g. `KnightAnisotropy`), alt = the
misalignment model (§3.4), each from its own joint fit. Curve identity is
preserved (same realigned traces), so no matching step is needed here.
Live ranking uses the existing `aic_weights` with
`AIC = total_chi_squared + 2·n_curves·n_params` per joint fit. The
"models agree within noise everywhere" warning carries over unchanged.

### 3.3 Assignment discrimination — which angle resolves the labelling

Competing hypotheses are **distinct EM outcomes**: today
`fit_assigned_angular_curves` tries several seeds (identity, continuity,
crossing-swap) and silently keeps the lowest-χ². The fit is extended to
also return runner-up outcomes whose assignment differs from the winner
and whose total χ² lies within a Δχ² window (default ~9, i.e. 3σ-ish) —
exactly the near-degenerate labellings worth testing.

Because a new run's components are unlabeled, two hypotheses are only
distinguishable at θ through the **sets** of values they predict. The
utility is the minimum-cost matching (Hungarian, reusing
`linear_sum_assignment`) between the two hypotheses' predicted N-vectors:

    U_assign(θ) = min_perm Σ_m [ f_m^A(θ) − f_perm(m)^B(θ) ]² / 2σ_m²(θ)

This reduces to the scalar disagreement utility for N = 1, is zero at
angles where the hypotheses predict the same value set (e.g. at the
crossing itself, where the labellings coincide by construction), and
peaks where the labellings imply genuinely different curve sets —
typically outside the measured span or between crossings.

### 3.4 The misalignment model — third `ANGULAR_MODELS` entry

New registered component (working name `AngularFourier2`; final name at
review):

    K(θ) = K_avg + K_1·cos(θ − θ1) + K_amp·cos(2(θ − θ2))     (degrees)

- **Physics**: a perfectly aligned rotation axis gives a pure second
  harmonic; a tilted axis leaks a first harmonic whose amplitude grows
  with the tilt (MuRef forward model / MnSi Sec. III.A). K_1 ≠ 0 is the
  fit-level evidence of misalignment. This is the phenomenological form —
  the full Euler-angle self-calibration (which would tie K_1, θ1 to a
  physical tilt) stays out of scope, as it does in MuRef.
- 5 parameters → needs ≥ 5 shared angles per curve; the existing
  `n_points < n_params` guard in `fit_assigned_angular_curves` already
  rejects thinner scans cleanly.
- **Canonicalisation** (extending `_canonicalize_theta0`): θ2 folds into
  (−45°, 45°] with a K_amp sign flip (as `AngularCos2` today); θ1 folds
  into (−90°, 90°] with a K_1 sign flip (period 360°, and
  (K_1, θ1) ≡ (−K_1, θ1 + 180°)).
- Registered in `PARAMETER_MODEL_COMPONENTS` with `scopes=("angle",)` and
  appended to `angular_assignment.ANGULAR_MODELS`; the window's model
  combo iterates `ANGULAR_MODELS`, so it appears without GUI changes.
- Seeding in `_seed_parameters`: K_avg = mean, K_amp = spread/2,
  K_1 = 0 (aligned start — the nested null), θ1 = θ2 = 0.

### 3.5 Covariance under the θ0 fold — exact, not approximate

`_canonicalize_theta0` currently adjusts marginal uncertainties in
quadrature and documents that the K_iso/K_ax covariance "is not
propagated here — a conservative approximation"
(`angular_assignment.py:368-370`). The fold is a **linear** map
(e.g. for the odd `KnightAnisotropy` flip:
K_iso' = K_iso + K_ax/2, K_ax' = −K_ax, θ0' = θ0 + const, so J is
constant), hence Σ' = J Σ Jᵀ is exact. Since BED consumes Σ, the fold
must transform the covariance — which also replaces the quadrature
approximation for the marginal errors and retires that caveat.

## 4. Codebase mapping

### 4.1 What exists

- `core/fitting/experiment_design.py` — the full local-tier kernel:
  `suggest_next_point` (:250), `suggest_discriminating_point` (:573),
  `aic_weights` (:671), `calibrate_suggestion` (:401),
  `cost_weighted_utility` (:710), plus `_sensitivities`,
  `_validated_covariance`, `_empirical_sigma`, `_candidate_grid`
  internals. GUI-free, oracle-tested.
- `ParameterModelFitResult.covariance` (`parameter_models.py:2338`) is
  populated from Minuit (`:3211-3230`) — so every per-curve fit inside
  the EM (`angular_assignment._fit_curves` → `fit_parameter_model`)
  **already carries a covariance in memory**.
- `core/fitting/angular_assignment.py` — the EM fit, seeds, Hungarian
  reassignment, `_canonicalize_theta0`.
- `core/fitting/knight_analysis.py` — `run_joint_fit` (:561) bridges the
  window to the EM fit; `KnightJointFitState`/`KnightJointCurve`
  persistence.
- `gui/windows/knight_shift_window.py` — sidebar controls, model combo
  (iterates `ANGULAR_MODELS`), off-thread joint fit via `TaskRunner`,
  plot canvas.

### 4.2 Gaps (prerequisites)

1. **Covariance is dropped at the bridge.** `KnightJointCurve` keeps only
   `(name, value, error)` triples; `run_joint_fit` discards
   `fit.covariance`. Fix: optional `covariance` field on
   `KnightJointCurve` (same `(names, matrix)` shape as
   `ParameterModelFitResult`), serialised in `to_dict`/`from_dict`,
   legacy dicts tolerated (`None` → suggestion degrades with a
   "re-run the joint fit" warning, mirroring the stale-fit pattern).
2. **The θ0 fold invalidates the covariance it now needs** (§3.5).
3. **Runner-up assignment hypotheses are discarded.**
   `fit_assigned_angular_curves` keeps only the best seed outcome. Fix:
   optional return of distinct-assignment runners-up within a Δχ² window
   (new keyword, default off, so existing callers are untouched).
4. **No vector-observable aggregation.** New GUI-free bridges in
   `knight_analysis.py` (matching the `run_joint_fit` bridge pattern),
   calling `experiment_design` per curve and aggregating:
   `suggest_next_angle(...)`, `suggest_model_discriminating_angle(...)`,
   `suggest_assignment_discriminating_angle(...)`, each returning the
   existing `NextPointSuggestion` (utility curve + argmax + warnings) so
   the GUI plumbing is uniform. Generic pieces that belong in
   `experiment_design.py` (e.g. a multi-series IG sum, the min-cost
   set-matching utility) go there; Knight-specific input assembly stays
   in `knight_analysis.py`.
5. **GUI surface** (settled 2026-07-10). One collapsible "Suggest next
   angle" `PanelSection` beneath *Model fit* with a **mode selector** —
   *Refine parameters* / *Test misalignment* / *Resolve assignment* —
   sharing one candidate-range row (`FloatLimitField` pair seeded from
   the measured span), one Suggest button, and one result/warning label
   pair. The target picker (curve × parameter) and the precision goal +
   events-factor/Mevents conversion apply in Refine mode only; the
   movement-cost grid is **omitted** (rotation is fast; add later if
   wanted). A muted disabled-hint (the dialog's
   `_suggest_disabled_hint` pattern) explains missing prerequisites:
   no joint fit, stale fit, legacy fit without covariance, non-angle
   axis. Utility evaluation is milliseconds — GUI thread, on demand.
   *Test misalignment* runs the alternative (Fourier) joint fit
   automatically off-thread via the window's existing `TaskRunner` path
   on Suggest, cached against (snapshot, model, unit, correction) so
   repeat clicks don't refit; no "Compare against" builder — the
   alternative is fixed. *Resolve assignment* consumes the EM
   runners-up kept **in memory** from the last fit run (never
   persisted); when the stored fit came from a project load, Suggest
   re-runs the EM off-thread first.
6. **Overlay reuse.** The utility-curve rendering (normalised band,
   best-x marker, extrapolation styling) lives inside
   `TrendPreviewCanvas`; the knight window draws on a plain axes in
   `_redraw`. Extract the overlay drawing into a shared `gui/widgets/`
   helper consumed by both surfaces, and add the new assignment-risk
   shading there so both render identically.

### 4.3 Entry point — menu always, panel button only when appropriate

Settled 2026-07-10: the workflow's entry points are re-balanced so the
menu action `Analysis → Knight shift analysis…` remains the
unconditional path (it already is — `mainwindow.py:1075`), while the
main-GUI exposure — the "Knight shift window…" button in the trend
panel's *Derived parameters* section — is shown **only when the trend is
likely a Knight-shift use case**. The signal already exists:
`_knight_observables` is non-empty exactly when the fitted model
contains a Knight-convertible component (a *local* precession
frequency/field via `CompositeModel.knight_observable_params`, which
already excludes applied-field muonium terms). Today the button is
enabled on the far weaker `bool(self._rows)`
(`fit_parameters_panel.py:1027`). Change: the button is **hidden**
unless `_knight_observables` is non-empty (and rows exist); pinned by
panel tests both ways.

### 4.4 Interface note (full-Bayesian later)

Per the decision log: the bridges accept "hypotheses" as (model,
parameters, covariance, predictor) tuples, so a sampler-backed posterior
predictive (MuRef-style emcee, if ever needed) can replace the Laplace
inputs behind the same `NextPointSuggestion` surface without touching
the GUI.

## 5. Verification plan

Closed-form oracles, following the trending study's pattern:

1. **cos 2θ geometry** — for a single `AngularCos2` curve: c-optimal for
   K_amp peaks at the antinodes (θ2, θ2 + 90°); for θ2 at the nodes
   (θ2 ± 45°, max |∂K/∂θ2|); K_avg utility is flat. D-optimal is
   90°-periodic.
2. **Vector sum** — two identical curves double IG_total; argmax
   unchanged. A curve with no covariance contributes nothing but a
   warning.
3. **Misalignment discrimination** — synthetic tilted data (K_1 ≠ 0):
   U_disc peaks near θ1 mod 360° (where the first harmonic separates the
   families); on aligned data the "agree within noise" warning fires and
   AIC prefers the 3-parameter model.
4. **Assignment discrimination** — the two-curve crossing fixture from
   `test_angular_assignment.py`: envelope vs crossed labellings within
   Δχ²; U_assign ≈ 0 at the crossing angle, peaks away from it; the
   Hungarian matching makes U_assign invariant to curve reordering.
5. **Fold covariance** — canonicalised fit's Σ' equals the Σ of a refit
   started in the canonical branch (tolerance); marginal errors match
   Σ'_kk exactly (retiring the quadrature approximation).
6. **Fourier model** — parameter recovery on synthetic tilt; K_1
   consistent with 0 on aligned data; < 5 angles rejected cleanly.
7. **Degradation** — legacy joint fit without covariance, stale-unit
   fits, all-points-excluded branches: warnings, never exceptions.
8. **Assignment-risk flag** — candidates at a predicted crossing carry
   the flag; well-separated angles do not.

## 6. Phased implementation plan

Delivered as a **single PR** on one feature branch. Phases are executed
by delegated workers (complexity → model tier) with an orchestrator
review gate after each phase; the orchestrator runs the focused tests at
each gate and `harness validate` once before the PR.

| Phase | Content | Complexity / worker |
|-------|---------|---------------------|
| 1 | Covariance plumbing: `KnightJointCurve.covariance` + serialisation + legacy tolerance; exact J Σ Jᵀ under `_canonicalize_theta0` (all models), replacing the quadrature marginals; tests (§5.5, 5.7) | moderate — Sonnet |
| 2 | `AngularFourier2` model: registration (`scopes=("angle",)`, param info, LaTeX), `ANGULAR_MODELS` append, seeding, canonicalisation, joint-fit eligibility, tests (§5.6) | moderate — Sonnet |
| 3 | Entry-point gating: hide the trend-panel "Knight shift window…" button unless `_knight_observables` is non-empty (§4.3); panel tests both ways | routine — Sonnet |
| 4 | Core BED: multi-curve IG sum + assignment-risk flag; EM runner-up hypotheses (opt-in Δχ² window); the three bridges + Hungarian set-matching utility; oracle tests (§5.1–5.4, 5.7–5.8) | **high — Opus** (the math kernel and its edge cases) |
| 5 | GUI: shared overlay helper extracted from `TrendPreviewCanvas`; the "Suggest next angle" section (§4.2 item 5); off-thread alternative fit + caching; overlay + assignment-risk shading in `_redraw`; GUI tests | moderate/high — Opus |
| 6 | Docs: `parameter_trending.rst` Knight section + `knight_shift_angle.rst` workflow update + screenshot scenario + CHANGELOG | routine — Sonnet |

Dependencies: Phases 1–3 are independent (run in parallel); Phase 4
needs 1 + 2; Phase 5 needs 4; Phase 6 needs 5 (UI strings quoted
verbatim from the shipped widgets).

## 7. Decision log

Settled with the maintainer (2026-07-10):

- **Local/Fisher tier only** — reuse the shipped Laplace machinery;
  full-Bayesian EIG deferred, but the bridge interface keeps a
  sampler-backed drop-in possible (§4.3).
- **Discrimination covers both models and assignment** — the
  assignment-ambiguity utility (§3.3) is in scope, not deferred.
- **Misalignment model is in scope** as the third `ANGULAR_MODELS`
  entry — motivated by the MuRef survey finding that the current two
  models are one family (§2), making it the piece that gives model
  discrimination meaning.
- **MuRef port review complete**: no BED and no new fit families to
  port (§1.1); demagnetisation tensors / error estimators / synthetic
  data generator noted as separate candidate ports, out of this study's
  scope.
- Docs-first workflow: this study precedes implementation; sign-off
  gates Phase 1.

Settled with the maintainer (2026-07-10, GUI review):

- **Entry point**: menu action stays unconditional; the trend-panel
  button is **hidden** (not disabled) when the trend has no
  Knight-convertible components (§4.3).
- **Suggest section scope**: precision goal + events-factor/Mevents
  conversion included; movement-cost weighting omitted (rotation is
  fast) — can be added later behind the same interface.
- **Mode UI**: one section with a Refine / Test misalignment / Resolve
  assignment mode selector; discrimination fits run automatically
  off-thread on Suggest (no "Compare against" builder).
- **Delivery**: single PR on a feature branch; phases executed by
  Sonnet/Opus workers with orchestrator review gates (§6).
