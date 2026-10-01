# Global Fit Wizard: recommend the fit that trends best

Status: plan, 2026-10-01, on `feat/global-wizard-trend-objective`. Phase 1
implemented; phases 2–5 not yet. Decisions D1–D5 and D16–D18 were taken with Ben; D6–D15 are lead proposals
recorded here so review can overturn them. Follows
[global-wizard-transitions.md](global-wizard-transitions.md) (phases) and
[global-wizard-stepper.md](global-wizard-stepper.md) (the Compare step).

## Problem

The wizard ranks candidates by an information criterion. What a series fit is
for is the trend: parameters that vary smoothly along temperature or field,
with error bars small against the variation, and with a sharp feature only
where the physics has one. The two disagree in a systematic way.

1. **All-local wins on high-statistics data.** With 2 000–90 000 points per
   run, a systematic difference the eye cannot see is worth tens of IC units,
   so localising any parameter "pays". The fit improves and the parameters
   stop being trackable: an amplitude and a background, or an amplitude and a
   rate, trade against each other run by run.
2. **The only trend signal punishes the wrong thing.**
   `_parameter_trace_roughness_from_results` is an unweighted second
   difference over run index. It ignores error bars and axis spacing, and a
   genuine transition scores as maximally rough; `_series_warnings` turns that
   into "changes abruptly", which demotes the candidate.
3. **The wizard often recommends nothing.** Run series-wide on four muon
   school series (YMnAl, Re₆Zr, copper, BiSCCO) it returned "No globally
   optimized candidate passed the automatic residual and continuity checks"
   on all four — on copper and Re₆Zr also after #348.
4. **The sharing search starts from the wrong place.** The separable engine
   scores sharing patterns with a Wald surrogate expanded around the
   all-local fits. Those fits are exactly where the degeneracy lives, so the
   surrogate is wrong where it matters (see Evidence).

## Evidence: the prototype

A scratch prototype ran phase 1 on seven muon school series, then real coupled
fits (`FitEngine.global_fit`, `strategy="least_squares"`) for a short ladder
of sharing patterns. χ²ᵣ is over the whole series; "trend quality" is the
score of D6 for the worst local parameter (0–1).

| Series (runs) | Model | Pattern | χ²ᵣ all-local → pattern | Trend quality |
|---|---|---|---|---|
| YMnAl LF 110 G (24573–24590) | Stretched exponential + constant | background shared | 1.0077 → 1.0082 | 0.61 → 0.78 |
| Copper ZF (20901–20917) | Dynamic Gaussian KT + constant | amplitude, background, Δ shared | 0.9729 → 0.9741 | 0.52 → 0.99 |
| BiSCCO TF 150 G (76976–76993) | two Gaussian lines, fraction group | total shared | 1.4553 → 1.4553 | unchanged |
| | | total, background width, phases shared | 1.4553 → 1.4953 | σ_vortex 0.56 → 0.84 |
| Re₆Zr ZF (38223–38260) | Dynamic Gaussian KT + constant | background, ν shared | 1.0177 → 1.0226 | 0.51 → 0.76 |
| | Static Gaussian KT × exponential + constant | background shared | 1.0389 → 1.0427 | 0.47 → 0.52 |
| PTFE ZF (17294–17322) | Dynamic F–μ–F + constant | none adequate | 1.2082 | 0.49 |
| Nickel ZF (124218–124248) | best all-local χ²ᵣ 1.83 | none adequate | — | 0.41 |
| Molecular AFM ZF (17094–17103) | best all-local χ²ᵣ 2.11 | background only | — | 0.55 → 0.56 |

What each case showed:

- **YMnAl.** Above ~110 K the all-local fits sit at amplitude ≈ 53 and
  background ≈ −32 with small errors: an unphysical basin. The surrogate
  predicted that sharing the background would raise one run's χ²ᵣ by 439 %;
  the real fit costs nothing. Sharing the amplitude as well fails on the
  eight coldest runs in one block, which is the real loss of asymmetry below
  the freezing temperature. Sharing the stretching exponent costs +3.0σ on
  two runs and is rejected, correctly.
- **Copper.** The textbook result: Δ shared, ν(T) rising monotonically from
  0.016 to 2.2 MHz.
- **BiSCCO.** A shared total amplitude with a free fraction costs nothing.
  The pattern that gives a smooth σ(T) is +2.7 % in χ²ᵣ on a model that
  only reaches 1.46: a fixed statistical threshold is too strict when the
  model itself is imperfect.
- **Re₆Zr.** Three runs (38223, 38230, 38231) carry ~2 % more asymmetry than
  their neighbours. Sharing the amplitude through them drives their
  exponential rate to zero. They need exempting, not tolerating.
- **PTFE.** The coarse pass and the infill pass give different parameters at
  neighbouring temperatures (`r_muF` 1.238, 1.366, 1.246 at 110, 115,
  120 K). No sharing fixes that; it is a property of the data worth naming.
- **Nickel, molecular AFM.** No one model fits across the transition.
  Sharing has to be judged per phase.

## Settled design (decision log)

Taken with Ben, 2026-10-01:

- **D1: trending is the default objective.** The wizard recommends the most
  trendable candidate among those that fit adequately. "Best statistical
  fit" stays available as the other objective and is today's behaviour.
- **D2: adequacy is a fit-quality tolerance, not an IC difference.**
- **D3: amplitude and rate rules are soft preferences.** Amplitudes and
  background are tried shared first. A component with a shared rate and a
  local amplitude is flagged "hard to justify" and ranks behind an unflagged
  candidate of equal trend quality; it is never forbidden.
- **D4: shared-total amplitudes are in scope.** Total shared, fractions
  local, for a volume fraction that changes through a transition.
- **D5: no searched Fixed role.** A shared parameter does the same job.

Lead proposals:

- **D6: trend quality of one local parameter** `Q = d · s · (1 − z)`, from
  its values `v_i ± σ_i` in axis order:
  - `d`, determined fraction: the share of runs where **not**
    (`σ_i > |v_i|/2` and `σ_i > span/4`). The span test keeps an order
    parameter that goes to zero from counting as undetermined.
  - `s = snr / (snr + 3)` with `snr = span / median(σ)` and `span` the
    10–90 % range: a parameter flat within its errors scores low, and should
    be shared.
  - `z`, zigzag fraction: the share of interior points that are a local
    extremum by more than 2σ (outside the interval their two neighbours
    span). It is scale-free and invariant under a monotone transform, so a
    rate rising over three decades is smooth and a single peak costs one
    point. Deviation from neighbour interpolation in σ units was tried and
    rejected: it penalised steep, perfectly smooth rates.
  - A candidate's trend quality is the minimum over its local parameters,
    then the mean, then the IC. A candidate with no local parameters is not
    a trend and is offered only when nothing else is adequate.
- **D7: a candidate is a template, a sharing pattern and an exemption
  list**, and every candidate the trend objective ranks is a real coupled
  fit. The surrogate keeps its job in the partition search, where a segment
  cost is a bound that tier 3 refits; it no longer ranks sharing patterns
  for the recommendation.
- **D8: the sharing ladder.** Per template, starting from all-local (free,
  from phase 1):
  1. share the background;
  2. share the amplitudes — as a shared total (D11) when the template has
     two or more signal amplitudes, then all of them;
  3. share one further parameter at a time, in kind order (D10): phase,
     shape, static width, then rates and frequencies last.
  Each rung is one coupled fit seeded from the series medians of the shared
  parameters and each run's own values for the rest — never from the
  surrogate's conditional shift. A rung that is not adequate (D9) is not
  climbed past for that parameter. That is at most P + 2 coupled fits per
  template; a template whose measured fit time (`fit_time_store`) puts the
  ladder over budget climbs only rungs 1–2.
- **D9: adequacy, and the ladder is shown.** Cost is measured in standard
  deviations of χ²ᵣ itself, `σ = √(2/ν)`, against the same template's
  all-local fit, per run and for the series.
  - The pre-selected rung is the highest trend quality whose series cost and
    every non-exempt run's cost are ≤ 2σ.
  - Every climbed rung is listed in Compare with its cost and its trend
    quality, as the break path is in Phases. The user can pick a costlier
    rung; BiSCCO is the case for that.
  - Between templates: a template competes on trend quality when its
    all-local series χ²ᵣ is within 3 % of the best template's. Re₆Zr's
    static KT × exponential (+2.1 %) is the case for the band.
- **D10: parameter kinds are declared, not guessed.** Each component
  declares a kind per parameter in the registry: amplitude, background,
  phase, shape, static width, rate, frequency, field, geometry.
  `global_search/heuristics.py` (name-prefix matching) is deleted and its
  callers read the kind.
- **D11: shared total.** For a template with two or more signal amplitudes
  the ladder also fits its fraction-group form (`(A + B){frac}`: one total,
  fractions local), with the total shared. It is the same template and one
  more pattern, not a new template key.
- **D12: exemptions.** When sharing an amplitude-kind parameter leaves runs
  over 2σ:
  - isolated runs (no three contiguous in axis order, and at most
    max(2, 10 %) of the series) are exempted: they keep a local value via
    `global_fit(local_param_groups=…)`, the rung is refitted, and they are
    listed on the candidate;
  - a contiguous block at either end is reported as "amplitude cannot be
    shared for runs X–Y: possible missing asymmetry", and the rung is
    offered for the remaining runs only when a phase boundary already
    separates them;
  - an interior block fails the rung.
- **D13: phases first.** When a partition is chosen, the ladder runs per
  phase. Series-wide it runs only when the wizard found no break.
- **D14: the old continuity checks go.** `_parameter_trace_roughness*`, the
  "changes abruptly" entry of `_series_warnings`, and the roughness fields
  of `GlobalParameterRecommendation` are deleted; the trend quality replaces
  them. The fingerprint-jump warning stays, because it is evidence for a
  break. A candidate is no longer disqualified by a series warning, which is
  what produced "No globally optimized candidate passed".
- **D15: acquisition-pass diagnostic.** When a local parameter zigzags in
  axis order (z ≥ 0.3) but not in run order (z ≤ 0.1), the candidate carries
  "runs taken in separate passes disagree" and names the passes. It is a
  caveat on the recommendation, not a veto.

## Settled in Phase 1 review (lead, 2026-10-01)

- **D19: widths, shapes and phases keep the role search's rate-first class.**
  The old name matching listed "delta", "beta" and "phase" as rate-like on
  purpose, so `role_policy` puts STATIC_WIDTH, SHAPE and PHASE in class 0 with
  RATE and FREQUENCY. The statistical objective's role decisions for `Delta`,
  `beta` and `phase` are then unchanged by the move to declared kinds; the
  ladder (D8) has its own order and does not read this table.
- **D20: the pass diagnostic needs about 22 runs as specified.** In run order
  the join between two passes costs up to two extrema, so `z ≤ 0.1` cannot be
  met by a shorter interleaved series. Phase 2 computes the run-order zigzag
  per pass (a pass is a maximal stretch of runs monotone in the axis) and
  names the passes.

## Code map (verify before editing)

- **Core.** `core/fitting/global_fit_wizard.py`:
  - ranking: `_assessment_sort_key`, `rerank_global_fit_wizard_recommendation`,
    `_COMPARABLE_SCORE_DELTA`, `_ROLE_DELTA_THRESHOLD`;
  - roughness and warnings: `_parameter_trace_roughness_from_results`,
    `_series_warnings`, `_fingerprint_jump_warnings`;
  - role search: `_run_separable_search` and the `_Separable*Task` family,
    `_build_parameter_recommendations_from_exact_cache`,
    `_assemble_assignment_assessment`;
  - per-phase optimisation and `OrderedCollapse` in
    `global_search/surrogate.py` (unchanged users of the surrogate);
  - `GlobalCandidateAssessment`, `GlobalParameterRecommendation`,
    serialisation at `_serialize_global_candidate_assessment`.
- **Kinds.** `core/fitting/global_search/heuristics.py`
  (`parameter_localisation_priority`, `is_amplitude_parameter`, used by
  `series_seeding.py` and `grouped_time_domain.py`); the component registry
  in `core/fitting/registration.py` and `models.py`.
- **Engine.** `FitEngine.global_fit(..., local_param_groups=…)` already
  supports grouped locals (used by `grouped_time_domain.py`, `joint.py`).
  `CompositeModel` fraction groups: `composite.py` (`{frac}`).
- **GUI.** `gui/widgets/model_compare_panel.py` (`ModelComparePanel`,
  `CompareRow`), `gui/windows/global_fit_wizard_window.py` (Compare and
  Phases steps), apply path in `gui/panels/fit/global_tab.py`
  (`_apply_fit_wizard_assessment`).
- **Docs.** `docs/reference/global_fit_wizard.rst`; scenarios
  `global_fit_wizard_*`.
- **Harness.** `tools/global_wizard_harness.py` has no case with a degenerate
  amplitude/background pair or a dynamic KT model; both are added here.

## Phases

Each phase is one subagent step with a lead review gate, in the main
checkout, in order.

1. **Trend quality and kinds (core).** `core/fitting/trend_quality.py` with
   D6 and the D15 diagnostic; parameter kinds in the registry (D10), callers
   moved, `heuristics.py` deleted. Tests on synthetic traces: steep smooth
   rise, single peak, step, noise within errors, noise beyond errors, an
   order parameter going to zero, tied axis values.
   *Landed.* Kinds live on `ComponentDefinition.param_kinds` and are read
   through `CompositeModel.parameter_kinds()`; the role-search classes are in
   `global_search/role_policy.py`. Decided here: a `FRACTION` kind for
   fraction weights (a group total is an `AMPLITUDE` with a `GroupAmplitude`
   identity); runs without an uncertainty are left out of span, median error
   and zigzag; user-function parameters default to `shape` at
   `register_component`.
2. **Sharing ladder (core).** D7–D9 and D12: the ladder, adequacy in σ
   units, exemptions, the "hard to justify" flag (D3). Synthetic harness
   cases modelled on YMnAl (amplitude/background degenerate at slow
   relaxation, amplitude lost in an end block) and copper (shared static
   width, local hop rate), plus one series with two anomalous-amplitude
   runs.
3. **Shared total (core).** D11, with a synthetic two-line series whose
   fraction changes through a transition.
4. **Verdict and persistence (core).** D1, D13, D14: the objective on
   `GlobalFitWizardRecommendation`, re-rank under either objective, phases
   path, deletions, schema bump with migration for stored recommendations.
5. **Compare ladder (GUI) and docs.** Objective switch; one row per rung
   with a cost bar (σ units) and trend quality; a trace strip per local
   parameter with error bars; flags for exemptions, missing asymmetry,
   hard-to-justify and pass disagreement; apply carries the pattern and the
   exemptions to the Batch tab. Sphinx page, screenshot scenarios, CHANGELOG.

## Acceptance

Synthetic cases are the tests. The muon school corpus is not in the repo, so
these are run by the lead before the PR and recorded in it:

| Series | The trend objective must recommend |
|---|---|
| YMnAl | stretched exponential, background shared; amplitude reported as not shareable below ~97 K |
| Copper | dynamic Gaussian KT with amplitude, background and Δ shared |
| BiSCCO | total shared pre-selected; the smooth-σ pattern listed with its cost |
| Re₆Zr | amplitude shared with runs 38223, 38230, 38231 exempt; static KT × exponential offered within the band |
| PTFE | a recommendation carrying the pass-disagreement caveat |
| Nickel, molecular AFM | per-phase patterns; no series-wide claim |

A recommendation, not "no candidate passed", on all seven.

## Settled after the plan (Ben, 2026-10-01)

- **D16: the shared total covers signal amplitudes only.** The background is
  shared separately (ladder rung 1). In transverse field the constant is a
  different quantity from the precessing signal.
- **D17: exempt runs are left out of the coupled series on apply.** The Batch
  tab keeps its Global and Local roles; exempt runs are listed, stay in the
  group, and are not given a third role.
- **D18: 2σ and the 3 % template band are constants.** The ladder in Compare
  is the control.

## Follow-ups (not this PR)

- The CLI's `fit-global` choosing a pattern with the same ladder.
- A wizard-harness case on a real dynamic KT decoupling series, to re-check
  the #347 changelog claim now that #348 has landed.
- The same objective for the single-run wizard's hand-off to a series.
