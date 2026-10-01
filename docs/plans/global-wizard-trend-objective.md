# Global Fit Wizard: recommend the fit that trends best

Status: plan, 2026-10-01, on `feat/global-wizard-trend-objective`. Phases 1–4
implemented; phase 5 not yet. Decisions D1–D5 and D16–D18 were taken with Ben; D6–D15 are lead proposals
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

## Settled in Phase 3 review (lead, 2026-10-01)

- **D21: the amplitude stage does not branch.** When the shared total and
  all amplitudes shared are both adequate, the ladder continues in the plain
  form only, and the shared-total rung stays listed as an alternative. On
  BiSCCO that gives amplitudes, phases and one frequency shared at +1.8σ with
  both widths trending (quality 0.84 and 0.93): a better outcome than the
  hand-run pattern of the prototype, at half the fits a branch would cost.
- **D22: the end block is where the amplitude departs.** On YMnAl that is the
  three coldest runs, not the eight the plain shared fit left over 2σ; the
  other five offended only because those three dragged the shared value.
- **D23: an exemption proposed from the amplitude trace is not confirmed by
  cost.** A run four scatters from the series median is exempt and listed.

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
   *Landed.* `global_search/sharing_ladder.py`: `climb_sharing_ladder`
   returns every rung (`LadderRung`) and the pre-selection
   (`SharingLadder.preselected`); it is not yet called by the wizard. The pass
   diagnostic counts zigzag per pass (D20). Decided here: exemptions and the
   end-block finding are judged only on the rung that shares the amplitudes,
   and later rungs inherit the exempt runs; a block spanning the whole series
   is not an end block; only a parameter free on every run can be shared; a
   rung's solve is capped at 1000 residual evaluations (`RUNG_MAX_CALLS`) and
   reported as failed beyond it, because a pattern the data reject crawls for
   thousands; pass disagreement needs two passes of three or more runs. The
   tests are in `tests/core/test_sharing_ladder.py`; the wizard harness scores
   a recommendation against planted roles, so its cases wait for Phase 4.
   Exemption detection was reworked in Phase 3 (below): offenders read from
   one plain fit failed once the anomaly was strong against the noise.
3. **Shared total (core).** D11, with a synthetic two-line series whose
   fraction changes through a transition.
   *Landed*, with the exemption rework.
   - **The form.** `core/fitting/fraction_form.py`: `signal_fraction_form`
     groups a plain sum's signal terms under one fraction group, leaves
     background terms outside (D16), and maps values both ways (total = Σ Aᵢ,
     fractionᵢ = Aᵢ / total). A model has the form when it is a sum of two or
     more added signal terms standing side by side plus background terms
     only; n amplitudes become a total and n − 1 fractions, so the parameter
     count is the same. A series has it when, further, every run's total has
     one sign and no signal amplitude opposes it by more than 2σ (fraction
     weights are clamped to [0, 1], so a line whose sign the amplitude
     carries has no fraction; an insignificantly negative amplitude is a term
     that has vanished and starts at fraction 0).
   - **The ladder** is a table of stages walked by one loop: background;
     amplitudes (total shared in the fraction form, then every amplitude in
     the template as given, both from the same rung); one further parameter
     at a time. Each stage continues from the last adequate rung of the stage
     before, in that rung's form, and `LadderRung.model` says which. The
     all-local fits are not repeated in the fraction form: their values are
     mapped into it and costs stay against the plain fits.
   - **D3** now reads a term's amplitude as local when its fraction is, and
     covers the factors multiplied onto an amplitude's component (the
     Gaussian of `Oscillatory * Gaussian`), so the two forms flag alike.
   - **Exemptions (D12), chosen design.** The rung below the amplitudes
     (background shared) fixes each run's amplitude, so the anomaly is read
     there: a run departs when its amplitude, or its total, lies more than
     4 scatters from the series median, the scatter combining the series'
     MAD with the run's own error. Isolated departing runs are exempt from
     the first fit; runs that still offend by cost are exempted in a second
     and last fit when all of them together are isolated. Two fits at most,
     as in Phase 2; the caps are unchanged. Compared on synthetic series
     (dynamic KT + constant, 12–38 runs):

     | Case | Phase 2 (plain fit's offenders) | Greedy (exempt the costliest, refit) | Chosen |
     |---|---|---|---|
     | 2 isolated, +2.5 % to +30 %, σ 0.05 | fails from +2.5 % | exempts them, 2 fits | exempts them, 1 fit |
     | 1 isolated, +10 % | fails | exempts it and a neighbour | exempts it |
     | 3 of 38, +2 % | exempts them | exempts them | exempts them |
     | end block, −3 % | names it | names it | names it |
     | end block, −10 % and −35 % | every run offends, nothing named | nothing named | names it |
     | interior block of 3 | fails | fails | fails |
     | drift or wander across the series | fails, names most of the series | the same | fails, names nothing |

     The trace alone is not enough either: on Re₆Zr the amplitude wanders by
     four times its error, the three runs sit 2.7–2.9 scatters out, and it is
     the cost that finds them — hence both.
   - **What the finding names.** The end block of the runs that depart in
     the rung below when they were too many to exempt, else of the exempt
     and offending runs of the fit. Nothing when the departure reaches both
     ends: a smooth drift fails the rung with no exemption and no finding,
     and the amplitude stays local, where its trace shows the drift. On YMnAl
     this names the three coldest runs, not seven: with those three exempt
     the amplitude shares at +0.6σ with no offender, so the other four
     offended only because the three dragged the shared value.
   - **Open for Phase 4.** On BiSCCO both amplitude rungs are adequate
     (total 0.0σ, every amplitude +0.9σ), so the ladder goes on in the
     template as given and the fraction form's further rungs are not fitted.
     If the acceptance row (total pre-selected, the smooth-σ pattern listed)
     is to hold, the stage has to branch when both are adequate.
4. **Verdict and persistence (core).** D1, D13, D14: the objective on
   `GlobalFitWizardRecommendation`, re-rank under either objective, phases
   path, deletions, schema bump with migration for stored recommendations.
   *Landed.* The wizard's default objective is the trend one.
   - **Where it lives.** `global_search/trend_objective.py` holds what the
     wizard's types need (`SelectionObjective`, `CandidateRung`, the band, the
     ranking and the summary); `global_search/trend_search.py` is the search,
     built from the wizard's own anchor task, pool drain and assembly, so the
     wizard imports it where it calls it.
   - **A candidate.** Every rung is a `GlobalCandidateAssessment` built by
     `_assemble_assignment_assessment`, with a `rung` (`CandidateRung`): cost
     in σ for the series and per run, the trend quality of each local
     parameter, exempt runs, hard-to-justify names, pass disagreements,
     whether it is its ladder's pre-selected rung, the ladder's end block and
     the template's all-local χ²ᵣ. A rung in the fraction form carries a
     `template` with the same key and title and the grouped model, so every
     name on the assessment is that model's and there is no second model field
     to remember. A shared amplitude with exempt runs stays in
     `global_param_names`; `exemptions` maps it to the runs that keep their
     own value, and the IC counts those columns.
   - **The search.** All-local nodes for every candidate (free when the
     pre-screen table is at the search resolution and carries its degrees of
     freedom; fits restored from a project carry neither those nor a
     covariance and are fitted again), then the band, then one pool task per
     competing template. Everything is fitted and reported at the series
     search resolution, where costs are against all-local fits of the same
     records.
   - **Who climbs, who contends.** A ladder is climbed for the templates
     inside the band, for those the data identified (the pattern and
     oscillatory keys the statistical shortlist forces) and, when the user
     ticked templates, for exactly those. Only templates inside the band of
     the climbed ones contend for the recommendation: on the hopping test
     series a forced template 25 % worse in χ²ᵣ had every parameter trending
     at 0.94, and trend quality must not buy back a fit the band rejected.
   - **Residual gates under the trend objective are a caveat.** Adequacy
     against the template's own all-local fits and the band are what admit a
     rung; the summary names the runs whose residual gate fails. Under the
     statistical objective a per-run gate failure still blocks, and a series
     warning (fingerprint jump, clustered failures) is a caveat under both.
     The staged-globalisation search that used the roughness was dead code
     (tests only) and is deleted.
   - **Lines that vanish.** A series with no break is one phase, so the trend
     search leaves out a multiplet or Overhauser rung whose lines are
     consistent with zero on a run, series-wide as well as per phase — the
     rule the per-phase role search already applied. A template whose
     pre-selected rung goes this way does not contend. The rule is read in the
     template as given: a fraction-form rung has no amplitude per line, so its
     total and fractions are turned back into amplitudes, with errors that
     leave out the covariance between the two. On YMnAl the two-cut-off
     Overhauser template otherwise won (worst trend 0.83 against the stretched
     exponential's 0.78) while describing two runs as plain relaxation.
   - **Phases (D13).** A phase's answer is the best pre-selected rung among
     the templates in the band on that phase's runs. The break is still scored
     by the best partition BIC among the phase's fits: a rung may cost 2σ and
     a template 3 %, which on real point counts is more than a break is worth.
   - **Budget (D8), changed.** The measured screening times are not a forecast
     of a rung: the store on the development machine holds 40 s per 1000
     points for dynamic Gaussian KT, which puts copper's coupled fit at over
     twenty minutes, and the whole copper ladder runs in under a second. The
     budget is a stopwatch instead: a ladder tries a further parameter only
     while a rung as long as its last would end inside the series budget
     (180 s; 1800 s per phase), and always climbs the background and amplitude
     rungs. `climb_sharing_ladder` takes that as `climb_further`. A template is
     lost only to the pool's 1800 s backstop. On YMnAl three of eleven
     templates were restricted and the search took about 200 s.
   - **Display order.** Template by template: contending templates in rank
     order, so the recommended rung is first, then the others by fit; within a
     template the pre-selected rung, the other adequate rungs by trend, then
     the costly ones by cost.
   - **Persistence.** Schema v24: `objective` on the recommendation, `rung` on
     each assessment, `total_variation`/`roughness` dropped; a stored
     recommendation migrates as statistical. Merging a result of one objective
     into a recommendation of the other replaces its optimised candidates.
   - **Corpus.** Copper: dynamic Gaussian KT + constant, amplitude, background
     and Δ shared, ν local (44 s). YMnAl: stretched exponential + constant,
     background shared, amplitude not shareable through the three coldest runs
     (about 200 s). Molecular AFM, per phase: one transition, and in each
     phase the two-line template with nothing shared (250 s).
   - **Open for Phase 5.** Parameter recommendations are empty on a rung, so
     the Compare step's role table is; apply must leave exempt runs out of the
     coupled series (D17) and takes the fraction-form model from the
     assessment's template; the end-block finding also fires on an amplitude
     that genuinely steps at one end; a rung far cheaper than all-local (a
     large negative cost) means the all-local fits were not at their minimum,
     which the ladder does not act on; the reference page still describes the
     statistical Compare step, and its result screenshot is pinned to that
     objective.
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
| YMnAl | stretched exponential, background shared; amplitude reported as not shareable through the three coldest runs (24573–24575) |
| Copper | dynamic Gaussian KT with amplitude, background and Δ shared |
| BiSCCO | two Gaussian lines with both amplitudes, both phases and one frequency shared, both widths local; the shared-total rung listed at no cost |
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
