# β estimator — implementation plan

Status: **in progress** (approved by Ben, 2026-07-19). Follows the study in this
directory and the expert consultation on the design note
(`beta-estimator-design-note.pdf`, 2026-07-19).

## Context and decision

The scalar β correction shipped in PR #272 (v0.15.0) with the estimator deliberately
deferred pending expert consult. The consultation outcome:

- **Protocol A** (simultaneous forward+backward count-domain fit, shared physics,
  β on the backward amplitude) is **endorsed as the measurement protocol** and will be
  the default.
- Some instrument scientists prefer **Protocol B** (paired single-histogram fits,
  β̂ = A₀,b/A₀,f by ratio); it will be offered as an alternative and cross-check.
- β is measured on a **weak-TF calibration run** — the same run type already used for α.
- β is broadly an **instrument-dependent** property of a detector pair. Typical values:
  ~0.8–0.9 on FLAME; just under 1 on GPS. (Goes in the docs as guidance.)

Protocol C (asymmetry-space fit) is dropped.

## What already exists (surveyed 2026-07-19)

The heavy lifting is done; Protocol A is a small delta on shipped code.

| Piece | Where |
| --- | --- |
| fgFB simultaneous F+B Poisson count fit with shared `N0`, free `alpha`, per-side backgrounds | `fit_fb_alpha` — `core/fitting/count_domain.py:746` |
| F/B count model (`N0·√α`/`N0/√α` split, sign ±1 on the polarization term) | `build_fb_count_model` — `core/fitting/grouped_time_domain.py:792` |
| Single-histogram count fit (Protocol B's building block) | `fit_single_histogram` — `core/fitting/count_domain.py:610` |
| Minimiser seam (migrad+simplex, HESSE, opt-in MINOS) | `drive_minuit` — `core/fitting/engine.py:333` |
| Cross-parameter covariance on results | `FitResult.covariance` + `covariance_parameters` — `core/fitting/engine.py:564` |
| Weak-TF calibration-run classifier + run-combo population | `classify_tf_calibration_run` — `core/data/calibration.py:91`; `populate_calibration_run_combo` — `gui/windows/grouping/alpha_section.py:265` |
| Off-thread estimate pattern (request → TaskRunner worker → result → policy signal) | `gui/windows/grouping/alpha_section.py` (`AlphaEstimateRequest`, `run_alpha_estimate`, `_on_estimate*`) |
| Provenance policy object + payload keys + promotion | `AlphaPolicy` — `core/project/profiles.py:211`; `_apply_alpha_policy` — `profiles.py:1222`; `promote_alpha_to_grouping` — `core/transform/promote.py:22` |
| Staleness (correction digest, banner, chip suffix, card tint) | `gui/windows/grouping/dialog.py:4536–4629`, `CorrectionCard.set_stale` |
| β card (fixed scalar, steel-blue stage colour, vector-mode hidden) | `gui/windows/grouping/beta_section.py`, `dialog.py:974–983` |

## Protocol specifications

### A — simultaneous count fit with β free (default; method id `"count_fit"`)

Extend the fgFB model so β scales only the backward polarization amplitude:

```
N_f(t) = N0·√α · e^(−t/τ_μ) · [1 + A·P(t)]        + b_f · (background term)
N_b(t) = N0/√α · e^(−t/τ_μ) · [1 − β·A·P(t)]      + b_b · (background term)
```

- `P(t)` is the built-in `oscillatory` model (damped cosine, ω/λ/φ) — fixed choice in
  v1; no model picker in the estimator UI.
- Shared free parameters: `N0`, `alpha`, `beta`, `amplitude`, ω, λ, φ. Per-side:
  `background`, `background_b`. Data handling (raw counts, window, exclude, deadtime
  options) identical to `fit_fb_alpha` today.
- Cost: Poisson (Cash) by default, via the existing `_solve`/`drive_minuit` path.
- Errors: HESSE symmetric by default; MINOS opt-in (threads through the existing
  `minos=` kwarg). Report the α̂–β̂ correlation from the covariance block.
- Bounds: β > 0 with a positive clamp like `_clamp_alpha_positive`
  (`count_domain.py:1053`); sane box [0.01, 10] on the fit (UI spin stays [0.01, 1000]).
- **Parameter-name collision**: the structural name `beta` clashes with e.g. the
  stretched-exponential stretch exponent. The collision guard
  (`_guard_model_param_collisions`) must reject `beta`-named model params **only on the
  β-estimation path** — `beta` must NOT be added to the global `RESERVED_COUNT_PARAMS`,
  or existing stretched-exponential count fits would regress.
- Seeding: ω from the run's field metadata (γ_μ·B), A ≈ 0.2, N0/backgrounds from count
  levels, λ small — follow the `_count_fit_seed_params` precedent
  (`gui/panels/fit/global_tab.py:2313`), but implemented in core so both GUI and
  scripts share it.

### B — paired single-histogram fits (alternative; method id `"single_histogram"`)

Two independent `fit_single_histogram` calls over the same window: forward
(sign +1) and backward (sign −1), each with its own `N0`, `background`, and
independently floating ω/λ/φ (no shared physics — that is the point of B, and its
statistical weakness). Then:

- β̂ = Â₀,b / Â₀,f, with σ_β = β̂·√[(σ_f/Â_f)² + (σ_b/Â_b)²] (independent fits).
- α̂ = N̂₀,f / N̂₀,b reported alongside, same ratio-error treatment. (Forward over
  backward — the opposite direction to β. In our α-on-backward convention the fgFB
  split is forward `N0·√α`, backward `N0/√α`, so α = N0,f/N0,b; the design note's
  α = N0,b/N0,f is the musrfit convention, i.e. our 1/α. Caught during M1.)
- No α–β correlation is available (report `None`); documented as a cross-check of A.

## Core API (new module `core/fitting/beta_calibration.py`)

Thin orchestrator over `count_domain` primitives; keeps `count_domain.py` focused.

```python
BETA_ESTIMATION_METHODS = ("count_fit", "single_histogram")

@dataclass(frozen=True)
class BetaEstimate:
    beta: float
    beta_error: float | None
    alpha: float               # fitted alongside, for consistency display
    alpha_error: float | None
    alpha_beta_correlation: float | None   # A only; None for B
    method: str
    n_bins_used: int
    reduced_chi2: float | None # per-side χ²/dof summary from the fit(s)
    ok: bool
    message: str

def estimate_beta_detailed(dataset, forward_group, backward_group, *,
    method="count_fit", t_min=None, t_max=None, exclude=None,
    field_tesla=None, minos=False, cancel_callback=None) -> BetaEstimate
```

Failure modes return `ok=False` with an informative message (no oscillation found /
fit did not converge / degenerate amplitude), mirroring `AlphaEstimate` semantics.

Changes inside existing modules:

- `build_fb_count_model(model_fn, *, with_beta=False)` — β multiplies the backward
  polarization term only; default path byte-identical to today.
- `fit_fb_alpha(..., estimate_beta=False)` — when true, adds shared free `beta`,
  scoped collision guard, positive clamp; result's `shared_parameters` gains `beta`.
  (Alternatively a thin `fit_fb_alpha_beta` wrapper — implementer's choice, but the
  existing function signature must not change behaviour for current callers.)

## Persistence — `BetaPolicy`

Mirror `AlphaPolicy` (`profiles.py:211`) exactly, minus the `per_run_estimate` mode
(not meaningful for β in v1):

- `BetaPolicy(mode: "fixed"|"calibrated", value, error, method, source_run)`.
- Grouping-payload keys: `beta_method`, `beta_error`, `beta_reference_run`,
  `beta_correction_digest` — emitted **only** in calibrated mode; `beta` itself keeps
  the emit-only-when-≠1 invariant (byte-identical round-trips for pre-β projects, and
  the `fourier/spectrum.py` `fb_beta` digest key stays untouched).
- No schema bump (additive, defaulted keys; `CURRENT_SCHEMA_VERSION` stays 17).
- Vector/projection profiles continue to never emit β or any β provenance
  (`resolve_effective_grouping` guard at `profiles.py:1116` extends to the new keys).

## GUI — grouping-dialog β card grows a "Measure from run" section

Mirror the α card's Flow A (`alpha_section.py`) inside `beta_section.py`:

- Run combo populated by `populate_calibration_run_combo` (weak-TF candidates
  highlighted, auto-select via `best_calibration_run_index`) — shared helper, not a
  copy.
- Protocol selector: "Count fit (recommended)" / "Single-histogram ratio" mapping to
  the two method ids; default `count_fit`.
- Estimate runs off-thread on the dialog's `TaskRunner` with an immutable
  `BetaEstimateRequest` snapshot and a token-guarded finish handler, exactly like
  `run_alpha_estimate`. Cooperative cancel via `worker.is_cancelled`.
- Result row: `β = 0.8732 ± 0.0041` plus the fitted α as a consistency readout
  (warn-tint when it disagrees with the card's current α beyond ~3σ).
- Apply → emits `beta_estimated(BetaPolicy(mode="calibrated", ...))`; the dialog
  writes the spin, records provenance and the current correction digest.
- Staleness: `_beta_is_stale()` reusing `_correction_digest()`; stale banner + " ·
  stale" pipeline-chip suffix + `card.set_stale`, all mirroring α (`dialog.py:4601+`).
- Hand-editing the β spin drops provenance (mirror `_on_alpha_spin_edited`).
- Vector mode: card stays hidden; estimator inherits the scalar-only rule.

## Documentation (same PR)

- New `docs/reference/data_reduction/beta_calibration.rst` modeled on
  `alpha_calibration.rst`: the two protocols, when β matters, instrument guidance
  (FLAME ~0.8–0.9, GPS just under 1; instrument-dependent, measured on the same
  weak-TF run as α), uncertainties, references (musrfit manual fit type 2).
- Update `detector_grouping.rst` (β card now has a measure action),
  `conventions.rst` cross-link, glossary if needed.
- Screenshot scenario for the β card with a completed estimate (register per
  `docs/README.md` rules; sibling scenario `alpha_calibration_dialog.py` is the
  template).
- `docs/porting/beta-correction/expert-consultation.md` recording the consultation
  outcome (content from the Context section above).
- `CHANGELOG.md` `[Unreleased]` entry.

## Milestones (single PR on `feat/beta-estimator`, sequential, orchestrator-reviewed)

Each milestone: one subagent implements; the orchestrator reviews the diff, runs the
focused tests plus the fast tier, and commits before the next starts. Subagents run
**only single-file focused tests** (never the suite); `validate` runs once, at the end,
by the orchestrator alone.

| # | Scope | Agent | Focused tests |
| --- | --- | --- | --- |
| M1 | Core: model extension + `fit_fb_alpha` β path + `beta_calibration.py` (both protocols, seeding, errors, correlation) + tests | **Opus** | new `tests/core/test_beta_estimation.py`; regression: `tests/core/test_count_domain_fits.py`, `tests/core/test_alpha_estimation.py` |
| M2 | Persistence: `BetaPolicy`, payload keys, digest key, resolution, promotion analogue + tests | **Sonnet** | `tests/core/test_beta_correction.py`, `tests/core/test_grouping_profiles.py`, `tests/gui/test_project_schema.py` |
| M3 | GUI: β-card measure section, TaskRunner flow, provenance label, staleness wiring, apply/drop rules + tests | **Opus** | `tests/gui/test_beta_section.py`, `tests/gui/test_alpha_section.py`, `tests/gui/test_grouping_dialog.py` |
| M4 | Docs page + screenshot scenario + consultation record + changelog | **Sonnet** | `python tools/harness.py docs` (orchestrator) |

M1 test intent (the load-bearing physics):

- Synthetic two-detector data at known (α, β) — e.g. α=1.15, β=0.85, weak-TF damped
  cosine — recovered within quoted errors by both protocols; A and B agree within
  combined errors.
- β=1 data → β̂ consistent with 1; `with_beta=False` path byte-identical to current
  `fit_fb_alpha` output.
- α̂–β̂ correlation reported and finite for A; `None` for B.
- Non-precessing (ZF-like) input → `ok=False` with an informative message, not a
  garbage number (β is degenerate without precession — design-note §4).
- Collision guard: stretched-exponential model rejected loudly on the β path; still
  accepted on the existing non-β count-fit paths.

## Decisions (Ben, 2026-07-19)

1. **Applying α from the Protocol A result:** the Apply action sets **β only**; the
   fitted α is shown as a consistency readout (warn-tinted beyond ~3σ from the card's
   current α), with an explicit secondary "also update α" affordance.
2. **Method id strings:** `"count_fit"` and `"single_histogram"`, as specified.
3. **Sanity warning:** yes — flag β̂ outside [0.5, 1.5] as suspicious in the result
   row (typical instruments sit 0.8–1.0).
