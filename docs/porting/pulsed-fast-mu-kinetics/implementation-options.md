# Implementation options — pulsed-source fast-Mu kinetics

Two routes recover λ_Mu past the truncation degeneracy (both share `A_Mu` across
the series); a third does nothing new and is rejected. After λ_Mu(run) the trend
maths is identical and uses existing components.

## Option A — Documentation-only recipe (rejected as the *primary*)

A user-guide page showing how to drive the existing `fit_global` with `A_Mu`
shared, then `fit_scan_model(..., "Linear")` for k_Mu, then a manual Arrhenius
fit. **No new code.**

- **Pro:** zero engine surface; everything already exists.
- **Con:** leaves the method **undiscoverable** in the API — the exact failure
  mode the corpus keeps hitting (`memory: api-discoverability-in-code-hints`:
  documented-but-not-pointed-to recurs). No clean tested entry point; every user
  re-derives "share `A_Mu`" and the Arrhenius unit handling. Rejected as the
  headline, but its doc page is still produced (it *is* the user guide for
  Option B/C).

## Option B — Shared-amplitude kinetics module (RECOMMENDED)

A thin, Qt-free core module (e.g. `core/fitting/mu_kinetics.py`) that **encodes**
the method as composition over implemented seams, plus the user-guide page.

Proposed public surface (names indicative, finalise at impl):

```python
# 1. Degeneracy-breaking per-run λ_Mu from a 2 G Mu series at one temperature.
fit_mu_relaxation_series(
    datasets,                    # 2 G MuonDatasets (one temperature)
    *, f_mu=2.78,                # MHz, fixed (γ_Mu·B); not free
    share_amplitude=True,        # A_Mu (+ phase) global; λ_Mu, dia local
    t_min=None, t_max=None,      # default = good-bin window
) -> MuRelaxationSeriesResult    # per-run λ_Mu ± σ, shared A_Mu, χ²r, diverged set
#   → wraps fit_global with model "Oscillatory*Exponential + Exponential"
#     (or MuoniumTF*Exponential), global_params=[A_1, frequency(fixed), phase],
#     local_params=[Lambda_2, A_3, Lambda_3].

# 2. Pseudo-first-order line λ_Mu = λ0 + k_Mu·[x]  (one temperature).
fit_bimolecular_rate(
    concentrations, lambdas, lambda_errors,
) -> BimolecularRateResult       # k_Mu (slope), λ0 (intercept), ± σ, χ²r
#   → weighted linear least squares (reuses the Linear-component fit path).

# 3. Arrhenius E_a from k_Mu(T).
fit_arrhenius(
    temperatures, k_values, k_errors, *, energy_unit="kJ/mol",
) -> ArrheniusResult             # E_a, log10 A, ± σ  (guide's log10/2.3R form)
```

- **Pro:** one discoverable, tested call per step; the "share `A_Mu`" decision is
  encoded once, correctly. Stays in core (GUI-free, per the invariant). Reuses
  `fit_global`, the composite model, `Linear`, and `Arrhenius` — no new minimiser
  or model maths. Matches the house pattern (`asymmetry-domain-global-fit`,
  `time-integral-asymmetry`: "every primitive exists; the port is composition").
- **Con:** ~3 small public functions + result dataclasses to design and test.

## Option C — Option B plus the amplitude/fraction inversion cross-check

Add to Option B an analytic-limit reducer:

```python
mu_relaxation_from_amplitude(
    dataset, *, reference_amplitude, f_mu=2.78, t_eff=None,
) -> (lambda_mu, sigma)          # λ = −ln(A_surv / A_ref) / t_eff
#   A_surv from a fixed-f, fixed-φ single-frequency amplitude fit (or FFT peak).
```

- **Pro:** an independent route that agrees with the global fit in the
  fully-truncated limit → strong correctness signal; matches the muon-school
  "fraction-loss" language literally; cheap (no series needed per point).
- **Con:** needs an explicit `A_Mu(0)` reference and an effective time origin
  `t_eff` (pulse-convolved) — more knobs, more ways to misuse. Best added **after**
  B works, as a verification cross-check and a documented manual fallback.

**Recommendation: implement Option B now; fold the Option C reducer in as the
RED-test cross-check and a documented fallback** (small, and it doubles as
independent verification of B on the truncated members).

## Trend maths (shared by all options — existing components)

- **Concentration line.** `λ_Mu = λ₀ + k_Mu·[x]` over [x] ∈ {0 (water), 1
  (quarter), 2 (half), 4 (full)} at room T → slope k_Mu (relative units),
  intercept λ₀. A small analytic weighted least-squares (`_weighted_linear_fit`)
  is used rather than routing the 2-parameter line through the iminuit `Linear`
  component — exact, lighter, and gives directly-propagated slope/intercept errors
  (the `Linear` component remains available for the GUI trend surface).
- **Arrhenius.** Repeat per T (278–358 K) → k_Mu(T); fit the guide's
  `log₁₀ k_Mu = log₁₀ A − E/(2.3·R·T)`. **Unit caveat:** the built-in `Arrhenius`
  component is `a*exp(-Ea/(k_B·T))` (`parameter_models.py:424`) → Ea is
  *per-particle* in k_B units. To emit the guide's **molar** E (kJ·mol⁻¹) either
  (a) fit log₁₀k vs 1/T linearly in the guide's exact form (slope = −E/2.3R), or
  (b) use the component and post-scale `E_molar = N_A·Ea`. Decide at impl;
  recommend (a) — it is literally the guide equation and avoids a k_B/R mismatch.

## Sign-off (2026-06-16, Ben)

**Option B approved** — encode the shared-amplitude method as a thin core
`mu_kinetics.py` (3 functions + result dataclasses) + the user-guide page;
Option-C amplitude inversion folded in as the RED-test cross-check.
**Decision 2: self-calibrated `A_Mu`** — fitted as a shared global from the series
itself (no external reference run required). **Decision 8: synthetic RED gate +
env-gated real-corpus sweep**, no binary fixtures. Remaining table rows below take
the recommended defaults unless grounding turns one up.

## Decisions needed (sign-off)

| # | Decision | Recommended | Alternatives |
|---|---|---|---|
| 1 | Primary engine | **B: shared-amplitude global fit** | A (docs only); C-inversion alone |
| 2 | `A_Mu(0)` reference | **self-calibrated shared global** (series pins it) | explicit reference run 78251 |
| 3 | Concentration axis | **caller-supplied `[x]` vector** (not in `.nxs`) | new metadata field / `order_key="concentration"` |
| 4 | E_a convention | **kJ·mol⁻¹ via guide `log10/2.3R` linear fit** | built-in `Arrhenius` + `N_A` post-scale |
| 5 | Module home / API | **new `core/fitting/mu_kinetics.py`**, 3 funcs + result dataclasses | extend `field_scan.py` / trend helpers |
| 6 | Mu model | **`Oscillatory*Exponential + Exponential`** (single f_Mu) | `MuoniumTF*Exponential` (four Mu⁰ lines) |
| 7 | f_Mu, φ | **f_Mu fixed = γ_Mu·B; φ shared global** | φ also fixed (phase-scan seed) |
| 8 | Verification | **synthetic RED gate + env-gated real-corpus sweep** | corpus-only (rejected: no binary fixtures) |

Open sub-questions to resolve during grounding:

- Is `select_period` needed at all here? EMU 2 G runs in this set appear
  single-period (the RF/DEVA period machinery was for benzene). Confirm
  `period_count == 1` for 78251–78302 and skip period selection if so.
- Effective time origin for the Option-C inversion: t_g (first good bin) vs a
  pulse-convolved `t_eff` accounting for the ~80 ns ISIS pulse. Affects only the
  inversion's absolute λ scale, not the global fit.
- Whether to share φ globally or fix it per-run from a phase scan (the prior
  session phase-scanned). Sharing is cleaner if the geometry is common.
