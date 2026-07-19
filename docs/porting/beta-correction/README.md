# Beta Correction (musrfit Asymmetry Fit Type 2) Study

Status: study complete; implementation pass in progress on `feat/beta-correction`.

This study ports the **beta correction** from musrfit's asymmetry fit
(fit type 2): a second detector-balance parameter, alongside `alpha`, that
accounts for the *intrinsic asymmetries* of the forward and backward detector
groups differing (solid-angle/absorption effects that scale the observable
asymmetry amplitude rather than the count rate).

Reference: musrfit user manual, "Asymmetry Fit (fit type 2)"
(<https://lmu.pages.psi.ch/musrfit-docu/user-manual.html>; also
`$MUSRFIT_SRC/doc/html/user-manual.html`), and the musrfit source
(`$MUSRFIT_SRC/src/classes/PRunAsymmetry.cpp`).

## Definitions and formulas

musrfit defines (their convention, α on the forward histogram):

- `α = N₀,b / N₀,f` — ratio of detector efficiencies / solid angles
  (count-rate balance).
- `β = A₀,b / A₀,f` — ratio of the intrinsic asymmetries of the two groups.

The corrected asymmetry (musrfit, `PRunAsymmetry.cpp:1412`):

    A(t) = [α·(N_f − B_f) − (N_b − B_b)] / [α·β·(N_f − B_f) + (N_b − B_b)]

Asymmetry's convention puts α on the **backward** group
(`core/transform/asymmetry.py`): `A = (F − αB) / (F + αB)`, i.e.
`α_ours = 1/α_musrfit`. The algebraically identical β extension in our
convention (multiply musrfit's numerator and denominator by `α_ours`):

    A = (F − αB) / (βF + αB),   with β = A₀,b / A₀,f  (same β as musrfit)

Derivation check: with `F = N₀,f(1 + A₀,f P(t))`, `B = N₀,b(1 − A₀,b P(t))`
and `α_ours = N₀,f/N₀,b`, the numerator is `N₀,f·P·(A₀,f + A₀,b)` and the
denominator `N₀,f[(β+1) + P(β·A₀,f − A₀,b)]`, whose P-term vanishes exactly
when `β = A₀,b/A₀,f`, giving `A(t) = A₀,f·P(t)` — the forward group's
asymmetry, the same result musrfit's form produces. `β = 1` reduces to the
current formula exactly.

Exact Poisson error propagation (var F = F, var B = B, covariance kept):

    ∂A/∂F = αB(1+β)/D²,  ∂A/∂B = −αF(1+β)/D²,  D = βF + αB
    var(A) = α²(1+β)²·F·B·(F+B) / D⁴

i.e. the current error expression with the factor `2|α|` generalized to
`|α|·(1+β)` and the denominator `F + αB` replaced by `βF + αB`. At `β = 1`
this is bit-for-bit today's formula. The count-error variant scales
`√(B²σ_F² + F²σ_B²)` by the same `|α|(1+β)/D²` factor.

## Key findings (musrfit source study, 2026-07-19)

- **β appears only as the product αβ in the denominator** — application is a
  single formula at asymmetry formation, after deadtime/background/packing
  (`PRunAsymmetry.cpp:1404–1421`; pipeline order `PrepareData` `:610+`).
- **Determination is independent of α.** β does not affect count totals, so no
  count-ratio estimator can see it; conversely β does not move the corrected
  asymmetry's zero crossing (`A = 0 ⇔ αF = B` regardless of β), so all three
  of our α estimators (`diamagnetic`, `general`, `ratio`) are unbiased by β.
- **musrfit has no estimator for β** (nor for α): both are msr-file references
  to FITPARAMETER entries, each independently fixed or fitted
  (`PMsrHandler.cpp:3549–3603`); musredit's dialog is a plain line edit. A
  parameter fixed at exactly 1.0 collapses to the "unity" tag and the
  correction is skipped (`PRunAsymmetry.cpp:152–183`).
- **musrfit's data-error propagation ignores both α and β**
  (`PRunAsymmetry.cpp:1418` propagates the raw asymmetry error only). We
  already diverge from this for α (we propagate exactly); β extends the same
  documented divergence (see `docs/porting/asymmetry-error-propagation/`).
- **WiMDA has no equivalent.** WiMDA's `AFbeta` / "Bsln beta"
  (`$WIMDA_SRC/src/Analyse.pas:7035`, `AsymFitFunction.pas:688`) is a
  stretched-exponential exponent on the baseline relaxation
  `exp(−(λt)^β)` — a relaxation shape parameter, not a detector balance. Do
  not conflate them. Mantid likewise exposes no asymmetry-amplitude balance.
- **No reference dataset with β ≠ 1 exists in the local corpora** — the msr
  examples' `beta` parameters are `generExpo` stretch exponents. Verification
  is therefore by algebraic identity and synthetic data (`test-data.md`).

## Decision log

- **2026-07-19 — design agreed with Ben (planning discussion).**
  1. β gets its **own card** in the Corrections editor, colour **blue** —
     justified by determination-independence from α (application coupling
     lives in the core formula, not the UI).
  2. v1 sets β as a **fixed user-entered scalar** only (default 1.0).
  3. The data-driven estimator (paired single-histogram / count-domain fit of
     both groups, which yields α and β simultaneously) is **deferred pending
     expert opinion**.
  4. **Scalar only** — no per-projection `beta_x/y/z` vector variants.
  5. Fittable β (musrfit lets MINUIT float it) is **deferred**; its natural
     home is the count-domain path (`fit_fb_alpha` / `FBCountModel`), as a
     per-side asymmetry amplitude `forward ∝ (1 + A·P)`,
     `backward ∝ (1 − β·A·P)` — note this is the same feature as the deferred
     estimator.

## Scope (implementation pass)

- Core: `beta` threaded through `compute_asymmetry`,
  `compute_asymmetry_with_count_errors`, `binned_fb_asymmetry`,
  `reduce_grouped_asymmetry`, `group_forward_backward`
  (`GroupedForwardBackward.beta`), the integral observable, simulate/combine
  reductions, and the F-B representation — with the re-derived exact error.
- Persistence: `beta` on `GroupingProfile` and in the grouping payload,
  emitted **only when ≠ 1** so existing projects/profiles round-trip
  byte-identically (the `t0_policy` precedent; no schema bump).
- GUI: blue `STAGE_BETA` card in the Corrections column with a value field,
  pipeline-strip chip, and a `compare_stage="beta"` before/after ghost
  (β = 1 vs β) in the shared preview.
- Docs: `detector_grouping.rst` card documentation, conventions/glossary
  formula updates.

Out of scope (deferred, recorded above): β estimator, fittable β,
per-projection β, BNMR-style helicity variants.

## Candidate port seams

1. **Formula seam** — `compute_asymmetry(..., beta=1.0)` keyword; every
   call site that threads `alpha` explicitly threads `beta` beside it
   (greppable, no hidden state).
2. **Grouping-read seam** — lenient `beta` read next to the `alpha` read in
   `group_forward_backward` (non-finite/≤0 → 1.0).
3. **Policy seam** — plain `beta: float` on `GroupingProfile` (a full
   `BetaPolicy` dataclass is deliberately deferred until the estimator gives
   it a second mode; recorded in `implementation-options.md`).
4. **Preview seam** — the existing `compare_stage` machinery gains a
   `"beta"` stop; the ghost recomputes the same corrected counts at β = 1.
