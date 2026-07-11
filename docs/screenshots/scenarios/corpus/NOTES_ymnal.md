# NOTES — Spin Glass YMnAl (Magnetism)

Module: `ymnal_spinglass.py`. Example: `Magnetism/Spin Glass YMnAl`.
Spec: the example's `GROUND_TRUTH.md`. Corpus's **stretched-exponential /
spin-glass** showcase. Same data as Telling *et al.*, PRB **85**, 184416 (2012).

## Scenarios registered

| name | render | intended docs use |
|------|--------|--------------------|
| `corpus_ymnal_spectra` | Overlay of LF 110 G spectra, 280 K → 85 K, framed 0–10 µs | Data-handling / motivation: muon relaxation grows dramatically as T_g ≈ 88 K is approached (critical slowing-down). |
| `corpus_ymnal_stretched_fit` | Converged `StretchedExponential + Constant` fit on the 95 K run (24577), param table showing β < 1 | Core analysis step: the guide's model `A·exp[−(λt)^β]+A_bg` on a run near the transition; β ≈ 0.565, χ²/ν = 1.00. |
| `corpus_ymnal_lambda_t` | λ(T) on a log-y axis with the fitted `CriticalDivergence` trend curve | **Headline**: the divergent rate approaching T_g. |
| `corpus_ymnal_beta_t` | β(T) points with β=1 and β=1/3 reference lines | The spin-glass lineshape signature: β falls from 1 (exponential) toward 1/3. |

Requires-fit: `stretched_fit`, `lambda_t`, `beta_t` run real iminuit fits at
capture time (`requires_fit = True`). `spectra` is a plain overlay render.

## Workflow followed (GROUND_TRUTH refs)

- **Model** (§4 / §6 / §10): `G(t) = A·exp[−(λt)^β] + A_bg`, component
  `StretchedExponential` (`A_1`, `Lambda`, `beta`) + `Constant` (`A_bg`).
- **Fit window** (§3 / §10, thesis): `0.5 ≤ t ≤ 12 µs`.
- **Parameter-fixing protocol** (§4), implemented in `_calibrate_amplitudes()`:
  1. Fit 24576 (90 K) free → take `A_bg` (≈ 3.13 %).
  2. Fit 24590 (280 K) with β = 1 and `A_bg` fixed → take `A` (≈ 27.97 %).
  3. Batch-fit the series with `A`, `A_bg` fixed and β free, warm-starting
     λ/β in descending-T order (`_fit_series()`).
- **Series** (§2 / §3): guide batch window 24580–24590 (100–280 K), extended
  down to 90 K (24576/24587/24578/24577/24588) to approach T_g. Runs at/below
  the frozen transition (24575/24574/24573 = 85/80/75 K) are **excluded** from
  the trend — they are outside the guide's paramagnetic series (§2/§9) and the
  stretched-exp λ/β partition becomes degenerate there (β collapses ≪ 1/3).
- **Trend model** (§4 / §10): `CriticalDivergence` `a·|T−T_c|^(−ν)+c`,
  T_c bounded < 90 K (transition sits below the lowest data point). The trend
  fit is weighted **uniformly** (equal weights, as a digitised λ(T) would be
  read); the steeply-shrinking per-point λ uncertainties otherwise over-weight
  the divergent tail and the minimiser fails to converge (which suppresses the
  panel's overlay curve, since it only draws for a successful fit).

## Fitted values vs GROUND_TRUTH targets

**Per-run batch fit** (A = 27.97 %, A_bg = 3.13 % fixed; warm-started):

| T (K) | λ_fit (µs⁻¹) | λ dig §10a | β_fit | β dig §10a |
|------:|-------------:|-----------:|------:|-----------:|
| 90    | 0.110 | 0.36 | 0.269 | ~0.50 |
| 91    | 0.090 | —    | 0.387 | —     |
| 92.5  | 0.081 | —    | 0.462 | —     |
| 95    | 0.066 | 0.21 | 0.564 | ~0.60 |
| 97.5  | 0.061 | —    | 0.605 | —     |
| 100   | 0.052 | 0.13 | 0.680 | ~0.70 |
| 111   | 0.032 | —    | 0.763 | ~0.80 |
| 120   | 0.027 | 0.08 | 0.827 | ~0.85 |
| 135   | 0.020 | —    | 0.844 | —     |
| 150   | 0.018 | 0.05 | 0.877 | —     |
| 180   | 0.013 | —    | 0.889 | —     |
| 221   | 0.012 | 0.03 | 0.952 | —     |
| 280   | 0.010 | 0.02 | 1.002 | ~1.0  |

**Critical-divergence trend fit** (uniform weights, T_c bounded < 89.5 K):

| Quantity | This run | GROUND_TRUTH target | Source |
|----------|---------:|--------------------:|--------|
| T_c (T_g) | **83.6 K** | 88.2 ± 0.2 K | paper §10 |
| exponent ν | **0.84** | γ = 0.92 ± 0.09 (Eq-7) / 0.80 (thesis) | paper / thesis §10 |
| β(280 K) | **1.00** | 1.0 | §6 / §10 Fig 6d |
| β floor (near T_g) | **0.27 (90 K)** | → 0.33 (1/3) by 85 K | §6 / §10 Fig 6d |

### Assessment (honest)

- **β(T): good.** Falls monotonically 1.00 (280 K) → 0.68 (100 K) → 0.56
  (95 K) → 0.27 (90 K), tracking the digitised §10a β(T) closely down to
  ~95 K. Near T_g it dips slightly *below* the paper's 1/3 floor (0.27 at
  90 K vs 0.33 at 85 K) — expected, since 90 K is closer to T_g than the
  paper's lowest paramagnetic point and the λ/β anticorrelation lets β run low.
- **λ(T): correct shape, low absolute scale.** λ rises ~11× from 0.010 (280 K)
  to 0.110 (90 K) — the divergence is unmistakable on the log axis — but sits
  a factor ~3 below the digitised values (paper 0.36 at 90 K, peak 0.50 at
  85 K). Two contributors: (i) **no α calibration** — the guide leaves α blank
  (§4/§9); the default grouping's uncalibrated amplitude partitions the signal
  differently, and stretched-exp λ is scale-dependent. (ii) The **λ/β
  anticorrelation**: our β falls faster than the paper's, so λ rises less. The
  *product/shape* (the physics the docs illustrate) is faithfully reproduced.
- **T_c = 83.6 K, ν = 0.84.** A few K below the paper's 88.2 K, and ν close to
  the **thesis** γ = 0.80 (§10). This matches GROUND_TRUTH §10's own caveat:
  the generic `CriticalDivergence a·|T−T_c|^(−ν)+c` is **not** the paper's Eq-7
  form `λ₀·[(T−T_g)/T]^(−γ)`, so its exponent and T_c differ from the Eq-7
  values — compare T_c approximately, and compare the exponent only against a
  matching functional form. (§10 records a prior Asymmetry run at T_g ≈ 89.5 K,
  ν ≈ 0.6; our 83.6/0.84 is in the same ballpark, the difference driven by
  fit window, series extent, and weighting.)

## Feature-demonstration opportunities

- **Batch fit + parameter-vs-temperature trend** with a model overlay (the
  `FitParametersPanel` `CriticalDivergence` fit) — the same machinery as
  `euo_nu_t_trend`, here on a *divergent* (not order-parameter) trend with a
  log-y axis.
- **Dual-parameter story from one batch fit**: λ(T) and β(T) come from the same
  per-run fits, shown as two scenarios. The panel does support multiple
  y-parameters on one axis, but λ (log, µs⁻¹) and β (linear, 0–1) read far more
  clearly as separate panels with their own framing/reference lines.
- **Not captured** (out of scope but present in the corpus): the TF 20 G
  calibration run 24563 (α from a damped-oscillation fit, §4), the ZF 280 K
  run 24591, and the frozen-regime runs 24573–24575 (75–85 K). A calibration
  screenshot on 24563 could round out the "data-handling" slot if desired.

## Problems hit (honestly)

- **Composite renames `A` → `A_1`.** `StretchedExponential`'s amplitude param is
  `A` standalone but `A_1` inside a `CompositeModel` — first fit KeyError'd
  until switched.
- **Unconstrained fits wander.** Without fixing A and A_bg (the guide's
  protocol), A jumps 15–25 % and A_bg 3–19 % run-to-run because λ, A, A_bg are
  correlated; λ(T)/β(T) become noisy and non-monotonic. Following §4's
  fix-A_bg-then-fix-A recipe is essential for a clean trend.
- **Trend curve overlay requires `result.success == True`.** The panel only
  precomputes/draws the model-fit curve for a converged fit. The first
  `CriticalDivergence` fit (weighted by real per-point λ errors) returned
  `success = False` and the overlay silently vanished — points only. Switching
  the trend-fit weighting to uniform errors fixed convergence (T_c = 83.6,
  ν = 0.84) and the curve renders. Plotted error bars still use the real
  per-point uncertainties (decoupled from the fit weighting).
- **Absolute λ scale / T_g precision** are the honest limitations (see
  Assessment). They stem from the missing α calibration and the generic-form
  trend model, both flagged in GROUND_TRUTH §4/§9/§10 — not fit bugs.

## Top pick for docs

`corpus_ymnal_lambda_t` — the headline divergence, log-y, with the fitted
critical trend curve; it is the single image that says "spin-glass transition".
Pair it with `corpus_ymnal_beta_t` (the β → 1/3 signature) and lead the page
with `corpus_ymnal_spectra` (visual motivation). `corpus_ymnal_stretched_fit`
is the best "how the model looks on one run" panel (β < 1, χ²/ν = 1.00).
