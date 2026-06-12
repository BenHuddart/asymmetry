# Spectral moments — test data

Moments have closed-form values on analytic distributions, so the core is tested
**without any external corpus** — the primary verification is synthetic and
self-checking. WiMDA arithmetic is pinned by a transcribed oracle on a shared
spectrum. External `.nxs`/`.bin` runs are an optional, env-gated sanity layer.

## 1. Synthetic distributions (closed-form, no corpus)

Built in-test as `(x, amplitude)` arrays on a dense field grid.

### 1.1 Gaussian line — symmetry + known width

`p(B) = exp(−(B−B₀)²/2σ²)` on a grid wide enough that the cutoff/window capture
the full line. Expected:

- `b_ave ≈ B₀`, `b_pk ≈ B₀` (parabolic peak sits on the centre), `b_diff ≈ 0`.
- `b_rms_mean ≈ σ` (to grid/discretisation tolerance).
- `skewness ≈ 0`, `skewness_g1 ≈ 0`, `beta ≈ 0`.

Tolerances scale with grid spacing; a fine grid (Δ ≪ σ) gives sub-percent
agreement. This is the zero-skew anchor.

### 1.2 Skewed two-Gaussian mixture — all moments analytic

`p(B) = w₁·N(μ₁,σ₁) + w₂·N(μ₂,σ₂)` with `μ₂ > μ₁` and `w₂ < w₁` (a dominant line
plus a smaller high-field satellite → positive skew). The raw moments of a
Gaussian mixture are exact:

```
m0 = Σ wⱼ
mean      = Σ wⱼμⱼ / m0
E[B²]     = Σ wⱼ(σⱼ² + μⱼ²) / m0
E[B³]     = Σ wⱼ(μⱼ³ + 3μⱼσⱼ²) / m0
```

from which `b_ave`, `b_rms_mean = √(E[B²]−mean²)`, `m₃ = E[(B−mean)³]` and hence
`skewness_g1`, WiMDA `α`, and the sign of `β` follow analytically. The test
asserts the **windowed, cutoff-free** core output against these closed forms
(cutoff 0, full range), and asserts `skewness_g1 > 0`, `beta > 0` for the
high-field-tailed choice. This exercises every moment at once with no oracle
dependency.

### 1.3 Vortex-lattice-like asymmetric lineshape — β sign check

A sharp low-field cutoff with a long high-field tail (a saddle-point-cut `p(B)`
proxy, e.g. a one-sided exponential-tailed profile). Assert `b_ave > b_pk`,
`beta > 0`, `skewness > 0` — the physically expected positive skew of the
mixed-state field distribution. This is the dedicated β-sign-convention test.

### 1.4 Cutoff / range sensitivity

Sweep `cutoff_fraction` (0 → 0.5) and `x_range` (full → tight) on 1.2's mixture
and assert the documented behaviours: tightening the window toward the main line
drives `skewness`/`beta` toward 0 (the satellite tail is excluded); raising the
cutoff drops `n_sample` and narrows `b_rms_*`. These are recorded as the
"moments are cutoff- and range-sensitive" demonstration, with the window captured
in the result's `recipe`.

### 1.5 Uncertainty behaviour

With a known `σ[i]`, assert bootstrap errors are finite and shrink ∝ 1/√(signal)
as noise drops; assert `b_pk_err`/`beta_err` **inflate** on a deliberately noisy,
near-flat-topped spectrum (the fragility made visible). Seeded → deterministic.

## 2. WiMDA transcribed oracle (shared spectrum)

`tests/porting/spectral-moments/wimda_oracle.py` — a direct, line-by-line
transcription of `Moments.pas`'s arithmetic (consistent `0…n-1` indexing; the D1
bug **not** reproduced), used purely to pin our core against WiMDA on **one
shared synthetic spectrum**. Asserts `b_pk`, `b_ave`, `b_diff`, `b_rms_mean`,
`b_rms_peak`, `α`, `β` agree to ~1e-9. Keeps us honest about the parabolic-vertex
formula, the cutoff-vs-discrete-peak gate, and the weighting. (GPL: this is a
behavioural oracle — independently re-derived arithmetic, not copied source.)

## 3. External corpus (optional, env-gated)

The superconductivity use case (a real vortex-lattice MaxEnt spectrum) is a
*sanity* layer, not a unit-test dependency. Any test that loads an external run
**skips** unless its env var is set, mirroring `tests/test_psi_loader.py`:

```python
data_root = os.environ.get("ASYMMETRY_MUSRFIT_DATA")  # or the project's corpus var
if not data_root:
    pytest.skip("external corpus not available")
```

Candidate runs (from the testing corpus): a transverse-field run with a clear
precession line → MaxEnt reconstruction → moments, checked for plausible
`b_ave ≈ applied field` and a finite width. No golden numbers are committed for
external data; the assertions are qualitative (finite, ordered, right ballpark).

## 4. Provenance

Every recorded result carries its `recipe` (range, cutoff, unit, mode, bootstrap
seed). Tests assert the recipe round-trips through `FitSeries.extra` and `.asymp`
save/load, and that re-sending the same selection **replaces** (same batch id)
rather than duplicating.
</content>
