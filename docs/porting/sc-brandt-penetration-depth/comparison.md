# Comparison — field-dependent vortex-lattice line width

## Reference programs

| Program | Has field-dependent Brandt B_rms(B₀)? | Notes |
|---|---|---|
| WiMDA | No | Ships σ(T) gap models and a London-limit λ readout, not σ(B₀). |
| musrfit | Partial (user functions) | Brandt's `b`-dependence is typically entered as a user-defined function; no built-in named model. |
| Mantid | No | Muon GUI has no vortex-lattice field model. |
| Asymmetry (today) | No | Only σ(T) `SC_*` (temperature scope) + the `b→0` London helper `lambda_nm_to_sigma_us`. |

So this is a theory addition; the "reference" is the published Brandt
formula plus the Pratt LiFeAs validation, not a program to match byte-for-byte.

## Formula provenance and self-consistency check

Brandt (PRB 68, 054506) ideal-triangular-lattice rms field width:

```
B_rms(b) = 0.0609 (Φ₀/λ²) (1−b)[1 + 1.21(1−√b)³],   b = B₀/B_c2
σ(B₀)    = γ_µ B_rms(B₀)
```

Numerical cross-checks against the LiFeAs corpus `GROUND_TRUTH.md`
(γ_µ = 0.8516 µs⁻¹ mT⁻¹, Φ₀ = 2.0678×10⁻¹⁵ Wb):

| Quantity | Formula value | Ground truth |
|---|---|---|
| Single-crystal prefactor A = 0.0609·γ_µ·Φ₀ | 4.85×10⁴ µs⁻¹·nm² | literature 4.83×10⁴ (rounding) |
| Bracket at b→0 | 2.21 | — |
| `lambda_nm_to_sigma_us(195 nm)` vs A·2.21/195² | 1.072e5/195² = 2.82 µs⁻¹ both | identical (helper = b→0 limit) |
| Powder B_rms(195 nm, b→0) = 0.0609·Φ₀/(3^¼·195nm)² | **1.91 mT** | Fig. 1 plateau ≈ **1.9 mT** ✓ |
| Powder B_rms(244 nm, b→0) | **1.22 mT** | Sample 2 low-T ≈ **1.0–1.3 mT** ✓ |

The powder relation with the `3^{1/4}` length and coefficient 0.0609
reproduces the published B_rms plateau to within graphical-read tolerance,
confirming the coefficient and the powder factor. (The `GROUND_TRUTH.md`
transcription of Eq. (3) writes the *variance* coefficient 0.00371 = 0.0609²
in a position that reads as linear; 0.0609 is the value that reproduces the
figure, and 0.0609² = 0.00371 closes the loop.)

## Domain choice: σ vs B_rms

`σ = γ_µ·B_rms` is exactly linear, so a fit in either domain yields the same
λ and B_c2 — only the data column and the prefactor are rescaled by the
constant γ_µ. The existing SC param-models and the GUI parameter-trend
workflow all operate in **σ [µs⁻¹]** (the time-domain fits report a Gaussian
rate σ, not a field width). Fitting in σ-domain therefore drops straight into
the existing pipeline with no new units plumbing; B_rms is recoverable as
`σ/γ_µ`. **Recommendation: σ-domain primary.**
