# Frequency-domain finishers — verification plan

How each study claim and ported behaviour is verified. Ladder:
`python tools/harness.py structural` (study/index) → `lint` →
`test -- <file>` → full `validate` (~2 min, green per phase) →
`docs` (user-guide build) → opportunistic corpus smoke. GUI tests run under
`QT_QPA_PLATFORM=offscreen`.

## Verification targets (from the brief) → tests

### Field axis (verify-only)
- **Claim:** a known-field TF run peaks at γ_μ·B in Gauss, and FFT + MaxEnt
  spectra of the same run agree on peak position in field units.
- **Test:** `test_fourier_finishers.py::test_field_axis_peak_at_gamma_b`
  (synthetic TF run, assert peak bin → B within one bin via `units.convert`);
  `::test_fft_and_maxent_agree_on_peak_field` (same run through both engines,
  peak field within one bin). Plus the corpus smoke variant.

### Exclusions
- **Claim:** the existing core `exclude_frequency_ranges` is wired; the PSI
  preset removes RF harmonics.
- **Tests:** `::test_exclusion_zeroes_requested_band` (line inside band → 0,
  outside survives); `::test_psi_preset_centres` (preset yields DC + 50.63×{1..5}
  centres with the expected width formula); `::test_diamag_slot_tracks_reference_field`
  (slot centre = γ_μ·B). Core-level exclusion math already covered by
  `test_fourier.py::test_exclude_frequency_ranges_*` — extend, do not duplicate.

### Pulse-rolloff compensation
- **Claim:** synthetic pulse-broadened data with known frequency content
  recovers flat amplitude vs frequency after compensation; the uncompensated
  distortion is documented.
- **Tests:** `::test_compensation_flattens_amplitude` (build signal broadened by
  `R(ν)` from `pulse.py`; after ÷R(ν) amplitude flat within tolerance across the
  band below the node); `::test_uncompensated_rolloff_is_monotone` (records the
  distortion the docs describe); `::test_compensation_guard_bounds_gain`
  (content past the first node → gain capped/cut, output finite, no overflow).
  Reuse assertion: the test imports `pulse_amplitude_phase` — a parallel pulse
  model in the compensation code would fail review.

### Baseline (iterative σ-clip)
- **Tests:** `::test_sigma_clip_removes_offset_preserves_peaks` (offset → <1
  noise-σ, peaks intact, converges within the iteration cap);
  `::test_sigma_clip_one_iteration_matches_wimda` (iterations=1 reproduces the
  single-pass 2σ-clipped mean — locks the parity special case).

### S/N and average error
- **Tests:** `::test_peak_sn_matches_analytic` (synthetic line + known noise);
  `::test_average_error_finite_nonzero`; `::test_sn_readout_guards_empty`
  (empty/zero-error spectrum → no crash, sensible sentinel). Builds on
  `average_fourier_display_values(estimate_error=True)`.

### Real+Imag view
- **Test:** `::test_real_imag_mode_returns_both_quadratures` (phased line: real
  channel peaks, imag ≈ 0 at the matching phase).

### Burg pole scan (the characterisation IS the docs content)
- **Claim:** document the window length where FFT merges a close doublet but
  Burg resolves it, AND the pole count where spurious splitting begins.
- **Tests / characterisation:**
  - `::test_burg_resolves_doublet_fft_merges` — two lines Δf below the FFT
    resolution `1/(2·N·Δt)`; assert FFT shows one peak, Burg shows two at the
    FPE-optimal order. The window length used is recorded in the user docs.
  - `::test_burg_fpe_tracks_line_count` — M known lines → FPE-optimal order
    near the expected pole count; `::test_burg_boundary_warning` — optimum at a
    scan edge raises the WiMDA-style warning.
  - `::test_burg_spurious_splitting_onset` — sweep pole count upward on a single
    strong line; record the order at which a spurious second peak appears (the
    number quoted in the docs' pathology section). Anchored to *Muon
    Spectroscopy* §15.5's stated pathologies.
  - Numerical oracle: `::test_burg_matches_wimda_memcof` — the Burg recursion
    reproduces WiMDA `memcof`/`evlmem` on a transcribed fixture (reflection
    coefficients, FPE, spectrum) to lock the port.

### Diamagnetic fit-and-subtract
- **Tests:** `::test_diamag_fit_recovers_field` (synthetic damped cosine at
  known f → fitted field ≈ truth, reported back to the reference control);
  `::test_diamag_subtract_flattens_line` (residual spectrum has no line at the
  fitted frequency). Corpus smoke: fitted field ≈ applied field on a TF run.

## Regression guards (must not break)
- `tests/test_fourier.py`, `test_fourier_spectrum.py`, `test_fourier_units.py`,
  `test_fourier_reference_methods.py` — existing FFT behaviour.
- `tests/test_maxent*.py` — MaxEnt and the pulse oracle (shared `pulse.py`).
- `tests/test_project_schema.py` — FFT-recipe round-trip; add a case for the
  new keys (save→load→recompute identical).
- Full `validate` green at the end of each phase; `docs` green after user-guide
  pages land.

## Divergence ledger (verified behaviours, both stated)
1. **Pulse compensation:** WiMDA `× exp((πfτ)²)` unbounded vs Asymmetry
   `× 1/R(ν)` (parabola×Lorentzian) with guard. Verified by the flatten +
   guard tests.
2. **Baseline:** WiMDA single-pass 2σ vs Asymmetry iterative σ-clip; the
   1-iteration equivalence test pins the relationship.
3. **Exclusions applied to the averaged display channel** (MHz) vs WiMDA's
   per-group cos/sin/power arrays — equivalent for the displayed mode.
4. **S/N:** Asymmetry per-bin peak max(|x|/e) vs WiMDA global (π/2)mean|x|/mean(e).
5. **Field constant:** CODATA `0.0135538817` vs WiMDA `0.01355342` MHz/G.
