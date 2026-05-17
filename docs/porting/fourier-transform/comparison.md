# Fourier Transform Comparison

## Scope

This comparison focuses on the parts of the Fourier workflow that decide user
visible behavior:

- what data is transformed
- which preprocessing steps are applied before FFT
- how phase is supplied, estimated, or stored
- whether phase is global, per-group, or per-detector
- which output components are available after the transform
- how similar quantities are named across programs

## Summary

All three reference programs expose FFT as more than a magnitude-only view of a
single asymmetry trace.

- WiMDA exposes manual and automatic phase correction around grouped FFT views,
  and its grouped-count FFT source also carries an implicit muon-lifetime
  correction before filtering.
- musrfit keeps explicit phase-aware Fourier options and works with real and
  imaginary spectra as first-class outputs. It adds an entropy-based automatic
  phase optimizer and a richer preprocessing pipeline.
- Mantid couples FFT with phase tables and can calculate phase information per
  group or per detector before later frequency-domain workflows.
- Asymmetry now keeps an explicit complex-spectrum seam and presents WiMDA-style
  FFT phase modes on top of its grouped, averaged Fourier workflow.

## Program Comparison

| Program | Entry points | Data scope | Phase scope | Other preprocessing | Output surface | Test coverage |
| --- | --- | --- | --- | --- | --- | --- |
| WiMDA | `FFTPar.*`, `WiMDA_Main.pas`, `Fourier.pas`, `PhaseTableUnit.pas` | grouped asymmetry views | manual global or per-group; automatic estimation modes | grouped-count lifetime correction, start/end time, apodization, zero-padding, t0 offset | `(Power)^1/2`, `Phase Spectrum`, `Cos`, `Sin`, `Phase` | no automated tests found in repo scan |
| musrfit | `musrFT.cpp`, `PPrepFourier.*`, `PFourier.*` | selected histograms and averaged outputs | fixed/manual constant phase and optional linear phase `c₀ + c₁·(i/N)`; entropy-based automatic optimizer | background subtraction, packing, apodization (qualitative levels), time range, lifetime correction + N₀ normalization, zero-padding | real, imag, real+imag, power, phase, phaseOptReal | no focused Fourier unit tests found in repo scan |
| Mantid | frequency-domain GUI docs, FFT presenter/model tests, phase-table docs | groups or all detectors depending on selected workflow | manual shift, phase table, per-group or per-detector calculation | apodization, padding, optional imaginary workspace, phase-table generation | FFT workspaces and later phase-quad / MaxEnt flows | presenter tests plus system references |
| Asymmetry | `src/asymmetry/core/fourier/fft.py`, `src/asymmetry/gui/panels/fourier_panel.py` | grouped detector sums only | manual global or per-group with automatic estimation in `Phase` mode | WiMDA-style grouped-count lifetime correction, subtract-average, time crop, Lorentzian/Gaussian/None filter with start and tau, zero-padding | averaged grouped spectrum on a per-run cached plot with WiMDA-style phase modes | `tests/test_fourier.py`, `tests/test_mainwindow_additional.py` |

## WiMDA Phase-Mode Formulas

WiMDA computes one complex FFT per selected grouped signal and then derives its
display channels from the cosine and sine parts of that spectrum.

Let `F(f) = C(f) + i S(f)`.

| WiMDA mode | Formula in WiMDA source | Notes |
| --- | --- | --- |
| `(Power)^1/2` | `sqrt(C(f)^2 + S(f)^2)` | WiMDA's `Powermode` label is split across a `(Power)` radio button and a nearby `1/2` label. This is FFT magnitude, not squared power. |
| `Phase Spectrum` | `atan2(S(f), C(f))` | Uses the raw spectral phase angle before phase correction. |
| `Cos` | `C(f)` | WiMDA forces the phase entry to `0` when this mode is selected. |
| `Sin` | `S(f)` | WiMDA forces the phase entry to `90` when this mode is selected. |
| `Phase` | `C(f) * cos(theta(f)) - S(f) * sin(theta(f))` | This is the phase-corrected projection used by WiMDA's manual, auto, and table-driven phase workflows. |

For `Phase`, WiMDA builds

- `theta(f) = 2π * (phase_deg / 360 + f * t0)`
- `phase_deg` from the manual entry, automatic estimate, or per-group phase table
- `t0` from the optional grouped `UseT0` path

This distinction matters because only `Phase` consumes the phase table,
automatic phase estimation, and `t0` correction. The other four modes are all
derived from the uncorrected complex FFT.

## Cross-Program Output Mode Name Map

The same physical quantity is referred to by different names in WiMDA and
musrfit. The table below maps each program's user-visible label to its formula
and notes where naming could cause confusion.

| WiMDA label | musrfit label | Formula | Notes on differences |
| --- | --- | --- | --- |
| `(Power)^1/2` | *(no direct equivalent)* | `√(C² + S²)` = `\|F\|` | **Critical difference:** musrfit `power` is `\|F\|²`, not `\|F\|`. WiMDA explicitly labels this the *square root* of power. Asymmetry follows WiMDA's convention. |
| `Phase Spectrum` | `phase` (`FOURIER_PLOT_PHASE`) | `atan2(Im, Re)` | **Naming collision:** musrfit's mode called "phase" is WiMDA's "Phase Spectrum", *not* WiMDA's "Phase" mode. This is a major source of confusion. |
| `Phase` | `real` (`FOURIER_PLOT_REAL`) | `C·cos(θ) − S·sin(θ)` | WiMDA's "Phase" mode (phase-corrected real projection) corresponds to musrfit's "real" output. After applying a phase correction, musrfit's real part is WiMDA's Phase output. |
| `Cos` | `real` with `c₀=c₁=0` | `C(f)` | WiMDA "Cos" is musrfit "real" with no phase correction applied. Not a separate output mode in musrfit. |
| `Sin` | `imag` with `c₀=c₁=0` | `S(f)` | WiMDA "Sin" is musrfit "imag" with no phase correction applied. Not a separate output mode in musrfit. |
| *(no equivalent)* | `imag` (`FOURIER_PLOT_IMAG`) | `−C·sin(θ) + S·cos(θ)` | The phase-corrected imaginary quadrature. WiMDA has no dedicated output mode for this after phase correction; "Sin" is the uncorrected imaginary only. |
| *(no equivalent)* | `real+imag` (`FOURIER_PLOT_REAL_AND_IMAG`) | Real and imaginary simultaneously | musrfit can plot both quadratures at once. WiMDA cannot. |
| *(no equivalent)* | `phaseOptReal` (`FOURIER_PLOT_PHASE_OPT_REAL`) | Real part after entropy-optimized phase | No WiMDA equivalent. Uses `PFTPhaseCorrection` to find the best phase automatically. |
| *(no equivalent)* | `power` (`FOURIER_PLOT_POWER`) | `\|F\|²` | **Different from WiMDA `(Power)^1/2`**. musrfit "power" is the squared magnitude (true power spectral density). Asymmetry does not expose this mode. |

### Summary of dangerous naming collisions

- **musrfit `phase` ≠ WiMDA `Phase`**: In musrfit, `phase` means spectral angle
  (`atan2`). In WiMDA, `Phase` means phase-*corrected* real projection. These
  are completely different quantities.
- **musrfit `power` ≠ WiMDA `(Power)^1/2`**: musrfit reports `|F|²`; WiMDA
  reports `|F|`. A spectrum plotted in musrfit "power" mode will look like the
  square of an Asymmetry "(Power)^1/2" spectrum.
- **musrfit `real` ≈ WiMDA `Phase` (after correction)**: These are the same
  phase-corrected real projection, but musrfit calls it "real" and WiMDA calls
  it "Phase".

## musrfit Apodization: Qualitative Levels vs Explicit Parameters

musrfit exposes apodization through four qualitative strength tags rather than
an explicit time constant:

| musrfit tag | Constant | Meaning |
| --- | --- | --- |
| `FOURIER_APOD_NONE` | 1 | No apodization |
| `FOURIER_APOD_WEAK` | 2 | Light smoothing |
| `FOURIER_APOD_MEDIUM` | 3 | Moderate smoothing |
| `FOURIER_APOD_STRONG` | 4 | Heavy smoothing |

WiMDA and Asymmetry instead expose an explicit time constant (`FFTtau` / filter
tau), a configurable start time (`FFTst` / filter start), and a named window
shape (Lorentzian or Gaussian). The WiMDA/Asymmetry approach gives the user
direct control over resolution and leakage suppression. musrfit's qualitative
levels are less reproducible because the actual window width is implicit.

**Recommendation**: Keep WiMDA-style explicit apodization as the default and
primary interface. musrfit's qualitative levels are not a useful model to port.

## musrfit Phase Correction Algorithm

musrfit's `PFTPhaseCorrection` class implements a two-parameter optimization
over a functional that combines entropy and a penalty for negative spectral
values.

### Phase correction form

```
φ(i) = c₀ + c₁ · (i / N)
```

where `i` is the frequency bin index and `N` is the total number of bins. `c₀`
is a constant phase offset (corrects for an uncertain time zero or a global
phase shift); `c₁` is a linear phase slope that corrects for
frequency-dependent timing dispersion between detectors.

The complex rotation at each bin is:

```
F'_re(ω) = F_re(ω) · cos(φ) − F_im(ω) · sin(φ)
F'_im(ω) = F_re(ω) · sin(φ) + F_im(ω) · cos(φ)
```

### Optimization functional

```
f(c₀, c₁) = Entropy term + γ × Penalty term
```

**Entropy term**: Shannon entropy of the derivative of the phase-corrected real
spectrum. A concentrated, smooth absorption-mode spectrum has *low* entropy, so
the optimizer minimizes entropy to make the real part compact and resolved.

```
p_i = |ΔF_re(i)| / Σ|ΔF_re|   where ΔF_re(i) = F_re(i+1) − F_re(i)
Entropy = −Σ p_i · ln(p_i)
```

**Penalty term**: Penalizes negative values in the real spectrum, which are
unphysical for absorption-mode muSR spectra.

```
Penalty = Σ [F_re(ω)² for all F_re(ω) < 0]
```

**Balance parameter γ** (default 1.0, typical range 0.1–10): controls the
relative weight of the smoothness and positivity objectives.

**Optimization engine**: Minuit2 `MnMinimize` with two free parameters (`c₀`,
`c₁`) and initial step sizes of 2.0 degrees each.

### Comparison with WiMDA automatic phase estimation

WiMDA offers two automatic estimation modes, both simpler than musrfit's
entropy approach:

- **Peak method**: finds the phase that maximizes the dominant-frequency real
  component.
- **Average method**: finds the phase that maximizes the power-weighted average
  of the real spectrum over a specified frequency range.

Both are frequency-independent (single phase constant `c₀` only; no `c₁`
slope). Asymmetry has ported both methods. musrfit's entropy optimizer is the
only method that handles frequency-dependent phase dispersion across the
spectrum without manual per-detector phase tables.

## Phase Family Investigation: Single-Constant vs Linear

The question was whether musrfit uses the linear phase family (`c₀ + c₁·i/N`)
exclusively, which would allow Asymmetry to replace its single-constant model
with the generalized form, or whether it retains constant-only paths that
require keeping the single-constant fallback.

### Finding 1: musrfit's linear term is phaseOptReal-only

Inspection of `PFourier.cpp` and `musrFT.cpp` shows that `c₁` is active in
exactly one path: the `phaseOptReal` entropy optimizer. All other musrfit
phase paths use a single constant:

- **CLI `--phase <value>`** (`musrFT.cpp`): parses a single `Double_t` scalar.
  `c₁` is never set; it stays at its default `0.0`.
- **Interactive ± increment** (`PFourier.cpp`, `IncrementFourierPhase`): adjusts
  only a scalar phase offset. No slope term.
- **Manual via msr file** (`PFourier.cpp`, `fPhaseParam`): reads `c₀` and
  optionally `c₁`; in practice, only `c₀` is set by the user. `c₁` defaults
  to `0.0` and is only ever non-zero when `phaseOptReal` writes it back.

The rotation formula is identical in all paths:
`F'_re(ω_i) = F_re·cos(φ_i) − F_im·sin(φ_i)`. The only difference is whether
`φ_i` includes a non-zero `c₁` contribution.

**Conclusion**: musrfit does not use linear exclusively. The linear form exists
solely as the backend for automatic optimization. Removing the single-constant
path would break all interactive and CLI usage.

### Finding 2: WiMDA has frequency-linear infrastructure but does not optimize the slope

`Plot.pas` builds phase as `phi := phi0 + f * Tz` for all five display modes
that honor the phase. `f` is the frequency of the current bin; `Tz` is
sourced from `UseT0`/`DeltaT0` in the GUI. This is a genuine linear-in-frequency
slope term. However:

- `Tz` is always a user-supplied preset, never automatically searched.
- Both auto-estimation methods (`PhaseEstimate`, `PhaseEstimateAve`) loop only
  over `phi0 ∈ [−180°, 180°]` with `Tz` fixed. Neither estimates or optimizes
  `Tz`.
- Per-group phase storage (`PhaseVal: array[1..maxhist] of single`) is a
  single scalar per group — no slope per group.

WiMDA's `f * Tz` term is therefore equivalent to Asymmetry's `t0_offset_us`
parameter: it shifts each frequency bin's phase by an amount proportional to
the bin's frequency, but the slope coefficient is always supplied manually.

### Finding 3: Asymmetry's t0_offset_us already covers the linear-in-frequency slope

`fft_complex_asymmetry` in `fft.py` builds the phase array as:

```python
phase = np.deg2rad(float(phase_degrees)) + 2.0 * np.pi * freqs * float(t0_offset_us)
```

The second term is `2πf·t₀`: a linear function of `freqs`. This is
conceptually the same correction as musrfit's `c₁·(i/N)` — both introduce a
phase that grows linearly across the spectrum — but their parameterizations
differ:

| Parameter | Domain | Physical meaning |
|---|---|---|
| `t0_offset_us` (Asymmetry / WiMDA) | time (µs) | timing offset between detector and reference clock |
| `c₁` (musrfit `phaseOptReal`) | dimensionless (0–1 of bin index) | abstract slope across the digitized spectrum |

The two are not directly interchangeable: `t0_offset_us` maps cleanly onto
a physical time-zero correction, while musrfit's `c₁` is an abstract
optimization variable that Minuit2 can adjust freely. For users who know their
detector timing offset, `t0_offset_us` is the better interface. For users who
want an automatic optimizer to find the best slope without a physical
interpretation, `c₁` via `phaseOptReal` is appropriate.

### Finding 4: Existing auto-estimation methods are inherently single-constant

`estimate_fft_phase` in `fft.py` returns a single `float` (degrees). The peak
method returns `np.angle(selected[np.argmax(np.abs(selected))])` — one number.
The average method returns the angle of a power-weighted phasor sum — also one
number. Neither method has a path to estimate a slope.

Extending these methods to estimate `c₁` would require a 2D grid search or a
2D optimizer, which is qualitatively different in cost and complexity. The
entropy optimizer already solves this problem. There is no value in duplicating
it within the existing peak/average framework.

### Conclusion: Do Not Replace the Single-Constant Model

The single-constant + `t0_offset_us` phase model should be retained as the
default for all manual and interactive use. Replacing it with a two-parameter
model would:

1. Break all existing manual phase workflows (CLI, interactive slider, msr
   file, per-group table) that supply only a scalar.
2. Expose `c₁` as a user-adjustable knob without an optimizer to set it,
   which musrfit itself never does in manual paths.
3. Duplicate the `t0_offset_us` parameter in a less physically meaningful
   form.

The linear model (`c₀ + c₁·i/N`) should be implemented **only** as the
internal backend for the `phaseOptReal` entropy optimizer. In that context,
`c₁` is an optimization variable — not a user-facing control — and the
optimizer's output can be described to the user as "optimized phase with slope
correction" without exposing `c₁` directly.

The recommended implementation boundary is therefore:

- **Manual / interactive / per-group phase**: keep single-constant `c₀` (plus
  `t0_offset_us` for timing correction). No `c₁` control.
- **phaseOptReal mode**: use linear `c₀ + c₁·(i−minBin)/(maxBin−minBin)` as
  the optimizer's internal parameterization, minimizing the entropy + penalty
  functional. Do not expose `c₁` as a panel control.

## musrfit Preprocessing: N₀ Normalization

WiMDA corrects for muon decay in the grouped-count FFT source by multiplying by
`exp(t / τ_μ)`. This expands the signal to a roughly constant amplitude but
keeps it in counts units. musrfit's `PPrepFourier::DoLifeTimeCorrection` goes
further:

1. Multiply each bin by `exp(t / τ_μ)` where `t = j × dt` and `dt` is the
   time resolution.
2. Estimate a baseline level `N₀` as the arithmetic mean of all corrected bins
   (scaled by an optional fudge factor, typically 1.0).
3. Subtract `N₀` and divide by `N₀`, producing an asymmetry-like trace:
   `(corrected_count − N₀) / N₀`.

This normalization anchors the data to zero mean with amplitude proportional to
the underlying muon polarization. It is the single-histogram equivalent of the
two-detector asymmetry `(F − αB) / (F + αB)`.

**Implication**: musrfit's FFT input is an asymmetry-like trace derived from a
single histogram; WiMDA's is decay-corrected grouped counts. For high-field
data where depolarization is small, both converge. For low-field or zero-field
data, the baseline estimation in the musrfit path becomes unreliable because the
mean of the corrected counts is no longer a stable estimate of `N₀`.

**Recommendation**: musrfit's N₀ normalization is the right approach when
working with single-histogram FFT inputs and high-field data. It should be
added as an advanced option rather than replacing the current WiMDA-style
grouped-count path, which remains the right default for grouped detector sums.

## musrfit Preprocessing: Background Subtraction

musrfit `PPrepFourier` offers two background handling modes, both more flexible
than WiMDA's subtract-average approach:

- **Range-based**: average counts in a specified bin range before the signal
  window and subtract that average from all bins.
- **Explicit value**: subtract a user-supplied constant.

WiMDA subtracts the error-weighted mean of the entire filtered window. Asymmetry
follows WiMDA here. musrfit's range-based approach is more physically motivated
because it estimates the true detector background from a pre-signal region
rather than a mean of the signal-containing window.

**Recommendation**: Add musrfit-style range-based background subtraction as an
advanced option. It is more appropriate for datasets where the pre-signal
background is meaningfully different from the signal-window mean.

## Detailed Notes

### WiMDA

- `FFTPar` exposes an explicit phase mode, a manual phase entry, automatic
  phase estimation, and a phase-table toggle.
- WiMDA also exposes `SubtractAve`, which removes an error-weighted average
  from the time-domain signal before filtering and FFT.
- In the grouped `Freq` FFT path, WiMDA first multiplies detector-group counts
  and their Poisson errors by `exp(t / tau_mu)` before optional zero
  extrapolation, mean subtraction, and apodisation. This is separate from the
  FB-asymmetry Fourier path.
- WiMDA's Lorentzian and Gaussian filters are not symmetric whole-trace
  windows. They use `FFTst` and `FFTtau` to define either a simple decay from
  time zero or a softened step around the configured filter start.
- WiMDA stores phase values per plotted group when the average/group FFT modes
  are active.
- The effective FFT window and maximum frequency are bunch-aware because the
  regrouped time axis uses `tres * cgrp.bunch`; grouped Fourier inputs are
  summed into wider time bins before the transform.
- In `Plot.pas`, WiMDA uses the same cosine and sine arrays to expose five
  display modes: raw magnitude, raw phase angle, raw cosine, raw sine, and a
  phase-corrected real projection.
- WiMDA only enables manual phase entry, automatic phase estimation, and the
  phase-table toggle when `Phase` is selected. `Cos` and `Sin` instead force
  the effective phase to `0` and `90` degrees respectively.
- The phase-corrected mode uses `phi0 = phase/360` and, when enabled, an added
  `f * tz` time-zero term before applying `cos(2πphi)` / `-sin(2πphi)`.
- This makes grouped FFT phase correction part of normal transform workflow,
  not a downstream plotting tweak.

### Asymmetry

- Asymmetry now mirrors WiMDA's `None` / `Lorentzian` / `Gaussian`
  apodisation modes in the Fourier dock and carries the same filter start and
  tau parameters through automatic phase estimation and final FFT generation.
- Asymmetry's grouped Fourier source now mirrors WiMDA's hidden grouped-count
  lifetime correction by applying `exp(t / tau_mu)` before the later
  subtract-average and filtering stages.
- Asymmetry now also exposes a dedicated `FFT Phase Mode` frame with the same
  five WiMDA-style display modes: `(Power)^1/2`, `Phase Spectrum`, `Cos`,
  `Sin`, and `Phase`.
- Asymmetry follows the same phase-mode contract as WiMDA: only `Phase` uses
  the manual phase value, automatic phase estimate, per-group phase table, and
  `t0` offset. The other modes derive from the uncorrected complex spectrum.
- Asymmetry deliberately fixes the Fourier source to grouped detector signals
  and deliberately plots only the average across the included groups. Those
  are product decisions, not missing parity work.
- The panel's `Info` button now documents those mode formulas in user-facing
  language and renders the equations directly in the application.
- Legacy whole-trace window helpers remain in the core API for existing
  script-side consumers, but the main FFT workflow now targets WiMDA parity.

### musrfit

- `musrFT` accepts Fourier options for real, imaginary, power, phase, real+imag,
  and phaseOptReal views (`FOURIER_PLOT_*` constants in `PMusr.h`).
- `PPrepFourier` applies preprocessing before the transform, including
  background subtraction and lifetime correction.
- `PPrepFourier::DoLifeTimeCorrection` multiplies by `exp(t/tau_mu)`, estimates
  `N0` as the mean of the corrected window, and renormalizes into an
  asymmetry-like trace before FFT.
- musrfit therefore goes further than WiMDA: WiMDA keeps decay-corrected counts
  as the FFT source, while musrfit additionally rescales by the estimated
  baseline level.
- `PFTPhaseCorrection` supports a linearly varying phase correction
  `phi = c0 + c1 * (i / N)`, which is broader than a single manual phase.
- The entropy + penalty functional in `PFTPhaseCorrection` uses Minuit2 to
  find the best `(c0, c1)` pair automatically; this is qualitatively different
  from WiMDA's simple peak-frequency or power-weighted average estimation.
- musrfit's apodization is specified by qualitative strength levels
  (NONE/WEAK/MEDIUM/STRONG) rather than explicit time constants. The actual
  window shape and width are implicit at each strength level, making results
  harder to reproduce or compare across programs.
- musrfit supports four x-axis units: Gauss, Tesla, MHz, and Mc/s. Asymmetry
  and WiMDA support Gauss and MHz only.

### Mantid

- Frequency-domain analysis accepts a phase table as a separate input surface.
- Phase calculation can be done for groups or all detectors, which preserves
  detector-specific phase information for later use.
- `PhaseQuadMuon` requires one phase-table row per detector and uses detector
  asymmetries and phases to solve a 2x2 coefficient system that reconstructs
  real and imaginary quadratures from detector residuals.
- Mantid's FFT surface therefore depends on a broader detector/group phase
  contract than Asymmetry currently has.

## Main Differences To Carry Forward

### Phase Estimation and Correction

- **WiMDA**: Manual, automatic (peak or power-weighted), and per-group phase
  correction. All phase corrections use a single constant `c₀`; no
  frequency-dependent term. Only "Phase" mode applies any correction.
- **musrfit**: Manual constant phase, linear phase `c₀ + c₁·(i/N)`, and
  entropy-based optimization with Minuit2. The linear term corrects for
  frequency-dependent phase dispersion that a single constant cannot address.
- **Asymmetry**: WiMDA-style phase handling only (manual `c₀`, `t₀` offset,
  per-group table, peak/average auto-estimation). No `c₁` slope; no
  entropy-based optimizer.

**Recommendation**: Port musrfit's linear phase correction and entropy-based
`phaseOptReal` as an *option* alongside WiMDA-style correction. These address
cases (detector timing dispersion, unresolved t₀ across groups) where the
single-constant approach leaves residual phase slope visible in the real
spectrum. WiMDA-style remains default for parity.

### Preprocessing Steps

- **WiMDA**: Grouped-count lifetime correction (`exp(t/τ_μ)`), mean subtraction,
  apodization, zero-padding. Background subtraction is simple (mean of the
  filtered window). No N₀ normalization.
- **musrfit**: Lifetime correction *plus* N₀ normalization (subtract and divide
  by estimated baseline). Flexible range-based or explicit background
  subtraction. Qualitative apodization levels (not explicit tau/start).
  Per-histogram packing.
- **Asymmetry**: Mirrors WiMDA grouped-count lifetime correction and
  WiMDA-style apodization. Background subtraction is mean of the signal window.
  Only grouped detector sums. No N₀ normalization, no range-based background.

**Recommendation**: Port musrfit's N₀ normalization and range-based background
subtraction as advanced options. Keep WiMDA-style as default. musrfit's
qualitative apodization levels are not worth porting; explicit tau/start
parameters give users more control.

### Output Surface and Naming

- **WiMDA**: `(Power)^1/2`, Phase Spectrum, Cos, Sin, Phase (phase-corrected
  real).
- **musrfit**: real, imag, real+imag, power (= `|F|²`!), phase (= spectral
  angle, NOT corrected real!), phaseOptReal.
- **Asymmetry**: Follows WiMDA's output modes and naming for grouped averages.
  No `phaseOptReal`, no per-detector outputs, no `real+imag` simultaneous view.

**Recommendation**: Add `phaseOptReal` as a sixth display mode alongside the
existing five WiMDA modes. Clarify the naming collision in documentation:
musrfit `phase` = WiMDA `Phase Spectrum`; musrfit `real` ≈ WiMDA `Phase`
(after correction). Do not adopt musrfit's `power` (|F|²) naming; the WiMDA
`(Power)^1/2` label is more honest about what is being displayed.

### Areas to Investigate Next

1. **Linear phase correction (`c₁` slope)**: Implement as the internal
   parameterization of the `phaseOptReal` entropy optimizer backend only.
   Do **not** expose `c₁` as a standalone user control. The existing
   `t0_offset_us` already provides a physically motivated linear-in-frequency
   slope for the manual phase path. See "Phase Family Investigation" section
   above for the full rationale.

2. **Entropy-based phase optimizer**: Port `PFTPhaseCorrection` algorithm as
   the backend for `phaseOptReal` mode. Requires a minimizer (SciPy `minimize`
   is the natural Python equivalent of Minuit2 for this problem).

3. **N₀ normalization**: Add as a toggle in `build_group_signal_dataset` or as
   a separate preprocessing step for single-histogram FFT inputs.

4. **Range-based background subtraction**: Add a pre-signal time range input to
   the Fourier panel for the grouped-count source.

5. **`real+imag` simultaneous view**: Useful for verifying phase correction
   quality interactively. Could be implemented as a split or dual-axis plot on
   the frequency tab.

6. **Tesla unit support**: Adding Tesla alongside MHz and Gauss is low-cost and
   covers common experimental reporting conventions.

### Summary Table

| Feature | WiMDA | musrfit | Asymmetry (current) | Recommendation |
|---|---|---|---|---|
| Phase family | Single constant `c₀` | Linear `c₀ + c₁·(i/N)` (optimizer only); constant `c₀` in all manual/CLI paths | Single constant `c₀` | Keep single-constant as default; use linear only as `phaseOptReal` optimizer backend |
| Phase optimizer | Peak / power-weighted average | Entropy + penalty (Minuit2) | Peak / power-weighted average | Add entropy optimizer for `phaseOptReal` |
| Lifetime correction | `exp(t/τ_μ)` on counts | `exp(t/τ_μ)` + N₀ normalization | `exp(t/τ_μ)` on counts | Add N₀ norm as advanced option |
| Background | Mean of signal window | Range-based or explicit value | Mean of signal window | Add range-based as advanced option |
| Apodization control | Explicit tau + start time | Qualitative levels (Weak/Medium/Strong) | Explicit tau + start time | Keep WiMDA style; don't adopt qualitative levels |
| Output: magnitude | `(Power)^1/2` = `\|F\|` | *(not separately exposed)* | `(Power)^1/2` = `\|F\|` | Keep; note musrfit `power` = `\|F\|²` is different |
| Output: power | *(not exposed)* | `power` = `\|F\|²` | *(not exposed)* | Don't add; would confuse with WiMDA convention |
| Output: spectral angle | `Phase Spectrum` = atan2 | `phase` = atan2 | `Phase Spectrum` = atan2 | Keep WiMDA name to avoid confusion with musrfit |
| Output: corrected real | `Phase` = `C·cos(θ)−S·sin(θ)` | `real` = phase-corrected Re | `Phase` = `C·cos(θ)−S·sin(θ)` | Keep WiMDA name; document musrfit equivalence |
| Output: corrected imag | *(not exposed)* | `imag` = phase-corrected Im | *(not exposed)* | Consider adding as optional paired view |
| Output: entropy-opt real | *(not available)* | `phaseOptReal` | *(not available)* | Add as sixth display mode |
| Output: simultaneous Re+Im | *(not available)* | `real+imag` | *(not available)* | Consider as inspection tool |
| X-axis units | MHz, Gauss | MHz, Gauss, Tesla, Mc/s | MHz, Gauss | Add Tesla as low-cost extension |
| Per-detector FFT | No | Yes | No | Defer; requires broader phase contract |

## Comparison Tests

The comparison harness lives in `tests/test_fourier_reference_methods.py`.

- `test_wimda_manual_phase_projection_matches_asymmetry_rotation`
- `test_wimda_grouped_fft_source_uses_decay_corrected_counts`
- `test_musrfit_lifetime_correction_flattens_decay_before_fft`
- `test_musrfit_linear_phase_profile_is_not_a_single_wimda_phase`
- `test_mantid_phase_table_recovers_quadratures_lost_by_group_sum`

These tests freeze the reference-method differences with synthetic data so later
Asymmetry implementation work can target a known contract.

Additional tests to add when implementing musrfit features:

- `test_entropy_phase_optimizer_recovers_known_phase_on_pure_cosine`: verify
  that the entropy + penalty functional converges on the true phase for a
  synthetic dataset with a known phase offset.
- `test_linear_phase_correction_removes_slope_from_multigroup_spectrum`: verify
  that `c₁ > 0` corrects a simulated detector timing gradient.
- `test_n0_normalization_produces_asymmetry_like_trace`: verify that after N₀
  normalization, the mean of the corrected signal is near zero.

---

**Next Steps:**

- Update this documentation as musrfit features are ported or exposed as options.
- Recommend porting musrfit's advanced phase and preprocessing as selectable
  options, not as a replacement for WiMDA parity.
- Document the rationale for each recommendation and the expected user impact.
