# Rotating-reference-frame display and fitting (rrf)

Date: 2026-06-12. Branch: `feat/rrf`. Umbrella: `wimda-parity-gap`, Wave B,
project 10. Promotes the `rrf-transform` candidate
(`docs/porting/candidates/rrf-transform/`), whose scoring history is retained
there; this study supersedes its technical content.

## Result

Asymmetry gains a rotating-reference-frame (RRF) representation of the
FB-asymmetry time view: the signal is demodulated by the complex carrier
e^{−i(2πν₀t + φ)} and low-passed, so a 2 T transverse-field run that
oscillates ~270 cycles across the plot window collapses to its slow
relaxation envelope (and any beat structure around ν₀). The transform is
**display-only**; quantitative results come from fitting the *raw* data, for
which a core-layer frequency-offset wrapper shifts oscillating model
components by ν₀ so envelope-scale parameters can be fitted in the rotating
frame with exact statistics. This follows the textbook recommendation that
the RRF transform be used as a visualization tool rather than a fit
preprocessor, because the low-pass filter both distorts lineshapes and
correlates neighbouring bins (Blundell, De Renzi, Lancaster & Pratt, *Muon
Spectroscopy*, OUP 2022, the rotating-reference-frame section of the
time-domain-analysis chapter).

The headline deviation from WiMDA: WiMDA demodulates with 2·cos(ωt+φ) and
box-smooths, which leaves an imperfectly-suppressed image at ν+ν₀ riding on
the envelope; complex demodulation followed by a designed low-pass suppresses
the image by construction. WiMDA's method ships in core as an explicitly
named comparison mode and is characterised in [comparison.md](comparison.md).

## Physics

In a transverse field B₀ the muon polarization precesses at ν = γ_μB/2π with
γ_μ/2π = 135.538817 MHz/T. Writing the measured FB asymmetry as
A(t) = a(t)·cos(2πνt + φ_d) with slowly varying envelope a(t), multiplying by
2e^{−i(2πν₀t+φ)} gives

> 2A(t)e^{−i(2πν₀t+φ)} = a(t)e^{i(2π(ν−ν₀)t + φ_d − φ)} + a(t)e^{−i(2π(ν+ν₀)t + φ_d + φ)}

The first term is the rotating-frame signal — the envelope, slowly rotating
at the offset δν = ν − ν₀; the second is the image at ν + ν₀, removed by a
low-pass with cutoff between |δν| + linewidth and 2ν₀. The real part is the
in-phase component, the imaginary part the quadrature, and the magnitude is
the phase-free envelope (noise-biased near zero — a Rician distribution, so
magnitude display is qualitative where the signal is comparable to its
errors). ν₀ is chosen close to the line but in a region without significant
spectral weight, so that genuine physics is not filtered away with the image.

Demodulating one real signal differs from the four-detector quadrature
geometry in the textbook (and Mantid's `RRFMuon`), where two signals measured
90° apart form a complex polarization and the frame rotation is exact with no
image and no filter. With only the FB asymmetry available the image is
unavoidable and the filter is the price; this is WiMDA's situation and ours.

## Sources verified (2026-06-12)

WiMDA (Object Pascal, `$WIMDA_SRC/src`, ignoring `__history`/`__recovery`):

- `PlotPar.pas` `RRFonClick`/`RRFvalueClick`/`RRFphase_editClick`/
  `RRFsmoothChange` (427–475): ν entry in MHz or Gauss (Gauss × 0.01355342 →
  MHz), phase in degrees, smoothing bin width in μs defaulting to
  1/(ν₀ + 0.01355342·B) — one period of the image frequency when ν₀ sits at
  the applied field, which is the box width that best nulls the image.
  Enabling RRF forces the FB-asym plot mode and **disables the Analyse (fit)
  form**; disabling re-enables it and zeroes `rotfreq`/`rotphas`.
- `Plot.pas` `plotdata` (1652–1695): display demodulation
  `2·cos(RRFfreq·t + RRFphase)` with `RRFfreq = 2π·ν₀` and phase in radians
  (2532–2536), then a running box average of half-width
  `trunc(RRFbin/tres) div 2`; output bins within half a box of either edge
  are zeroed. Errors are scaled by |2cos| and then *linearly averaged* over
  the box. 1712–1727: a single-group variant demodulates
  background-corrected, lifetime-unrolled counts by cos (no factor 2).
- `Analyse.pas` `MusrFun` (1124–1140) and the threaded
  `AsymFitFunction.pas` (440–470): with RRF on, the three rotation component
  types evaluate cos(2π((p_ν − rotfreq)t + (p_φ − rotphas)/360)) — frequency
  parameters shifted down by ν₀ (the field-parameterised type after γ_μ
  conversion), all phases shifted by −φ₀.
- `Analyse.pas` `dofit` (805–867): the fitted y-vector is assembled from raw
  per-bin counts — **the demodulated display data are never fitted**. See the
  ledger note in [comparison.md](comparison.md) on what this implies.

Mantid `RRFMuon` (`$MANTID_SRC/Framework/Muon/src/RRFMuon.cpp`, GPL — oracle
only): takes a two-histogram workspace of lab-frame real/imaginary
polarization plus ν₀ (MHz, Gauss, or Mrad/s; Gauss via 2π·135.538817×10⁻⁴)
and phase, and applies the exact frame rotation — no filter, because the
quadrature pair is assumed measured. Cross-program oracle for the rotation
convention (e^{−i(ωt+φ)}, same sign as ours) and the Gauss conversion
(identical constant to `core/utils/constants.py`).

musrfit `PRunAsymmetryRRF` (`$MUSRFIT_SRC/src/classes/PRunAsymmetryRRF.cpp`):
same 2·cos demodulation as WiMDA, but filtered by decimating mean ("RRF
packing") with errors combined in quadrature carrying the √2 average of
(2cosθ)² — statistically more honest than WiMDA's linear error average.

Textbook: *Muon Spectroscopy* (Blundell, De Renzi, Lancaster & Pratt, OUP
2022), rotating-reference-frame section of the time-domain-analysis chapter:
the quadrature construction and frame rotation, the choice of ν₀, the
sum-frequency image and its filtering, and the caveat that RRF filtering
introduces lineshape distortion — "preferable to use the RRF transformation
as a visualization tool, rather than using it to preprocess the data before
fitting". Primary references it cites:

- T. M. Riseman and J. H. Brewer, Hyperfine Interact. 65, 1107 (1991).
- B. D. Rainford, in *Muon Science*, eds. S. L. Lee, S. H. Kilcoyne, and
  R. Cywinski (CRC Press, 1999), p. 463.

## Decisions (Ben, 2026-06-12)

Scope round:

| Question | Decision |
|---|---|
| GUI display component | Selector Real / Imaginary / Magnitude, default Real; core returns the complex curve regardless. Magnitude documented as Rician-biased near zero. |
| ν₀ seeding | Auto-seed from the run's field metadata (γ_μB/2π via `core/fourier/units.py`) when enabling with no prior value; always editable. |
| Multi-run overlay | One frame applies to every overlaid run (envelope comparison across same-field runs is the workflow); the annotation states the common frame. |
| WiMDA 2·cos mode | Core + tests + docs only; the GUI exposes complex demodulation alone. |
| Fitting route | Core-only frequency-offset wrapper (Ben, 2026-06-12, Wave B pre-implementation pass): fit raw data with component frequencies offset by ν₀; **zero edits to `fit_panel.py`** (owned by the fit-workflow-diagnostics session this wave). GUI exposure recorded as a follow-on. |

Implementation round (after study):

| Question | Decision |
|---|---|
| Low-pass design | Windowed-sinc FIR (Blackman, odd taps, zero-phase, `scipy.signal.firwin`), GUI bandwidth = single-sided cutoff in MHz, default ν₀/2 clamped below the image; decimating mean rejected as primary (sinc sidelobes, coarse grid) but effectively available via WiMDA mode comparisons. |
| Demodulated errors | Exact per-point propagation σ²ᵢ = Σₖ h²ₖ·(2σ)²·{cos², sin²} through the filter; inter-bin correlation length ≈ filter support, stated in the curve metadata and docs; the demodulated curve is never offered to the fit layer. |
| Fit-offset parameterisation | Registry of rotation components (`Oscillatory` → `frequency` MHz; `OscillatoryField` → `field` Gauss) + a `CompositeModel` wrapper adding ν₀ (or its Gauss equivalent) to the mapped unique parameters; fitted frequencies are rotating-frame offsets δν; unsupported oscillating components (muonium, Bessel, F-μ-F families) fail loudly. Follow-up flagged: an engine-level `frequency_offset` argument once `engine.py` is free this wave — the registry/resolver split keeps that migration cheap. |
| Annotation & export | In-axes badge "frame: ν₀ = … MHz" (+ φ when non-zero) drawn on the plot so figure exports are self-describing; data exports carry frame parameters in the curve label/header. |

## Scope

In: `core/transform/rrf.py` (complex demodulation + WiMDA comparison mode,
Qt-free); `core/fitting/rrf_offset.py` (frequency-offset model wrapper,
core-only); `gui/widgets/rrf_controls.py` (enable, ν₀ with MHz⇄Gauss toggle
via `FieldUnit.convert`, phase, bandwidth, component selector) shown only on
the FB-asymmetry time view; `plot_state["rrf"]` persistence (additive, no
schema bump); axis annotation; user-guide page.

Out (deliberate): per-detector / per-group RRF (needs the quadrature
machinery of Rainford's method — record as a future candidate if multi-group
phase tables land); RRF inside MaxEnt (MaxEnt already handles high-TF in the
frequency domain); fit-panel GUI exposure of the offset wrapper (follow-on,
see above); automatic ν₀ tracking of the fitted line.

## Documents

- [comparison.md](comparison.md) — four-way comparison and the WiMDA quirk
  ledger, including the 2ω-image characterisation.
- [implementation-options.md](implementation-options.md) — options weighed,
  seams, and the chosen design.
- [test-data.md](test-data.md) — synthetic recipes and the 8 T HAL-9500
  corpus target.
- [verification-plan.md](verification-plan.md) — exactness, image
  suppression, fit-equivalence, and corpus checks.
