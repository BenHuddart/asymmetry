# RRF verification plan

Numbered checks; each lands as a pytest beside the behaviour it protects
unless marked docs-only.

1. **Envelope exactness (core).** Demodulation at exactly the generating
   frequency and phase returns the generating relaxation envelope in Re
   inside the valid range, exact up to the filter's stopband leakage of the
   image (Blackman −74 dB → residual ≲ 2×10⁻⁴ of the initial amplitude on
   noiseless data); Im ≈ 0 at the same level. Magnitude equals the envelope
   regardless of φ error.
2. **Beat correctness (core).** Detuned by δ, the demodulated signal
   oscillates at δ (cross-check by zero-crossing count or FFT peak) with the
   undistorted envelope in magnitude.
3. **Image suppression (core + docs).** On identical synthetic data, peak
   residual at ν + ν₀ for the FIR path is below the Blackman stopband
   prediction; the WiMDA mode shows the expected sinc leakage when the box
   width is detuned from the image period. The numbers go into the user-guide
   comparison; the test asserts the FIR bound and that WiMDA-mode output
   matches a direct re-implementation of the Pascal loop (including edge
   zeroing and linear error averaging) bin-for-bin.
4. **Error propagation (core).** With seeded Poisson noise: per-point pulls
   ~N(0,1) per quadrature; predicted neighbour correlation matches measured
   (sample over many seeds); magnitude bias appears only where |z| ≲ σ.
5. **Fit-offset exactness (core).** Direct lab-frame fit vs `rrf_offset_model`
   fit of the same raw synthetic data: identical χ² and parameters with
   ν̂_lab = δν̂ + ν₀ for both `Oscillatory` (MHz) and `OscillatoryField`
   (Gauss) routes; composites containing unregistered oscillating components
   raise naming the component.
6. **Corpus envelope-vs-fit (skip-if-missing).** HAL-9500 8 T run: the
   demodulated envelope's decay (log-magnitude slope over the valid range)
   agrees with the directly-fitted relaxation rate of the raw data within
   combined uncertainties.
7. **GUI behaviour (offscreen).** Controls visible only on the
   `fb_asymmetry` time view; ν₀ MHz⇄Gauss round-trips through
   `FieldUnit.convert`; auto-seed fills γ_μB/2π from run metadata; one frame
   applied to every overlaid run; fit-curve overlay demodulated in step with
   data; frame badge present on the axes and in export labels.
8. **Persistence (GUI).** `plot_state["rrf"]` round-trips through a save/load
   cycle; projects without the key (and with unknown sub-keys) restore
   cleanly with RRF off — no schema bump.
9. **No-regression.** Existing plot-panel, fit, and project tests green; the
   full `tools/harness.py validate` passes.
