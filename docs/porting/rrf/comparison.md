# RRF across the reference programs

All four implementations agree on the goal — show the slow envelope of a fast
transverse-field precession — and on the rotation convention (demodulate by
e^{−i(2πν₀t+φ)}; equivalently multiply by cos for the in-phase part). They
differ in what signal they start from, how they remove the sum-frequency
image, and what happens to the error bars.

| | WiMDA | musrfit `PRunAsymmetryRRF` | Mantid `RRFMuon` | Asymmetry (this port) |
|---|---|---|---|---|
| Input | FB asymmetry (one real signal) | asymmetry (one real signal) | lab-frame quadrature pair (two histograms) | FB asymmetry (one real signal) |
| Demodulation | ×2·cos(ωt+φ) | ×2·cos(ωt+φ) | exact rotation of (Re, Im) | ×2·e^{−i(ωt+φ)}, complex |
| Image at ν+ν₀ | imperfectly nulled by box width | aliased by packing | none (exact rotation) | suppressed by designed low-pass |
| Filter | running box average, default width one image period | decimating mean ("RRF packing") | none | windowed-sinc FIR (Hamming), documented cutoff |
| Output | in-phase only | in-phase only | in-phase + quadrature | Real / Imag / Magnitude (complex in core) |
| Errors | ×|2cos|, then *linear* average over box | quadrature with √2 = ⟨(2cosθ)²⟩^{1/2} factor | untouched | exact per-point propagation through demod + FIR, correlation documented |
| Edges | output zeroed within half a box of each edge | partial pack dropped | n/a | filter edge region flagged (shortened valid range) |
| ν₀ units | MHz or Gauss (×0.01355342) | Mc (MHz→rad/μs), phase degrees | MHz / Gauss / Mrad·s⁻¹ (×2π·135.538817×10⁻⁴ for G) | MHz or Gauss via `FieldUnit.convert` (135.538817 MHz/T) |
| Fitting in RRF | model frequencies shifted −ν₀ in `MusrFun`; raw data fitted; Analyse form disabled while RRF display on | RRF run *type*: the packed demodulated curve **is** the fit data | n/a (display algorithm) | frequency-offset wrapper: raw data fitted, model components shifted +ν₀, reported parameters are rotating-frame offsets δν |

## The 2ω image: why 2·cos + box is not enough

Multiplying a real precession signal by 2·cos(ω₀t) is the real part of
complex demodulation: it produces the difference line at δν = ν − ν₀ *plus*
an equal-amplitude image at ν + ν₀. WiMDA's box average is a low-pass with
transfer function sinc(πfW) (box width W): it has nulls at f = k/W, so its
default W = 1/(ν₀ + ν_field) places the *first null exactly on the image*
when ν₀ is tuned to the applied field — a genuinely clever default. But the
null is a single point: detune ν₀ off the field (the normal use — you detune
to create a visible beat), broaden the line, or change W, and the image leaks
through the sinc sidelobes (first sidelobe −13 dB ≈ 22% amplitude) as a fast
ripple superimposed on the envelope. musrfit's decimating mean has the same
sinc response sampled coarsely, plus aliasing of the residual ripple onto the
packed grid.

Complex demodulation keeps both quadratures, so the image is a clean spectral
line at ν + ν₀ that a windowed-sinc FIR with cutoff ≪ ν + ν₀ removes by
design (Hamming sidelobes ≤ −53 dB) rather than by hoping a null lands on it.
The verification plan quantifies the residual ripple of both methods on the
same synthetic data; the figure belongs in the user guide.

Mantid avoids the problem entirely by demanding the measured quadrature pair
— the right answer when detector geometry provides it, and the reason
per-detector/quadrature RRF is recorded as a future candidate rather than
folded in here.

## WiMDA quirk ledger (reference-program findings)

1. **Errors through the box smooth are linearly averaged** (`Plot.pas`
   1671–1689): σ_out = mean(|2cosθᵢ|σᵢ), not a quadrature combination, and
   with no reduction for the box width N. For a flat-error region this
   overestimates the smoothed point's standard deviation by ~√N·⟨|cos|⟩/√⟨cos²⟩
   relative to the exact propagation — a display-only quantity in practice,
   since WiMDA never fits the smoothed curve, but the bars drawn on the RRF
   display are systematically wrong. musrfit gets this right.
2. **Bins near zeros of the carrier get near-zero values *and* near-zero
   errors** (value and error both scale with |2cos|): the box average then
   implicitly de-weights them. Correct on average, but the per-bin bars again
   misrepresent the local information content. Complex demodulation has no
   such zeros (|e^{−iωt}| = 1).
3. **Edge bins are zeroed but keep finite averaged errors** (`Plot.pas`
   1684–1687): the first/last half-box of the display shows artificial
   zero-value points with error bars.
4. **The rotating-frame fit shift is self-consistent only as a display
   overlay.** `MusrFun` evaluates components at p_ν − ν₀ while `dofit`
   assembles raw lab-frame data; a χ² fit run in that state would converge to
   p̂_ν = ν_lab + ν₀, biased by one full ν₀ (and consistent with *neither*
   frame's reading of the parameter). WiMDA guards this by disabling the
   Analyse form while RRF display is on — the shifted model exists so the
   *fitted curve overlay* drawn on the RRF display is demodulated in step
   with the data. The guard does not obviously cover batch/multifit entry
   points. Our design avoids the trap structurally: the offset wrapper owns
   the parameter semantics (fitted ν ≡ δν, lab ν = δν + ν₀) and the display
   demodulates the stored fit curve through the same pipeline as the data.
5. **Rounded constant**: WiMDA's Gauss→MHz factor 0.01355342 is the
   fifth-significant-figure-rounded γ_μ/2π (cf. 0.0135538817); the same
   rounding family was found in `rmatch` during the radical-correlation
   study. We use the CODATA constant from `core/utils/constants.py`
   throughout; at 8 T the difference is ~0.04 MHz — visible at RRF scale.
6. **Single-group RRF exists in WiMDA's display** (`Plot.pas` 1712–1727,
   cos demodulation of lifetime-unrolled single-group counts, no factor 2 and
   no smoothing) — undocumented and inconsistent with the FB path; recorded
   here, deliberately not ported (per-group RRF is out of scope).

## What we keep from WiMDA

- The control vocabulary: enable, ν₀ with MHz⇄Gauss toggle, phase, and a
  width-type control (our bandwidth in MHz replaces its smoothing bin in μs —
  the GUI label says what it is).
- Forcing the FB-asymmetry context: our controls only appear on the
  `fb_asymmetry` time view (the post-#53 representation seam), which is the
  same statement made declaratively.
- The image-period intuition behind its default box width survives as the
  default cutoff choice (ν₀/2, safely between envelope and image).
- The frequency-offset idea for fitting — implemented with the parameter
  semantics made explicit instead of implicit.
