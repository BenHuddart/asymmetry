# RRF test data

## Synthetic (primary; fixed seeds, no external dependency)

1. **Pure single line, on resonance.** A(t) = a₀e^{−λt}cos(2πνt + φ_d) on a
   realistic grid (e.g. ν = 30 MHz, λ = 0.3 μs⁻¹, dt = 16 ns, 8 μs window).
   Demodulation at exactly ν₀ = ν with φ = φ_d must return the generating
   envelope a₀e^{−λt} in the real part (and ~0 in the imaginary part) to
   numerical precision inside the filter's valid range.
2. **Off resonance by δ.** Same signal demodulated at ν₀ = ν − δ
   (δ ~ 0.5–2 MHz) must show the δ beat: Re ∝ e^{−λt}cos(2πδt + …),
   magnitude equal to the pure envelope.
3. **Image-suppression comparison.** The same signal through `method="fir"`
   vs `method="wimda"` (box width = WiMDA's default image period and a
   detuned variant): quantify residual ripple at ν + ν₀ in both outputs. This
   dataset generates the comparison figure for the docs.
4. **Noise + errors.** Poisson-realistic noise via `core/simulate`
   (`simulate_count_run`, fixed seed) → FB asymmetry → demodulation: pull
   distribution of (demodulated − truth)/σ per quadrature must be ~N(0,1)
   point-wise; neighbouring-bin correlation must match the documented
   Σh²-based prediction.
5. **Fit-offset equivalence.** Synthetic oscillatory dataset fitted (a)
   directly with `Oscillatory ⊕ …` at lab frequency, (b) through
   `rrf_offset_model` with ν₀ and the frequency seeded at δν: identical
   χ², amplitudes/relaxations, and ν̂_lab = δν̂ + ν₀, to optimizer tolerance.
   Repeat with `OscillatoryField` (Gauss path).

## Corpus (skip-if-missing, `~/Documents/WiMDA muon school`, the committed
corpus-smoke pattern)

- **HAL-9500 8 T**: `Magnetism/AFM transition in high TF/data/`
  `tdc_hifi_2020_00739.mdu` (κ-ET crystal, 8 T, 100 K; dt = 24.4 ps,
  ~3.9×10⁵ bins, ν_μ ≈ 1084 MHz — ~10⁴ visible cycles, the canonical RRF
  case). Demodulated envelope at ν₀ = γ_μB/2π must decay consistently with
  the directly-fitted relaxation rate (verification-plan item 4). Companion
  temperatures 00730–00738 (4–75 K) and 6 T series 00692–00693 available for
  by-eye beat checks across the AFM transition.
- **HiFi 20 G** (`Basics/data_hdf5/hifi00062798.nxs`, Ag): low-field sanity —
  RRF at ν₀ = 0.27 MHz on slow precession exercises the long-filter /
  short-window edge handling.

## Cross-program oracle

Mantid `RRFMuon` is GPL — read-only oracle. Its exact-rotation output on a
synthetic quadrature pair (built analytically, no Mantid run needed) is the
zero-filter limit our complex demodulation must approach as bandwidth → ∞ on
noiseless data; the rotation-sign and Gauss-conversion conventions were
verified against its source directly ($MANTID_SRC). No numerical Mantid
artefacts are vendored.
