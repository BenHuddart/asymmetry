# CdS shallow-donor scenarios

The CdS example uses 100 G transverse-field ISIS EMU runs. The cold run 20721
has a logged sample temperature of 5.175 K and a cryostat setpoint of 1 K.
Run 20729, at about 50 K, provides the warm control. The temperature-trend
scenario uses the stable-setpoint block, where the logged data and setpoints
agree.

| Scenario | What it shows |
|---|---|
| `corpus_cds_low_t_lineshape` | Cold grouped FFT around the diamagnetic line with fixed guides from the limiting hyperfine tensor components. |
| `corpus_cds_cold_vs_warm` | Cold and warm grouped FFTs on a common frequency axis, each scaled to its central-band maximum. |
| `corpus_cds_maxent_triplet` | Matched cold and warm grouped FFTs and MaxEnt spectra, together with weighted MaxEnt residuals for both detector groups. |
| `corpus_cds_time_beats` | Cold-run time-domain asymmetry over the first 12 µs. |
| `corpus_cds_neutral_fraction_t` | Temperature dependence of the satellite band-peak ratio, fitted with `OrderParameter`. |

## Literature and markers

The central Larmor line comes from the diamagnetic Mu⁺ state. The satellites
come from neutral shallow muonium. Gil *et al.* report an inner-pair spacing
of 0.214(5) MHz and an outer-pair spacing of 0.335(7) MHz in their Fourier
spectrum at 2.1 K and 10 mT along the crystal's hexagonal axis. Their Eq. (1)
relates the pair spacing to the orientation of the axial interaction, whose
principal values are A∥ = 335(7) kHz and A⊥ = 199(6) kHz.
[Gil *et al.*, Phys. Rev. Lett. **83**, 5294 (1999)](https://doi.org/10.1103/PhysRevLett.83.5294).

The comparison figure uses the published inner-pair spacing and 1.38 MHz
Larmor frequency to place the dotted guides. The circles mark prominent local
maxima on each displayed spectrum's grid. The figure supplies qualitative
evidence for the side peaks. Fitted centres and a splitting uncertainty
require a line model and a fit.

## Matched processing

The comparison transforms both forward/backward groups using the grouped FFT
with `(Power)^1/2`, average subtraction, a Lorentzian filter starting at 0 µs
with τ = 6 µs, a 0–31 µs time window, and padding of 4. Padding refines the
sampled grid. The cold side peaks persist in the unweighted FFT with shorter
time windows, with weaker side structure in the warm FFT.

The MaxEnt comparison uses both groups, the good-bin time range, ×4 time
binning, a frequency window of 0.9–1.9 MHz, 128 points, and 20 cycles. The cold
spectrum's broad three-peak pattern persists when the window widens to
0.85–1.95 MHz or the grid is reduced to 96 points. The warm reconstruction has
one dominant peak. The displayed χ²/N is computed from the same weighted
residuals as **Show time-domain reconstruction**. The larger cold-run
excursions are concentrated early in the record, and changing the window or
grid leaves structured residuals.

The neutral-fraction trend uses Hann-weighted FFT band maxima as a proxy for
relative satellite strength. An ionisation energy would require calibrated
line amplitudes and an Arrhenius fit.
