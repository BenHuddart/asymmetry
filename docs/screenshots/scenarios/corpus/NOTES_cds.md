# NOTES — Shallow donor state in cadmium sulphide (Semiconductors)

Module: `cds_shallow_donor.py`. Example: `Semiconductors/Shallow donor state in
cadmium sulphide`. Spec: the example's `GROUND_TRUTH.md` (audit-corrected) +
`CdS 2026.docx` teaching guide. Data: real ISIS EMU HDF4 NeXus,
`EMU00020711`–`20733` (23 runs), TF **100 G**, logged sample T ≈ 5–285 K.

Reference papers: Gil *et al.*, PRL **83** (1999) 5294 (novel Mu state);
PRB **64** (2001) 075205 (ionisation fitting); anisotropy restated in Prokscha
*et al.*, PRB **90** (2014) 235303.

## Scenarios registered

| Scenario | Render | Intended docs use |
|---|---|---|
| `corpus_cds_low_t_lineshape` | Program FFT of the coldest run (20721, 5.2 K) zoomed on the 100 G Larmor line: central Mu⁺ line + two symmetric Mu⁰ satellites, with expected A∥ / A⊥ positions marked. | The shallow-donor lineshape; how the frequency-domain view exposes the sub-MHz hyperfine splitting. |
| `corpus_cds_cold_vs_warm` | **Headline.** 5.2 K spectrum (3 lines) overlaid on 50 K (single sharp line), each normalised to its central-line peak. | The signature result: warming ionises the donor → satellites vanish. |
| `corpus_cds_time_beats` | Time-domain asymmetry of the 5.2 K run showing the beat envelope (node ~3–4 µs, antinode ~8–9 µs). | The guide's "note the pronounced beats"; the time-domain counterpart of the frequency splitting. |
| `corpus_cds_neutral_fraction_t` | Mu⁰ neutral fraction vs T across the stable runs 20722–20733, OrderParameter fit → onset Tc ≈ 30 K. | The ionisation crossover / order-parameter trend (Model-Fit panel). |

Top pick for the docs: **`corpus_cds_cold_vs_warm`** — one figure that tells the
whole shallow-donor story (satellites present cold, gone warm).

## Run selection & workflow (GROUND_TRUTH refs)

- **Coldest run = 20721** (Tlog **5.175 K**, 45.9 MEv — the highest-statistics
  run, deepest in the Mu⁰ phase). Used for the lineshape and beat scenarios.
- **Warm reference = 20729** (Tlog **50.0 K**, ionised → single Mu⁺ line).
- **T-trend = the stable-setpoint block 20722–20733** only (Tset = Tlog,
  10–50 K). This already brackets the Mu⁰ onset, so it sidesteps the
  temperature pitfall entirely (see below). GROUND_TRUTH §3.
- ⚠ **Temperature pitfall (GROUND_TRUTH §3, §9):** runs 20711–20721 park the
  cryostat **setpoint** at 1.000 K while the sample actually cools 285 → 5.175 K;
  the **logged** temperature is the physical axis. The program's data browser
  shows the parked setpoint by default (visible in `corpus_cds_time_beats`:
  "CdS T=1.0 F=100"). 20721's *logged* 5.175 K is the coldest point in the whole
  scan despite the "1 K" setpoint — that is why it is the coldest lineshape run.
- **FFT workflow:** every spectrum runs through the program's own
  `compute_average_group_spectrum` (the average grouped FFT the Fourier panel
  drives), with a **Hann** apodisation and the config's **`padding`**
  zero-padding — the long-time coherent signal + gentle apodisation the brief
  prescribes. GROUND_TRUTH §4 (frequency-domain display, MaxEnt/FFT compare).

## Fitted / measured values vs ground-truth targets

| Quantity | Measured (this module) | Target (GROUND_TRUTH) | Verdict |
|---|---|---|---|
| Central diamagnetic (Mu⁺) Larmor line | **1.388 MHz** | γ_µ·100 G = 1.355 MHz; program note (§7) fit 1.389 MHz | matches program note; ~2.4 % above the ideal γ_µ·B (calibration offset flagged in §7) |
| Satellite splitting = A_µ (cold 20721) | **0.214 MHz** (satellites 1.287 / 1.501 MHz, offsets −0.101 / +0.113) | anisotropic A∥ = 335(8) kHz, A⊥ = 199(6) kHz; observed splitting 0.199–0.335; inner pair Δν_I = **0.214(5) MHz** (§6) | on target — lands essentially on the inner-pair Δν_I and inside the A⊥–A∥ range; consistent with the program note's A_µ = 0.242 MHz (§7) |
| A_µ / vacuum-Mu ratio | 0.214 / 4463 ≈ **4.8×10⁻⁵** | "order 10⁻⁴ of free muonium" (§5/§6) | matches (shallow-donor scale) |
| Mu⁰ base-T neutral fraction | **≈ 0.68** (satellite/(satellite+central) peak proxy, 10 K) | program note base-T ≈ 0.66 (§7, not authoritative) | matches the program note |
| Mu⁰ onset temperature | OrderParameter fit **Tc ≈ 30.6 K** | "satellites appear below ~30 K" (§4/§6) | on target |
| Ionisation energy E_i | **not extracted** | tens of meV (§6, literature — verify); program note E_i ≈ 13 meV (§7) | see limitations |

The guide deliberately prints no numeric A_µ or E_i (they are the experiment's
deliverables); the A_µ / onset numbers above are graded against the
literature-confirmed §6 values, and the neutral-fraction/E_i comparisons against
the non-authoritative program note in §7 (recorded for traceability only).

## Satellite splitting vs the 0.2–0.34 MHz expectation

At 100 G the muon Zeeman splitting (1.36 MHz) dwarfs the ~0.2–0.3 MHz hyperfine
(deep Paschen-Back), so the TF line is a **central Mu⁺ line ± satellites offset
by A/2**, i.e. the satellite–satellite splitting equals A directly (guide). The
CdS donor hyperfine is **anisotropic** (A∥ = 0.335, A⊥ = 0.199 MHz); a real
sample averages Δν(Θ) = A∥cos²Θ + A⊥sin²Θ over orientation, so the *observed*
splitting is a single effective pair inside the 0.199–0.335 range rather than
the two textbook pairs of a perfectly aligned single crystal. The measured
**0.214 MHz** sits right at the inner-pair value Δν_I = 0.214(5) MHz (GROUND_TRUTH
§6), i.e. the powder/dominant-orientation average — an honest, defensible match.
The `corpus_cds_low_t_lineshape` figure marks **both** the A⊥ (±0.0995) and A∥
(±0.1675) expected positions so the reader sees the observed satellites falling
between them.

## Resolution limits (honest)

This is spectroscopy **at** the FFT resolution limit, and the module is framed
around that:

- The EMU loader delivers **1979 bins at 16 ns over a 31.75 µs window** ⇒ a raw
  FFT bin of **~32 kHz**. The central–satellite offset (~0.11 MHz) is only ~4
  raw bins, and the two satellites straddle the central line by ~7 bins.
- A **Hann** window (suppresses the sinc side-lobes that otherwise masquerade as
  satellites) plus **`padding` = 4–8** zero-padding (interpolates the bin to
  ~4–8 kHz) is what makes the three lines separate cleanly. Without apodisation
  the raw FFT of even the best run is a lumpy comb of side-lobes (verified during
  development) — the satellites are real but only survive gentle apodisation.
- The satellites resolve **clearly only on the highest-statistics cold run
  (20721, 45.9 MEv)**. On the 37-MEv stable runs (10–25 K) they are present but
  noisier; the `corpus_cds_neutral_fraction_t` trend therefore uses a *band-peak
  ratio* (robust to the exact peak position) rather than a per-run line fit, and
  still traces a clean order-parameter curve through the onset.
- Consequently the module renders the **honest best views**: a resolved
  three-line cold spectrum, a normalised cold-vs-warm overlay where the
  satellite presence/absence is unambiguous, the time-domain beat envelope (the
  same information with no resolution penalty), and a robust amplitude-vs-T
  trend — rather than over-claiming a crystal-resolved two-pair spectrum.

## E_i not extracted — why, and the opportunity

The ionisation energy needs an **Arrhenius fit of the neutral fraction vs 1/T**
(GROUND_TRUTH §4, PRB 64 (2001) 075205 method). The `neutral_fraction_t`
scenario fits the **OrderParameter** power law instead — it cleanly recovers the
**onset Tc ≈ 30 K** but is not an Arrhenius slope, so it does not yield E_i. The
band-peak-ratio proxy is also not an absolute neutral fraction (it is not α- or
amplitude-calibrated), so quoting a benchmark E_i from it would be
over-reaching. An Arrhenius fit on a properly amplitude-calibrated multi-line
fit would give the tens-of-meV E_i (program note: ~13 meV) — flagged as a future
extension, not shipped.

## Feature-demonstration opportunities spotted

- **Multi-line tied time-domain fit** (central + 2 satellites, ties for equal
  spacing / common phase): the guide's headline analysis (§4). The program note
  (`ANALYSIS_asymmetry.md`) records that Asymmetry currently offers **no
  equal-spacing Ties** and the **parameter table does not scroll** past ~5 of the
  13 rows, so a hand-seeded tied 3-oscillation fit is impractical in the GUI —
  hence this module does the spectroscopy in the frequency domain and via the
  robust amplitude trend rather than a GUI tied fit. Worth a dedicated
  UX/feature note.
- **MaxEnt vs FFT comparison** (guide §4): a MaxEnt reconstruction of the cold
  spectrum would be a natural 5th scenario (the κ-Cl module has the MaxEnt
  pattern). Not attempted here — MaxEnt at this sub-MHz resolution over a 32 µs
  window is delicate and the FFT already tells the story; left as an option.
- The **parked-setpoint artefact** is itself a teachable data-hygiene moment
  (the "T=1.0" browser label on a 5 K run) — visible in `corpus_cds_time_beats`.

## Problems hit

- **Missing font glyphs.** The frequency PlotPanel styles annotation text with
  IBM Plex Mono, which lacks ⁺ (U+207A) and ∥ (U+2225); a Unicode "Mu⁺"/"A∥"
  rendered as tofu boxes. Fixed by drawing the marker labels as **mathtext**
  (`$\mathrm{Mu}^+$`, `$A_\parallel$`, `$A_\perp$`) and using ASCII ("Mu+",
  "Mu0") in the legend `run_label`s.
- **Normalisation for the overlay.** Warming collapses all coherent weight into
  the single Mu⁺ line, so the raw warm central peak is ~4× the cold one and
  squashed the cold satellites off the bottom of the frame. The headline overlay
  therefore normalises each spectrum to its **own central-line peak**, making the
  satellite presence/absence — not the absolute height — the message.
- **Beat plot density.** Overlaying cold + warm time traces over 16 µs was an
  illegible tangle of two oscillating bands; the beat envelope reads far more
  clearly from the **cold run alone over 0–12 µs** (~1.5 beat periods before the
  statistics thin), so the beat scenario dropped the warm overlay.
- **Bonus:** the panel's built-in **γ_µ·B reference marker** (from the "Reference:
  100.00 G" field) lands at 1.355 MHz and sits just left of the observed 1.388 MHz
  central line — a free, honest illustration of the §7 calibration offset.
