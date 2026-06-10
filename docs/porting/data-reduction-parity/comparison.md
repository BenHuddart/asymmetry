# Data-reduction parity — cross-program comparison

Date: 2026-06-10. All WiMDA citations are from a direct reading of the Pascal
at `/Users/bhuddart/Source/WiMDA/src` (ignoring `__history/`/`__recovery/`);
Asymmetry citations are against `main` at `19f242b`. Textbook citations are to
Blundell, De Renzi, Lancaster & Pratt, *Muon Spectroscopy: An Introduction*
(book page numbers). The umbrella brief was verified against the source —
discrepancies found in the brief are flagged inline.

WiMDA constants (`globals.pas:13`): τ_μ = 2.1969811 μs, λ_μ = 1/τ_μ.
Asymmetry should use the same CODATA/PDG value, defined once in core.

---

## 1. Alpha estimation (Phase 1)

### WiMDA (`Group.pas:1775 EstimateButtonClick`)

Both estimators run on the **current reduced output bins** (`groupd`,
`timed[i]`, `np` points): grouped counts after period/RG selection, deadtime
correction, background-run subtraction, and rebinning, restricted to the
good-bin window. Bins with `f ≤ 0` or `b ≤ 0` (or `f + αb = 0`) are skipped.
The result therefore depends on the live bunching/correction settings.

**Diamagnetic method** (`cgrp.method = diamag`). Minimises over α

    S(α) = Σᵢ (Aᵢ/σᵢ)²,  Aᵢ = (fᵢ − α bᵢ)/(fᵢ + α bᵢ),
    σᵢ = 2α √(fᵢ bᵢ (fᵢ + bᵢ)) / (fᵢ + α bᵢ)²

— the σᵢ expression (written in the source as
`2·a·(f/b)·√(1/f + 1/b)/(f/b + a)²`) is algebraically the exact Poisson
propagation of the asymmetry. On a TF run the oscillation averages to zero
when α balances the detectors, so the weighted asymmetry power is minimal at
the correct α. Optimiser: coarse-to-fine **grid walk** from the current α with
steps 0.1 → 0.01 → 0.001, internally clamping trial α to [0.1, 10], aborting
outright if α > 4 on entry. Reports a bare number; no uncertainty.

**General method** (`cgrp.method = general`). For each bin forms the
lifetime-corrected balanced count

    Nᵢ(α) = (fᵢ/√α + bᵢ √α) · e^{tᵢ λ_μ},  σᵢ = √|fᵢ/√α + bᵢ √α| · e^{tᵢ λ_μ}

and minimises the **weighted relative scatter** of Nᵢ about its weighted mean:
with m₁ = Σ(Nᵢ/σᵢ)/Σ(1/σᵢ), m₂ = Σ(Nᵢ/σᵢ)²/Σ(1/σᵢ²), c = Σ(Nᵢ/σᵢ²)/Σ(1/σᵢ²),
the objective is √(m₂ + m₁² − 2 m₁ c)/m₁. Same grid walk.

*Why it works*: writing F(t) = N_F e^{−λt}(1 + a₀P(t)) and
B(t) = N_B e^{−λt}(1 − a₀P(t)), the combination F/√α + B√α equals
2√(N_F N_B)·e^{−λt} **exactly when α = N_F/N_B** — the polarization term
cancels for any P(t). The lifetime-corrected combination is then flat in
time, so minimising its relative scatter finds α on relaxing LF/ZF data where
no zero-mean oscillation exists and where the plain count ratio is biased by
the relaxing polarization.

### Asymmetry current state

`estimate_alpha(forward, backward, first_good_bin, last_good_bin)`
([core/transform/asymmetry.py:107]) — Mantid `AlphaCalc` convention,
α = ΣF/ΣB over the good-bin window; returns 1.0 on degenerate input. GUI:
"Estimate" buttons in the grouping dialog (single-α and per-axis vector
variants, [gui/windows/grouping_dialog.py:356,406]); the value lands in the
`alpha` (or `alpha_x/y/z`) grouping key and is applied to all selected
datasets (deadtime-Estimate precedent). No uncertainty, no method choice.

*Bias note*: ΣF/ΣB equals N_F/N_B only when the polarization integrates to
zero over the window — true for many-cycle TF data, increasingly wrong for
LF/ZF (relaxing P(t) > 0 biases α upward by ≈ a₀⟨P⟩ to first order).

### Mantid / musrfit

- **Mantid** `AlphaCalc` ("Guess Alpha" on the Grouping tab): ΣF/ΣB over the
  good range (the current Asymmetry estimator is this); a single number, no
  uncertainty; the GUI course frames it as a TF-data operation. The ALC
  interface locks α = 1.0 for multi-period data.
- **musrfit**: no reduction-time estimator — α (and β) are RUN-block fit
  parameters of the asymmetry fit type, so MINUIT gives an honest σ_α; the
  asymmetry model has α inside it rather than baked into the data. The
  α-as-fit-parameter route is Asymmetry's separate `count-domain-fit-modes`
  project and complements (does not replace) reduction-time estimation —
  the WiMDA manual itself calls the fitted route "the most accurate way",
  with the two estimators as the quick calibration tools.
- **Published workflow** (WiMDA manual LF tutorial): determine α on a weak-TF
  ("T20", ~20 G) calibration run on the same sample and mounting, then fix
  it for the LF/ZF runs. The General method's value is that it can sanity-
  check α directly on the LF data when no T20 run exists.

### Textbook physics

α is the balancing factor N_F⁰/N_B⁰ in
A(t) = (N_F − αN_B)/(N_F + αN_B) (§2.4); calibration practice is a weak-TF
run on a diamagnetic state, adjusting α until A(t) "oscillates symmetrically
about zero" (§15.2, p. 220) — exactly the diamagnetic estimator's objective.
Errors per bin are Poisson, propagated to A (§15.3–15.4).

### Design implications

Port both objectives **verbatim as objectives** but replace the grid walk
with bounded continuous optimisation (`scipy.optimize.minimize_scalar`,
bounded method on ln α to make the bounds symmetric), and report an
uncertainty (decision recorded in implementation-options.md). Operate on
**raw grouped counts inside the good-bin window** (deterministic, independent
of display bunching) rather than WiMDA's live-binned `groupd` — divergence
D3 below. Keep ΣF/ΣB as a third labelled method ("Count ratio — TF
calibration runs only").

---

## 2. Backgrounds (Phase 2)

### WiMDA tail fit (`Group.pas:1114 BGfit`, `:1125 estBG`)

Triggered by the "BG correction" checkbox (`doBG`); runs inside `Regroup`
per group. Takes the **late half** of the current output bins
(`n0 = np div 2`), and fits two parameters (initial rate p₁, flat rate p₂) of
the bin-integrated model

    C(t, w) = [p₁ e^{−λ_μ t} · (e^{wλ_μ/2} − e^{−wλ_μ/2})/(wλ_μ) + p₂] · w

(w = bin width; the bracket is the exponential averaged across the bin) to
the per-bin counts with σᵢ = √Cᵢ, except **bins with ≤ 4 counts get
σ = 10¹⁰** — i.e. they are effectively deleted rather than treated with
Poisson statistics. Optimiser: the 1972 Gauss–Newton `FITE` with fixed
starting values (1000, 0.001) and no convergence reporting. The flat rate p₂
(counts/μs per group) is stored per histogram as `background[h] = p₂/n_hist`
and subtracted as `groupd[i] −= p₂·wᵢ`. No uncertainty.

Note the model has **no muon-correlated asymmetry term** — on a relaxing or
oscillating group the late-time window must be long enough that P(t) has
died; WiMDA leaves that to the user implicitly (window is always the late
half of the *displayed* range).

### WiMDA background-run subtraction (`Group.pas Regroup` FileBG path, `BGform.pas`)

"File BG" checkbox loads a complete second run (`BGmuonrun`, by run number,
through the normal loaders — `BGform.pas:32`). During `Regroup`, if the BG
run differs from the sample run, every raw bin gets

    scale = frames_sample / frames_BG    (`Group.pas:1437–1441`)
    groupd −= scale · BG.histos[h, i + t0ₕ]   (`:1506–1508`; R/G variants :1462–1471)

i.e. the **raw, deadtime-uncorrected** BG counts are subtracted from the
**deadtime-corrected** sample counts, aligned by the *sample's* t0 table, and
the error arrays (`groupunc`) keep the sample counts only — the subtraction
adds no variance in WiMDA's error model. Exclusive with the other BG modes.

### Asymmetry current state

[core/transform/background.py]: fixed-value mode and range-average mode
(pre-t0 window, musrfit-style default 0.1·t0–0.6·t0, beam-period-aware for
PSI/TRIUMF), with propagated errors. `supports_background_correction`
(:41) gates the whole feature to PSI/LEM data — **pulsed ISIS data currently
has no background mode at all**, and there is no background-run subtraction.

### Mantid / musrfit

- **musrfit**: three mutually exclusive RUN-block mechanisms — `background
  <first> <last>` bin intervals (pre-prompt-peak, continuous sources),
  `backgr.fix` fixed per-histogram constants, or `backgr.fit` (background as
  a free parameter in single-histogram fits). Subtraction is per histogram
  **before** the asymmetry ratio is formed — the same count-level convention
  Asymmetry uses. No tail-fit estimator, no background-run subtraction.
- **Mantid**: the muon Corrections tab has background None / **Auto** /
  Manual, where Auto fits "Flat Background + Exp Decay" over a user X-range
  with the decay constant **fixed to 1/τ_μ** and subtracts the flat part —
  functionally WiMDA's tail fit (least-squares, optionally on rebinned
  data, no bin-integration factor). Run subtraction is generic workspace
  algebra (`Minus`), unscaled.
- WiMDA's frame-ratio-scaled background-run subtraction has no counterpart
  in either (sample-holder / silver / laser-off references); the nearest
  ISIS practice is period-level on/off subtraction within one run, which
  needs no exposure scaling.

### Textbook physics

Two backgrounds must not be confused (§15.4 footnote, p. 225): the
**uncorrelated time-independent count rate** (this project) and the
**background asymmetry** from holder muons (a real muon signal — that one is
handled by fit models / silver-vs-hematite calibration, §15.2, and is *not*
in scope here). At continuous sources the uncorrelated rate is significant
and is estimated **from the pre-t0 bins** or fitted per detector (§15.4,
p. 224) — Asymmetry's existing range mode. At pulsed sources the duty factor
(~1.6 × 10⁻³ at ISIS) suppresses it to "virtually unmeasurable" (§14.3) and
the book gives no late-time estimation procedure. WiMDA nevertheless ships
one because real pulsed spectra can carry a small flat rate (dark counts,
light leaks, upstream decays surviving the veto) that matters for long-time
relaxation work at 20–32 μs — the tail fit *quantifies* smallness instead of
assuming it. The user docs should carry exactly this framing: at ISIS expect
p₂ consistent with zero; a significantly non-zero value is a diagnostic.
Poisson statistics: σ = √n is the right per-bin error only at high counts
(§15.3, p. 221) — at 20+ μs the counts per raw bin are few, which is why the
fit must be Poisson-likelihood rather than weighted least squares.

### Design implications

Tail fit: same two-parameter bin-integrated model, but (a) **Poisson MLE**
(minimise the Poisson deviance) instead of √N weights with the ≤ 4-count
amputation; (b) fit window explicit and configurable (default: late half of
the good-bin window on **raw grouped counts**, not display bins); (c) report
p₂ with uncertainty; (d) flag when p₂ < 2σ (background consistent with
zero). Gating: replace the binary `supports_background_correction` with
per-mode availability — range-average needs a pre-t0 region (continuous
sources), tail-fit needs a long counting window (pulsed sources qualify),
fixed/manual is universal.

Background-run subtraction: same frame-ratio scale, but subtract
**consistently in the count domain with error propagation**
(σ² = N_sample + scale²·N_BG) and deadtime-correct both runs the same way —
divergences D6/D7. Data model: the reference run attaches at the grouping
level (decision in implementation-options.md). Shared frame-scaled count
arithmetic should live where the Wave B `run-arithmetic` project can reuse
it; co-add/co-subtract of datasets themselves is **out of scope here**.

---

## 3. Binning modes (Phase 3a)

### WiMDA (`Group.pas:1411–1418`, `gtype = (fixed, variable, const_error)`)

Inside the `Regroup` accumulation loop, the target width of the output bin
starting at time t₁ is

| Mode | Width(t₁) | Controls (`Group.dfm`) |
|---|---|---|
| `fixed` | `tres · bunch` | Bunching factor |
| `variable` | `bin0 · exp(λ_μ · t₁ · 0.22 · ln(bin10/bin0))` | Initial Bin (μs), Late Bin (μs) |
| `const_error` | `bin0 · exp(λ_μ · t₁)` | Initial Bin (μs) |

Raw bins are accumulated until the running edge passes t₁ + width; output
centres are `(t₁+t₂)/2` with recorded per-bin widths `binwid[j]`. Defaults
bin0 = 0.08 μs, bin10 = 0.25 μs (`InitializeGlobalVars`).

Decoding `variable`: λ_μ·0.22 = 0.10014 μs⁻¹, so
width(t) ≈ bin0 · (bin10/bin0)^(t/10 μs) — **bin0 is the width at t = 0 and
bin10 the width at t = 10 μs**, with exponential growth between (the umbrella
brief's "per decade" gloss is wrong; the 0.22 is 1/(10·λ_μ) folded into the
exponent so the user-facing knobs are the two widths). The growth ratio is
(bin10/bin0)^1.0014 at 10 μs — WiMDA's constant makes it approximate; we
implement the exact form.

Decoding `const_error`: counts per output bin ≈ rate(t)·width(t) ∝
e^{−λ_μ t}·e^{+λ_μ t} = const, so the **Poisson error per output bin is
~constant** while the polarization is slowly varying — the textbook's scheme
(d) in Fig. 15.7. bin0 sets the statistics level.

### Asymmetry current state

`rebin(time, values, errors, factor)` ([core/transform/rebin.py:9]) —
fixed integer bunching of the reduced curve (mean of values, quadrature/n
errors). Consumers: plot path ([core/representation/time.py:78],
[gui/mainwindow.py:2746,3089]), count-level fixed rebin inside grouped
fitting ([core/fitting/grouped_time_domain.py:218]) and Fourier
([core/fourier/grouped.py:290]), MaxEnt bunch coupling
([core/maxent/engine.py:420]). `bunching_factor` is the grouping-dict key.

### Mantid / musrfit / textbook

musrfit has fixed `packing` only (display can differ via `view_packing`).
Mantid's generic `Rebin` accepts variable edge lists and logarithmic widths
(negative step ⇒ x_{j+1} = x_j(1+|Δx|)), with quadrature-by-overlap error
propagation, but the muon GUI exposes only None/Fixed/Variable rebin args
and recommends fitting **raw** data, using rebinning for display and
background estimation. All three tools pack **histograms** then form the
asymmetry rather than rebinning the asymmetry ratio directly — the safe
order. The textbook (Fig. 15.7, p. 222) explicitly names
all four schemes — raw, fixed, variable, constant-error — so the two
non-fixed modes are established μSR practice that only WiMDA automated.
Caveat from §15.5: rebinning lowers the Nyquist frequency; check for high-
frequency content first. Variable-width output is incompatible with the FFT
(uniform sampling) — Fourier and MaxEnt inputs must stay on fixed binning.

### Design implications

Implement as **edge generation + aggregation onto edges** in `rebin.py`:
`variable_bin_edges(t_range, bin0_us, bin10_us)` /
`constant_error_bin_edges(t_range, bin0_us)` plus an aggregation step —
count-level vs curve-level is a design decision recorded in
implementation-options.md (count-level is essentially forced: in
constant-error mode the late-time raw bins hold 0–2 counts, where a
weighted mean of per-bin asymmetry ratios is undefined/NaN-prone, while
summed counts stay exactly Poisson — and count-then-ratio is the order all
three reference tools use). Display/fit-input only; raw histograms
untouched (provenance invariant). Fourier and MaxEnt require uniform
sampling and stay fixed-mode; variable-width *count-domain fitting* belongs
with `count-domain-fit-modes` if ever needed. Time-domain fitting of an
unevenly-binned asymmetry curve is already weight-correct (point-wise χ²
with per-point σ).

---

## 4. Automatic t0 search (Phase 3b)

### WiMDA (`Group.pas:2225 SearchT0ButtonClick`)

Per group: scan the **raw group histogram** downward from bin `np` to 1 and
take the bin of maximum counts (ties → earliest bin, because the comparison
is strict and the scan descends); set `tzero[group] = argmax`. For
continuous (non-pulsed) data the per-histogram t0 of the group's *first*
member is also updated. Two quirks: the scan ceiling `np` is the current
number of *output display bins*, not raw bins (a latent bug — with heavy
bunching the scan window shrinks to a fraction of the histogram, though the
peak is always early so it rarely bites); and a comment shows it once used
the maximum *derivative* before being simplified to the maximum bin. Enabled
only when file values are overridden and per-histogram t0 editing is active
(`SameT0` off ⇒ per-hist; `FileVals` off).

### Asymmetry current state

t0 comes from loaders (per-histogram `t0_bin`; NeXus `time_zero`/attrs,
PSI header) with manual override spinners in the grouping dialog
(`t0_bin` + `t_good_offset` + `last_good_bin` grouping keys,
[gui/windows/grouping_dialog.py:242–258]). No automatic search.

### Mantid / musrfit / textbook

Mantid has no t0 estimator at all — t0/first-good/last-good come from the
file header ("determined by the instrument scientist"), applied via
`MuonPreProcess`'s per-detector `TimeZeroTable`, with manual override in the
Home tab. musrfit ships `musrt0`: interactive, or
`--getT0FromPromptPeak [offset]` — **maximum bin of each histogram**, with
the first good bin a fixed offset after it; its docs flag the prompt-peak
assumption (silently wrong on pulsed data). The WiMDA manual (§5.5–5.6)
quantifies the pulsed convention: t0 is "the middle of the muon pulse"
(MuSR ≈ bin 40, EMU ≈ bin 18 at 16 ns), t_good after "the entire pulse has
arrived" — offset ≈ 7 bins at ISIS (less critical for FB than TF analysis),
≈ 3 bins at PSI, and recommends Search for T0 for continuous-source files.
The textbook (§15.3,
pp. 223–224 and §14.2 Fig. 14.4) distinguishes: continuous sources — t0 is
the sharp **prompt peak** (good to a few tenths of ns); pulsed sources — t0
is the **centre of the muon pulse**, "in practice found from the midpoint of
the rising edge", with *first good data* after the pulse has fully arrived.
And the blunt practice warning: "never rely on information stored in the
data file, if you have not recorded it yourself!" — exactly the use case for
a Find t0 action.

### Design implications

Per-detector estimator with two strategies selected by source type:
continuous → argmax of counts (prompt peak, WiMDA/musrfit-compatible);
pulsed → half-maximum crossing of the leading edge (linear interpolation,
rounded to a bin) — divergence D9 records that WiMDA uses peak-max for both,
which on ISIS data systematically places t0 at the pulse *peak* rather than
its centre-of-rise. Results surface as a "Find t0" action in the grouping
dialog that fills the existing override controls (user confirms before
apply); never silently overwrite loader values.

---

## 5. Detector exclusion (Phase 3c)

### WiMDA (`Group2.pas:126 ExcludeDetectorsClick`, `Group.pas:1885 loadexclude/saveexclude`, `nexusunit.pas:651 count`)

A text list ("1,5,10-15"; ranges in either direction) parsed into a
`Exclude[1..maxdet]` boolean array; consumed inside the NeXus read path —
`count()` returns 0 for excluded detectors — so exclusion requires
**reloading the run** and the excluded detectors contribute zero counts
while the detector→histogram mapping is unchanged. Persisted as a
`.exclude` sidecar next to the grouping file (line 1 instrument name,
line 2 the list; applied only when the instrument matches). The
`default.exclude` auto-load described in the umbrella brief is **commented
out** in current WiMDA (`Group.pas:2203–2215`), as is `default.mgp`.

### Asymmetry current state

None. Grouping edits can omit detectors from groups manually (group table /
detector layout dialog), but there is no per-detector exclude list, no
persistence, no schematic toggle.

### Mantid / musrfit / textbook

Mantid muon interfaces support custom dead-detector handling only via
grouping strings; musrfit via editing the forward/backward histogram lists
in the `.msr`. The textbook covers grouping and deadtime but has **no
treatment of dead/hot detector exclusion** (index confirmed) — the user docs
for this feature will be the reference register (what a dead vs hot detector
looks like in per-detector totals, why excluding beats correcting).

### Design implications

Exclusion belongs at the **grouping level, applied at reduction time**
(zero-weight excluded detectors when summing groups in
`grouping.py apply_grouping*`), not at load time — no reload, raw histograms
intact, provenance explicit. Grouping-dict key `excluded_detectors`
(sorted list of 1-based detector ids) persisted with the grouping in the
project; parser accepts the WiMDA-style range text. GUI: click-to-exclude on
the detector schematic plus a list editor (decision on exact UX in
implementation-options.md). Alpha/deadtime estimators and group sums all see
the exclusion consistently because they consume the same grouped counts.

---

## 6. Multi-period subset mapping (Phase 3d)

### WiMDA (`PeriodMappingUnit.pas`, `Group.pas:1177 MapPeriods`, `:818 RGselect`)

For runs with > 2 periods a mapping form shows up to **8 fixed slots**, one
per period, each a three-way radio {Ignore, Red, Green} with per-period
metadata (name, frames, sequence frames, binary output tag). Periods with
`mode = 2` (dwell periods — no DAQ) are forced to Ignore and hidden;
defaults: first DAQ period → Red, second → Green. `MapPeriods` then sums the
selected period histograms bin-wise into `Rhistos`/`Ghistos`, and the
existing RG box (Red / Green / G−R / G+R) consumes them. Frame bookkeeping:
`redfraction = redframes/periodDAQframes` (and green alike) feed the
RG deadtime correction (`ccorrectRG` normalises the per-frame rate by the
fraction) and the G−R combination is `G·n_red/n_green − R` with integer
error arithmetic on the uncorrected counts (`:1484–1486`).

### Asymmetry current state

[core/io/periods.py]: `select_period` (single period, scriptable),
`select_period_histograms` (count-level, shared with GUI),
`combine_period_asymmetry` (G±R **on reduced asymmetry curves**, quadrature
errors), `period_count/labels/resolve_period_index`; grouping keys
`period_histograms`, `period_good_frames`, `period_dead_time_us`,
`period_mode`. GUI: Red/Green/G±R radio set in the grouping dialog
(2-period assumption). No subset summation, no Ignore, no per-period frame
display, no N > 2 mapping UI.

### Mantid / musrfit / textbook

Mantid loads periods as workspace-group members and sums arbitrary subsets
via `MuonPreProcess`'s `SummedPeriods`/`SubtractedPeriods` lists — the same
capability as WiMDA's mapping, expressed as two index lists; the Grouping
tab's table has a Periods column accepting comma/dash lists, and
`PlotAsymmetryByLogValue` takes Red and Green period numbers (outputs
difference, each, and sum). Period sums happen on **counts before**
deadtime/asymmetry steps. musrfit has no first-class period concept —
grouping lists and `ADDRUN` summation cover the same ground. The textbook (Ch. 19) documents the acquisition physics:
alternating light-on/off frames at half the muon pulse rate (photo-μSR,
p. 289), RF on/off alternating every ~10 s (p. 294) — N > 2 period
structures arise from multi-condition sequences (e.g. laser delay scans),
which is what subset summation serves.

### Design implications

Extend `periods.py` with a **count-level mapping reducer**:
`map_periods(histograms_by_period, mapping)` where
`mapping: dict[int, "red"|"green"|"ignore"]` sums counts and good-frames per
set and returns the two synthetic period payloads the existing
G±R machinery already consumes — the current 2-period modes become the
trivial mapping, one shared code path. Frame-fraction bookkeeping carries
into deadtime correction as per-set good-frames (equivalent to WiMDA's
`framefraction`, but exact per set). G±R stays at the asymmetry level
(existing convention) — divergence D11 records WiMDA's count-level
`G·n_red/n_green − R` versus Asymmetry's asymmetry-level combination.
Persist the mapping in the grouping (`period_mapping` key); the matrix
dialog (period × {Red, Green, Ignore}) hangs off the period selector.

---

## Divergences from WiMDA

Every intentional divergence, with both behaviours. "Improvement" entries
follow the umbrella's physics-correctness mandate.

| # | Area | WiMDA behaviour | Asymmetry behaviour | Class |
|---|---|---|---|---|
| D1 | α optimiser | Coarse-to-fine grid walk (0.1/0.01/0.001), clamp [0.1, 10], abort if α > 4 | Bounded continuous minimisation on ln α; converges to machine precision; explicit failure status | improvement |
| D2 | α uncertainty | Bare number | σ_α reported with the estimate (method per implementation-options.md) | improvement |
| D3 | α input data | Live display bins (`groupd`): depends on bunching, BG and deadtime settings at click time | Raw grouped counts in the good-bin window with current corrections applied deterministically; documented contract | improvement (reproducibility) |
| D4 | Tail-fit weighting | σ = √N, bins with ≤ 4 counts deleted via σ = 10¹⁰ | Poisson MLE (deviance), all bins retained | improvement (low-count correctness) |
| D5 | Tail-fit window | Late half of current display bins, implicit | Explicit window on raw grouped counts; default late half of good-bin range, user-adjustable; uncertainty + consistent-with-zero flag | improvement |
| D6 | BG-run deadtime | BG counts subtracted raw from deadtime-corrected sample counts | Both runs deadtime-corrected identically before subtraction | improvement (consistency) |
| D7 | BG-run errors | Subtraction adds no variance (`groupunc` keeps sample counts only) | σ² = N_sample + scale²·N_BG propagated | improvement |
| D8 | Variable-bin law | width = bin0·exp(λ_μ·0.22·ln(r)·t), i.e. ≈ (r)^(t/10 μs) with r = bin10/bin0 (exponent off by ×1.0014) | Exact width = bin0·(bin10/bin0)^(t/10 μs); same knobs, same intent | cosmetic exactness (values differ < 0.2%) |
| D9 | t0 search, pulsed | Max-count bin (pulse peak) for all sources; scan ceiling = current output-bin count (latent bug) | Continuous: prompt-peak argmax (parity). Pulsed: half-maximum of rising edge (textbook convention). Full-histogram scan | improvement + parity split |
| D10 | Exclusion mechanics | Detectors zeroed at file-read time (`count()` = 0), run reload required; `.exclude` sidecar | Zero-weighted at grouping time; no reload; raw histograms intact; persisted in project grouping | improvement (provenance invariant) |
| D11 | G−R combination | Count-level `G·n_red/n_green − R`, integer error arithmetic | Asymmetry-level combination, quadrature errors (existing `combine_period_asymmetry` convention); subset *summation* is count-level (exact) | convention (documented since PR #29 era) |
| D12 | Period slots | 8 fixed UI slots (`maxperiods`) | Arbitrary N (matrix rows generated per file) | improvement |
| D13 | Estimator persistence | α method stored in binary `.mgp` grouping record | α method + estimate provenance in project grouping dict (schema-additive) | modernisation |

Non-divergences worth stating: the two α objectives, the bin-integrated
tail-fit model C(t, w), the frame-ratio BG-run scale, the constant-error
width law bin0·e^{λ_μ t}, the {Ignore, Red, Green} mapping semantics, and
dwell-period (mode 2) forced-Ignore are all ported exactly.

## Brief-verification notes (umbrella brief vs source)

- "Variable binning (width grows per decade from bin0)" — wrong gloss; bin10
  is the width **at 10 μs**, not a per-decade growth factor (§3 above).
- `default.mgp`/`default.exclude` auto-load is dead code in current WiMDA
  (commented out, `Group.pas:2203–2215`) — reinforces the out-of-scope call.
- The "EMU LF series 10–100 G @ 350 K" named in the umbrella test-data is
  actually **HIFI** runs 118222–118240 (corannulene; see test-data.md).
- Everything else in the brief checked out against the source.

## References

1. S. J. Blundell, R. De Renzi, T. Lancaster, and F. L. Pratt, *Muon
   Spectroscopy: An Introduction* (Oxford University Press, Oxford, 2022).
2. F. L. Pratt, Physica B **289–290**, 710 (2000) — WiMDA.
3. A. Suter and B. M. Wojek, Phys. Procedia **30**, 69 (2012) — musrfit.
4. O. Arnold *et al.*, Nucl. Instrum. Methods Phys. Res. A **764**, 156
   (2014) — Mantid.
5. R. L. Workman *et al.* (Particle Data Group), Prog. Theor. Exp. Phys.
   **2022**, 083C01 (2022) — τ_μ = 2.1969811(22) μs.

Documentation consulted (June 2026): musrfit user manual
(lmu.web.psi.ch/musrfit/user/html/user-manual.html); WiMDA manual rev. 2018
(shadow.nd.rl.ac.uk/wimda/); Mantid algorithm and muon-interface docs
(docs.mantidproject.org — AlphaCalc, MuonPreProcess, ApplyDeadTimeCorr,
EstimateMuonAsymmetryFromCounts, PlotAsymmetryByLogValue, Rebin,
LoadMuonNexus v2, Corrections/Grouping tabs, muon GUI course).
