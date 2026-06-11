# Frequency-domain finishers — comparison

WiMDA → Asymmetry, feature by feature. Each section quotes the WiMDA Pascal
(file:line, from `/Users/bhuddart/Source/WiMDA/src`, ignoring `__history/`,
`__recovery/`), states the Asymmetry target, and records any divergence with
**both** behaviours stated. Parity of *functionality*, not implementation:
modern numerics and physical correctness win where they tension with WiMDA.

References (APS style, collected): Blundell, De Renzi, Lancaster & Pratt
(eds.), *Muon Spectroscopy: An Introduction*, OUP (2022). J. P. Burg,
*Geophysics* **37**, 375 (1972). B. D. Rainford and G. J. Daniell, *Hyperfine
Interact.* **87**, 1129 (1994). J. Skilling and R. K. Bryan, *Mon. Not. R.
Astron. Soc.* **211**, 111 (1984). T. M. Riseman and E. M. Forgan, *Physica B*
**326**, 230 (2003). Á. Sánchez-Monge *et al.*, *Astron. Astrophys.* **609**,
A101 (2018) [STATCONT].

---

## 1. Field-axis display — VERIFY-ONLY (already built)

**WiMDA.** `FFTPar.pas:210-213` computes the field axis maximum from the Nyquist
frequency using the muon constant `gmu2 = 0.01355342` MHz/G
(`Analyse.pas:566`): `bmax := fmax / 0.01355342`. A `field_in_tesla` global
(`globals.pas:32`) exists but is an unused stub — WiMDA works in Gauss only.

**Asymmetry.** Already implemented and *more* complete. The frequency plot
panel has a `Frequency (MHz) / Field (G) / Field (T)` combo
([`plot_panel.py:437`](../../../src/asymmetry/gui/panels/plot_panel.py)) routed
through `core/fourier/units.py`, with applied-field reference handling
(`_frequency_reference_for_dataset`, `_display_frequency_reference`) and
per-unit axis-limit memory; the unit + reference persist in the project
(`get_state`/`restore_state`, ~`plot_panel.py:4961`/`5121`). FFT and MaxEnt
share one `_frequency_plot_panel`, so they show identical peak positions in
field units by construction.

**Divergence (constant).** WiMDA's `gmu2 = 0.01355342` MHz/G vs Asymmetry's
CODATA `γ_μ/2π = 135.538817` MHz/T → `0.0135538817` MHz/G. They agree to ~6
significant figures (Δ ≈ 3×10⁻⁵ relative). **Behaviour:** Asymmetry keeps the
CODATA value (single-source in `units.py`); WiMDA's rounded constant is
documented, not adopted. Negligible for any real field.

**Action:** no new build. Add verification tests (§verification-plan) that a
known-field TF spectrum peaks at γ_μ·B in Gauss and that FFT≡MaxEnt peak
positions agree.

---

## 2. Frequency-range exclusions — WIRE EXISTING CORE

**WiMDA.** Ten ranges, each a centre±halfwidth pair (`RangeMid[j]`,
`RangeWid[j]`, j∈1..10). Applied in `Plot.pas:1993-2007`:

```pascal
if FFTparams.ExcludeRange.checked then
  for i := 1 to nf2 do begin
    f1 := fmin + (i-1)*fint;
    for j := 1 to 10 do
      if (f1 >= RangeMid[j]-RangeWid[j]) and (f1 <= RangeMid[j]+RangeWid[j]) then
        begin fd^[i-1]:=0; fc^[i-1]:=0; fs^[i-1]:=0 end;
  end;
```

Range 1 is the **diamag slot** (UI label "Diamag", `FFTPar.pas:365-378`). The
**PSI RF-harmonics preset** (`FFTPar.pas:327-358`) sets DC + 50.63 MHz ×{1..5}
(plus a second DC entry), each with width `artwid = 2/FFTtau` (fallbacks
`2/(tres·npts)`, ×4 in power mode):

```
RangeMid = [DC, 50.63, 101.26, 151.89, 202.52, 252.15, DC];  // MHz
RangeWid = artwid (each)
```

**Asymmetry.** `exclude_frequency_ranges(freqs, values, ranges)`
([`fft.py:422`](../../../src/asymmetry/core/fourier/fft.py)) already takes
`(centre_mhz, half_width_mhz)` pairs and zeroes matching bins — an exact match
for WiMDA's parameterisation — but is **unwired**. Phase 1 connects it to a new
"Exclusions" panel section: a small editable table (≤10 rows), a diamag-linked
row whose centre tracks the run's reference field (γ_μ·B), and a "PSI RF
harmonics" preset button.

**Divergence (where applied).** WiMDA zeroes the *pre-derivation* cos/sin/power
arrays (`fc/fs/fd`) for every group before averaging. Asymmetry has already
collapsed to one averaged display channel; it applies `exclude_frequency_ranges`
to that final channel on the canonical MHz axis. **Behaviour:** identical for
the displayed mode; Asymmetry's is simpler and mode-correct. Exclusions operate
in MHz regardless of the chosen display unit (the plot panel converts the axis
for display). Documented.

---

## 3. Pulse frequency-response compensation — REUSE pulse.py, GUARDED

**WiMDA** (`Plot.pas:1931-1944`): a Gaussian rolloff inverse,

```pascal
pw := pwid * pi / 1000;                 { Pwidth (ns) → π·τ(µs) }
for i := 1 to nf2 do begin
  fff := fmin + (i-1)*fint;
  fcfactor := ex(sqr(fff*pw));          { = exp((π f τ)²) }
  fd^[i-1]:=fd^[i-1]*fcfactor; fe^[..]:=..; fc^[..]:=..; fs^[..]:=.. end;
```

i.e. multiply every bin by `exp((π f τ)²)`, with τ from a user text box and
**no high-frequency cap** — it grows without bound (overflow at IEEE limits).
This models the pulse as a Gaussian whose transform is `exp(−(π f τ)²)`.

**Physical basis** (*Muon Spectroscopy* §15.5, §14.2; §17.3): at a pulsed
source the finite muon-arrival pulse "suppresses higher frequencies before they
are recorded, acting as a passband filter." Compensation inverts that
suppression — exactly the forward response that `maxent-completion` already
folds into the MaxEnt kernel.

**Asymmetry — the reconciliation.** `core/maxent/pulse.py` models the *actual*
ISIS arrival distribution as a parabolic proton pulse (cosine transform
`G(x)=3[sin x/x³ − cos x/x²]`, `x=ωw`) times a pion-decay single-pole
`1/(1+(ωτ_π)²)`, exposing the per-frequency amplitude
`R(ν)=√(P_cos²+P_sin²)` via `pulse_amplitude_phase(...)`. The
physically-coherent FFT-side correction is **divide the spectrum by `R(ν)`** —
the inverse of the same response MaxEnt uses, so the two methods agree on a
common pulse model. WiMDA's `exp((πfτ)²)` is the small-x Gaussian caricature of
this (for the parabola `1/G ≈ 1 + x²/10`, with no pion term).

**Divergence (lineshape + guard).** **WiMDA:** `× exp((πfτ)²)`, unbounded.
**Asymmetry:** `× 1/R(ν)` from `pulse.py`, with a high-frequency guard
(checkpoint-3 decision — see implementation-options.md) because `1/R` also
diverges as `R→0` near the first node of `G`. The guard is the *physically
correct* design point: beyond the node the pulse has destroyed the information
and no compensation can recover it, so the correction is capped and/or cut off.
Pulse width defaults from instrument metadata where available, else the panel
field. Documented as the headline divergence.

---

## 4. Spectrum baseline offset — ITERATIVE σ-CLIP (modern default)

**WiMDA** (`Plot.pas:1959-1984`): a single-pass 2σ-clipped mean over the whole
spectrum, subtracted from the power channel `fd` only:

```pascal
s := mean(fd); serr := rms_dev(fd);              { pass 1 }
s := mean(fd where |fd-s| < 2*serr);             { pass 2, one iteration }
fd := fd - s;
```

**Asymmetry.** The brief asks to "evaluate a modern robust-baseline alternative
and pick with evidence." The literature-standard robust continuum estimator is
**iterative σ-clipping** (STATCONT, Sánchez-Monge et al. 2018): re-estimate the
median/σ of the inlier set until σ converges; it is the most accurate and least
input-sensitive of the surveyed estimators (<5% bias on line-rich spectra),
needs no histogram binning, and its converged σ doubles as a baseline-noise
estimate (useful for S/N, §6). WiMDA's single-pass 2σ is exactly its
one-iteration truncation.

**Divergence.** **WiMDA:** one iteration, mean of inliers. **Asymmetry:**
iterate to σ-convergence (capped iterations, configurable κ default 2.0),
robust **median** location. Offered as the default; a "single-pass (WiMDA)"
equivalent is reachable by capping iterations at 1. Applied to the averaged
display channel.

---

## 5. Burg all-poles pole scan — NEW `core/fourier/burg.py` (Phase 2)

**Textbook anchor** (§15.5, "Method 1: Autoregression"): the all-poles AR
spectrum `P(ν)=a₀/|1+Σ_{k=1}^{M} aₖ zᵏ|²`, `z=e^{2πiνΔt}`, M≪N poles; Burg's
algorithm estimates the coefficients; the Final Prediction Error (FPE) "goes
through a minimum versus M" giving the optimum pole count. Stated advantages:
*better intrinsic frequency resolution, suited to sharp features; no phase
correction; works on short data sets.* Stated pathologies: *spurious splitting
of strong features; spurious baseline peaks; small offsets to peak positions;
time-dependent errors not propagated.* These sentences are the load-bearing
content of the Burg user docs.

**WiMDA** (`MaxEnt.pas`): `memcof` (Burg recursion, lines 34-83), `evlmem`
(AR power-spectrum evaluation, 85-107), FPE order scan (171-203); wired in
`Plot.pas:1913-1927`, inheriting the full FFT preprocessing chain and forcing
power mode. Transcribed equations to port:

- **Initial power:** `P₀ = mean(x²)`; `fp1 = P₀·(N+1)/(N−1)`.
- **Reflection coefficient (Burg):** `κ_k = 2·Σ f·b / Σ(f²+b²)` over the n−k
  forward/backward residuals.
- **Power update:** `P_k = P_{k−1}·(1 − κ_k²)`.
- **AR coefficient (Levinson) update:** `a_i^{(k)} = a_i^{(k−1)} − κ_k·a_{k−i}^{(k−1)}`.
- **Residual update:** `f_k[j] = f_{k−1}[j] − κ_k·b_{k−1}[j]`,
  `b_k[j] = b_{k−1}[j+1] − κ_k·f_k[j+1]`.
- **FPE:** `FPE_m = P_m·(N+m)/((N−m)·fp1)`; pick `argmin_m log₁₀(FPE_m)` over the
  scan range; warn if the optimum hits a scan boundary (WiMDA does).
- **Spectrum evaluation:** `S(ν) = √( P_m / |1 − Σ aₖ e^{−2πi k νΔt}|² )`
  (amplitude-like, matching the `(Power)^1/2` convention).

**Asymmetry.** New `core/fourier/burg.py` (~100 lines numpy): `burg_coefficients`,
`ar_power_spectrum`, `fpe_order_scan`. It consumes the **same preprocessed
time-domain grouped signal** that feeds the FFT (lifetime-corrected,
average-subtracted, filtered — exactly WiMDA's chain), and produces an AR
spectrum on the same MHz frequency grid. Surfaced as a clearly-badged
diagnostic display mode (checkpoint-3 wording/range). Docs state plainly: it is
a qualitative super-resolution **diagnostic** and a line-count hint via the
FPE-optimal pole count — **never** the quantitative result; frequency-domain
fitting and MaxEnt are.

---

## 6. S/N-at-peak and average-error readouts — EXTEND existing

**WiMDA** (`Plot.pas:1352-1385`, readout `2259-2262`): `S/N = (π/2)·mean(|fd|)/mean(fe)`;
`freq_error` reassigns a uniform per-bin error `sbar/SN`; the dialog shows
`Ave error` (mean of `fe`) and `S/N at peak` (peak in the middle 7/8 of the
band).

**Asymmetry.** Already largely present: `set_average_summary(mean_error,
peak_signal_to_noise, group_count)`
([`fourier_panel.py:693`](../../../src/asymmetry/gui/panels/fourier_panel.py)),
fed from `mainwindow._on_compute_fourier` with `peak = max(display/error)` and
`mean_error = mean(error)` over the averaged spectrum, error from
`average_fourier_display_values(..., estimate_error=True)` ([`fft.py:453`](../../../src/asymmetry/core/fourier/fft.py)).

**Divergence.** **WiMDA:** global S/N = (π/2)·mean|x|/mean(e), single number.
**Asymmetry:** per-bin **peak** S/N = max(|display|/error) — a sharper, more
honest peak metric — plus mean error. Phase 1 hardens this readout (guard
empty/zero-error, optionally restrict the peak search away from DC like WiMDA's
7/8 window) and surfaces a baseline-noise S/N when the σ-clip baseline (§4) is
active. Documented; WiMDA's average-S/N formula recorded as the parity
alternative.

---

## 7. Real+imag combined display — NEW display mode (Phase 1)

**WiMDA:** no simultaneous real+imag view. **musrfit:** has `real+imag`
(`fourier-transform/comparison.md`). **Asymmetry.** Add a "Real+Imag" display
mode that overlays the cosine (real) and sine (imag) channels of the averaged
complex spectrum on one frequency axis — useful for judging phase-correction
quality. Implementation: the averaged path retains both quadratures for this
mode; the plot panel draws a primary + secondary trace. Kept simple (overlay,
no dual y-axis); a polish pass can refine later.

---

## 8. Diamagnetic fit-and-subtract — time-domain, pre-FFT (Phase 2)

**WiMDA** (`Plot.pas:1832-1890`, model `246-255`): a 5-parameter damped cosine
fit in the **time domain** before the transform,

```
f(t) = p₁·cos(2π(p₂·t + p₃))·exp(−|p₄·t|) + p₅
```

(p₁ amplitude, p₂ frequency MHz, p₃ phase cycles, p₄ damping µs⁻¹, p₅ baseline;
`|p₄t|>100 ⇒ clamp to p₅`). Frequency seeded from the header field
(`p₂ = field·0.01355342`); on success the fitted cosine is subtracted from the
signal and the fitted field `p₂/0.01355342` is written back to the `CorrField`
box.

**Asymmetry.** Phase 2: a pre-FFT time-domain damped-cosine fit-and-subtract on
the grouped signal, reusing the existing fitting engine
(`core/fitting`) and an Oscillatory×Exponential-style model; report the fitted
field back to the reference-field control (which drives the diamag exclusion
slot and the field axis). UX (visibility of the fitted line) is a checkpoint-3
decision. Divergence: modern minimiser (iminuit) vs WiMDA's `fite`; phase
in cycles vs radians is normalised consistently. Documented.

---

## 9. Muonium-radical correlation spectrum — DEFER (follow-on)

**WiMDA** (`Plot.pas:515-523`, `1387-1394`, `2149-2230`): `rmatch(f,B)` pairs a
muon frequency with its Breit-Rabi partner
(`ω_e=2.81555·B`, coupling `1.394225·B`), interpolates the spectrum at the
partner frequency, combines via an order-weighted mean `CorrFn`, and plots
against the hyperfine axis `|f₁+f₂|`. Niche (muoniated-radical chemistry),
high-complexity, and entangles a second quantum system.

**Asymmetry.** Both study agents and the brief converge on **defer**. The
Breit-Rabi relations in `core/fitting/muonium.py` (`_tf_levels`, transition
frequencies) could anchor a future port, but the correlation Hamiltonian is
distinct from the muonium line-shapes and should not be grafted onto them.
Recorded as the natural follow-on; ships only if checkpoint-3 elects it.

> **PROMOTED (2026-06-11).** This deferred follow-on is now its own study,
> [`radical-correlation-spectrum`](../radical-correlation-spectrum/README.md)
> (branch `feat/radical-correlation-spectrum`). It confirms the transcription
> above against the Pascal source directly and reconciles the Breit-Rabi
> relation across three sources: `_tf_levels` *is* the anchor (the study builds
> the spectrum by the **exact forward map** `A = ν₁₂ + ν₃₄` rather than
> transliterating `rmatch`'s approximate inverse), exactly as anticipated here.

---

## Reused-API reconciliation (no duplication)

| Need | Reused API | Status |
|---|---|---|
| Field-axis units | `core/fourier/units.py` `FieldUnit`/`convert` | Already wired in `plot_panel`; verify-only |
| Exclusions | `core/fourier/fft.py:422` `exclude_frequency_ranges` | Exists, **wire** to panel — do not reimplement |
| Pulse compensation lineshape | `core/maxent/pulse.py` `pulse_amplitude_phase` | Reuse `R(ν)`; do not re-derive a parallel pulse model |
| S/N & error | `fft.py:453` `average_fourier_display_values(estimate_error=True)` | Extend the existing readout |
| Spectrum builder seam | `core/fourier/spectrum.py:184` `compute_average_group_spectrum` | Insert conditioning chain after averaging (line ~276) |

## Out of scope (recorded with rationale)

- **FB t=0 extrapolation** — moot under Asymmetry's grouped-counts Fourier
  source (no per-detector t=0 zero-fill stage to extrapolate).
- **Per-detector FFT** — deliberately deferred in the `fourier-transform`
  study; needs a broader per-detector phase contract.
- **N₀-normalised single-histogram FFT input** — follow-on; it interacts with
  the open count-domain PR #41 and the WiMDA-style grouped-count path remains
  the correct default (see `fourier-transform/comparison.md` "N₀
  Normalization"). Confirmed at checkpoint-3.
