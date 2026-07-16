# NOTES — Magnetic ordering in EuO (Magnetism)

Scenario module: `euo_ordering.py`
Example: `Magnetism/Magnetic ordering in EuO`
Corpus data: real PSI GPS `.bin` histograms `deltat_pta_gps_2923–2973`
(51 runs; ZF T-scan 2923–2960, TF-60 G 2961–2973).
Spec: `GROUND_TRUTH.md` (Blundell *et al.*, Phys. Rev. B **81**, 092407 (2010)).

## Scenarios registered

| name | render | intended docs use |
|---|---|---|
| `corpus_euo_load_browse` | Data browser with a 12-run ZF T-scan (1.5–200 K) loaded via the real `deltat_pta_gps` loader; base-T ZF spectrum in the plot. | "PSI-bin / GPS format support" + data-handling / browse step. |
| `corpus_euo_zf_fit` | Converged `Oscillatory*Exponential + Constant` fit on the 1.6 K ZF run (2960), zoomed to 0–0.45 µs so the damped precession resolves. **f = 30.18 MHz, λ = 3.09 µs⁻¹, χ²ᵣ = 1.30.** | Core below-T_C analysis step; the "ordered oscillation" result (ν(0) ≈ 30 MHz). |
| `corpus_euo_fft` | Frequency-domain view of the 1.6 K run, F/B pair + Lorentzian apodisation, framed 20–42 MHz: a single precession line peaking at ~30 MHz. | "Single precession frequency observed" (Fig. 1(c)); FFT workflow. |
| `corpus_euo_nu_t_trend` | **Headline.** ν(T) order parameter from 18 real per-run ZF fits (1.6 → 68.7 K) with the fitted `OrderParameter` power law overlaid. | Parameter-trending workflow; the critical-behaviour result (T_C, β). |
| `corpus_euo_waterfall` | Waterfall of 6 ZF spectra (1.6 → 68.3 K) stacked; precession slows as ν(T) → 0 toward T_C. | Distinctive overlay/waterfall feature; qualitative order-parameter collapse. |

## Run selection & workflow (GROUND_TRUTH refs)

- **Detector pair (GT §4).** The loader's default grouping already picks the
  transverse **Forward/Back** pair (`forward_group=2, backward_group=1`,
  α = 1). Verified empirically: of all 10 GPS detector-pair combinations, only
  Forw/Back carries a coherent ZF oscillation; Up/Down/Righ pairs give only a
  DC/low-frequency baseline. So `G_z(t)` is the F–B asymmetry with no manual
  grouping needed. α is left at 1 (uncalibrated) — this leaves a large ~28 %
  constant baseline the additive `Constant` term absorbs.
- **Per-run model (GT §4).** `Oscillatory * Exponential + Constant`
  = `A·cos(2πνt+φ)·e^(−λt) + A_bg`, fit range 0–6 µs via the core `FitEngine`
  (the same engine the single-fit panel drives).
- **Warm-starting (GT §4).** ν is seeded at ν(0) ≈ 30 MHz at base T and
  warm-started downward as T → T_C. This is essential: a seed near the true
  frequency lands A ≈ 5 %, χ²ᵣ ≈ 1.0; a seed ~6 MHz too low collapses to a
  spurious A ≈ 0.5 % minimum. The trend scenario therefore fits in **ascending
  temperature order**, carrying ν forward.
- **Runs used for ν(T) (GT §3).** 18 ZF runs, ordered by *measured* sample
  temperature: 2960, 2925, 2928–2937, 2938, 2939, 2940, 2941, 2942, 2943
  (1.6 → 68.7 K). Runs **2923/2924 dropped** (bad thermometry / low statistics,
  §3 note). TF-60 G runs 2961–2973 excluded (paramagnetic, not order-parameter
  data). Very-near-T_C runs 2944–2956 (ν ≲ 5 MHz, heavily damped) omitted from
  the trend — their fits are unstable and the phenomenological full-range fit
  does not need them.
- **Trend model (GT §4).** `OrderParameter`: ν(T) = ν0·[1 − (T/T_C)^α]^β, fit
  with the core `fit_parameter_model` (the panel's Model-Fit machinery),
  data-aware `suggest_trend_seeds`.

## Fitted values vs ground truth

### Per-run ν(T) (fitted here vs digitised Fig. 1(d), GT §11)

| T (K) | ν fitted (MHz) | ν GT (MHz) | Δ |
|---:|---:|---:|---:|
| 1.6 | 30.19 | 30.0 | +0.2 |
| 10.1 | 29.86 | 29.7 | +0.2 |
| 17.2 | 29.22 | 29.1 | +0.1 |
| 24.2 | 27.98 | 27.8 | +0.2 |
| 30.1 | 26.61 | 26.5 | +0.1 |
| 36.3 | 24.88 | 24.8 | +0.1 |
| 41.3 | 23.41 | 23.3 | +0.1 |
| 46.2 | 21.58 | 21.4 | +0.2 |
| 50.3 | 20.06 | 19.9 | +0.2 |
| 52.8 | 18.79 | 18.7 | +0.1 |
| 57.8 | 16.46 | 16.4 | +0.1 |
| 61.3 | 14.24 | 14.2 | 0.0 |
| 65.9 | 10.66 | 10.3 | +0.4 |
| 68.7 | 5.53 | 5.5 | 0.0 |

Every point is within the GT read-uncertainty (±0.3 MHz, ±0.5 near T_C).
ν(0) = 30.2 MHz reproduces the hard text target **ν(0) ≈ 30 MHz / B_µ(0) ≈ 0.22 T**
(GT §6). B_µ = ν/(γ_µ/2π) = 30.2/135.5 = 0.223 T ✓.

### OrderParameter trend fit (18 runs, full range)

| Quantity | Fitted here | GT target | Note |
|---|---|---|---|
| ν0 (= ν(0)) | **30.6 MHz** | ν(0) ≈ 30 MHz (§6) | ✓ |
| α | **1.51** | α ≈ 1.5 (§4/§6, full-range) | ✓ |
| β | **0.44** | β ≈ 0.4 (§4/§6, full-range phenomenological) | ✓ matches the *full-range* value |
| T_C | **69.9 K** | 69.05(1) K critical / 69 K literature (§6) | ~0.9 K high (see caveat) |

**Regime caveat (GT §4/§9 — the important one).** The `OrderParameter`
full-range fit reproduces the paper's own **full-range phenomenological**
numbers (α ≈ 1.5, β ≈ 0.4), which GROUND_TRUTH explicitly flags as **NOT** the
reliable critical exponent. The authoritative **β = 0.32(1), T_C = 69.05(1) K**
comes from a log–log fit restricted to the critical regime (small 1 − T/T_C,
Fig. 3), which this trending panel does not perform. So the headline render is
correct *as a full-range order-parameter fit* and matches the paper's Fig. 1(d)
red curve; it should be captioned as such, not as the critical β. The full-range
fit also runs T_C slightly high (69.9 vs 69.05) because ν has not quite reached
0 at the last included run (68.7 K, ν = 5.5 MHz) — expected behaviour for the
phenomenological fit, and consistent with the paper noting T_C is sensitive to
the fit regime.

## Feature-demonstration opportunities

- **PSI-bin loader** — real GPS 5-detector `.bin` with metadata (sample "EuO",
  instrument GPS@PSI, start/stop timestamps, `piM3.2` beamline). Good "we read
  PSI format natively" evidence.
- **Order-parameter trending** — the headline; a genuine critical-behaviour
  inference on real data (`corpus_euo_nu_t_trend`).
- **Warm-started batch fitting across T** — the ascending-T warm-start is a real
  workflow lesson worth a docs callout (single-frequency ZF rotation collapsing
  to zero at T_C).
- **Not captured but available:** (a) a **critical-regime β** render — restrict
  the trend to the near-T_C runs on a log–log axis to recover β = 0.32; the
  panel has a `log` toggle on both axes, so this is capturable and would let the
  docs show *both* regimes side by side (strongly recommended follow-up).
  (b) **Paramagnetic λ(T)** above T_C (GT §5 Q6, Fig. 4) from the TF-60 G runs.
  (c) **MaxEnt** frequency view as an alternative to FFT.

## Problems / quirks hit

- **FFT view needed conditioning.** The frequency panel computes an *averaged
  grouped* FFT (per-detector lifetime-corrected signals, averaged). Over all
  five GPS detectors this is dominated by each detector's slowly-varying
  baseline — low-frequency power that completely buries the small precession
  line (first capture showed a monotonic decay, no 30 MHz peak). Fix, all
  through real panel controls: restrict to the F/B pair (`set_group_enabled`),
  apply a **Lorentzian apodisation** (τ = 0.5 µs), and frame 20–42 MHz. The
  peak then stands ~2.4× over the in-window floor. Note the cleaner transform
  physically is the FFT of the F–B *asymmetry* (`fft_complex_asymmetry`, ~3.6×
  contrast), but the panel does not expose that as a display mode — a genuine
  UI gap worth logging.
- **Empty run titles.** PSI `.bin` headers carry no title for these runs, so the
  browser Title column is blank (temperatures/fields populate fine). Cosmetic.
- **α uncalibrated (≈28 % baseline).** With α = 1 the F–B asymmetry sits on a
  large constant offset; harmless (absorbed by the `Constant`) but it dominates
  the y-scale, so the ZF-fit and waterfall renders are zoomed in time and the
  fit plot is Y-framed to the oscillation window.
- **No numpy ≥ 2.3 issue observed** here, but the two fit-running scenarios are
  flagged `requires_fit = True` per house rules (real iminuit fits at capture).

## Top pick for the docs

`corpus_euo_nu_t_trend` — the ν(T) order-parameter curve is the headline result
and the render is clean and self-explanatory. Pair it with `corpus_euo_zf_fit`
(the below-T_C oscillation that *produces* each ν point, f = 30.18 MHz) as the
two-image story.

## Rebase onto main (PR #265) — 2026-07-16 — FFT workaround retired

- **`corpus_euo_fft` switched to the F−B asymmetry Fourier source (#265).**
  - *Before:* restricted the grouped average to the transverse F/B pair
    (`set_group_enabled({1,2 on; 3,4,5 off})`) + Lorentzian τ = 0.5 µs, to keep
    each detector's lifetime baseline from burying the small ~30 MHz line.
  - *After:* select the **Signal source → F−B asymmetry** radio
    (`_fourier_panel._signal_fb_radio`), which transforms the forward−backward
    difference directly, cancelling the common baseline. The line stands ~5×
    clearer (PR #265 measured peak/floor 7.27 vs 1.50 on this exact run 2960);
    the recapture shows a single clean peak at ν ≈ 30 MHz (peak ≈ 6.8, floor
    ≈ 1). In F−B mode the per-group include table is inert, so the old
    group-mask line is gone. Render is cleaner — retired the workaround.
