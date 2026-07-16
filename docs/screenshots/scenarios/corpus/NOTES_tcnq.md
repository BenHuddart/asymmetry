# NOTES — ALC resonance in TCNQ (Chemistry)

Scenario module: `tcnq_alc.py`
Example: `Chemistry/ALC resonance in TCNQ`
Corpus data: real EMU `.nxs` (ISIS NeXus HDF4) runs `emu00019485–19612`
(128 runs). Four longitudinal-field ALC scans, each 31 runs stepped
2000 → 5000 G / 100 G: 350 K (19489–19519), 100 K (19520–19550),
10 K (19551–19581), 50 K (19582–19612).
Spec: `GROUND_TRUTH.md` (guide `ALC in TCNQ 2026.docx`).

## Scenarios registered

| name | render | intended docs use |
|---|---|---|
| `corpus_tcnq_integral_scan` | 350 K scan (31 runs) built as **integral asymmetry vs field** in the integral-scan view; the D1 ALC dip near ~3.1 kG stands clear of a flat ~25 % baseline. No fit — Baseline/Peaks panel awaiting input. | "Field-domain / integral-scan (ALC) mode" data-handling step; the model-free observable. |
| `corpus_tcnq_alc_fit` | **Converged** Cubic-background + Lorentzian fit on the 350 K scan, driven through the real Baseline→Peaks buttons. Green centre line + panel read-out: **B₀ = 3104 ± 0.88 G, FWHM = 325.2 G, amp = −6.01 %**. | Core ALC analysis step; the headline resonance-field result → A_µ. `requires_fit`. |
| `corpus_tcnq_temperature` | All four T scans (350/100/50/10 K) overlaid, each with its fitted Lorentzian+cubic curve; legend carries FWHM + dip depth. The dip **deepens and narrows as T rises** (motional narrowing). | Distinctive multi-scan overlay; the qualitative dynamics story (GT Q2/Q5). `requires_fit`. |
| `corpus_tcnq_dmu_trend` | **Headline deliverable.** Two stacked panels: A_µ(T) (≈82–85 MHz, near the ≈80 MHz target) and D_µ(T)=FWHM/68 from the four fitted resonances; D_µ falls as T rises. | Hyperfine-extraction workflow (GT §4D); the A_µ(T)/D_µ(T) deliverable. `requires_fit`. |

## Run selection & workflow (GROUND_TRUTH refs)

- **Observable (GT §1/§4B).** ALC is a **field-domain** technique: each run is
  reduced to one time-integral asymmetry number and the scan is that number vs
  longitudinal field. Built with the core `build_field_scan(method="integral",
  order_key="field")` — the exact reduction the GUI's `_on_scan_requested`
  runs. The GUI scenarios drive it through the real `integral_scan` view
  (select all runs → `build_requested`).
- **350 K scan runs (GT §3/§9).** 19489–19519 (2000–5000 G). Run 19488 (the
  duplicate 350 K/2000 G setup point) is **excluded** per the guide's explicit
  batch-add range 19489–19519.
- **Integration window.** Left at the loader default (full good-time window,
  ~0.1–31.75 µs, shown in the integration strip). The `"integral"` method sums
  native-bin counts and is bunching-invariant, so the guide's "bunch 500 → 8 µs
  bins, ~4 µs average" instruction (a WiMDA display convenience) does not change
  the reduced value; the scan is byte-identical whatever the display bunching.
- **Fit model (GT §4C).** **Cubic** background over the two non-resonant edge
  windows `(2000–2600 G)` and `(3400–5000 G)`, then a **Lorentzian** (`LorentzianLCR`,
  `f/(1+((B−B0)/Bwid)²)`, FWHM = 2·Bwid) on the baseline-corrected scan. Uses the
  core `fit_scan_baseline` / `fit_scan_model` the Baseline/Peaks buttons call.
  See "Problems" for the guide's "2 Lorentzians" wording.
- **Hyperfine inversion (GT §4D/§6).** A_µ[MHz] = B_res[G]/36.71 (from
  B_res = (A_µ/2)(γ_µ⁻¹−γ_e⁻¹)); D_µ[MHz] = FWHM[G]/68.

## Fitted values vs ground truth

Cubic + single Lorentzian per scan (module `_fit_alc`; the 350 K row is the one
shown live in `corpus_tcnq_alc_fit`):

| T (K) | B_res (G) | FWHM (G) | dip depth (%) | A_µ (MHz) | D_µ (MHz) |
|---:|---:|---:|---:|---:|---:|
| 350 | 3104.3 ± 0.9 | 325.2 | 6.01 | **84.6** | 4.78 |
| 100 | 3047.6 ± 2.0 | 396.1 | 2.78 | 83.0 | 5.83 |
| 50  | 3036.2 ± 2.6 | 448.0 | 2.44 | 82.7 | 6.59 |
| 10  | 3006.7 ± 2.7 | 430.6 | 2.22 | 81.9 | 6.33 |

- **B_res / A_µ vs target (GT §6).** Guide target A_µ ≈ 80 MHz ↔ B_res ≈ 2937 G.
  The fitted resonance sits a bit higher (3006–3104 G → A_µ ≈ 82–85 MHz),
  within ~2–6 % of the ≈80 MHz *target*. GT §6/§9 are explicit that 2937 G is
  only the "expected neighbourhood" and the precise B_res/A_µ are **student
  deliverables read off the fit**, not a tabulated answer key — so this is a
  legitimate result, not a miss. The coldest scan (10 K, 81.9 MHz) lands closest
  to the nominal 80 MHz.
- **Temperature story (GT Q2/Q5).** As T rises the dip **deepens** (2.2 → 6.0 %)
  and **narrows** (FWHM 431 → 325 G), i.e. D_µ falls from ~6.3 to ~4.8 MHz:
  motional averaging of the dipolar hyperfine coupling as molecular motion grows
  — exactly the dynamics the guide asks about. The trend is cleanest 50 → 350 K;
  the 10 K point breaks strict monotonicity (D_µ 6.33 < 6.59 at 50 K), consistent
  with the first 10 K runs (19551/19552) "still settling" per GT §3.

## Feature-demonstration opportunities

- **Captured:** integral-scan build, Cubic+Lorentzian ALC fit with readable
  B_res, multi-scan overlay, parameter-vs-T trend. Together they exercise the
  full field-domain surface (`integral_scan` view, Baseline/Peaks analysis
  panel, integration-window strip).
- **Not captured but available:**
  - *Time-resolved view (GT §4A).* Bunch-factor-3 time spectra per run, with
    on-resonance **oscillations** and a relaxation rate that peaks at the ALC
    field (GT Q1–Q3). A "relaxation-rate vs field" fit-table trend would be a
    strong companion but needs a per-run time-domain relaxation fit across 31
    runs — heavier, and off the field-domain headline.
  - *Scan-point exclusion.* The `alc_scan_exclusion` synthetic already shows the
    greyed-point mechanism; no genuinely bad point stands out in the 350 K scan
    here, so a real-data exclusion would be contrived. Skipped deliberately.
  - *dA/dB derivative view* (the scan view's built-in toggle) — the WiMDA
    derivative presentation of the same dip.

## Problems / caveats (honest)

- **"2 Lorentzians" wording (GT §9).** The guide names a *two*-Lorentzian +
  cubic model, but only **one** genuine D1 resonance exists in this window; the
  second Lorentzian is a template default. Fitting a single Lorentzian is the
  physically correct choice and gives a clean, well-determined B_res
  (±0.9 G at 350 K); a second line would be unconstrained. Noted rather than
  forced.
- **Lorentzian slightly over-peaks the 350 K dip.** The fitted curve bottoms
  ~0.6 % below the deepest data point at 3100 G (visible in both fit renders):
  the true lineshape is a touch rounder than a pure Lorentzian. Immaterial to
  B_res; it would matter for a precise depth/area, which is not a deliverable.
- **Determinism — twin-axis edge artifact.** `corpus_tcnq_dmu_trend` originally
  used a twin-y axis; under offscreen Qt the twinned right spine settled the
  **last canvas pixel column** two ways ~1/3 of runs (`QWidget.grab`
  first-paint artifact, same failure the `alc_scan_exclusion` scenario notes).
  Fixed two ways: rebuilt as two stacked single-axis subplots, and saved from
  the figure's **Agg RGBA buffer** (byte-deterministic) instead of the Qt grab.
  All four PNGs are now byte-identical across repeated full-set runs. The two
  MainWindow scenarios and the single-axis temperature overlay are stable under
  the standard `widget.grab()`.
- **No answer key (GT §9).** All fitted B_res/FWHM/A_µ/D_µ are deliverables, not
  tabulated ground truth; grading is against the ≈80 MHz target and the guide
  formulae only.
- **Grouping / α / t0 not prescribed (GT §9).** The loader defaults are used
  throughout; the guide gives no grouping/α/deadtime recipe for this example.
