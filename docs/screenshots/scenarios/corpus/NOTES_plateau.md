# NOTES — Dynamics in a magnetic plateau system (Ca₃Co₂O₆)

Corpus example: `Magnetism/Dynamics in a magnetic plateau system`
Module: `plateau_ca3co2o6.py` · Ground truth: that folder's `GROUND_TRUTH.md`.
Real HiFi HDF4-NeXus `.nxs` corpus files, runs **9023–9051**: a TF20 calibration
(9023, 300 K) plus a 15 K longitudinal-field decoupling scan 0 → 3.8 T
(9031–9051). Same data/campaign as Baker, Lord & Prabhakaran, *J. Phys.:
Condens. Matter* **23**, 306001 (2011), arXiv:1105.2200 (author-matched in the
NeXus headers, GT §11). Ca₃Co₂O₆ is a frustrated Ising-chain magnet with a
partial 1/3 magnetization plateau; the muon decouples static from **dynamic**
internal fields inside it.

## Scenarios registered

| name | render | intended docs use | requires_fit |
|---|---|---|---|
| `corpus_plateau_lf_overlay` | Raw LF spectra at ZF / 0.5 / 1.5 / 3.5 T (runs 9031/9039/9045/9050) overlaid, 0–12 µs. The 3.5 T trace is flat and high (~40 %); low-field traces relax fast and sit lower — the decoupling + asymmetry-recovery picture. | Data-handling / overlay feature; qualitative decoupling. | no |
| `corpus_plateau_exp_fit` | Converged `Exponential + Constant` fit on the 1.0 T run (9044): **A = 8.72 %, λ = 1.332 µs⁻¹, A_bg = 7.23 %, χ²ᵣ = 1.25**. Display bunched ×5, framed 0–10 µs so the decay + fit dominate. | Core analysis step — the per-run λ that feeds the trend. | **yes** |
| `corpus_plateau_lambda_field` | λ(µ₀H) trend in the parameter-trending panel (paper Fig. 2(a)): three-regime falloff, ~4.2 µs⁻¹ at 0.2 T → ~0.3 µs⁻¹ at 3.8 T. B on the panel's native "B (G)" axis. | Trend-building step; the λ(B) shape. | **yes** |
| `corpus_plateau_redfield` | **Headline.** λ⁻¹ vs µ₀²H² (paper Fig. 2(b)) **in the real trending panel** (PR 248 axis transforms): Y→`reciprocal` (1/λ), X→`square` (B²), a `Linear` model fit on the plateau; the 0.4 T and 3.8 T points are `include_in_trend=False` (ringed grey, off the line). Provenance chip "8/10 members in trend · 2 excluded (0.4 T, 3.8 T)". | Parameter-trending → Redfield linearity; the Δ/τ result. **The axis-transform showcase.** | **yes** |

## Run selection & workflow (GROUND_TRUTH refs)

- **Per-run model (GT §4 / eq.(3)).** `Exponential + Constant`
  = `A·exp(−λt) + A_bg`, fit 0–16 µs via the core `FitEngine` (the engine the
  single-fit panel drives). The additive `A_bg` absorbs the large field-growing
  baseline from positron spiralling (A_bg ≈ 7 % at 1 T rising to ~37 % at
  3.5 T; GT §4).
- **Warm-starting in field order (README lesson).** λ is seeded high (~9 µs⁻¹)
  and carried downward run-to-run. Essential on the low-field runs, which are
  degenerate (see Problems).
- **λ(B) trend runs.** 0.2 → 3.8 T (9036–9051). ZF (9031) and 0.1 T (9035)
  excluded: below 0.2 T the fast relaxation is unresolved at the ISIS pulse
  width and the asymmetry is suppressed, so the exponential is degenerate and λ
  unphysical (GT §4/§9). Runs 9024–9030 (thermally unsettled cooldown, GT §9)
  and 9032–9034 (redundant ZF) not used.
- **Redfield plateau fit (GT §4/§6).** λ⁻¹ = slope·(µ₀H)² + intercept,
  weighted, over **0.5–3.6 T** (runs 9039–9050). Solve
  `slope = τ/(2Δ²)`, `intercept = 1/(2γ_µ²Δ²τ)` with γ_µ = 2π×135.5 MHz/T for
  (Δ, τ). The 3.8 T point (9051, saturated) is plotted but excluded from the fit.

## Fitted values vs ground truth

### Per-run λ(µ₀H) (fitted here vs digitised Fig. 2(a), GT §11)

| µ₀H (T) | λ fitted (µs⁻¹) | λ GT (MHz) | | µ₀H (T) | λ fitted (µs⁻¹) | λ GT (MHz) |
|---:|---:|---:|---|---:|---:|---:|
| 0.2 | 4.17 | ~3.4 | | 1.0 | 1.33 | ~1.35 |
| 0.3 | 2.66 | ~2.4 | | 1.5 | 0.90 | ~0.85 |
| 0.4 | 2.50 | ~2.3 | | 2.0 | 0.69 | ~0.65 |
| 0.5 | 2.05 | ~1.95 | | 2.5 | 0.47 | ~0.5 |
| 0.6 | 1.86 | — | | 2.9 | 0.33 | ~0.4 |
| 0.7 | 1.78 | — | | 3.2 | 0.24 | ~0.3 |
| 0.8 | 1.62 | — | | 3.5 | 0.28 | ~0.3 |
| 0.9 | 1.44 | — | | 3.8 | 0.28 | ~0.3 |

Plateau points (0.5–3.8 T) sit right on the digitised targets; the 0.2–0.4 T
points run slightly high (steep-drop regime, large read-uncertainty in the GT).

### Redfield line → Δ, τ (headline)

Values below are from the **PR 248 transform path** (`AxisTransform.preset` +
the core `fit_parameter_model` trend minimiser on the transformed 1/λ, B²
arrays — exactly what the panel's Model-Fit dialog does once the axes are set).
The old standalone-matplotlib figure gave 0.278 / 0.434 / 41.1 mT / 940 ps; the
transform path agrees within rounding (they differ only in the minimiser:
iminuit trend fit vs a hand-rolled weighted normal equation).

| Quantity | Old (matplotlib) | New (transform panel) | GT target (paper §3/§6) | Note |
|---|---:|---:|---:|---|
| slope d(λ⁻¹)/d(µ₀²H²) | 0.278(9) | **0.276(9) µs T⁻²** | ≈ 0.267 (GT §11) | ✓ |
| intercept (H→0) | 0.434(19) | **0.442(27) µs** | ≈ 0.48 (GT §11) | ✓ (~8 % low) |
| **Δ (= σ_int)** | 41.1 mT | **41.0 mT** | **40.6(3) mT** | ✓ within ~0.4 mT |
| **τ** | 940 ps | **929 ps** | **880(30) ps** | ✓ ~6 % high (~2σ) |
| χ²ᵣ (Linear on plateau) | — | 1.84 | — | 3.2 T point scatters |

Δ lands essentially on the paper value; τ is ~6 % high, traceable to the two
scattered high-field points (the paper notes these "scatter around/below" the
line, GT §11) pulling the slope up slightly. The fit is, as the paper states,
insensitive to small changes in the window; dropping 3.2 T brings τ closer to
880 ps but the full 0.5–3.6 T window is kept for faithfulness to GT §4.

**Transform usage (PR 248).** Field is stored in the row dicts in **tesla** (not
the panel's native gauss) so X→`square` yields B² in T² and the Redfield slope
comes out in µs T⁻². Y→`reciprocal` propagates σ(1/λ)=σ(λ)/λ². The `Linear` fit
runs on the transformed plateau; the panel samples its overlay in the B² domain
(0.25–12.25 T²), matching the scatter. The two excluded points ride off the
line: 0.4 T near the intercept, 3.8 T (saturated) below the extrapolation at
high B². **Axis-unit caveat:** the transformed plot labels read `B²` and `1/λ`
with **no units** (the transform strips the base unit before wrapping the
symbol), so the reader cannot tell B² is in T² or 1/λ in µs — see PR-248 VERDICT.

## Feature-demonstration opportunities

- **HiFi HDF4-NeXus loader** — real ISIS `.nxs` (HDF4-based) read natively;
  field/temperature metadata populate the browser (`Ca3Co2O6 T=15 F=…`). Good
  "we read HiFi format" evidence.
- **Redfield linearisation** — the λ⁻¹-vs-µ₀²H² transform is the distinctive
  physics of this example. **PR 248 delivers exactly this "derived-axis" mode**
  (the earlier feature request), so the headline is now the *real trending panel*
  (Y→reciprocal, X→square, Linear fit) rather than a standalone matplotlib
  figure. This is the single best corpus demonstration of the axis-transform
  feature: a genuine three-regime µSR field scan linearised in the GUI.
- **Not captured but available:** (a) the **TF20 α-calibration** on run 9023
  (the inline α-calibration pattern from `ionic_motion_llz.py`) — GT §4 Q1
  data-prep step. (b) A **P_z(t) waterfall** across the full scan. (c) The
  **sub-plateau regime** (Δ ≃ 30 mT, τ ≃ 5 ns, GT §6) if the <0.5 T fits could
  be stabilised (they can't with a single exponential — see below).

## Problems / quirks hit

- **Low-field (<0.2 T) fits are degenerate.** At ZF and 0.1 T the true λ (~7.7,
  ~6 µs⁻¹) relaxes the signal within the first ~0.1 µs, unresolved at the ISIS
  pulse width; the observed asymmetry is also suppressed (GT §4). The remaining
  data is a flat baseline, so `A·exp(−λt)+bg` is under-determined — λ collapses
  to a spurious near-zero minimum (0.1 T fitted 0.005 µs⁻¹ from a high seed).
  These points are physically meaningless and are **excluded from the λ(B)
  trend**; this is exactly the GT §9 caveat, not a program bug. Only Δ and τ
  from the plateau are grading targets anyway (GT §9).
- **Baseline grows strongly with field.** A_bg rises from ~7 % (1 T) to ~37 %
  (3.5 T) via positron spiralling (GT §4), so the raw-spectra overlay separates
  the traces vertically by baseline as much as by relaxation. Framed Y 0–48 % to
  hold all four; the decoupling (flattening) still reads clearly.
- **Late-time F–B noise.** Past ~8 µs the F–B asymmetry fans out as counts
  vanish (±100 % bins). The fit runs on the unbinned 0–16 µs data (χ²ᵣ = 1.25);
  the `corpus_plateau_exp_fit` *display* is bunched ×5 and framed 0–10 µs so the
  decay is clean. The app tags χ²ᵣ = 1.25 as "poor" (its threshold wording) —
  the fit is in fact good.
- **Trend axis is gauss (native).** `corpus_plateau_lambda_field` stores field in
  gauss (native "B (G)" axis); `corpus_plateau_redfield` stores it in **tesla** so
  the X→`square` transform gives B² in T² for the physics. Because a preset can
  only square whatever number is in the column, getting T² from the native gauss
  column would need a **Custom** `(x/10000)**2` transform — an ergonomics gap (see
  VERDICT: transforms carry no unit awareness).
- **PR 248 — transformed-axis labels drop units.** The plot shows `B²` and `1/λ`
  with **no units**; a reader cannot tell B² is T² or 1/λ is µs. Reproduction:
  set Y→reciprocal, X→square; the axis titles are bare `1/λ`, `B²`. (Root cause:
  `_transformed_*_axis_label` calls `_axis_symbol()` to strip the `(unit)` before
  `AxisTransform.describe()` wraps the symbol — so the unit is discarded.)
- **PR 248 — GLE/TSV export ignores the active transform (significant).** With
  Y→reciprocal, X→square active, **Export TSV** and **Export to GLE** stay enabled
  and silently write the **raw** λ and raw field (headers still `Lambda (µs⁻¹)`,
  `B (G)`), not 1/λ vs B². Worse for GLE: the model-overlay curve *is* sampled in
  the transformed B² domain (0.25–12.25 T²) while the data `errorbar_from_file`
  uses the raw field column (`_gle_x_column("field")→col 2`), so the exported
  figure draws the Linear fit on a different x-scale than the points — a broken
  plot. See VERDICT for the reproduction.

## Top pick for the docs

`corpus_plateau_redfield` — the λ⁻¹-vs-µ₀²H² Redfield line, now the **real
trending panel** (PR 248 axis transforms), is the headline result and reproduces
the paper's Δ = 40.6 mT / τ = 880 ps within uncertainty (41.0 mT / 929 ps). Pair
it with `corpus_plateau_exp_fit` (the per-run exponential that produces each λ
point) as the two-image story; `corpus_plateau_lambda_field` (Fig. 2(a)) is the
natural bridge between them.

## PR 248 round 2 (re-test, 2026-07-12) — exports fixed, but a NEW unit-label defect

Re-tested commit 4a91420 on the real `corpus_plateau_redfield` panel (per-run
exponential fits + the injected Linear Redfield fit, Y→reciprocal, X→square).

- **CONFIRMED FIXED — TSV/GLE now export the transformed coordinate.** The
  round-1 merge-blocker ("export ignores the active transform") is closed.
  Verified by reading the files:
  - `redfield.tsv` — provenance comments `# X transform: x**2` /
    `# Y transform: 1/x`; **raw columns retained** (`B (G)`, `Lambda (µs⁻¹)`, …)
    and transformed columns **appended**: `B² (G²)`, `1/λ (µs)`, `err_1/λ (µs)`.
    Numeric spot-check row 0: raw B=0.4 → B²=0.16 (exact); raw λ=2.49752 →
    1/λ=0.400398 (=1/2.49752 ✓); err_1/λ=0.024334 (=σ_λ/λ² ✓).
  - `redfield.dat` — same provenance, raw columns c1–c7 unchanged, transformed
    appended c8=`B² (G²)`, c9=`1/λ (µs)`, c10=err. `_gle_effective_x_column`
    → col 8 (transformed) and `_gle_transformed_columns_for_param("Lambda")`
    → (9,10), so the GLE **plot + errorbar read the transformed columns**.
- **CONFIRMED FIXED — GLE fit curve shares the data coordinate.** The sampled
  Linear fit curve spans xs ∈ [0.25, 12.25] (= (0.5 T)²…(3.5 T)², the plateau
  fit window) and the data B² range is [0.16, 14.44] — both in B². Round-1's
  "model curve in B² vs data errorbar in raw field" broken-plot is fixed.
- **CONFIRMED FIXED — stale-transform guard.** After changing X (square→
  reciprocal) post-fit, `_overlay_suppressed_for_transform("Lambda")` → True and
  `_iter_active_fit_ranges("field")` → `[]` (the `.fit` sidecar is *not*
  emitted); restoring square brings the curve back. So the export can no longer
  emit a fit curve the screen hides.
- **NEW ISSUE (label correctness regression on the headline figure).** Round 1's
  caveat was "transformed labels drop units" (bare `B²`, `1/λ` — incomplete but
  not *wrong*). Round 2 makes labels unit-aware — but this scenario stores field
  in **tesla** in the panel's gauss column (to get B² in T² for the physics),
  and the panel hard-codes the field unit as `G`. So the X axis (and the TSV/GLE
  headers) now read **`B² (G²)`** while the data is really **T²** — an
  *affirmatively incorrect* unit on the shipped headline image and its exports.
  Y is correct (`1/λ (µs)`). Root cause is `_x_axis_display_label("field")`
  always returning "B (G)"; the scenario's tesla-in-gauss hack is now surfaced
  as a wrong unit rather than hidden. Options: (a) reframe the caption to state
  "B² is in T²" explicitly; (b) upstream, teach the panel a per-series field
  unit. Not a fit/physics problem — Δ/τ are unchanged.
- **Physics regression: none.** Δ = 41.0 mT, τ = 929 ps (NOTES table 41.0/929;
  paper 40.6 mT/880 ps). Unchanged.

## Rebase onto main (PR #262) — 2026-07-16 — tesla-storage workaround retired

- **`corpus_plateau_redfield` now stores/plots field in GAUSS; the `B² (G²)`
  label is correct.**
  - *Before:* field was stored in **tesla** in the row dicts, so the X→square
    transform yielded B² in T² while the panel's axis label (derived from the
    registered gauss field unit) read `B² (G²)` — a mislabel.
  - *After:* field is stored in gauss (`SCAN_FIELDS_T × 1e4`; #262 normalises
    NeXus field to gauss natively — verified run 9044 → 10000 G), so the axis
    reads correctly as **`B² (G²)`** with a 1e9 scale. `_build_redfield_linear_fit`
    fits 1/λ vs B² in G² and `_redfield_from_line` uses γ_µ in µs⁻¹ G⁻¹.
  - **Δ/τ unchanged (invariant held):** recomputed **Δ = 41.01 mT, τ = 929.0 ps**
    (was Δ ≈ 41 mT, τ ≈ 0.93 ns) — the fit units cancel out of the physics.
