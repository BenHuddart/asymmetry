# NOTES — Ionic motion in a solid electrolyte (Al-doped LLZ garnet)

Corpus example: `Nuclear magnetism and ionic motion/Ionic motion in a solid electrolyte`
Module: `ionic_motion_llz.py` · Ground truth: that folder's `GROUND_TRUTH.md`.
This is the corpus's flagship **global-fit** example: EMU (ISIS) longitudinal-field
muon decoupling on Al-Li₇La₃Zr₂O₁₂, 13 temperatures × (0/5/10 G) triplets, fitted
simultaneously with the **Keren** relaxation + flat background.

## Scenarios registered

| Name | What the render shows | Intended docs use | requires_fit |
|---|---|---|---|
| `corpus_llz_calibration` | Alpha-calibration dialog on the TF 20 G run 51315; α = 0.876(1) fitted, before/after asymmetry preview | Data-prep: fix α before the science fits (GT §4 calibration) | no |
| `corpus_llz_lf_triplet` | 160 K 0/5/10 G triplet overlaid, 0–12 µs; the three fields visibly separate (0 G relaxes fastest, 10 G least) | The raw signature of weak LF decoupling | no |
| `corpus_llz_global_setup` | Batch/global fit panel (1680×1000, fit dock widened to 560 px in `settle()`): full Keren+Constant formula, Δ/ν/amplitudes **Global**, B_L **From File**, bounds column fully visible, 12 µs window, guide seeds, all 3 runs selected — Run Batch Fit armed, batch-results box readable | THE parameter-tying showcase | no |
| `corpus_llz_global_result` | Converged triplet fit: fitted shared Δ=0.358 µs⁻¹, ν=0.267 MHz, χ²ᵣ=1.67, red Keren curve over the ZF data | Headline global-fit result | **yes** |
| `corpus_llz_nu_arrhenius` | **Reworked for PR 248.** ν(T) as a native **Arrhenius plot**: X→`reciprocal` (1/T), Y→**Custom** `log(x − 0.274324)` (baseline-subtracted ln), a `Linear` model fit on the activated branch (T≥264 K, 8 pts). Slope → Eₐ. The 5 plateau points are `include_in_trend=False`: 3 near-plateau sit ringed-grey low with huge error bars, 2 sub-baseline (ν<c) drop out as NaN. Chip "1/x · log(y - 0.274324)". | Trend → activation energy, **and the axis-transform + baseline interplay showcase** | **yes** |

## Run selection & workflow (GROUND_TRUTH.md refs)

- **Calibration (GT §4):** run 51315, TF 20 G at 300 K. The alpha dialog fits
  α = 0.876(1) from the F/B TF oscillation. Note: `classify_tf_calibration_run`
  returns *not a candidate* for this run (`field_gauss=None` — it reads a metadata
  key the EMU loader leaves unset), so it would not auto-highlight in the run
  dropdown; the scenario selects it explicitly. This is a minor classifier gap,
  not a fit problem — the estimate itself is correct.
- **Triplets (GT §3):** each temperature's triplet is `(zf, zf+1, zf+2)` =
  `(0 G, 5 G, 10 G)`; ZF run opens each set (51341, 51344, … 51377). The loader
  reads field (0/5/10/20) and setpoint T (160…404) straight from the `.nxs`
  headers — no manual entry needed, so B_L can be tied **From File** per run.
- **Model (GT §4, §10):** **Keren** exists natively
  (`asymmetry.core.fitting.models.keren`, params A, Δ, ν, B_L) — **no substitution
  needed**. Used as `CompositeModel(["Keren", "Constant"])`; the Constant is the
  flat background term the guide asks for. A_1 = sample-signal amplitude,
  A_bg = background amplitude.
- **Ties (GT §4):** A_1, Δ, ν, A_bg shared (**Global**) across the triplet;
  B_L fixed per run (**From File**, its 0/5/10 G set value). This is exactly the
  UI state the `corpus_llz_global_setup` render freezes.
- **Seeds @160 K (GT §4, starting values):** A_1=15 %, A_bg=5 %, Δ=0.3, ν=0.2.
  Used verbatim; the series is warm-started up in temperature (each triplet seeds
  from the previous fit's globals — the guide's "propagate up in T" workflow).
- **Fit window (GT §4):** t ≤ 12 µs (`t_max=12`, also shown in the panel's Fit
  Range). Past ~13 µs the F–B asymmetry diverges as counts vanish; all plots are
  clipped to 0–12 µs and lightly bunched (×6) for legibility.
- **Keren low-T ZF exclusion (GT §4):** the guide drops the 0 G run for Keren at
  low T because Keren fails for ZF + low ν. **Not applied here** — the fits
  converge cleanly with all three fields included (χ²ᵣ ≤ 1.7 at every T), so the
  ZF run is kept for a fuller triplet demo. Documented as an available option;
  a "ZF-excluded low-T" variant is a feature opportunity (see below).

## Fitted values vs targets

The guide supplies **no benchmark fitted numbers** (deliverable-only); the paper
(Amores et al., J. Mater. Chem. A 4, 1729 (2016) — the published dataset) is the
graded target (GT §10).

| Quantity | This reproduction | Target | Source |
|---|---|---|---|
| α (TF 20 G, run 51315) | 0.876(1) | — (just "fix α first") | GT §4 |
| Δ(160 K) | 0.358 µs⁻¹ | seed 0.3; expect smooth decrease | GT §4/§10 |
| ν(160 K) | 0.267 MHz | seed 0.2; expect plateau then rise | GT §4/§10 |
| Δ(T) trend | 0.358 → 0.27 µs⁻¹ (smooth decrease, min ~0.27 near 340 K) | "smooth decrease as Li⁺ mobilises" | GT §10 (paper) |
| ν(T) trend | flat ~0.27 MHz to ~250 K, activated rise to 1.10 MHz at 404 K | "plateau then exponential rise above ~290 K" | GT §10 (paper) |
| ν(T) plateau χ²ᵣ | 1.1–1.7 across the series | (no benchmark χ²) | — |
| **Eₐ (Li⁺, from ν(T))** — Custom `log(ν−c)` transform, T≥264 K | **0.222(8) eV** (χ²ᵣ = 1.26, c = 0.2743 MHz) | **0.19(1) eV** (µSR) | GT §10 (paper) |

**PR 248 transform + baseline interplay (the key finding this rework exposes).**
The task's literal instruction — Y→`log` preset (plain ln ν), X→`reciprocal`,
Linear on the activated branch — does **not** land in the 0.19–0.25 eV range,
because the activated ν(T) sits on a ν ≈ 0.274 MHz plateau baseline the bare log
cannot subtract. With the plain `ln x` preset the extracted Eₐ is both far too
small and **branch-dependent**:

| Transform / branch | Eₐ | note |
|---|---:|---|
| plain `ln ν`, T ≥ 264 K (8 pts) | **71 meV** | baseline flattens the slope |
| plain `ln ν`, T ≥ 284 K (7 pts) | 87 meV | " (branch-dependent!) |
| plain `ln ν`, T ≥ 304 K (6 pts) | 102 meV | " |
| **Custom `log(ν − 0.2743)`, T ≥ 264 K** | **222 meV** | branch-**in**sensitive (222→224 meV) |
| (old) Arrhenius+Constant model, all 13 T | 221(10) meV | the previous scenario's model |

So the axis-transform feature **needs a baseline-subtracted series to give a clean
Arrhenius line**, and the only way to express that with the current API is a
Custom expression `log(x − c)` with the numeric baseline typed in by hand
(0.2743 here). That recovers Eₐ = **0.222(8) eV** — matching the old
Arrhenius+Constant fit (0.221 eV) and ~15 % above the paper's 0.19(1) eV (the
same residual gap GT §9 attributes to the unpinned background model / MHz↔µs⁻¹ 2π
convention). This is the rework's headline **API-ergonomics finding**: the
transform layer has no "subtract fitted baseline before log" affordance, so the
constant is baked into the axis label (`log(y - 0.274324)` — self-documenting but
inelegant), and the near-plateau points' errors blow up as σ/(ν−c) (visible as the
±2 bars on the ringed points). The **physics is still reproduced** (flat-then-
activated ν(T), Li⁺ Eₐ of the right magnitude) — but only via the Custom route.
(The corpus folder's `ANALYSIS_asymmetry.md` quotes Eₐ ≈ 27 meV, but that is
program self-output, **not** ground truth per GT §7, and disagrees with the paper
by an order of magnitude — not used.)

## Feature-demonstration opportunities

- **Keren low-T ZF exclusion** — a `corpus_llz_..._zf_excluded` variant fitting
  only the 5/10 G pair at low T would illustrate the guide's Keren caveat (GT §4).
- **Dynamic-KT alternative** — the guide names dynamic Kubo–Toyabe *or* Keren;
  a side-by-side `DynamicGaussianKT + Constant` fit would show the model-choice
  comparison the guide invites (`DynamicGaussianKT` exists in the model set).
- **Δ(T) trend** — a companion to `corpus_llz_nu_arrhenius` trending Δ vs T would
  show the "smooth decrease as Li⁺ mobilises" the paper reports (data already
  computed in `_fit_nu_of_t`; would need Δ carried through).
- **Global Fit Wizard on real data** — the synthetic `global_fit_wizard_*`
  scenarios could be reproduced on this triplet to show automated role selection
  recommending Δ/ν Global, B_L Local.
- **Logbook view** — this example ships a rich `logbook.rtf` (40 runs, per-run T);
  the `logbook_view` feature would render well on it.

## Problems / quirks (honest)

- **F–B asymmetry divergence past ~13 µs.** Real EMU pulsed data: as forward/
  backward counts vanish the asymmetry ratio blows up to ±100 %, swamping the
  signal. Worked around by clipping every plot to the 0–12 µs fit window +
  bunch ×6. The alpha-calibration dialog draws its own full-range plot (can't be
  clipped from the scenario), so its tail past ~25 µs is visibly noisy — the TF
  oscillation and α correction over 0–20 µs remain clear.
- **`global_fit_completed` raises the Parameters dock.** Emitting the fit success
  auto-switches the right dock to the Parameters panel (a per-run B_L plot). For
  `corpus_llz_global_result` the Fit dock is re-raised afterwards
  (`window._dock_fit.raise_()`) so the Batch fit results with the fitted shared
  Δ/ν stay visible. Product behaviour, worked around in-scenario.
- **Fit-range spin reset by dataset selection.** Setting the 12 µs `t_max` before
  `_on_dataset_selected` gets overwritten by the auto data-range refresh; it must
  be set *after* selection. Minor UI-ordering quirk.
- **Inspector-dock resize clobbered by showEvent.** `MainWindow.showEvent`
  applies the adaptive default inspector width (`_apply_default_dock_widths`),
  so a `resizeDocks` on `_dock_fit` issued inside `build()` (pre-show) is
  silently overridden. To widen the fit dock for a capture it must be resized
  in `settle()` (post-show) — `corpus_llz_global_setup` does this so the
  parameter table's bounds column is not clipped. Gotcha for scenario authors.
- **B_L display value in the result table.** With B_L typed *File*, the
  classification table shows a single representative field (5) rather than the
  selected ZF run's 0 G — cosmetic; the fit itself reads the correct per-run
  field from each file (verified: ZF/5G/10G fitted with their own B_L).
- **TF classifier miss on run 51315** (see calibration note above) — a genuine
  small gap in `classify_tf_calibration_run` for EMU-loaded metadata.
- **Fit speed is a non-issue.** All 13 triplets fit in well under 1 s combined
  (joint global strategy), so the full series is used for the ν(T) trend — no
  subset needed. `corpus_llz_nu_arrhenius` runs 13 global fits + the trend fit in
  ~2 s at capture time.

## Top pick for the docs

**`corpus_llz_global_setup`** — the parameter-tying render (Keren+Constant, Δ/ν
Global, B_L From File, 12 µs window, guide seeds) is the single clearest picture
of what a global/"Multi Fit" is and is the feature this example demonstrates
better than any other. Pair it with **`corpus_llz_nu_arrhenius`** (the headline
ν(T) activated rise → Eₐ) for the two-image story: *how you tie it* and *what it
buys you*.

## PR 248 round 2 (re-test, 2026-07-12)

Re-tested commit 4a91420 on the real `corpus_llz_nu_arrhenius` panel
(13 global Keren fits + Custom-transform Linear trend fit).

- **CONFIRMED FIXED — unit-aware transformed labels.** Round 1's caveat was
  bare, unit-less transformed axes. The re-captured render now reads:
  - X (reciprocal on T): **`1/T (K⁻¹)`**
  - Y (Custom `log(x − 0.274324)` on ν): **`log([ν (MHz)] − 0.274324)`** — the
    whole dimensioned quantity is bracketed, so the MHz unit is carried into the
    Custom expression. Section chip: `1/x · log(y - 0.274324)`.
- Excluded-member overlay unchanged: "8/13 members in trend · 5 excluded
  (160 K, 180 K, 200 K, 220 K, 240 K)"; the 5 plateau points ring grey with the
  large propagated σ/(ν−c) bars.
- **Physics regression: none.** The activated-branch Linear fit is unchanged
  (E_a ≈ 0.222 eV; the Custom `log(ν−c)` baseline-subtract route is now also
  documented in `parameter_trending.rst`'s new Arrhenius-on-a-plateau note).
