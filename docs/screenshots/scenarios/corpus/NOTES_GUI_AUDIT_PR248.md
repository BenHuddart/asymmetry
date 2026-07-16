# GUI-integration audit — PR 248 (trend-panel axis transforms + multi-series overlay + Quadratic)

Reviewer question: *"whether this new functionality is cleanly integrated into the
GUI or whether this needs tightening up."*

Method: drove the **real** `FitParametersPanel` UI offscreen (`QT_QPA_PLATFORM=offscreen`)
on **real corpus data** — the Ca₃Co₂O₆ plateau λ(B) LF scan (per-run `Exponential+Constant`
fits) and the BiSCCO 400 G / 200 G σ(T) two-field scans (per-run TF-Gaussian fits),
reusing the batch-fit helpers in `plateau_ca3co2o6.py` / `cuprate_bscco.py`. Gestures
were exercised through their real signal paths (combo `activated`, `QTest` Shift+click on
the series pills, the public `select_series`, the real `AxisTransformDialog` /
`ModelFitDialog` shown with `show()`), never by poking private state. Evidence PNGs are in
`docs/_generated/gui_audit/` (gitignored under `_generated`).

## VERDICT: mostly clean — needs light tightening

The feature is well-integrated overall: it uses the shared `PanelSection`
(collapsible, collapsed-by-default, `settings_key`-persisted), a `TEXT_MUTED` header chip,
`tokens.ERROR/OK` status colours, `dialog_width()` metrics, and unit-aware axis labels — it
reads as part of the app, and it obeys `GUI_GUIDELINES.md`. Two items genuinely need
tightening before it reads as finished; both are exactly the "injection path works, real
user path confuses" trap. Everything else is polish or fine-as-is.

Notably, the corpus scenario `PlateauRedfieldScenario` (`plateau_ca3co2o6.py:481-482`) sets
`_model_fit_transform_sig["Lambda"] = panel._transform_signature()` *immediately after*
setting the transform, so the demo **never** exercises a stale fit — which is why finding A1
below slipped through. A real user fits first and re-linearises second, and hits it at once.

---

## (a) Genuine integration gaps — fix these

### A1. Stale fit gives no visible signal — the star and the plot disagree
Evidence: `09_stalefit_active_curve_drawn.png` → `10_stalefit_after_transform_curve_gone.png`.

Sequence (real user path): set Redfield transforms (Y→`1/y`, X→`x²`), run a `Linear`
Model Fit → the straight line draws and the button reads **`Model Fit*`** (09). Then change
one axis transform (X→`1/x`) from the combo. The fit's ranges now live in the old
coordinate, so `_overlay_suppressed_for_transform("Lambda")` returns `True` and
`_draw_model_overlay_mpl` correctly drops the curve — **but the button still reads
`Model Fit*`** and the provenance line is unchanged ("7/9 members in trend · 2 excluded")
(10). `_refresh_model_fit_button_labels` (panel line ~3580) only tests
`fit.active and _has_successful_fit_curve(fit)`; it never consults the transform signature
that `_overlay_suppressed_for_transform` uses. The docs promise *"Changing a transform marks
an existing trend fit for re-fit"* — the GUI draws no such mark. Result: starred button +
empty plot + no explanation.

Fix: when `_overlay_suppressed_for_transform(name)` is true, relabel the button (e.g.
`Model Fit⚠` with tooltip "Re-fit — transform changed since this fit") and/or draw a muted
"fit hidden — re-fit under current transform" note on the plot. `refit_active_model_fits`
already exists; surfacing a one-click re-fit here would close the loop.

### A2. Points dropped by a transform are dropped silently — no count (the LLZ ν case)
Evidence: `06_custom_dialog_log_undefined.png`.

`AxisTransform.apply` maps `1/0`, `ln(≤0)`, `√(<0)` to NaN, and the panel then drops those
points exactly like any NaN — but **surfaces no count anywhere**. The custom-transform
dialog only previews a *single* representative sample ("x = -3 → undefined here"); it does
not scan the column. The trend provenance line counts only `include_in_trend` exclusions and
quality flags, not transform-NaNs. Compare the custom-*column* x-axis path, which *does*
surface "⚠ N/total skipped (empty/non-numeric)" (panel line ~3190). So a user applying `ln`
to a ν(T) column that holds a couple of non-positive values loses them with no warning — the
exact LLZ case. Fix: count transform-undefined points at the data-assembly boundary and
surface "N points dropped (transform undefined)" in the provenance line (and ideally the
dialog once a real column is available to it).

---

## (b) Coherence / polish

- **B1. "Show table" under a transform is raw and unlabelled-in-context.** Evidence
  `07_show_table_under_transform.png`: headers stay `λ (µs⁻¹) (fit)` / `B (G)` with raw
  values while the plot shows `1/λ` vs `B²`. The only bridge is the section hint; the table
  dialog itself says nothing. A one-line note in the dialog when a transform is active would
  remove a real "why don't these match the plot?" moment.
- **B2. Table dialog header mislabeled.** The "Show table" dialog (window title "Fitted
  Variable Parameters") opens with a hardcoded body header **"Global fitting parameters"** /
  "None" for a plain single-series table (panel line ~5763). Pre-existing (not from PR 248)
  but visible in `07`. Confusing; should read "Fitted parameters" when there is no global fit.
- **B3. Section hint slightly misstates exports.** Hint: *"the table and exports stay raw."*
  Per commit 4a91420 exports keep raw columns **and append** the transformed ones (with
  `# X/Y transform` provenance). Reword to "…the table stays raw; exports keep raw columns and
  append the transformed ones."
- **B4. Quadratic ordering.** In the `Edit Model` basis library the models are alphabetical,
  so `Quadratic` lands after `PowerLawQuadBG`, far from `Linear`/`Cubic` (Cubic sits near the
  top). The registry places Quadratic right after Linear; the picker re-sorts. Consistent, but
  not a natural degree ordering. Low priority.
- **B5. Overlay gesture undiscoverable in the panel.** Evidence
  `11_overlay_two_series_legend.png`: Shift+click on a second pill overlays it, but the pills
  carry no tooltip/hint saying so — it lives only in the docs. Add a pill tooltip
  "Shift+click to overlay another series".
- **B6. GLE overlay warning is click-time only.** With two series selected, Export to GLE is
  enabled and only warns (active-series-only) *after* the user commits (modal). Acceptable; a
  proactive hint near the button when an overlay is active would be cleaner.

---

## (c) Fine as-is — confirmed working on the real user path

- Section collapsible, **collapsed by default**, state **round-trips via QSettings**
  `settings_key="parameters/sections/transforms"` (confirmed: a `setExpanded` in one process
  persisted into the next). `01_panel_transforms_collapsed.png`.
- **Header chip communicates active transforms** — `x² · 1/y` on both the collapsed and
  expanded header, the across-the-room signal. `01b_collapsed_header_with_chip.png`,
  `02_transforms_section_expanded.png`.
- **Unit-aware transformed axis labels on the plot**: `B² (G²)`, `1/λ (µs)`. `09`.
- **log-transform vs log-checkbox guard**: selecting `ln`/`log₁₀` on an axis disables that
  axis's `log` scale checkbox (both the global X box and the per-parameter Y box) with the
  tooltip "Values are already log-transformed — clear the … transform to use a log axis
  scale." `03_log_guard_lnX_lnY.png`.
- **Custom transform dialog**: live validation; invalid → OK disabled + red
  "Use only the variable 'x' (found: banana, q)"; valid → green "Preview: x = 1.5 → 666.7"
  (propagated ± shown when the sample has an error); non-positive `log` sample → "undefined
  here"; **default focus on the expression field**. `04`, `05`, `06`.
- **Model Fit dialog is transform-aware**: header reads "X variable: **B²**", data range and
  live preview are in transformed coordinates, the fitted `Linear` line is the Redfield line.
  `08_model_fit_dialog_catalogue.png`.
- **Overlay UX**: `(active)` flag on the active series in the legend, colour = series, twin
  axis suppressed for ≥2 series; both pills select via real Shift+click and `select_series`;
  the row button relabels to **"Global fit (2 groups)…"**; TSV + GLE both enabled.
  `11_overlay_two_series_legend.png`.
- **Both overlaid series share the transform** (`_plot_overlay` → `_plot_series_param` →
  `_series_y_arrays`, which applies `_y_transform` per series).
- **Transform scope is axis-global, not per-parameter** — coherent with an "axis transform"
  framing; switching Y parameter keeps the axis transform.
- **Quadratic present on every axis** (`scopes=("common","field","temperature")`), the plain
  `c₀+c₁x+c₂x²` parabola.
- **GUI_GUIDELINES compliance**: uppercase section header via `PanelSection`/
  `make_section_header` matching `PARAMETER SETTINGS` / `DERIVED PARAMETERS`; chip in
  `footer_font` + `TEXT_MUTED`; dialog width via `dialog_width(64)`; status colours via
  `tokens.ERROR`/`tokens.OK`; no literal geometry or hex.

## Evidence index (`docs/_generated/gui_audit/`)
| PNG | Shows |
|---|---|
| `01_panel_transforms_collapsed.png` | Section collapsed by default, in panel context |
| `01b_collapsed_header_with_chip.png` | Collapsed header with active-transform chip `x² · 1/y` |
| `02_transforms_section_expanded.png` | Expanded choosers (X: x², Y: 1/y) + hint + chip |
| `03_log_guard_lnX_lnY.png` | ln on both axes greys the log checkboxes |
| `04_custom_dialog_invalid.png` | Invalid expr → OK disabled + red message |
| `05_custom_dialog_valid_preview.png` | Valid `1000/x` → green preview, OK enabled |
| `06_custom_dialog_log_undefined.png` | `log(x)` on a non-positive sample → "undefined here" |
| `07_show_table_under_transform.png` | Table shows raw λ / B under an active transform (B1) |
| `08_model_fit_dialog_catalogue.png` | Model Fit dialog transform-aware (X variable: B²) |
| `09_stalefit_active_curve_drawn.png` | Redfield line drawn, button `Model Fit*` |
| `10_stalefit_after_transform_curve_gone.png` | Curve suppressed but button STILL `Model Fit*` (A1) |
| `11_overlay_two_series_legend.png` | Two σ(T) series, `(active)` legend flag, colour=series |
