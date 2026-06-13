# Implementation options — workflow-visualisation

For each ADOPT/ADAPT item: the concrete seam in the **current** code (post-#53,
post-responsiveness, post-② RRF gating), the chosen shape, the additive panel-state
key, and the divergences from WiMDA. REJECT items (1, 5, 9) carry only their
deferred-nicety notes. Item 10 is hook-design only.

All file paths are under `src/asymmetry/`. No `schema_version` bumps; every new
persisted key is additive.

---

## 2. Data-only / ASCII export (ADAPT)

**Seam.** `gui/panels/plot_panel.py`:
- `export_plots_to_gle()` (~5371) — current entry; folder dialog → `.gleplot` →
  `_plot_export_payloads_on_axis` → schedules `_write_data_file`/`_write_fit_file`
  → GLE script + compile.
- `get_current_plot_export_data()` (~4780) — assembles the per-dataset payload
  (data, fit, components, grouping, run_metadata, histogram_info incl.
  `events_grouped`).
- `_write_data_file()` (~5063) — writes the `!`-header + `time asymmetry error`.
- `_write_fit_file()` (~5029) — writes the `.fit` sidecar.
- Button: `self._export_gle_btn` (~576).

**Chosen shape.**
1. Add a content enum threaded into the payload write: **data only / data + fit /
   fit only**. `data+fit` = current behaviour (`.dat` + `.fit`); `data only` skips
   the `.fit` write and the component shading payload; `fit only` writes a
   resampled-model `.fit`-style file and no `.dat` (matches WiMDA `stFit`, optional
   — lowest value, gate behind "if cheap").
2. Add `export_plotted_data_as_text(content, dest_dir, *, x_range=None)` that calls
   `get_current_plot_export_data()` and the **existing** `_write_data_file` /
   `_write_fit_file`, into a plain destination directory **without** the GLE script
   or compile. Factor the shared payload-assembly + file-write loop out of
   `export_plots_to_gle` so both paths call one helper (no duplication — addresses
   the umbrella's "no parallel exporter" rule directly).
3. **One** button, no clutter: convert `self._export_gle_btn` to a `QToolButton`
   with a popup menu — *"Export to GLE…"* (current) and *"Export plotted data
   (text)…"* (new). Or, equivalently, a `File → Export` submenu entry that calls the
   same method. Decision: QToolButton menu on the existing button (keeps it in the
   plot workspace where the data lives).
4. Optional x-range: today `.dat` writes every point ("fits/transforms/exports
   always use every point"). Offer an *optional* "limit to current x-range" checkbox
   in the text-export dialog that filters rows to `[x_min, x_max]` (the plot's
   spin-box values) — additive, defaulting **off** to preserve current behaviour.

**Divergences from WiMDA.** Comment char stays `!` (matches the existing `.dat`/GLE,
not `#`). The header is already a **superset** of WiMDA's (adds histogram bin
counts, good-range events, binning mode). Batch-over-selection deferred unless the
single-plot path makes it free.

**Persisted keys (additive):** last-used text-export content choice and the
limit-to-range flag, under an existing plot-state dict (no new schema).

---

## 3. Events columns (ADOPT)

**Seam.** `gui/panels/data_browser.py`:
- `_RUN_INFO_FIELD_LABELS` (~402) — label map for `run_info.*` columns.
- `_resolve_run_info_value()` (~1461) — computes each `run_info.*` value
  (`counts_mev = total/1e6`, `counts_per_detector`, `points`, `histograms`, …).
- `add_extra_column`/`remove_extra_column`/`get_extra_columns` (~1276–1301).
- `_open_header_context_menu()` (~2438) — currently offers only "Remove from Data
  Browser" for extra columns.
- Frame count: `core/transform/grouping.good_frames(grouping, default=1.0)` (~108).

**Chosen shape.**
1. Two new resolver branches + labels:
   - `run_info.good_events_mev` → "Good Events (MEv)": sum the grouped detectors'
     counts over `[first_good_bin, last_good_bin]` (clamp to bin count), ÷1e6.
     Reuse the good-range summation logic already proven in
     `plot_panel.get_current_plot_export_data` (`events_grouped`) — factor it to a
     small core helper so browser and export share one implementation (avoids a
     duplicate).
   - `run_info.events_per_frame` → "Events/frame": `good_events / good_frames`
     (frame count from the grouping helper); show "—" when `good_frames` ≤ 0 or
     absent (synthetic runs).
2. **Add-column surface:** extend `_open_header_context_menu` with an **"Add
   column…"** action (available on any header) opening a small checkable list of the
   available `run_info.*` fields not already shown; toggling calls
   `add_extra_column`/`remove_extra_column`. Persist the chosen set (additive key;
   the browser already restores extra columns from project/panel state).

**Divergences from WiMDA.** WiMDA conflates all-bins vs good-range under one "events"
number; Asymmetry keeps the existing all-bins `counts_mev` **and** adds the
good-range `good_events_mev` as a separate, unambiguous column. `events_per_frame`
uses `good_frames` (the deadtime normaliser) as the divisor — the Asymmetry-native
frame count — rather than WiMDA's `framestotal`.

---

## 4. B-from-log (ADOPT)

**Seam.** `gui/panels/data_browser.py`:
- `set_use_temperature_from_log()` (~1308) / `set_dataset_temperature_from_log()`
  (~1324) — global + per-run toggles.
- `_temperature_uses_log_for_display` (~1371) / `_temperature_from_log_for_display`
  (~1378) — display decision + event-weighted mean for combined runs.
- `_series_mean_for_field(dataset, field_key)` (~1411) — generic series mean.
- `_series_path_score(path, role, primary)` (~1433) — **temperature-only** scoring.
- B column populated from `metadata['field']` (~971); `_LOG_TEMPERATURE_FOREGROUND`
  tint (~57).

**Chosen shape.** Mirror temperature exactly for field:
1. `set_use_field_from_log(enabled)` (global) + `set_dataset_field_from_log(run,
   enabled)` (per-run override), with a `_field_from_log_overrides` dict.
2. A field-scoring branch — either parameterise `_series_path_score` by quantity or
   add `_series_path_score_field` — favouring roles/paths
   `sample_magnetic_field`/`magnet`/`b_field`/`field`/`b` with the same
   primary/role bonus structure temperature uses.
3. `_field_from_log_for_display(dataset)` reusing `_series_mean_for_field(dataset,
   "field")` and the **same** event-weighted-mean path for combined runs.
4. A log tint on the **B** cell when the value is log-sourced (reuse the existing
   foreground colour or a sibling constant).
5. A menu/context entry to toggle field-from-log alongside the temperature one
   (wherever temperature-from-log is exposed in `mainwindow`/browser context menu).

**Persisted keys (additive):** the global field-from-log flag + per-run overrides,
parallel to the temperature keys.

**Divergences from WiMDA.** WiMDA falls back to a field-coil channel when no field
channel is tagged; Asymmetry's scoring degrades to the best-matching series by the
same heuristic ladder temperature uses, returning `None` (→ header scalar) when none
qualifies — no silent coil substitution.

---

## 6. Log-count diagnostic (ADOPT)

**Seam.** `gui/panels/plot_panel.py`:
- `set_time_view_modes()` (~1418) + `_TIME_VIEW_FIELDS` (~90):
  `fb_asymmetry`/`groups`/`raw_counts`.
- `plot_grouped_time_domain_subplots()` (~2591) — renders the raw-counts stacked
  subplots; `_is_raw_counts_dataset` (~2092).
- `_on_time_view_mode_changed` (~1404) / `time_view_changed` signal.

**Chosen shape (recommended): a log-y toggle on the raw-counts view, not a 4th
mode.** Rationale: the data shown is identical; only the y-scale changes, so a mode
is the wrong abstraction and would proliferate view tokens (the reconciliation study
warns against concept proliferation). Add a small **"Log scale"** checkbox that is
visible only when the active time view is `raw_counts` (gate it exactly as the RRF
controls gate on `applies_to_current_view`, minus the Advanced flag — this is
mainstream, not specialist). On toggle, set the subplot y-axes to log₁₀ and redraw.

- **Non-positive bins:** masked (log undefined); record the count masked in a small
  annotation/status note rather than silently dropping.
- **Error bars:** kept where the log axis renders them sanely (matplotlib handles
  asymmetric log error bars; clamp the lower whisker to a floor).
- **Alternative weighed:** a distinct `log_counts` entry in `_TIME_VIEW_FIELDS`
  (closer to WiMDA's named mode). Rejected as primary because it duplicates the
  raw-counts render path for a pure axis change.

**Divergence from WiMDA.** log₁₀ on a log **axis** (counts stay readable) vs WiMDA's
`ln(count)` on a linear axis; non-positive bins masked vs WiMDA's `log(0) → −∞`;
errors retained vs dropped.

**Optional pairing (from §9):** a **bins vs µs** x-unit toggle is most useful on
this diagnostic (t0 lands on an integer bin); offer it only here if cheap, else
defer to the polish pass.

**Persisted key (additive):** the log-scale flag for the raw-counts view.

---

## 7. F,B balance overlay (ADAPT, borderline)

**Seam.** Grouped-counts data path feeding `plot_grouped_time_domain_subplots`; the
grouping carries `forward`/`backward` group ids and `alpha`. α estimators in
`gui/panels/grouping_dialog.py` + `core/.../asymmetry.py`; α-free fit
`core/fitting/count_domain.fit_fb_alpha`.

**Chosen shape.** A diagnostic rendering — **forward group vs α·(backward group)**
on **shared** axes (single subplot), reached from the grouping dialog or a plot
diagnostic action (not a new top-level view token, to avoid proliferation). The user
reads balance directly: under the correct α the two envelopes coincide; a vertical
gap is a mis-set α. Reuse the grouped-counts assembly; apply the current grouping's
α to the backward trace.

- Lifetime handling: overlay on the **raw-count** (un-lifetime-corrected) scale so
  the comparison is of the actual histograms, consistent with `_is_raw_counts_dataset`.
- This is **display-only**: no new α value, no promote path — so it does not touch
  the α collision set (F7).

**Divergence from WiMDA.** WiMDA overlays raw F vs raw B; Asymmetry overlays
**F vs α·B** because the diagnostic question is "do they balance *under this α*?",
which raw-vs-raw cannot answer.

**If REJECTED at the checkpoint:** ship nothing here; the estimators + α-free fit +
groups view stand. (Recommended fallback if Ben is unconvinced.)

---

## 8. Data-snapped cursor + readouts (ADAPT)

**Seam.** `gui/panels/plot_panel.py`:
- `_on_canvas_motion_notify` (~4363) emits `cursor_coords_changed(x, y)` on hover
  over the main axis.
- Cached arrays `_last_plot_time/_last_plot_asymmetry/_last_plot_error` (~366).
- `mainwindow._on_cursor_coords_changed` (~9020) → `_status_coords_label`.
- `core/transform/integral.integrate_curve(time, asym, err, t_min, t_max)` (~158)
  → `(mean, mean_error)`.

**Chosen shape.**
1. **Snap (carrier):** in `_on_canvas_motion_notify`, find the nearest cached
   `_last_plot_time` index to `event.xdata` and emit the *snapped* (x, y, err)
   instead of (or alongside) the free coordinate. Status bar shows `t, A±err`.
2. **S/N:** at the snapped point, append `S/N = |y/err|` to the status readout
   (cheap; guard err=0).
3. **Windowed average ± err:** a drag-select (or the existing fit-range handles in a
   "measure" mode) over `[t1, t2]` → call `integrate_curve` over that window → show
   `⟨A⟩ = mean ± err (n pts)`. Reuses the core helper exactly (no new math).
4. **Parabolic peak:** at the snapped index, fit `y = ax² + bx + c` to the point and
   its two neighbours; if `a < 0`, report vertex `x_pk = −b/2a`, `y_pk` to sub-bin
   precision; reject (no readout) if `a ≥ 0` or the vertex falls outside the
   neighbour span (matches WiMDA's validity guard). A tiny pure-numpy helper in
   `core/` (e.g. `core/analysis/peakfit.py` `parabolic_peak(x3, y3)`), unit-tested
   against the WiMDA formula.

**Subset is Ben's call** (re-asked at the checkpoint with the §8 workflow framing).
Recommendation: windowed-average + parabolic-peak + snap; S/N as a cheap extra.

**Divergence from WiMDA.** Snap-on-hover to the nearest cached point (Asymmetry's
mouse idiom) rather than WiMDA's keyboard-stepped cursor index. Readouts go to the
existing status bar (the inspector idiom), not a bespoke cursor panel.

**Persisted key:** none required (transient UI state); a "show snapped readout"
preference is optional and additive if wanted.

---

## 10. Live current-run monitoring (DESIGN-ONLY — hook shape, not implemented)

**Goal (deferred).** Surface the in-progress run ("run 0") from the DAE temp files
(freshest of `macq*.tmp`/`auto_*.tmp`), auto-refreshing as counts accumulate.

**Hook design (the study's only deliverable here).**
- A **refreshable dataset** abstraction: a `MuonDataset` whose backing source can be
  re-read on demand. Add a loader capability flag (e.g. `source_is_live: bool`) and
  a `reload_from_source()` that re-runs the loader against the same path and swaps
  histograms/metadata in place, emitting a "dataset updated" signal the browser/plot
  already understand (reuses the existing render-on-selection path).
- A **source resolver** that, given a data directory, picks the freshest live temp
  file by mtime — isolated in the loader registry (`core/io/`) so it is testable
  without a beamline (feed it a temp directory of fixture files).
- A **refresh cadence** owned by the GUI (a `QTimer` on the GUI thread that triggers
  an off-thread `reload_from_source` via the existing `TaskRunner`), never blocking
  the UI; a browser badge marks the live row.
- **No implementation now:** verification requires a live DAE. The contract above is
  recorded so the loader seam (`source_is_live` + `reload_from_source`) can be added
  cheaply and tested with fixtures when a beamline is available.
