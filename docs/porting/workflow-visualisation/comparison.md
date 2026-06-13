# Comparison & verdicts — workflow-visualisation

The centrepiece of the study. Each item is weighed against the **established
Asymmetry workflow** (browser-centric; view selector; inspector deck;
Options → Advanced gate), not against WiMDA's single-run main window. WiMDA source
citations are oracle-only (`$WIMDA_SRC`, GPL — behaviour read, never copied).

## Verdict table

| # | Item | WiMDA mechanism | Asymmetry-native alternative considered | Verdict | Rationale (one line) |
|---|------|-----------------|------------------------------------------|---------|----------------------|
| 1 | Run stepping / pattern walker | `WiMDA_Main.pas:248–303,1301–1345` `GetPrefixSuffix`/`PrevRun`/`NextRun`: parse prefix+padded#+suffix, file-exists check, step without loading | Browser already lists the directory; native ↑/↓ on the `ExtendedSelection` table moves the row → `dataset_selected` → `_render_current_selection_plot()` auto-loads | **REJECT** | The browser is a superset of the walker; keyboard next/prev + auto-load already exists |
| 2 | ASCII / data-only export | `WiMDA_Main.pas:1414–1869` `SaveAs` (stData/stBoth/stFit, x-range, `!`-header); batch `Plot.pas:3082` | GLE export already writes `.dat` (`!`-header + columns) + `.fit`, but only inside a `.gleplot` folder with a compiled script | **ADAPT** | Strengthen the GLE path: data-only / data+fit / fit-only text export reusing `_write_data_file`, surfaced on the **existing** export button (no second button) |
| 3 | Events columns | `LogbookUnit.pas:595–626`: good-range Σcounts/1e6 (MEv); Σcounts/`framestotal` (ev/frame) | `run_info.counts_mev` (all-bins) + `counts_per_detector` exist as removable extra columns; **no UI to add columns**; no good-range or per-frame column | **ADOPT** | Add good-range MEv + events/frame resolvers and a user surface to add columns — the cheap run-quality win, made real |
| 4 | B-from-log | `WiMDA_Main.pas:729–733` `Bfromaveinblog`: field = avg of the B log channel (coil fallback) | Temperature-from-log is fully built (`_series_mean_for_field`, per-run override, log tint); **no field analogue** — field is always `metadata['field']` | **ADOPT** | Mirror the temperature-from-log machinery for field; a clean native pattern already exists to copy |
| 5 | Deadtime-file auto-discovery + staleness | `WiMDA_Main:925–975`: discover a `.cal` in the data dir; warn if its date predates the run | estimate/calibrate/promote family (`calibrate_deadtime_from_histograms` → `promote_deadtime_to_grouping`); grouping persists in `.asymp`; `parse_deadtime_calibration_text` exists but unused | **REJECT** | Calibrating from *this run* is strictly fresher than a dated external file; project persistence replaces the file store. (Optional deferred nicety: wire the existing parser to a manual import — not auto-discovery) |
| 6 | Log-count display mode | `Plot.pas:1542–1548` `GetDataGroup` "Logp": `ln(count/nbin)`, no error bars | Raw-counts view shipped (#53) via `set_time_view_modes` + `plot_grouped_time_domain_subplots` | **ADOPT** | Standard t0/background/deadtime diagnostic: a pure decay is a straight line on a log axis, so deviations pop out. Add a log-y rendering of the raw-counts view |
| 7 | F,B **balance** overlay | `Plot.pas:2701–2711` `FBOverlay`: overlay forward vs backward group histograms; visual α check | Three α estimators (diamagnetic/general/ratio) + α-free count fit; groups view shows each group in its **own** subplot, never F vs α·B on shared axes | **ADAPT (borderline)** | The estimators give a *number*; the missing piece is the *eyeball* — F and α·B on shared axes — which matters when an estimator silently clamps (ledger #6). Display-only; may drop to REJECT on Ben's call |
| 8 | Data-snapped cursor + readouts | `Plot.pas:1159–1228,2962–3006`: snap to point; S/N=\|y/err\|; 3-pt parabola vertex; windowed mean±err | `cursor_coords_changed` → `_status_coords_label` gives free (x,y); cached `_last_plot_*` arrays; `integrate_curve` gives windowed mean±err | **ADAPT** | Build on the existing cursor signal + status bar. Subset is Ben's call — see §8 for the workflow this belongs to |
| 9 | Cosmetic basket | error-bar toggle, marker styles, tick spacing, ns/bins x-units | The plot panel's existing styling | **REJECT here** | UI-polish-pass territory; no workflow reason found (the one near-miss, bins x-units, is noted under §6 as an optional pairing) |
| 10 | Live current-run monitoring | `muondata.pas:1376` "run 0": freshest `macq*.tmp`/`auto_*.tmp`, auto-refresh | none | **DESIGN-ONLY** | Beamline access required to test (decision 2026-06-10); design the refreshable-loader hook, implement later |

Net: **3 ADOPT** (3, 4, 6), **3 ADAPT** (2, 7-borderline, 8), **3 REJECT** (1, 5,
9), **1 design-only** (10). Three rejects is a healthy outcome — it is the thesis
working: the browser and the existing analysis surfaces already absorb a third of
WiMDA's "instrument feel" basket.

---

## Per-item reasoning

### 1. Run stepping — REJECT (Ben asked me to try to refute it; here is the case)

**What WiMDA does.** WiMDA has no persistent run list. To move between runs you
press **Prev/Next**, which (`WiMDA_Main.pas:248–303,1301–1345`) splits the current
filename into `prefix + zero-padded number + suffix + ext`, increments/decrements
the number, checks the file exists on disk, and (separately, on demand) loads it.
Stepping *is* WiMDA's navigation because there is nothing else.

**What Asymmetry already does.** The Data Browser lists the whole directory as a
sortable, multi-select table (`ExtendedSelection`). Native ↑/↓ keys move the
current row; a single-row selection emits `dataset_selected(run_number)`, which
`mainwindow._on_dataset_selected` turns into `_render_current_selection_plot()`.
**So the keyboard next/prev-with-auto-load that WiMDA's walker delivers is already
present** — it is just the table widget doing its job, against runs that are
already parsed and loaded. The browser is a *superset*: it shows T, B, title and
events at a glance, sorts and filters, and multi-selects for overlay — none of
which a one-at-a-time walker can do.

**Is there any residual benefit?** Two candidates, both weak:

- *Pattern-walk to a run not yet in the table.* In Asymmetry you bulk-load a
  directory into the browser once; "step to an unloaded file" collapses into the
  ordinary Open action. Re-deriving a filename pattern to fetch one more file is
  effort the Open dialog (multi-select, whole directory) already covers better.
- *A global PgUp/PgDn that advances the selection+plot while focus is in the plot
  or fit panel.* This is the only genuinely-additive micro-affordance — but it is a
  one-line keyboard nicety on the *existing* selection model, not WiMDA's walker,
  and it is easy to add later in a polish pass if the keyboard-from-the-table path
  proves insufficient in practice.

**Verdict: REJECT** the filename-pattern walker as mimicry. The benefit
("flip quickly through consecutive runs") is already served by the browser; the
walker would re-implement, worse, what the table gives for free. The optional
global-shortcut nicety is recorded but **not** adopted in this session — it is not
WiMDA's mechanism and does not need this study to justify it.

### 2. ASCII / data-only export — ADAPT

**What WiMDA does.** `SaveAs` (`WiMDA_Main.pas:1414–1869`) writes a `!`-commented
provenance header (run info, grouping, α, deadtime) followed by columns, with a
three-way content switch — **stData** (x, y, err), **stBoth** (x, y, err, fit), and
**stFit** (model resampled on a uniform grid over the view range) — plus an optional
x-range and a batch loop over a run range (`Plot.pas:3082`).

**What Asymmetry already does.** The GLE export (`plot_panel.export_plots_to_gle`,
`get_current_plot_export_data`, `_write_data_file`, `_write_fit_file`) *already*
writes `.dat` sidecars with a richer `!`-header than WiMDA (run info, grouping,
histogram counts incl. good-range `events_grouped`, binning mode, deadtime) and a
`time asymmetry error` column block, plus a `.fit` sidecar — **but only as part of
a `.gleplot` folder, alongside a GLE script that gets compiled to PDF/EPS.** There
is no way to get just the text, and no data-only (suppress-fit) choice.

**Ben's steer (this session):** *"most of this is exported when we export to GLE.
Strengthen that rather than developing a parallel export. Allow data-only exports
without going through the GLE path, while being careful not to clutter the GUI with
multiple export buttons."*

**Verdict: ADAPT.** Reuse `_write_data_file`/`_write_fit_file` verbatim; add a
content selector (**data only / data + fit / fit only**) and a text-only
destination that writes the `.dat`/`.fit` columns **without** the `.gleplot` folder
or GLE compile. Surface it on the **existing** "Export Plot(s) to GLE" control as a
small popup menu (a `QToolButton`/menu split: *"Export to GLE…"* vs *"Export
plotted data (text)…"*) — **one** button, no clutter. The provenance header already
exists and is a superset of WiMDA's, so this is mostly plumbing a content flag and
a no-compile path through code that is already there. Single-curve / current-plot
first; batch-over-selection is a stretch (the GLE path is already per-plot, so
batch reuses the run loop, but it is the lower-value half — defer unless cheap).
Divergence kept: comment char stays `!` (matches the existing `.dat` and GLE), not
`#`.

### 3. Events columns — ADOPT

**What WiMDA does.** `LogbookUnit.pas:595–626`: **good events (MEv)** = Σ over the
good-bin range `[tgood_beg, tgood_end]` across all histograms, ÷10⁶; **events per
frame** = that sum ÷ `framestotal` (the acquisition frame count).

**What Asymmetry already does.** `data_browser._resolve_run_info_value` already
serves `run_info.counts_mev` (**but over *all* bins**, not the good range) and
`run_info.counts_per_detector`, surfaced as optional extra columns. Critically,
there is **no end-user UI to *add* a column** — only a header-context "Remove from
Data Browser"; columns appear only via programmatic/project-state restore.

**Verdict: ADOPT.** Two new resolvers plus a surface to turn them on:

- `run_info.good_events_mev` — Σ counts over `[first_good_bin, last_good_bin]`
  across grouped detectors ÷10⁶ (the WiMDA-faithful "good events"). Kept **distinct
  from** the existing all-bins `counts_mev` so no behaviour/column changes silently.
- `run_info.events_per_frame` — good events ÷ `good_frames` (the frame count lives
  in the grouping; helper `transform.grouping.good_frames`). Falls back to "—" when
  `good_frames` is absent or ≤0 (synthetic runs).
- A user surface to add columns: extend the existing header context menu with an
  **"Add column…"** entry listing the available `run_info.*` fields (mirrors the
  existing Remove path; additive panel-state key for the chosen set).

Divergence from WiMDA: WiMDA's all-bins vs good-range distinction is collapsed into
*two named columns* so both are available and neither is ambiguous.

### 4. B-from-log — ADOPT

**What WiMDA does.** `Bfromaveinblog` (`WiMDA_Main.pas:729–733`): when ticked, the
run's field becomes the **average of the magnetic-field log channel** (falling back
to the field-coil channel), instead of the header scalar — the exact analogue of
WiMDA's temperature-from-log.

**What Asymmetry already does.** Temperature-from-log is fully built:
`set_use_temperature_from_log` (global) + `set_dataset_temperature_from_log`
(per-run override) + `_series_mean_for_field(dataset, "temperature")` reading
`metadata['nexus_time_series']` + `_series_path_score` (role/heuristic scoring) +
a red tint on log-sourced cells. **There is no field analogue** — `field` is always
`metadata['field']` from the loader scalar.

**Verdict: ADOPT.** Mirror the machinery for field. `_series_mean_for_field` already
takes a `field_key`, so the work is: a field-specific scoring branch in
`_series_path_score` (favouring roles/paths like `sample_magnetic_field`,
`magnet`/`field`/`b_field`, with the same primary/role bonuses temperature uses), a
`set_use_field_from_log` global toggle + per-run override, the B-column tint, and an
event-weighted mean for combined runs (already generic). Native pattern, copied.

### 5. Deadtime-file auto-discovery + staleness — REJECT

**What WiMDA does.** `WiMDA_Main:925–975`: looks in the data directory for a
deadtime calibration file and warns if the file's date is older than the run —
because in WiMDA deadtime *lives in external files*.

**What Asymmetry already does.** Deadtime is estimated **from the run itself**
(`calibrate_deadtime_from_histograms`, the "Cal" path), optionally entered
manually, and **promoted into the grouping** (`promote_deadtime_to_grouping`),
which persists in the `.asymp` project. A WiMDA-format parser
(`parse_deadtime_calibration_text`) even exists, unused.

**The governing test.** Auto-discovery and a *staleness date* are only meaningful
when calibration lives in a dated external file. Asymmetry's model is the opposite:
calibrate from *this* run (always maximally fresh — there is no staleness to warn
about) or carry the value in the project. Re-introducing a directory file store, a
discovery scan, and a date-comparison warning would import WiMDA's *persistence
model*, not a benefit — and would create a *fourth* deadtime write path beside the
three the reconciliation study already tracked (collision F6).

**Verdict: REJECT** auto-discovery + staleness as mimicry. **Optional deferred
nicety** (not adopted now, recorded for honesty): the unused
`parse_deadtime_calibration_text` could be wired to a manual *"Load deadtime
calibration…"* file picker in the grouping dialog — a one-off import, **no**
auto-discovery, **no** staleness — for the rare facility that ships a deadtime
table as text. That is a small import affordance, not WiMDA's feature.

### 6. Log-count display mode — ADOPT

**What WiMDA does.** `Plot.pas:1542–1548`: a "Logp" mode plots `ln(count/nbin)` (no
error bars) so an exponential decay reads as a straight line.

**The diagnostic.** This is the **standard t0 / background / deadtime sanity view**.
A pure muon-decay histogram is `N(t) = N₀ e^(−t/τ_μ) + bg`; on a log count axis it
is a **straight line of slope −1/τ_μ**. Three faults jump out that are invisible on
a linear axis: a mis-placed t0 kinks the line at early time; a wrong background
level bends the tail upward (the constant dominates as the exponential dies); and
deadtime at high instantaneous rate flattens the earliest bins. None of these need
a fit to *see*.

**What Asymmetry already does.** The raw-counts view (#53,
`plot_grouped_time_domain_subplots`) renders grouped counts linearly via the
`set_time_view_modes` seam.

**Verdict: ADOPT.** Add a log-scaled rendering of the raw-counts view. Two shapes
are weighed in `implementation-options.md`; the recommendation is a **log-y toggle**
that is visible only on the raw-counts view (additive panel-state key) rather than a
fourth top-level view mode — it is orthogonal to the data being shown and avoids
mode proliferation. **Divergence from WiMDA, on physical-correctness grounds:** plot
counts on a **log₁₀ y-axis** (tick labels stay in counts, the decade structure is
legible) rather than WiMDA's `ln(count)` on a linear axis; **mask non-positive
bins** (log undefined) and keep Poisson error bars where the axis permits, instead
of WiMDA's silent `log(0) → −∞` and dropped errors.

### 7. F,B balance overlay — ADAPT (borderline; Ben to confirm or drop to REJECT)

**What WiMDA does.** `FBOverlay` (`Plot.pas:2701–2711`) draws the forward and
backward group histograms on one set of axes so the eye can judge **α balance**.

**What Asymmetry already does.** A strong α suite: three estimators (diamagnetic,
general two-window, count-ratio) with bootstrap uncertainty and a method label, and
an α-free count fit that recovers α "the statistically proper way" with a covariance
and χ². The groups view plots each group, but in its **own stacked subplot** — never
forward and backward on **shared axes**, and never with α applied.

**The honest weighing.** The estimators give the *number*; what is genuinely missing
is the *picture* — F and **α·B** overlaid on one axis so you can see whether they
coincide in the relaxing envelope. That matters precisely when an estimator misleads:
ledger #6 records that WiMDA's General-α has no interior minimum at realistic
statistics and silently returns the clamp; Asymmetry pins that behaviour in tests,
but a "why does α look wrong?" eyeball is still the fastest triage. Because it is
**display-only** (it adds no new α value or promote path), it does not touch the α
collisions (F7) the reconciliation study flagged.

**Verdict: ADAPT (light, borderline).** Provide an **F,B balance overlay** — forward
group and α-scaled backward group on shared axes, reusing the grouped-counts data
path — as a diagnostic the estimators cannot give. Reshaped from WiMDA: overlay
**F vs α·B** (balance-aware), not raw F vs raw B, because "do they balance under the
current α?" is the actual question. This is the **most marginal** ADOPT in the
table; if Ben judges the estimator suite + α-free fit sufficient, **REJECT** is a
defensible alternative and the cleaner outcome. Flagged explicitly for the checkpoint.

### 8. Data-snapped cursor + readouts — ADAPT (workflow explained; subset is Ben's call)

Ben's instruction was: *"What workflow does this belong to? Explain this to me, then
ask me again."* So, the workflow first.

**What WiMDA provides** (`Plot.pas:1159–1228,2962–3006`): a keyboard-stepped cursor
that **snaps to a data point**, and at that point reports **S/N = |y/err|**, a
**3-point parabolic peak** (fit `y=ax²+bx+c` to the point and its two neighbours,
report the vertex `x=−b/2a` as a sub-bin peak position/height), and a **windowed
average ± error** over the visible range (`mean = Σy/n`, `σ = √(Σy²/n − mean²)/n`).

**The workflow it belongs to: spectrum reading / quick quantitation *without a
fit*.** These four readouts are the gestures of *inspecting* a curve rather than
modelling it:

- **Windowed average ± err** — reading a *level* off the curve: a baseline, an
  asymmetry plateau, an ALC/repolarisation step height. This is exactly what
  `core/transform/integral.integrate_curve` already computes (windowed mean and
  error-on-the-mean) — the cursor would just make it a drag-to-select gesture.
- **3-point parabolic peak** — locating a **peak to sub-bin precision without
  fitting it**: a line in an **FFT/MaxEnt frequency spectrum**, or an **ALC
  resonance centre**. This is the genuinely-missed capability; today you either
  read the nearest bin or set up a fit.
- **S/N at the snapped point** — judging whether a spectral feature is *real* before
  committing to a fit.
- **Snap** — the carrier that makes the three land on actual data rather than
  free-floating pixels.

So this belongs to the **frequency-domain / spectrum-reading workflow** (and the
ALC/integral "read a level" workflow) far more than to time-domain fitting. It is
the "I see a peak — where is it, how tall, is it significant?" and "what's the mean
level across here?" pair of gestures.

**What Asymmetry already does.** `cursor_coords_changed` already streams free (x, y)
to `_status_coords_label`; `_last_plot_time/asymmetry/error` cache the plotted
arrays; `integrate_curve` is ready for the windowed average. The carrier is in
place; only the readouts are missing.

**Verdict: ADAPT**, building on the existing cursor signal + status bar (not WiMDA's
keyboard-index model — Asymmetry snaps on hover to the nearest cached point). The
**subset** is Ben's decision, re-asked at the checkpoint with this workflow framing.
Recommendation: **windowed-average ± err** (reuses `integrate_curve`; serves the
level-reading workflow) and **parabolic peak** (the real gap; serves spectrum
reading) are the two worth doing; **snap** is their carrier; **S/N** is a cheap
add-on on the snapped point.

### 9. Cosmetic basket — REJECT here

Error-bar toggle, marker styles, tick spacing, ns/bins x-units: these are
appearance preferences, not workflow. They belong to a dedicated UI-polish pass, not
to a parity study judged on workflow benefit. The single near-miss is **bins (vs µs)
x-units**, which pairs naturally with the log-count t0 diagnostic (§6) — recorded
there as an optional add-on, not adopted as part of this basket.

### 10. Live current-run monitoring — DESIGN-ONLY

Per the umbrella decision (2026-06-10) and the brief, this is contingent on beamline
access for testing and is **not implemented** in this session. The study contributes
the **hook design** only — see `implementation-options.md` §10 for the shape of a
*refreshable dataset* whose source file can change underneath it (freshest
`macq*.tmp`/`auto_*.tmp`, periodic re-read, browser badge), so that when a beamline
becomes available the loader contract is already specified.
