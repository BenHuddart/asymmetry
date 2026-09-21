# Grouping preview: legible compares and a Counts view

Status: implemented 2026-09-21 on `feat/grouping-preview`, PR to follow.
Landed as **one PR** built in three phases by subagents with a lead review
gate after each. Mockup (Design canvas, Ben-reviewed through
three rounds): <https://claude.ai/artifact/1Xcbr5vT3qfSpn1vhiWtQ5>. Follows the
corrections-UX work recorded in
[porting/correction-order-alpha-estimation/corrections-tab-ux-plan.md](../porting/correction-order-alpha-estimation/corrections-tab-ux-plan.md),
whose "ghost-the-removal" model stays; this plan changes how the ghost is
drawn, what the pane says about itself, and adds a count-domain view.

## Problem

Ben's report: the preview is unintuitive and it is not obvious what each
parameter or correction does to the spectrum. Verified on 2026-09-21 by
reading the pane and the dialog's compare wiring and by rendering the window
offscreen in seven states (line numbers are anchors; re-check before editing):

1. **The ghost is nearly invisible.** It is drawn *behind* the solid
   (`preview_pane.py:463-484`: ghost zorder 2, errorbar zorder 3), in
   `TEXT_DIM` at 0.9 px and 55 % opacity. The solid is ~2,000 markers at 2 px
   (`_MAX_PREVIEW_POINTS`, `:73`) across a ~750 px axis, so the dots overlap
   into a band that hides both the error bars and the ghost wherever the
   curves are close, which is everywhere a correction is small.
2. **The ghost's name is a clipped inline label** placed at its rightmost
   in-view sample and clamped to the axis edge when off-scale
   (`_ghost_label_spec`, `:746-773`); in the renders it overlaps the data at
   the right edge and reads as noise.
3. **Count-domain settings have no visual.** t0, t_good offset, last good bin,
   deadtime, the subtracted background level and exclusions all act on the F
   and B histograms, but the pane only ever draws the asymmetry over the good
   window (`_form_asymmetry`, `:685-719`), so a t0 change is a sub-pixel
   x-shift, the window is just where the curve stops, and the pre-t0 region
   (where the `range` background mode reads) is never shown.
4. **The pane does not say what it shows.** The status strip reads "Preview:
   run 7101" (`:513-519`); it names neither the F and B groups nor the
   binning, and in vector mode it does not say that only the primary pair is
   previewed. The α residual ⟨A⟩ is quoted there as a bare number rather than
   drawn on the plot it describes.
5. **"Compare vs raw (uncorrected)" duplicates the pager's last stop** and
   shows a compound ghost (no deadtime, no background, α = β = 1;
   `_run_reduction`, `:654-663`) that cannot be attributed to any one
   setting. It is the ghost that can sit ~10⁷ % off-scale on a FLAME run,
   which is what forced the clamped-label machinery in (2).

## Settled design (decision log)

Decisions taken with Ben on 2026-09-21 over the mockup rounds.

- **D1 — the preview follows the scope-panel selection and says so.** No
  separate preview-run picker (rejected: the run would be set in two places).
  The status strip leads with an uppercase `PREVIEW` label and the run in
  bold — `run 7101 · TF 200 G (selected run)` — followed, muted, by
  `· F = Det 1 (0°) · B = Det 2 (90°) · bin 5 · 0.2 – 9.5 µs` and the active
  count-domain corrections (`· tail-fit background`). In vector mode it
  appends `· P_z pair`. The run title comes from the dataset's `run_label`;
  group names from the draft's `group_names`. This is how a user sees the
  profile's α, calibrated on a TF run in the α card, acting on a ZF run: they
  select the ZF run.
- **D2 — the solid curve is a line with a ±σ band, not a dot cloud.** When the
  reduced curve has more than 400 points the pane draws a 1.2 px `ACCENT`
  line with a `fill_between` band at 18 % opacity; at 400 points or fewer
  (after bunching) it draws markers with error bars as today. The decimation
  cap stays for the line mode.
- **D3 — the ghost is drawn on top, in the stage's identity colour.** 1.4 px,
  90 % opacity, `tokens.STAGE_DEADTIME / STAGE_BACKGROUND / STAGE_ALPHA /
  STAGE_BETA` — the same colour the stage's chip outline and card stripe
  already wear, so chip, card and ghost read as one thing. A fixed caption in
  the axes' top-left corner names the curves, one row each with a colour
  swatch: `as reduced · α = 1.080` and `α = 1 (ghost)` (or `without deadtime
  (ghost)`, `without background (ghost)`, `β = 1 (ghost)`). The inline
  rightmost label and its off-scale clamp are deleted. The solid-only
  autoscale contract stands (a deadtime-removed FLAME ghost is still far
  off-scale); an off-scale ghost is simply named by the caption.
- **D4 — ⟨A⟩ is drawn, not quoted.** For the α compare the residual baseline
  is a dashed horizontal line in `STAGE_ALPHA` with its value beside it on the
  plot (`⟨A⟩ = −6.11 ± 0.00 % (residual baseline)`), and the status strip no
  longer carries it.
- **D5 — no Δ strip and no effect readout.** Both were mocked and rejected by
  Ben: the Δ strip costs vertical space the window does not have, and an
  `effect: rms …, max …` suffix on the caption is more than the minimal
  caption he wants. Recorded so neither is re-proposed. (The deferred Δ-strip
  design in the corrections-tab plan is superseded by this decision.)
- **D6 — the compound "raw" compare is removed entirely.** The pager-row
  checkbox, the `"raw"` stop in `_COMPARE_CYCLE`, its label and availability
  branch, the worker's raw ghost pass, and the `_compare_toggles` dict with
  its sync loop and `_on_compare_toggled` slot all go (the dict's only
  remaining entry was the checkbox). The pager cycles off → deadtime →
  background → α → β. Rationale: it does not answer "what does this control
  do", it is the only source of the 10⁷ % ghost, and it duplicated a pager
  stop.
- **D7 — a Counts view, toggled beside the pager.** A two-button segmented
  control `Asymmetry | Counts` at the left of the pager row; preview-only
  state like `_compare_stage`, never persisted. Counts draws the corrected
  F and B group counts — deadtime-corrected, grouped, background-subtracted,
  i.e. `CorrectedGroupedCounts.forward / backward` — over the **full**
  histogram on a log₁₀ y-axis (counts clipped at 1 so a subtracted bin at or
  below zero sits on the floor), x = time after t0 in µs. F is `ACCENT`, B is
  `PLOT_AXIS` at 75 %. Markers: the pre-t0 region and the bins outside the
  good window shaded `SURFACE_HI`; vertical lines at t0 (`t0 · bin 100 (from
  file)` / `(manual)` / `(detected)`), at the first good bin (`good window:
  t_good offset N bins`) and the last good bin (`last good bin N`); a dashed
  `STAGE_BACKGROUND` line at the subtracted level, labelled `background level
  · F 7119 / B 7168 counts per bin (tail fit)`, for the constant-level modes
  (`fixed`, `range`, `tail_fit`) — `reference_run` subtracts a spectrum and
  draws no line. Deadtime and background compares ghost the same view: the
  stage-removed F and B counts on top in the stage colour (F solid, B dashed).
  α and β act when the asymmetry is formed, so in Counts their compare draws
  no ghost and the caption says `α acts when the asymmetry is formed — see
  the Asymmetry view`.
- **D8 — one worker pass serves both views.** `_PreviewResult` gains the
  decimated F and B count arrays, their time axis, and the t0 / good-window /
  background-level facts for the markers, alongside the asymmetry it carries
  today; switching the view is a redraw, never a recompute. The deadtime and
  background compares already run a second corrected pass for their
  asymmetry ghost; its counts become the Counts ghost at no extra cost. The
  threading contract in the module docstring is unchanged.
- **D9 — the pane is 300 px tall, still fixed.** `_PANE_HEIGHT` 200 → 300.
  The dialog measures its preferred size from the panes' minimums
  (`dialog.py:2903`), so the default window grows by 100 px on its own; the
  two columns are untouched, so
  `test_both_columns_fit_without_scroll_at_default_size` keeps its meaning.
  A splitter child (user-draggable pane) was considered and deferred: it
  adds chrome for a size the mockup already settles.
- **D10 — out of scope, recorded as follow-ups.** Collapsing the compare
  controls to the chips alone (the pager row stays as it is, minus the
  checkbox); a cross-request reduction cache keyed on the grouping digest so
  paging through compares is instant; a converging build-up stepper.

## Code map (verified 2026-09-21)

- `src/asymmetry/gui/windows/grouping/preview_pane.py` — the whole pane.
  `_PreviewRequest` (`:79-112`, `overlay` legacy flag + `compare_stage`);
  `_PreviewResult` (`:115-134`); `GroupingPreviewPane.__init__` (`:146-234`,
  fixed height `:149`, canvas via `create_canvas`, compact pan/zoom/home
  buttons on the status row `:217-234`); `request_preview` /
  `request_preview_from_profile` (`:238-318`); `_draw` (`:421-519`);
  `_run_reduction` (`:533-682`, raw branch `:654-663`); `_form_asymmetry`
  (`:685`); `_solid_ylimits` (`:722`); `_ghost_label_spec` (`:746`, delete);
  `_weighted_centre` (`:776`).
- `src/asymmetry/gui/windows/grouping/dialog.py` — compare surfaces.
  `_COMPARE_CYCLE` / `_COMPARE_STAGE_LABELS` / `_CARD_COMPARING_TEXTS`
  (`:158-181`); `_compare_toggles` dict (`:1130`); pager + preview placement
  (`:1292-1302`); α auto-focus after calibration (`:5313-5317`);
  `_build_pipeline_strip` (`:3854`); `_on_compare_toggled` (`:4083`);
  `_set_compare_stage` (`:4090`); `_sync_compare_toggles` (`:4096`);
  `_build_compare_pager` (`:4123`, checkbox `:4157-4167`); `_step_compare`
  (`:4171`); `_sync_compare_pager` (`:4188`); `_compare_stage_available`
  (`:4212`, raw branch `:4231-4237`); `_refresh_preview` (`:4238-4294`, the
  seam that must also pass the D1 status facts); `_preview_run_number`
  (`:4296`); group names via `_load_group_names` (`:2378`).
- `src/asymmetry/core/transform/reduce.py` — `CorrectedGroupedCounts`
  (`:144-177`: `forward`, `backward`, `common_t0`, `bin_width`, `t0_time_us`,
  `background_level`), `corrected_grouped_counts` (`:199`). Everything the
  Counts view needs is already returned; no core change.
- `src/asymmetry/gui/styles/tokens.py` — `STAGE_*` (`:117-126`), `ACCENT`,
  `PLOT_AXIS` (`:177`), `SURFACE_HI` (`:15`).
- `src/asymmetry/gui/utils/plot_decimation.py` — `decimate_for_preview`.
- Tests: `tests/gui/test_grouping_preview_pane.py` (raw: `:757`, `:764-800`,
  `:824-901` use `compare_stage="raw"` to exercise autoscale and the clamped
  label — rewrite onto the deadtime ghost); `tests/gui/test_compare_pager.py`
  (cycle with `raw`, `:136-212`); `tests/gui/test_beta_section.py:261-284`
  (`raw` availability and order); `tests/gui/test_grouping_dialog.py:1437`
  and `:397` (size budget).
- Docs: `docs/reference/detector_grouping.rst` "Live asymmetry preview"
  bullet (`:269-274`) and "Comparing a correction's effect" (`:791-866`, the
  raw checkbox paragraphs `:816-830`, `:865`). Screenshot scenarios that show
  the pane: `grouping_window_profile_editor` (1180×720),
  `alpha_calibration_dialog` and `beta_calibration_dialog` (1220×760, α/β
  compare focused) — all regenerate on merge and must fit the new height.

## Phases

Lead runs `python tools/harness.py validate` once at the end; agents run only
focused files and `--tier fast`. Agents leave the tree uncommitted; the lead
reviews the diff, renders the affected states offscreen (never `dialog.close()`
after driving state directly — use `_teardown_workers()` + `deleteLater()`),
and commits. Briefs carry `AGENTS.md` § Lean Code Rules verbatim.

- **P1 — pane rendering and raw removal (Opus).** `preview_pane.py`: D2 line
  + band with the 400-point rule, D3 ghost on top in the stage colour with the
  fixed corner caption, D4 ⟨A⟩ line, delete `_ghost_label_spec` and the
  `overlay` request flag (every caller passes `compare_stage`), D9 height, D1
  status strip from facts passed on the request (run label, F/B names, bunch,
  window, active corrections, vector pair). `dialog.py`: D6 removals, pass the
  D1 facts from `_refresh_preview`. Tests: rewrite the raw-based autoscale and
  label tests onto the deadtime ghost; delete the raw ghost and label-clamp
  tests; new pins — ghost artist above the solid, ghost colour equals the
  stage token, caption text per stage, ⟨A⟩ line present only for α, line vs
  markers either side of 400 points, status strip text, pager cycle without
  raw, `_compare_toggles` gone.
- **P2 — Counts view (Opus).** D7 toggle at the left of the pager row (the
  dialog owns the widget, the pane owns the view state through a
  `set_view("asymmetry" | "counts")` seam that redraws from `_last_result`);
  D8 result fields; `_draw_counts` with the markers and the per-mode t0 /
  background labels (t0 mode from the resolved grouping's policy, background
  mode from `background_mode`); the deadtime/background ghost in Counts; the
  α/β no-ghost caption. Tests: view toggle is a redraw (worker not
  re-dispatched); counts arrays bounded like the asymmetry; log clip at 1;
  markers at the resolved t0 / window bins; background line only for
  constant-level modes; ghost only for deadtime/background.
- **P3 — docs (Sonnet).** `detector_grouping.rst`: rewrite the preview bullet
  and the "Comparing a correction's effect" section quoting the new UI strings
  verbatim (caption texts, `PREVIEW` strip, pager without the checkbox), add a
  "Counts view" subsection; a `grouping_window_counts_view` screenshot
  scenario (Counts with background focused) registered in `capture.py` and
  referenced from the page; re-check the three existing pane scenarios at the
  new height; `CHANGELOG.md` `[Unreleased]` (Added: Counts view, line + band,
  coloured ghost, PREVIEW strip; Removed: "Compare vs raw (uncorrected)");
  `docs/PLANS.md` entry pointing here.

## Acceptance

1. With α focused on the TF sample run, the ghost is visibly a red line over
   a blue line with a band, the caption names both, and ⟨A⟩ is a dashed line
   on the plot; the status strip names the run and both groups.
2. Selecting a different run in the scope panel changes the `PREVIEW` run in
   the strip and the curve, and nothing else in the pane.
3. The Counts view shows t0, both good-window edges, the pre-t0 region and
   the background level on the same run; toggling views does not start a
   worker.
4. No `raw` string survives in `preview_pane.py`, `dialog.py`, the tests or
   the grouping docs' compare section.
5. `python tools/harness.py validate`, `gui-smoke` and `docs` are green, and
   the regenerated screenshots fit their scenario sizes.
