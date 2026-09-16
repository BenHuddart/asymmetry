# Series workflow: the Batch tab edits one series, series never overwrite each other

Status: planned 2026-09-16, implemented on `feat/series-workflow`, one PR at
the end; built phase by phase by subagents with a lead review gate after every
phase. Mockups (Claude Design canvas):
https://claude.ai/artifact/8D3T4paNTXSMm6s2i3yxTx — page "Recommended" is the
proposal (Batch tab as series editor, series menu, hints, chip rail, overlay
pills, delete/replace dialogs, lifecycle and storage), page "Directions" holds
the three selector placements considered (A, the top-of-tab selector, was
chosen).

## Problem

Working on a project with several data groups, each with its own batch or
global fit, loses work in four independent ways. All four were verified in
code on 2026-09-16; the line numbers are anchors for the phases below and
must be re-checked before editing.

1. **A run can hold exactly one fit.** Each run has one `FitSlot` per
   representation (`core/representation/base.py:54`). Every batch recording
   path overwrites its members' slots (`mainwindow.py:11262`,
   `_record_fit_series`), so a run in two campaigns keeps only the last one,
   and the `diverged` / `include_in_trend` flags on that shared slot are
   shared by every series containing the run. The series keeps its own
   `results_by_run`, so the trend survives, but overlay, Single tab and
   trend gating all follow the last writer. Divergence exists *only* because
   of this overwrite.
2. **The Batch tab is a scratchpad reset by every browser click.** A
   selection change calls `FitPanel.set_datasets` (`mainwindow.py:14521`),
   which rebuilds the member pool, clears exclusions and the group binding,
   reseeds the table, replaces the results card
   (`GlobalFitTab.set_datasets` → `_update_mode_ui(preserve_result=False)`)
   and, when the selected runs share a single-fit model, swaps the composite
   model (`global_tab.py:1634`, `_refresh_inherited_single_fit_defaults`).
   One `global_fit_state` per domain is saved, so nothing can bring a
   previous setup back.
3. **No route from a recorded series back to the Batch tab.** The chip menu
   offers Rename / Select members / Delete (`fit_parameters_panel.py:1891`);
   selecting a chip only tints browser rows (`mainwindow.py:11001`). The
   docs promise re-running "from the Batch tab, still bound to the same
   group", but the only route is the browser's "Fit this group…", which
   restores members only. The fit range is plot-owned and project-wide;
   series identity (`project_model.py:44`, `_series_signature`) deliberately
   ignores range and roles, so re-running the same group + model with a new
   range **replaces** the old series in place. Pinned by
   `tests/core/test_project_model.py::test_remove_superseded_ignores_fit_range_and_roles`.
4. **Deleting a series clears every member run's single-fit state and
   overlays** (`mainwindow.py:11062` → `_on_fit_parameters_group_fits_deleted`
   → `FitPanel.clear_fits_for_runs`, `PlotPanel.clear_fits_for_runs`)
   regardless of other fits on those runs.

Persistence: `save_project` (`core/project/schema.py:1426`) overwrites the
file in place — no temp file, rename, backup or autosave. Real projects are
~300 kB with ~1 kB per run per series (measured on four of Ben's files), so
size is not the pressing risk; the symptom is one sample spread across six
`.asymp` files named by campaign.

## Settled design (decision log)

Decisions taken with Ben on 2026-09-16 unless marked *lead*.

**D1 — The Batch tab edits one series.** The tab always has an *open* series
(recorded, or a draft that has never run). Its model, parameter table, fit
range, members, exclusions, seeding and co-add settings are that series'
recipe. A recorded series' recipe is immutable except through a run (D3) and
`Rename`. Editing the table edits a draft over the open series; a passive
"Edited · results show last run" tag says so; there is no prompt and no
"Save as new" action. Re-opening the series from the selector drops the draft.

**D2 — The recipe lives on the series.** `FitSeries.recipe: dict` (JSON,
~2 kB) holds: `parameters` (the Batch tab's table rows: name, value, type,
bounds, seeded), `fit_range` (`{"min", "max"}` numeric, domain unit implied
by `rep_type`), `seeding` (`"auto"|…`, the Batch tab's `_batch_seeding_mode`),
`coadd` (`{"mode", "window"}`). Existing fields keep their roles:
`canonical_model`, `param_roles` (now derived from `parameters[].type` at
record time and kept for readers), `order_key`, `group_id`,
`excluded_run_numbers`, `last_fitted_members`. Results stay a snapshot in
`results_by_run` (summary only, no curves, no HTML).

**D3 — Run = identical replaces, anything else is a new series.** On run,
the draft's normalised recipe plus effective member set is compared with the
open series' recorded recipe. Identical → results replaced in place, same
`batch_id`, same label. Different in any field → a new `FitSeries` with a
fresh id and a default label from its recipe becomes the open and active
series; the original is untouched. Signature-based superseding
(`remove_superseded_batches`, `superseded_batch_ids`, `dedupe_batches`) is
removed — the load-time dedupe would collapse series that differ only by
range. Renaming never changes identity.

**D4 — Per-run state is the single fit only.** Batch/global/grouped/scan
recording paths stop writing member `FitSlot`s. `Representation.fit` (and
`projection_fits`) hold the Single tab's fit only; `FitSlot` loses
`batch_id`, `diverged`, `include_in_trend`; `provenance` keeps `"none"`,
`"single"`, `"wizard"` (`"batch"`/`"global"` are never written again; the
migration drops such slots — their results already live on the series).
Divergence (`refresh_divergence`, `canonical_model_matches`,
`FitSeries.diverged_runs`, `mark/clear/is_diverged`, the ⚠ divergence glyph
and the browser's amber "diverged" state) is deleted: with D4 it cannot
occur. Trend gating moves to the series: `FitSeries.trend_excluded_runs:
list[int]`, toggled by the existing Parameters-panel control. Membership
staleness (`is_stale`) is unchanged.

**D5 — One active series per representation.** `ProjectModel.active_series:
dict[str, str]` (rep_type value → batch_id), persisted top-level as
`active_series`. The Parameters chip, the Batch tab's open series and the
plot overlay read and write the same pointer: pressing a chip, opening a
series in the Batch tab or double-clicking an overlay pill all set it. The
plot draws the active series' fit for every run it covers. Which *other*
fits are shown on a run (the "Fits on this run" pills: other series, the
single fit) is transient view state in the plot panel, default = active
series only. A single fit that has just completed on the Single tab shows
its overlay for that run, as today, until the active series changes or the
pills are toggled. The Single tab stays project-wide and exploratory; a run
with no single slot but a series result restores its form from the active
series' result (today's `build_single_fit_payload_from_slot`, fed from the
series), labelled as such, and `register_global_fit_results` keeps seeding
the single form from batch results.

**D6 — Delete touches only the series.** `remove_batch` removes the series,
clears `active_series` entries that point to it, and the GUI clears only
that series' overlays. `FitPanel.clear_fits_for_runs` /
`PlotPanel.clear_fits_for_runs` are no longer called on series delete. The
dialog lists what is kept (other series on these runs, single fits, the
group). Group delete keeps the "Keep fits / Delete fits" prompt. The Global
Fit Wizard's re-partition still deletes the previous phase series (it is
replacing them); out of scope here.

**D7 — Browser selection never resets an open series.** While a series is
open, a selection change shows the "Selection differs" hint with `New
series from selection` / `Open a series for these runs ▾` / `Keep editing`;
nothing in the tab changes. `New series ▾` offers *from browser selection*,
*from a data group*, *copy of current* (= Duplicate). "Fit this group…"
opens a draft bound to the group seeded from the group's newest series
recipe when one exists (so the natural next run is an identical re-run or a
deliberate variation), else from the tab defaults. A fresh project has no
open series: the tab is a draft over the browser selection, exactly today's
behaviour, until the first run records it.

**D8 — Fit range.** The Batch tab's range fields edit the open series'
`recipe.fit_range`; a batch run crops its member datasets with that range
(`_get_fit_dataset` gains an explicit range argument; the memo key already
embeds the range). While the Batch tab is visible, the plot's range guides
show the open series' range. The Single tab's range remains the plot-owned,
project-wide range. Ben, 2026-09-16: single fit is exploration, batch is
production.

**D9 — Persistence is crash-safe.** `save_project` writes to `<path>.tmp`
and `os.replace`s it over the target, keeping the previous file as
`<path>.bak` (one generation). While the project is dirty, an autosave
timer (default 5 min, `QSettings` key in `gui/fit_settings.py`'s style)
writes `<stem>.autosave.asymp` beside the project (or under the app data dir
for an unsaved session); a successful save deletes it, and opening a
project whose autosave is newer offers to recover it. Per-run
`states_by_run` duplication with `FitSlot.ui_state` is *not* touched here
(follow-up).

**D10 — Naming.** The word is "series" everywhere (chips, selector, menus,
docs). Default label becomes `<model> · <range>[ · <group>]`
(`default_series_label`), with ` (2)` appended when a group already holds a
series with the same model and range. Existing user labels are untouched.

**D11 — Schema v20** (*lead*). Migration from v19: for every series, seed
`recipe` from `canonical_model` + `param_roles` + the first member slot's
`parameters` template (values/bounds) + the `fit_range` provenance string
parsed from the first `results_by_run` entry (`"a – b µs"` → numbers; absent
→ `null` range = "as fitted, unknown"), `seeding="auto"`,
`coadd={"mode":"off","window":2}`; `trend_excluded_runs` = member runs whose
slot had `include_in_trend=False`; drop slots whose `provenance` is
`"batch"`/`"global"`; drop `diverged`/`include_in_trend`/`batch_id` from the
remaining slots; `active_series[rep]` = newest series of that
representation. Tolerant: a malformed slot or series never aborts the load.

## Key code map (verify anchors before editing)

| Concern | Location |
| --- | --- |
| `FitSlot`, `Representation.fit` / `projection_fits` / `fit_for` / `set_fit_for` / `to_dict` | `core/representation/base.py` (~54–302) |
| `FitSeries` fields, `is_global`, `effective_members`, `is_stale`, divergence helpers, `to_dict`/`from_dict` | `core/representation/series.py` (~58–400) |
| `ProjectModel`: `_series_signature`, `superseded_batch_ids`, `remove_superseded_batches`, `dedupe_batches`, `remove_batch`, `_relink_unlinked_members`, `refresh_divergence`, `trend_runs_for_batch`, `set_member_trend_inclusion`, `to_dict`/`from_project_state`/`write_to_project_state` | `core/representation/project_model.py` (~44–120, ~513–800, ~827–918) |
| Default series label | `core/representation/naming.py` |
| Schema version, migrations, `save_project`/`load_project` | `core/project/schema.py` (`CURRENT_SCHEMA_VERSION` ~213, `_migrate_v18_to_v19` ~1173, `save_project` ~1426) |
| Recording choke points | `mainwindow.py` `_record_fit_series` (~11262), `_record_single_fit_slot` (~11296), `_record_global_fit_batch` (~11346), `_on_scan_requested` (~11440), `_record_grouped_fit_series` (~12000) |
| Group resolution for a batch | `mainwindow.py` `_resolve_batch_group` (~12221), `_series_exclusions_for_group` (~12389) |
| Selection → Batch tab reset, binding clear | `mainwindow.py` (~14480–14530); `_on_fit_group_requested` (~12433) |
| Series delete / rename / select-members / trend inclusion handlers | `mainwindow.py` (~11041–11110), `_on_fit_parameters_group_fits_deleted` (search) |
| Trend panel reload (rows, `include_in_trend` at ~10754, stale ids) | `mainwindow.py` `_refresh_trend_panel` (~10781–10960) |
| Single-fit restore precedence from slot | `mainwindow.py` `_single_fit_restore_payload` (~11140–11200) |
| Fit dataset crop (memoised, plot range) | `mainwindow.py` `_get_fit_dataset` (~14649) |
| Project save/collect/restore | `mainwindow.py` `_write_project` (~15071), `collect_project_state` (~15280–15420), restore (~15960–16060) |
| Batch tab: `set_datasets`, `set_bound_group`, members list, `_reseed_batch_parameter_table`, `_refresh_inherited_single_fit_defaults`, `_run_global_fit`, `FitLaunch`, `get_state`/`restore_state`, `_update_mode_ui` | `gui/panels/fit/global_tab.py` (~371, ~1140–1260, ~1383, ~1634, ~2449, ~4545–4760, ~4766) |
| Fit panel: `clear_fits_for_runs`, `register_global_fit_results`, `get_global_state`, domain state | `gui/panels/fit/panel.py` (~717, ~1010, ~1100–1190) |
| Fit range fields, provenance text | `gui/panels/fit/tab_base.py` (~2459–2560) |
| Parameters panel: chip rail (`_rebuild_group_buttons` ~1843), chip context menu (~1891), `_delete_group_fits` (~1921), `load_representation_series` (~1591), member inclusion signal | `gui/panels/fit_parameters_panel.py` |
| Plot overlays keyed by run (`_fit_curves`, `_fit_curves_by_key[(run, axis)]`, `plot_fit` ~6819, `clear_fits_for_runs` ~7081) | `gui/panels/plot_panel.py` |
| Browser: `fit_group_requested`, `show_group_series_requested`, group context menu (~4085), highlight | `gui/panels/data_browser.py` |
| Colour tokens (`ACCENT_RED`, group header colours) | `gui/styles/tokens.py` |
| Screenshot scenarios | `docs/screenshots/scenarios/batch_tab_group_binding.py`, registry in `docs/screenshots/capture.py` |
| Docs pages | `docs/reference/parameter_trending.rst` § "Group-bound series and staleness", `fitting.rst`, `gui_usage.rst` (~1102), `project_files.rst`, `docs/ARCHITECTURE.md` § "DataGroup and FitSeries" |

## Execution plan

Branch `feat/series-workflow` off `main`, one PR. Phases are strictly
ordered; each is one or more commits by one subagent working **in the main
checkout on this branch** (no worktrees — sequential phases share one tree,
and worktree agents import `asymmetry` from the wrong checkout unless
`PYTHONPATH` is set). Subagents cannot be messaged after launch, so each
brief is complete up front: the decision log above, the phase's scope and
checklist, the lean rules, and the test commands.

**Lean rules for every agent brief.** Prefer deleting machinery to adapting
it (divergence, superseding, member-slot pointers all go). No new
`hasattr`/`isinstance` guards on internal objects and no fallbacks for
states the design prevents — ask the lead (by stopping with a clear note)
rather than guarding. Reuse `gui/widgets/` and `gui/styles/` foundations
(`FloatLimitField`, `style_group_state_button`, `build_primary_button_qss`,
`PanelSection`); no new `QThread`, no `processEvents`. No new project-file
payloads beyond D2/D5/D9. Tests beside the behaviour, focused; run
`python tools/harness.py test -- <files>` while iterating and `--tier fast`
before handing back; never `validate` mid-phase.

**Lead review gate (after every phase).** Diff read against the phase
checklist and the decision log; focused tests plus `--tier fast` (plus the
phase's named GUI files) green; `harness structural` and `lint` green; no
Qt in `core`. Fix-ups land inside the phase. `validate` and `gui-smoke` run
once, in Phase 7.

### Phase 1 — Core model and schema v20 (agent: Opus)

Scope: `core/representation/{base,series,project_model,naming}.py`,
`core/project/schema.py`, tests under `tests/core/` and `tests/project/`
(plus the core halves of `tests/gui/test_series_identity.py`, which may move
to `tests/core/`). No GUI edits.

- `FitSeries.recipe` (D2) with `recipe_identity()` returning a canonical
  JSON string of normalised recipe + `sorted(effective members)`;
  `trend_excluded_runs`; `trend_member_run_numbers()` reads it. Remove
  `diverged_runs` and its helpers. `is_global()` may read `param_roles`
  as today (roles are still recorded).
- `FitSlot`: drop `batch_id`, `diverged`, `include_in_trend`;
  `from_dict` ignores them. `Representation` unchanged otherwise.
- `ProjectModel`: remove `_series_signature`, `superseded_batch_ids`,
  `remove_superseded_batches`, `dedupe_batches`, `_relink_unlinked_members`,
  `refresh_divergence` and both `_refresh_*_divergence`,
  `set_member_trend_inclusion` (replace with a series-level setter),
  `trend_runs_for_batch` (read the series). Add `active_series` (D5) with
  `set_active_series(rep_type, batch_id | None)`, cleared by `remove_batch`
  and `remove_data_group(orphan_series=False)`. `to_dict`/`from_dict`/
  `from_project_state`/`write_to_project_state` carry it. `remove_batch`
  no longer touches representations.
- `naming.default_series_label` per D10.
- Schema v20 + `_migrate_v19_to_v20` per D11, with round-trip tests: a v19
  project with (a) a batch series and member slots, (b) an
  `include_in_trend=False` member, (c) a single-fit slot with `ui_state`,
  (d) a computed (scan) series, (e) a grouped (`member_kind="groups"`)
  series, (f) a project with no batches. Slots with provenance
  `"batch"`/`"global"` are dropped; single slots keep `ui_state`.
- Update the identity tests: replace the superseding tests with
  `recipe_identity` tests (range differs → different; label differs → same;
  member excluded → different; bounds differ → different).

Checklist: no Qt imports; every removed helper has no remaining caller in
`core` (`grep`); `results_by_run` untouched; migration tolerant.

### Phase 2 — Recording, active series, overlays and delete in MainWindow (agent: Opus)

Scope: `mainwindow.py`, `gui/panels/plot_panel.py`, `gui/panels/fit/panel.py`
(the write-back and restore seams only), tests in `tests/gui/`.

- Recording (`_record_fit_series`, `_record_global_fit_batch`,
  `_record_grouped_fit_series`, `_on_scan_requested`): build the recipe from
  the Batch tab (`get_global_state` rows, range, seeding, co-add — Phase 3
  adds the accessor; until then read the tab's attributes directly and leave
  a one-line note), apply D3 against the open series id handed in by the
  tab (Phase 3) — until then the newest series of that representation with
  an identical `recipe_identity` counts as "open" — and stop writing member
  slots. Set `active_series` to the recorded series.
- Delete (D6): `_on_series_delete_requested` removes the series and clears
  only its overlays; `_on_fit_parameters_group_fits_deleted` and
  `FitPanel.clear_fits_for_runs` are no longer called from series delete
  (delete them if they have no other caller). The confirmation text lists
  what is kept.
- Trend refresh: rows read `series.trend_excluded_runs`; the member
  inclusion handler writes it; stale ids unchanged; the divergence glyph
  path in `fit_parameters_panel.py` and the browser's "diverged" tint are
  removed.
- Overlays: key plot fit curves by `(run, fit_id)` where `fit_id` is a
  `batch_id` or `"single"`; `plot_fit` gains `fit_id`; the panel draws, per
  run, the active series' curve (accent colour, solid) plus any fit ids in
  a transient `shown_fits_by_run` set (next trace colours); `set_active_series`
  on the panel redraws. `_single_fit_restore_payload` falls back to the
  active series' result (D5). `register_global_fit_results` unchanged.
- `_get_fit_dataset(dataset, fit_range=None)`; batch paths pass the recipe
  range (Phase 3 wires the tab; until then the plot range).
- Project restore: `active_series` restored before the trend panel reload;
  the reload selects it.

Checklist: no member slot written anywhere (`grep "representation.fit ="`
shows only the single path); deleting a series leaves other series' and
single overlays in place (test); a run in two series draws the active one
(test); `_refresh_trend_panel` no longer reads `FitSlot` flags.

### Phase 3 — Batch tab as series editor (agent: Opus)

Scope: `gui/panels/fit/global_tab.py`, `gui/panels/fit/panel.py`,
`gui/panels/fit/tab_base.py`, the selection seam in `mainwindow.py`
(~14480–14530), `_on_fit_group_requested`, tests in `tests/gui/`.

- Series row above Model (mockup "Batch tab · series editor"): a selector
  (`QComboBox`-style popup listing the current representation's series
  sectioned by data group with model · range · status, plus "New series
  from browser selection"), status tag (`Fitted n/n · time` /
  `Edited · results show last run` / `Draft`), buttons `New series ▾`
  (from selection / from a data group… / copy of current), `Duplicate`,
  `Rename…`, `Delete…`. Reuse the Parameters panel's chip/tag styling
  helpers rather than new widgets.
- `open_series(series)` restores the recipe into the tab (model, rows,
  range, members from the owning group with exclusions ticked off, binding
  label, results card from the series' last outcome, seeding, co-add).
  `current_recipe()` returns the draft; `open_series_id()` is what Phase 2's
  recording consults. The edited tag compares `current_recipe()` with the
  open series' recipe on `itemChanged` / range commit (cheap: the identity
  string).
- Selection seam (D7): `set_datasets` is called only when no series is
  open; otherwise the hint bar appears with its three actions. "Fit this
  group…" per D7. After `New series ▾ → from selection` the tab becomes a
  draft over the selection (today's flow).
- Fit range (D8): the tab's range fields write the draft; the plot range
  guides follow the Batch tab's range while it is visible
  (`set_fit_range_display` is no longer pushed *into* the Batch tab from
  the plot); `fit_range_edit_committed` from the Batch tab no longer moves
  the plot range.
- Run button reads `Run series`; the post-run notice states replaced vs
  new (mockup "Delete · replace · keep", B); `restore_state`/`get_state`
  persist only the open series id and any draft (drafts are session-only;
  a draft that never ran is not saved).

Checklist: clicking runs in the browser while a series is open changes
nothing in the tab (test); opening series A then B then A restores A's
recipe exactly (test); identical re-run keeps the id and label; a range
change then run creates a new series and leaves the old one (test);
project reload reopens the active series.

### Phase 4 — Parameters rail sections, chip menu, overlay pills (agent: Sonnet)

Scope: `gui/panels/fit_parameters_panel.py`, `gui/panels/plot_panel.py`
(pill strip only), `mainwindow.py` wiring, tests in `tests/gui/`.

- Chip rail sectioned by owning data group with the group's colour swatch
  and name; standalone series under "Standalone" (mockup "Parameters · chip
  rail"). Sorting inside a section by recorded time.
- Chip context menu: `Open in Batch tab`, `Duplicate…`, `Rename…`, `Select
  members in browser`, `Show fit overlay` (sets active), `Delete series…`.
  Double-click a chip = Open in Batch tab. New signals routed through
  `MainWindow` to `FitPanel.open_series` / duplicate.
- Plot "Fits on this run" pill strip above the time plot for the current
  run: one pill per series covering the run plus the single fit; click
  toggles shown, double-click makes active; hidden when the run has one
  fit. Uses the chip styling.
- Active-series sync: chip press, pill double-click and Batch tab open all
  go through `ProjectModel.set_active_series` and one `MainWindow`
  refresher.

Checklist: no second chip implementation; the pill strip is hidden for
single-fit runs; toggling a pill never records anything.

### Phase 5 — Crash-safe save and autosave (agent: Sonnet)

Scope: `core/project/schema.py` (`save_project`), `mainwindow.py` save/open
seams, `gui/fit_settings.py`-style setting, tests in `tests/project/` and
`tests/gui/`.

- `save_project`: temp file + `os.replace`, previous file kept as `.bak`
  (D9). Tests: a write that raises mid-way leaves the original intact; the
  `.bak` equals the previous content.
- Autosave: a `QTimer` in `MainWindow` armed by `_mark_dirty`, writing
  `<stem>.autosave.asymp` off-thread through the existing `_tasks` runner
  (never while a save is active); cleared on successful save and on clean
  close. On open, an autosave newer than the project prompts
  "Recover unsaved changes from <time>?". Interval setting with a default of
  5 min; 0 disables.

Checklist: no second file-writing path; the autosave never replaces the
project file itself; `_project_save_active` guards both.

### Phase 6 — Docs, screenshots, changelog (agent: Sonnet)

- `docs/reference/parameter_trending.rst` § "Group-bound series and
  staleness": rewrite for D1/D3/D5/D6 (drop divergence text); add the chip
  menu items and overlay pills, quoting UI strings verbatim.
- `docs/reference/fitting.rst` Batch tab section and
  `docs/reference/gui_usage.rst` (~1102): the series row, `New series ▾`,
  the selection hint, per-series range, "Fit this group…" behaviour.
- `docs/reference/project_files.rst`: `recipe`, `trend_excluded_runs`,
  `active_series`, dropped slot fields, `.bak` / autosave files.
- `docs/ARCHITECTURE.md` § "DataGroup and FitSeries" and § "ProjectModel":
  remove divergence/superseding, describe the recipe, active series and
  overlay keying.
- Screenshots: update `batch_tab_group_binding` (now shows the series row)
  and add `batch_tab_series_menu` (selector open) and
  `plot_fits_on_run` (two series on one run); register both in
  `docs/screenshots/capture.py`; stay within the size budget.
- `CHANGELOG.md` `[Unreleased]`: Added (series editor, active series,
  overlay pills, crash-safe save, autosave), Changed (identical re-run
  replaces, otherwise new series; single fits no longer join or diverge a
  series), Removed (divergence marking).

### Phase 7 — Validation and PR (lead)

`python tools/harness.py structural`, `lint`, `validate`, `gui-smoke`,
`docs`; open the PR with the decision log summary, the canvas link and the
migration notes; propose the release bump afterwards per `RELEASING.md`.

## Risks / watch items

- **`mainwindow.py` blast radius.** Phases 2 and 3 touch the highest-traffic
  seams. Mitigation: each brief lists the exact functions; the review gate
  greps for leftover member-slot writes and `FitSlot` flag reads.
- **Grouped (Individual groups) surface.** `MultiGroupFitWindow` records
  `member_kind="groups"` series through the same choke point; D4 applies
  (no member slots), but the series editor UI (Phase 3) is the F-B Batch tab
  only. The grouped window keeps its current single-state form; note it in
  the docs as unchanged.
- **Wizard flows.** The Global Fit Wizard applies phases through the Batch
  tab with a bound group; Phase 3 must keep `apply_wizard_phase_assessments`
  working (it opens a draft per phase and runs it). Test
  `tests/gui/test_global_fit_wizard_phases_apply.py` is the gate.
- **Migration of orphan pointers.** v19 projects may hold member slots whose
  `batch_id` no longer exists; D11 drops them, losing nothing the series
  did not already hold. Pin with a test.
- **Overlay keying change.** Every reader of `_fit_curves[run]` must move to
  the keyed form; Phase 2 greps for both maps and `fit_metadata`.
