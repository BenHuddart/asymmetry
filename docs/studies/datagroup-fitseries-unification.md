# DataGroup / FitSeries Unification (Option C)

Status: implemented on branch `feat/datagroup-fitseries-unification`
(Phases 1–5 complete: core model, browser, fit flows, carry-forward, docs);
awaiting orchestrator's Phase 6 validation and PR review.

This document is written to be executed by subagents that have **no access to
the design conversation**. Each phase states its scope, the invariants it must
preserve, the tests that gate it, and the model tier recommended for the
executing agent. Line numbers are anchors into the tree at the time of
writing — always re-locate by symbol name (grep) before editing.

## 1. Problem

`DataGroup` and `FitSeries` are separate entities with a deliberately weak
link ("D1 Option B"): a series records `source_group_id` as provenance only,
groups never reference series, membership is frozen at record time, and the
browser enforces a one-group-per-run partition. Three overlapping affordances
("Send to Batch", "Fit this group…", "Share with Group") mutate three
different things, and none of them is "the group's batch fit". Two concrete
workflows the current model cannot express:

- one run participating in two scans (a field scan **and** a temperature
  scan) — blocked by the browser partition;
- a run that belongs to a scan organisationally but cannot share the scan's
  fit function — group membership and fit membership are the same thing today.

Additional structural debt this plan retires:

- Two `DataGroup` classes (core `core/representation/group.py`; GUI dataclass
  in `gui/panels/data_browser.py` ~line 193) mirrored only at save/load by
  `MainWindow._sync_data_groups_to_project_model` /
  `_seed_browser_groups_from_project_model` — an acknowledged stopgap.
- `MainWindow._data_group_id_for_runs` guesses provenance ("do all these runs
  share exactly one group?") because batch fits have no explicit group.
- `share_single_function_state` (`gui/panels/fit/panel.py` ~line 669)
  overwrites every group member's stored single-fit state unconditionally —
  no guard for existing fits.
- Naming collision: the cross-group trend fit's `_GroupFitData.source_group_id`
  (`gui/panels/fit_parameters_panel.py` ~line 4175) is unrelated to
  `FitSeries.source_group_id`.

## 2. Settled design (decision log)

All decisions below were made explicitly by the maintainer; do not re-litigate
them during execution.

**D1 — Groups own their fits.** A run-membered `FitSeries` belongs to a
`DataGroup`: `group_id` becomes a structural field (not provenance). A group
has zero or more series ("analyses"); the same scan can be fit with several
models. Effective series membership is **live-derived**:
`group.member_run_numbers − series.excluded_run_numbers`. Results remain a
snapshot of what was actually fit; when the derived membership differs from
the last-fitted membership the series is **stale** (surfaced like divergence,
cleared by re-running).

**D2 — Multi-group membership, duplicated rows.** A run may belong to any
number of groups. The browser shows one row per membership: the run's
*primary* membership (the first group it was added to) renders as today; each
additional copy carries a small marker glyph (①, ② …) and a tooltip naming
the run's other groups. Selection always dedupes to the underlying dataset —
a run selected via two copies reaches plotting/fitting/co-add exactly once.
Removing the primary membership promotes the earliest remaining copy.

**D3 — Batch fits auto-create groups.** Every batch/global run-series fit has
an explicit group: either the group it was launched from, or an auto-created
one built from the ad-hoc selection. Re-running over a member set identical
to an existing auto-group **reuses** that group (no proliferation). Auto-group
names derive from the member range and order key (follow
`core/representation/naming.py` conventions, e.g. "Runs 1001–1010").
`_data_group_id_for_runs` is deleted.

**D4 — Auto vs user groups are visually distinct.** `DataGroup` gains
`kind ∈ {"user", "auto"}`. User groups keep the blue ramp
(`GROUP_HEADER_BG` #c8d2e1 / `GROUP_MEMBER_BG` #ebeff7 in
`gui/styles/tokens.py`). Auto groups get a **new red-family ramp** — new
tokens mirroring the blue ramp (header ≈ #e1cdc8, member tint lighter than
`ACCENT_RED_SOFT` #f5dcd8). Red is the existing FitSeries brand colour
(`data_browser.py` ~line 147: series-selected rows highlight with
`ACCENT_RED_SOFT`), so the auto-group tint must stay clearly lighter/greyer
than the selected-series highlight, which must still read on top of it.
**Renaming an auto group promotes it to a user group** (kind flips, red →
blue).

**D5 — "Share with Group" is retired; carry-forward becomes
refresh-unless-fitted.** The Single tab's carry-forward
(`FitPanel._carry_forward_single_fit_form`) is upgraded:

- **Protected:** any run with a recorded fit result (a `FitSlot` whose result
  is populated — single fits *and* batch write-backs). Selecting it always
  restores its own fitted state. Never auto-overwritten.
- **Refreshable:** everything else — carried forms, domain defaults, and
  hand-edited-but-never-fit forms (deliberately: no dirty tracking; the
  protection trigger is "did you commit by fitting"). On selection these
  refresh from the most recent **fitted** function in the session.
- **Carry source is "last fitted", not "last displayed".** Fall back to
  today's last-displayed behaviour only when no fit exists yet.
- Field-dependent reseeding moves into carry-forward: frequency peaks
  (already done) **and** the per-target file-value reseeding (e.g. `B_L`)
  currently done by `share_single_function_state` via
  `_get_file_value_for_parameter`.
- The "Share with Group" action (`single_tab.py` ~line 190), the grouped
  button (`global_tab.py` ~line 362), `share_single_function_state`,
  `share_single_grouped_function_state`, and their MainWindow handlers
  (`_on_share_single_function_with_group`,
  `_on_share_grouped_function_with_group`, `_data_group_peer_runs`) are all
  removed.
- Protection is **derived, not persisted**: check the slot's recorded result
  directly (batch member slots are pointer slots with no `ui_state`, and the
  restore mediator `_single_fit_restore_payload` returns `None` for them, so
  do not use the mediator's return as the protection signal).

**D6 — Single source of truth for groups.** `ProjectModel.data_groups`
becomes canonical (name, members, order, kind). `DataBrowserPanel` becomes a
view/controller over it; the GUI `DataGroup` dataclass and the save/load
mirroring (`_sync_data_groups_to_project_model`,
`_seed_browser_groups_from_project_model`) are deleted. `collapsed` stays
GUI-side (it is view state; persist it in `browser_state` as today).

**D7 — Series identity and lifecycle.** Series identity/dedupe
(`_series_signature`, `remove_superseded_batches`,
`ProjectModel.dedupe_batches`) re-keys on
`(group_id, model signature, exclusions)` for group-bound series; re-running
a group analysis replaces in place by construction. Deleting a group with
analyses prompts the user: delete its series too, or keep them as **frozen
legacy series** (`group_id=None`, snapshot membership — exactly the pre-D1
semantics). Dangling `source_group_id` pointers no longer occur for new fits.

**D8 — Scope limits.** Detector-group series (`member_kind="groups"`,
`MultiGroupFitWindow`) keep their current frozen semantics — they are not
DataGroup-based. UI wording keeps "Batch" / "series" (no rename to
"Analysis" in this PR). The cross-group trend fit's unrelated
`source_group_id` field is renamed (e.g. `trend_group_id`) to end the
collision.

**D9 — Schema v14 → v15 migration.**

- `data_groups` entries gain `kind` (default `"user"`).
- Run-membered series gain `group_id` (from `source_group_id` when it
  resolves to a live group, else `None`), `excluded_run_numbers` (default
  empty), and `last_fitted_members` (initialised from the existing
  `member_run_numbers`).
- Group-less legacy series migrate to **frozen legacy** (`group_id=None`),
  not synthesized groups — old projects must not sprout red groups the user
  never made. A later re-run of such a fit creates its auto-group then.
- Detector-group series are untouched.
- Multi-membership requires no data migration (newly permitted).

## 3. Key code map (verify anchors before editing)

| Concern | Location |
| --- | --- |
| Core `DataGroup` | `src/asymmetry/core/representation/group.py` |
| GUI `DataGroup` dataclass, partition (`_run_to_group`), group CRUD, context menu | `src/asymmetry/gui/panels/data_browser.py` (~193, ~969–1060, ~3156–3360) |
| `FitSeries` (fields, `source_group_id`, signature exclusion) | `src/asymmetry/core/representation/series.py` (~58–121) |
| `ProjectModel` (batches, data_groups, `series_for_group`, `remove_data_group`, `_series_signature`, dedupe/supersede) | `src/asymmetry/core/representation/project_model.py` |
| Schema version + migrations | `src/asymmetry/core/project/schema.py` (`CURRENT_SCHEMA_VERSION`) |
| Series recording choke points | `mainwindow.py` `_record_fit_series` (~10599), run-series ctor (~10736), grouped (~11443), field-scan (~11747) |
| Group→fit entry, provenance guess, group-series filter | `mainwindow.py` `_on_fit_group_requested` (~11548), `_data_group_id_for_runs` (~11507), `_on_show_group_series_requested` (~11587) |
| Group registry mirroring (to delete) | `mainwindow.py` `_sync_data_groups_to_project_model` (~14350), `_seed_browser_groups_from_project_model` (~14323) |
| Single-fit state store, restore precedence, carry-forward, batch write-back, share (to retire) | `gui/panels/fit/panel.py` (`_single_state_by_run` ~98, `set_dataset` ~315, `_carry_forward_single_fit_form` ~424, `register_global_fit_results` ~822, `share_single_function_state` ~669) |
| Single-tab actions (Share with Group ~190, Send to Batch ~194, Add to Series ~199) | `gui/panels/fit/single_tab.py` |
| Grouped share mirror (to retire) | `gui/windows/multi_group_fit_window.py` (~436) |
| Colour tokens | `src/asymmetry/gui/styles/tokens.py` |
| Cross-group `source_group_id` collision | `gui/panels/fit_parameters_panel.py` (~4175), `cross_group_config.py` |

## 4. Execution plan

Branch: `feat/datagroup-fitseries-unification` off `main`. One PR; each phase
is one or more commits. Phases are strictly ordered — later phases assume the
earlier ones' invariants.

**Orchestrator review gate (every phase):** before starting the next phase,
the orchestrating agent reviews the diff against the phase checklist below,
runs the phase's focused tests plus `python tools/harness.py test --tier
fast` (and the named GUI test files for GUI phases), and checks the standing
invariants: no Qt/matplotlib imports in `core`, `harness structural` green,
no new bespoke widgets where `gui/widgets/` foundations exist. Fix-ups happen
inside the phase, not as debt carried forward. Run `python tools/harness.py
validate` once, at the end of Phase 6, not per phase.

### Phase 1 — Core model rework (agent: Opus)

Scope: `core/representation/{group,series,project_model}.py`,
`core/project/schema.py`, tests under `tests/core/` and `tests/project/`.
Pure core — no GUI edits.

- `DataGroup`: add `kind` (`"user"`/`"auto"`, validated, default `"user"`);
  drop any implicit single-membership assumptions (core already tolerates
  overlap — make it explicit in the docstring and tests).
- `FitSeries`: add `group_id: str | None` (structural), `excluded_run_numbers:
  list[int]`, `last_fitted_members: list[int]`. Effective membership helper
  `effective_members(group)` = group members − exclusions, in group order.
  Staleness helper `is_stale(group)` = effective ≠ last-fitted. Keep
  `source_group_id` reading tolerated on load but stop writing it (fold into
  `group_id` migration).
- `ProjectModel`: `data_groups` is canonical (add mutation API the browser
  will call: create/rename/set-members/set-kind/delete, emitting no Qt —
  plain methods). `remove_data_group(group_id, *, orphan_series)` implements
  D7 (delete series or freeze them to `group_id=None` snapshots).
  `series_for_group` keys on `group_id`. Re-key `_series_signature`,
  `remove_superseded_batches`, `dedupe_batches` per D7 (group-bound series:
  `(group_id, model signature, exclusions)`; frozen/legacy and
  detector-group series: unchanged frozen keying).
- Auto-group reuse helper: find an existing `kind="auto"` group with an
  identical member set.
- Schema v15 + migration per D9, with round-trip tests: v14 project with
  (a) group-linked series, (b) group-less series, (c) detector-group series,
  (d) pre-Phase-7 project with no `data_groups` block.

Checklist for review gate: field semantics match D1/D7/D9 exactly; migration
is tolerant (never raises on legacy shapes); `source_runs()`/naming fallbacks
still work for frozen series; no behaviour change for
`member_kind="groups"`.

### Phase 2 — Browser: canonical registry + multi-membership + palette (agent: Opus)

Scope: `gui/panels/data_browser.py`, `gui/styles/tokens.py`, MainWindow
wiring; tests under `tests/gui/`.

- Rewire `DataBrowserPanel` group state onto `ProjectModel.data_groups`
  (panel holds a reference; `collapsed` stays panel-local keyed by group id).
  Delete the GUI `DataGroup` dataclass and the two mirroring methods in
  `mainwindow.py`. This touches ~30 read/write call sites — enumerate them
  first (`grep -n "_groups\|_run_to_group" data_browser.py mainwindow.py`)
  and convert mechanically.
- Multi-membership per D2: `_run_to_group` → `_run_to_groups` (ordered;
  first = primary); `_display_order` may list a run once per membership;
  duplicated rows get marker glyphs + tooltip; selection APIs
  (`get_selected_datasets` etc.) dedupe. "Send to Group" adds a membership
  (no longer steals); "Remove from Group" removes the clicked membership;
  primary promotion on removal per D2. `create_data_group` no longer strips
  runs from prior groups.
- Palette per D4: new tokens (`AUTO_GROUP_HEADER_BG`, `AUTO_GROUP_MEMBER_BG`,
  plus sel/focus variants mirroring the blue ramp), header/member painting
  switches on `kind`. Rename-promotes-to-user in the rename handler.
  Verify visually (offscreen screenshot) that the selected-series
  `ACCENT_RED_SOFT` highlight still reads on top of an auto-group tint.
- Keep `DataBrowserPanel.batch_updates()` semantics; group rebuilds stay
  O(n).

Checklist: no `_sync_data_groups_to_project_model` remains; partition
assumptions gone (grep for `get_group_id_for_run` single-value uses);
selection dedup covered by a test (same run via two groups → one dataset);
palette tokens named and reused, no literal hex in the panel.

### Phase 3 — Fit flows: group-bound batches, auto-groups, staleness (agent: Opus)

Scope: `mainwindow.py` record/entry choke points, `gui/panels/fit/panel.py`,
`gui/panels/fit/global_tab.py`, trend panel filter; tests under `tests/gui/`
and `tests/integration/`.

- "Fit this group…" binds the Batch tab to the group: member datasets =
  effective membership; a per-member include/exclude affordance in the Batch
  tab's member list edits `excluded_run_numbers` (exclusion is per-series,
  not per-group).
- Running a batch over an ad-hoc selection auto-creates (or reuses, D3) an
  auto-group before recording; `_record_fit_series` and the run-series /
  field-scan constructors set `group_id`, write `last_fitted_members`, and
  stop calling `_data_group_id_for_runs` (delete it and
  `_data_group_peer_runs`).
- Staleness surfacing: reuse the divergence display channel (trend panel row
  state / browser tint) for `is_stale`; re-run clears it. Membership edits
  to a group refresh staleness of its series (hook the ProjectModel mutation
  API from Phase 2).
- "Show series from this group" filters on `group_id`.
- Group deletion prompt per D7 (dialog: delete fits / keep frozen).
- Rename cross-group `_GroupFitData.source_group_id` → `trend_group_id`
  (mechanical; includes `cross_group_config.py`).

Checklist: every new run-membered series has a non-None `group_id`; re-run
of a group analysis replaces in place (no duplicate trend pills); ad-hoc
re-run reuses its auto-group; no caller of `_data_group_id_for_runs`
survives; detector-group recording path untouched.

### Phase 4 — Carry-forward rework, retire "Share with Group" (agent: Sonnet)

Scope: `gui/panels/fit/panel.py`, `single_tab.py`, `global_tab.py`,
`multi_group_fit_window.py`, `mainwindow.py` handler wiring; tests under
`tests/gui/` (existing carry-forward and share tests are the map — find them
with `grep -rl "carry\|share_single_function" tests/`).

- Implement refresh-unless-fitted per D5: track the session's most recent
  fitted single-tab state ("last fitted source"); in `set_dataset`, protected
  runs restore their own state (branches A/B as today when the slot has a
  recorded result); non-protected runs refresh from the last fitted source
  (superseding stale `carried_session` cache entries), badge updated to name
  the source run; fall back to last-displayed carry when no fit exists.
- Fold per-target field reseeding (`_get_file_value_for_parameter` logic on
  e.g. `B_L`) into the carry/refresh path alongside the existing frequency
  reseed.
- Remove the two "Share with Group" affordances, both handlers, both share
  methods, and their tests; adjust any docs strings referencing them.
- Batch write-back (`register_global_fit_results`) states count as fits —
  ensure the protection check treats batch member slots (pointer slots,
  no `ui_state`, mediator returns `None`) as protected.

Checklist: a run with a completed fit is never auto-overwritten (test both
single-fit and batch-member cases, and across save/load); an unfit run
refreshes when a newer fit lands elsewhere; hand-edited-unfit forms are
refreshable by design (assert, so the behaviour is pinned deliberately); no
`share_single_function_state` references remain.

### Phase 5 — Docs, screenshots, changelog (agent: Sonnet)

Scope: `docs/` reference pages + screenshot scenarios, `CHANGELOG.md`.

- Update the owning pages (find via `grep -rl "data group\|Share with Group\|Send to Batch\|FitSeries" docs/`) — at minimum
  `docs/reference/project_files.rst` (schema v15, new fields),
  `docs/reference/parameter_trending.rst` (group-bound series, staleness),
  the data-browser page (multi-membership markers, auto vs user groups,
  colours), and `docs/getting_started/key_concepts.rst`. Quote UI strings
  verbatim from the widget code. Remove "Share with Group" everywhere;
  document carry-forward refresh-unless-fitted.
- Update/add screenshot scenarios for: an auto (red) group beside a user
  (blue) group; a duplicated marked row; the Batch tab bound to a group with
  an exclusion. Respect the per-image size budget (`harness structural`
  enforces drift).
- `CHANGELOG.md` `[Unreleased]`: user-facing summary (groups drive batch
  fits; multi-group membership; auto-groups; Share with Group removed —
  call out the removal explicitly as a behaviour change).
- `docs/ARCHITECTURE.md`: rewrite the FitSeries/DataGroup sections (the
  "D1 Option B" description is now historical); update `group.py` /
  `series.py` module docstrings if Phase 1 left them stale.

### Phase 6 — Full validation + PR (orchestrator)

- `python tools/harness.py validate`; fix fallout.
- `python tools/harness.py gui-smoke`.
- Manual offscreen screenshot pass over the new browser visuals.
- Squash-tidy commits if needed; open the PR with a summary keyed to the
  decision log (D1–D9) and the migration notes.

## 5. Risks / watch items

- **Blast radius in `mainwindow.py`:** the record choke points and browser
  wiring are the highest-traffic code in the app. Mitigation: phases 2–3
  enumerate call sites before editing, and the review gate diffs against the
  checklist rather than trusting green tests alone.
- **Selection dedup regressions:** co-add, subtract, and batch member lists
  all consume browser selection; a missed dedup double-counts silently.
  A dedicated test (run in two groups, both copies selected) is mandatory.
- **Red-on-red legibility (D4):** if the auto-group tint and the
  selected-series highlight cannot be made distinct in practice, fall back to
  a neutral/violet auto-group ramp and keep red exclusively for the selected
  series — flag to the maintainer at the Phase 2 gate rather than shipping a
  muddy palette.
- **Carry-forward behaviour change (D5):** losing a hand-built unfit draft on
  navigation is accepted by design, but must be pinned by a test and called
  out in the changelog so it reads as intent, not regression.
- **Migration tolerance:** v14 projects in the wild include pre-Phase-7
  saves (no `data_groups`), duplicate-era saves (pre-dedupe), and partial
  `member_source_run` maps; the migration tests in Phase 1 must cover all
  three.
