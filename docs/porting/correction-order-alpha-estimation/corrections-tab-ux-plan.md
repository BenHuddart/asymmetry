# Corrections-tab UX follow-ups — implementation plan

Status: **DONE** (designed and executed 2026-07-17 with Ben; landed as five
commits on `feat/correction-order-alpha-estimation`: M1 `8b52aa0`, M2 `952cac9`,
saturation-readout fix `3f9d9cb`, M3 `a4df936`, integration fix `9f73250`).
Built on the completed interactive Corrections tab (compare toggles + pipeline
strip, commits `2dc2f57`..`1773312`).

**M4 addendum (2026-07-17, after the original three milestones): the tabs are
gone.** Ben-approved follow-on landed as `6299152` (plus live-GUI fixes
`5927252`, `342e6e3`, `1ae6117` in between): the right pane is now a full-width
pipeline strip over two side-by-side columns — "Grouping and timing" (natural-
width fields, stretch 0) and "Corrections" (stretch 1) — with the pager and
preview below both, at a 1220×680 default that needs no scrolling in either
axis (pinned by `test_both_columns_fit_without_scroll_at_default_size`).
Everything α (value, provenance, result, staleness banner, vector table) is
unified in an α area in the corrections column; the ⚠ tab marker became a
" · stale" suffix + tooltip on the pipeline α chip (`_alpha_is_stale`). The
adaptive-deadtime collapse (M1) is what made the columns short enough to
coexist — the tabs existed to stack content that no longer needed stacking.
Known accepted looseness: vector mode scrolls ~38px at the default height
(pill covers it); cross-OS font metrics give ~21px vertical / ~32px horizontal
slack on the fit test (measured on macOS; Linux CI will confirm).

**M5 addendum (2026-07-18): correction cards + consolidated compare + preview
polish (`e8a65b4`, Ben-approved via mockups).** Each correction is a
`CorrectionCard` (grouping-local widget — clickable status header, stale
warn-tint, accent "comparing: …" state; expanded-iff-active default recomputed
per open, deliberately NOT QSettings-persisted so a saved collapse can never
hide an active correction on another profile). The three per-section
"Compare in preview" checkboxes are retired — compare focus is controlled by
the pipeline chips + pager only and displayed on the focused card; column
headers use `make_section_header`. Preview: y-axis follows the solid curve
alone (an uncorrected FLAME ghost reaches ~1e7% and used to crush the view);
ghost thinner/dimmer with an inline clamped label; legend removed. Also fixed
the pre-existing α run-combo minimum-width overflow (alpha_section.py). Vector
mode now fits the default height entirely. **Deferred with recorded design: the
difference strip** — a ~40px Δ = (with − without) trace under the main axes,
shown only while a compare is focused; the only display that shows a
correction's effect when it is smaller than a line width; both curves are
already retained in `_PreviewResult`, so it costs the same whenever built.

Execution notes (what the plan didn't foresee):

- **The pager row consumed the M1 fit budget.** Adding the M3 pager row under
  the tabs shrank the tab viewport 379→348 px, silently re-breaking M1's
  "default state fits without outer scrolling" acceptance (content 366 px) —
  caught at the closeout advisor review, not at either milestone gate, because
  no test pinned the whole-tab budget. Fix (`9f73250`): the compare footer was
  redundant once the pager shipped ("vs raw" is a pager stop), so the
  "Compare vs raw (uncorrected)" checkbox moved into the pager row and the
  badge sentence became the pager label's tooltip (content now 306 px);
  `test_corrections_tab_fits_without_scroll_at_reference_size` pins the budget.
- The deadtime max-correction readout could display a clamped ~1e8% when the
  correction saturates; fixed (`3f9d9cb`) with a "deadtime saturates the t=0
  correction" warning past 100%.
- Headless-render gotcha: driving dialog state directly (bypassing section
  seams) marks the draft dirty, so `dialog.close()` pops the discard-guard
  modal and hangs offscreen scripts — use `_teardown_workers()` +
  `deleteLater()` in verification scripts instead.
- Benign leftover: the file/estimate "Show per-detector values" disclosure
  keeps its expanded state across mode switches (collapsed on fresh
  `configure()`, which is what the spec required).

## Problem and agreed scope

At the default dialog size (1180×760) the Corrections tab viewport is ~379 px
but its content is ~640 px, because the deadtime section is 313 px tall (192 px
of it the per-detector table) **even when deadtime is off**. Background (42 px)
and α (90 px) — the sections users need to discover — sit entirely below the
fold. Measured with an offscreen render; the Grouping tab fits in single mode
(309 px) but can overflow in vector mode (per-projection α table).

Three milestones, in order:

1. **M1 — adaptive deadtime section**: the section only shows the controls its
   mode uses; the default state fits the tab without outer scrolling.
2. **M2 — named overflow indicator**: a viewport-overlay pill listing the
   sections hidden below the fold, for smaller windows; attached to both tabs.
3. **M3 — compare pager**: `◀ ▶` at the preview that walks `_compare_stage`
   through the configured corrections (replaces the earlier hold-to-swap idea).

Explicitly deferred (recorded so the decisions aren't lost):

- **Cumulative build-up stepper** — if ever built, use the *converging-ghost*
  shape: solid stays the full reduction, the ghost steps Raw → +Deadtime →
  +Background → +α and converges onto it. Steps 1/3/4 are the existing
  `compare_stage` ghosts ("raw", "alpha", none); only "+Deadtime"
  (`_reduce(True, False)`, α=1) is new. Never a partial-pipeline solid, never a
  demonstration mode.
- **Hold-to-swap peek flip** — momentary ghost↔solid promotion; redraw-only
  (requires retaining the last `_PreviewResult` and the ghost's real errors).
  Superseded by the pager; still cheap to add later if wanted.
- Changelog enumeration of the whole corrections-UX commit set: merge-time
  task, not part of these milestones.

## Execution model

Orchestrator: Fable (this session) — designs stay fixed, reviews every diff,
owns all commits. Implementation: one subagent per milestone, run
**sequentially** (shared `dialog.py`, and the never-two-concurrent-test-suites
rule).

Ground rules for every subagent prompt:

- All checks via `python tools/harness.py …` (re-execs into `.venv`,
  offscreen Qt). Subagents run **only focused test files**
  (`test -- tests/gui/test_x.py`), never `--tier fast`/`validate`.
- Leave the working tree **uncommitted**; the orchestrator reviews, requests
  fixes, then commits. No push, no PR.
- Match surrounding code style; comments state constraints, not narration.
- The hard preview invariant stands: the solid curve is always the full
  configured reduction; nothing preview-only reaches
  `_current_grouping_payload`.

Per-milestone review gate (orchestrator):

1. Read the full diff.
2. Headless render of the affected states to PNG (offscreen Qt) and inspect —
   the geometry numbers above are the acceptance baseline.
3. `harness.py lint` + `structural`; re-run the changed test files once
   (serialized after the subagent finishes).
4. Fix-or-bounce; then commit with a `feat(grouping):`/`fix(grouping):`
   message.

After M3: consult the built-in advisor on the combined result, run
`harness.py validate` once (orchestrator, alone), regenerate + inspect the
`alpha_calibration_dialog` screenshot scenario, then close out docs + memory.

---

## M1 — adaptive deadtime section  *(agent: Opus)*

**Goal**: default Corrections-tab state (deadtime off) fits ~340 px < 379 px
viewport with no outer scroll; every mode shows only what it uses.

**Files**: `src/asymmetry/gui/windows/grouping/deadtime_section.py`,
`tests/gui/test_deadtime_section.py`,
`docs/screenshots/scenarios/alpha_calibration_dialog.py`,
`docs/reference/data_reduction/detector_grouping.rst`.

Per-mode layout (drive from `_on_mode_or_state_changed`, switching from
`setEnabled` to `setVisible` for whole rows):

- **Off**: mode radios + the "disabled" hint only. No estimate row, no table
  controls, no table, no summary.
- **From file**: radios + one summary line — mean value × N detectors + the
  max-correction-at-t=0 figure (extend `_refresh_summary`) — plus a
  "Show per-detector values" disclosure (checkable `QToolButton` with arrow)
  that reveals the read-only table on demand, collapsed by default.
- **Estimate from run**: radios + the source-run/Estimate row + the same
  summary line + the same collapsed disclosure.
- **Manual**: radios + table controls (Fill-all / Cal) + the table, always
  visible but height-capped at ~6 rows (`rowHeight×6 + header + frame`,
  `setMaximumHeight`); rows beyond that scroll *inside the table* — no outer
  scrolling by design (Ben's call). Summary line shown.

Constraints:

- `configure()` / `get_policy()` contracts unchanged — the dialog integration
  and the section's source-of-truth pattern must not move.
- Mode changes must invalidate layout so the scroll-content height actually
  shrinks (`updateGeometry`/`adjustSize` on the section as needed).
- Keep the section header row (title + "Compare in preview") untouched — it
  belongs to the dialog, not this widget.

Tests (extend `test_deadtime_section.py`): per-mode visibility matrix (off
hides table+controls+estimate row; file/estimate show summary, disclosure
starts collapsed and toggles the table; manual shows capped table); height cap
holds at 64 detectors; all existing behaviour tests stay green unmodified
unless they asserted the old always-visible layout (then retarget, don't
delete).

Screenshot scenario: with α now above the fold, drop the blind
scroll-to-maximum in favour of `ensureWidgetVisible(alpha section)` (harmless
no-op at default size, still correct on small captures).

Verification renders: PNG per mode at 1180×760 + printed
`content.sizeHint().height()` vs viewport; off/file/estimate must fit, manual
may exceed only via the table cap being reached.

---

## M2 — section overflow indicator  *(agent: Opus)*

**Goal**: on windows small enough that sections still fall below the fold, a
compact pill overlays the scroll viewport's bottom edge naming them —
`↓ Background · α (detector balance)` — click scrolls the first one into view;
the pill vanishes when nothing is hidden.

**Files**: new `src/asymmetry/gui/widgets/section_overflow_indicator.py`, new
`tests/gui/test_section_overflow_indicator.py`, wiring in
`src/asymmetry/gui/windows/grouping/dialog.py`,
`docs/ARCHITECTURE.md` (gui/widgets inventory),
`docs/reference/data_reduction/detector_grouping.rst` (one sentence).

Design:

- `SectionOverflowIndicator(scroll_area, sections)` where `sections` is a
  callable returning the current `list[(label, QWidget)]` (callable because
  vector mode hides the α section and M1 changes section heights live).
- Overlay child of the scroll area, right-aligned just above the viewport's
  bottom edge, semi-opaque pill styled from `tokens`; `raise_()`d; accepts
  clicks on itself only (small footprint, never intercepts wheel events).
- Hidden-below test per section: `isVisible()` and
  `mapTo(viewport, rect.topLeft()).y() >= viewport.height()`.
- Recompute on: vertical scrollbar `valueChanged` + `rangeChanged` (covers
  content-height changes from M1's mode switching), viewport `resizeEvent`
  (via event filter), and an explicit public `refresh()` the dialog calls when
  section visibility changes (vector-mode toggle).
- Click → `ensureWidgetVisible(first hidden)`; recompute follows via
  `valueChanged`.

Wiring: attach to `_corrections_scroll` with the three correction headers
(Deadtime / Background / α — label text without the "correction"/"subtraction"
suffixes, α omitted when vector) and to `_grouping_scroll` with coarse
landmarks: per-projection α table (vector only), "t0 and binning" (the t0 row),
"Periods". Landmark granularity is a review point — err coarse.

Tests: offscreen scroll area with tall content — pill visible + labels correct
at top, click scrolls and pill updates/hides at bottom, `refresh()` reflects a
section hidden via `setVisible(False)`, pill absent when content fits.
Structural gate: new widget file + test location must pass
`harness.py structural` untouched (no new harness rule needed).

Verification renders: Corrections tab at a deliberately short size (e.g.
1180×560) showing the pill; default size showing no pill.

---

## M3 — compare pager  *(agent: Sonnet)*

**Goal**: `◀ ▶` + label in a dialog-owned row directly above the preview pane,
cycling `_compare_stage` through `[None, "deadtime", "background", "alpha",
"raw"]`, skipping stages `_compare_stage_available` rejects. Works from both
tabs (the preview is pinned below both).

**Files**: `src/asymmetry/gui/windows/grouping/dialog.py`, new
`tests/gui/test_compare_pager.py` (beside the other grouping GUI tests),
`docs/reference/data_reduction/detector_grouping.rst` ("Comparing a
correction's effect" section).

Design:

- `_build_compare_pager()` → row inserted in `right_layout` between `_tabs`
  and `_preview_pane`: two arrow `QToolButton`s (autoDefault/default False,
  tooltips "Previous/Next comparison") + a muted label.
- Step logic: from the current stage, advance through the cycle to the next
  available stage (None is always available). Pure wrapper over
  `_set_compare_stage` — no new state beyond the widgets.
- Label: `Comparing: off` / `Comparing: without deadtime (1/4)` /
  `Comparing: α = 1 (3/4)` / `Comparing: vs raw (4/4)`, where the count is the
  number of currently-available stages (recomputed live, so it reads (…/3)
  with background off, etc.).
- Sync: `_sync_compare_toggles` (the existing single sync seam) also refreshes
  the pager label + disables both arrows when no stage is available. Toggles,
  chips and pager all drive the one `_compare_stage`; no new invariants.
- No preview-pane changes at all; each step rides the existing debounced
  off-thread request path.

Tests: cycle order + wraparound; unconfigured stages skipped (deadtime off →
▶ from None lands on background); vector mode skips α; label text matches the
stage and live count; toggling a section checkbox updates the pager label
(shared-state sync); pager click issues a preview request with the right
`compare_stage` (mirror the existing `test_compare_toggles_*` request
assertions). Run the new file + `test_grouping_dialog.py` 3× serially (the
change drives the preview worker cadence).

---

## Closeout (orchestrator, docs sub-tasks to Sonnet)

- Advisor consult on the combined diff before declaring done.
- `harness.py validate` once (orchestrator, alone).
- Regenerate the `alpha_calibration_dialog` scenario PNG locally; inspect.
- Docs sweep: this file's status flipped to done;
  `README.md` decision log entry (adaptive layout, pill, pager; hold-to-swap
  dropped; stepper deferred with the converging-ghost shape);
  `ARCHITECTURE.md`; `detector_grouping.rst` already updated per milestone.
- Memory: update `project_correction_order_alpha.md`.
- Three local commits (one per milestone), nothing pushed.
