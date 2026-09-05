# Axis limits: one per-axis Auto/Hold policy for every plot representation

Status: implemented 2026-09-05 on `feat/axis-limit-policy` (one PR), built
phase by phase by subagents with a lead-agent review gate after every phase
(see "Gates"). Decisions that changed during implementation are listed under
"Decisions recorded".

## Problem

The plot panel decides *when* to reframe its axes with hidden state that the
Auto X / Auto Y toggles do not describe: a one-shot first-paint latch
(`_limits_initialized`), a content signature (`_framed_identity`), a single
shared hold flag (`_limits_user_locked`) set by any typed limit or zoom/pan
gesture and cleared by clicking *either* Auto button, and a sticky per-subplot
y cache (`_y_limits_by_polarization`). The main window layers more on top for
the frequency tab (capture-and-restore of limits around every render,
`clear(preserve_view_state=True)`, `view_reframed_on_last_draw`) and for
startup (QSettings limits that are applied and then discarded).

Verified consequences (2026-09-05 investigation, headless probes):

- Zoom on run A, browse to run B, click **Auto Y**: the x window jumps to the
  full data span and the drawn points stay clipped to the old window. The
  locked switch never refreshed the framed identity; releasing the shared
  hold let the deferred redraw re-arm first-paint framing for both axes,
  inside a refresh that could not re-decimate.
- **Auto X** discards a typed Y (and vice versa) on the next run switch,
  because the hold is one flag for both axes. The docs promise per-axis.
- **Auto X** on a time panel pads 5 % below t0; first paint clamps to t0.
- The stacked view never refits y on a run switch (cache is only cleared by
  `clear()` / `restore_state`); the single view refits both axes.
- Project restore locks the view silently and permanently; QSettings restore
  is discarded on the first plot.
- A mouse zoom or pan turns off *both* Auto toggles even when only x moved.

The ALC panel already has the model we want (`_auto_x`/`_auto_y` booleans,
fields as the held values, one `_axis_target(auto, bounds, fields)` resolver
run at the end of every render) — it just is not shared.

## Design (settled)

**Mental model.** Each axis is either **Auto** (follows the data on every
redraw: run switch, x edit, view-mode change) or **Held** (keeps its value
across everything). The toggle is the *only* answer to "does this axis
follow the data". Nothing reframes behind the user's back.

1. **`AxisLimitPolicy`** (new, Qt-free) in
   `src/asymmetry/gui/widgets/axis_limits.py`, beside `AxisLimitControls`.
   Axis states keyed by string id (`"x"`, `"y"`, and in stacked views one y
   per subplot: `"y:P_x"`, `"y:group:2"`, …). Per axis: `auto: bool`,
   `held: (lo, hi) | None`, `quantity: str | None`.
   - `resolve(bounds, quantities) -> dict[axis, (lo, hi)]` is the single
     decision point: every limit a render applies comes from it (a render
     resolves x first, then each y inside that window). Per axis: Auto → use the
     supplied bounds; Held → use `held`; **an axis with no `held` value
     has nothing to hold and takes the bounds** (this *is* the first frame —
     no latch); a changed `quantity` on a Held axis refits it once from the
     bounds (view mode, waterfall stacking, domain unit without a converter).
     Bounds `None` (no data) leave the axis unchanged. Every resolved value
     becomes the new `held`, so the fields and the axes never disagree.
   - `set_manual(axis, lo, hi)` (typed field), `set_auto(axis, on)`
     (toggle), `record_gesture(before, after)` (zoom/pan end: only axes whose
     limits moved become Held at their new value), `convert(axis, fn,
     quantity)` (unit switch with a converter, e.g. MHz↔field: converts
     `held` in place and stamps the new quantity so no refit happens),
     `reset()` (forget every `held` — used only at project teardown),
     `state()` / `restore(state)` for project files.
   - New axes default to **Auto off** (see "Decisions recorded"): the
     first render frames an axis that has nothing held, and every later
     render holds until the user turns a toggle on.
2. **Fields and buttons are a view of the policy.** Fields display the
   resolved values; committing a field calls `set_manual`. The Auto buttons
   are the truth for the Auto flags: every resolve first syncs them into the
   policy (`_sync_auto_from_buttons`), a click simply re-renders, and a
   gesture or restore mirrors the policy back onto the buttons — so the two
   can never disagree.
3. **Render order flips to one pass:** resolve x from full-resolution
   bounds → decimate for that window → draw → resolve y within the window →
   apply → draw once. No "one view behind" refresh, no
   `schedule_viewport_refresh` for reframes. (Gestures still redraw at
   gesture end, through the same path.)
4. **Gestures.** At nav-mode button press capture every axis's limits; at
   release compare and call `record_gesture`. The `xlim_changed`/
   `ylim_changed` callbacks, `_sync_limits_from_axes`, the
   `_syncing_limits_from_axes` re-entrancy flag, and the draw-event sync are
   deleted; fields update at gesture end.
5. **Quantities.** Time panel: x `"time"`, y = the time-view mode plus the
   waterfall stacking key (enabled + Δ). Frequency panel: x = unit/mode key,
   converted via `convert` by the existing unit-switch code; y = the density
   key, likewise converted. Log-count scaling stays matplotlib's (out of
   scope).
6. **Stacked / grouped / MaxEnt views** register one y axis per subplot. The
   Y fields drive the selected (fit-target) subplot's axis; the Auto Y
   button sets every y axis. Per-projection memory falls out of the axis
   ids (a `"y:P_x"` hold survives visiting `"y:P_y"`), replacing
   `_y_limits_by_polarization` and the cache/restore pair. MaxEnt joins the
   fields (today it bypasses them).
7. **Main window.** View modes apply limits via `set_manual`. The frequency
   tab's capture/restore of limits, `preserve_x_limits`,
   `clear(preserve_view_state=...)` and `view_reframed_on_last_draw` go:
   a blank canvas never touches held values, so browsing onto an uncomputed
   run needs nothing. `policy.reset()` is called only from project
   close/new. QSettings limit save/restore is removed. `restore_state`
   restores toggles + held values; no lock.
8. **Guard rails.** A harness structural rule
   (`find_axis_limit_policy_violations`): in the policy-owning panels a
   `set_xlim(`/`set_ylim(` call may appear only inside the function that
   applies the resolved dict (`_apply_limits`, the GLE export helper,
   `_apply_axis_limits`); elsewhere only in an explicit allowlist of
   surfaces that are not axis-limit-policy plots. A contract test
   (`tests/gui/test_axis_limit_contract.py`) runs one scenario table against
   six representations. Docs rewritten once, referenced from the ALC page.

Decisions recorded: **Auto defaults off for a new session** (changed from the
draft: the policy's "an axis with nothing held takes the bounds" rule frames
the first dataset on its own, so buttons-off gives held-by-default browsing
directly); a blank canvas never resets holds; only project teardown resets;
QSettings limits are dropped; zoom/pan marks only the moved axes Held; an
all-NaN stacked pane frames to the neutral ±0.3 range and then holds like
any pane; the ALC scan's Auto Y frames the whole scan (not the visible x
window) so drag handles stay reachable — the contract test documents this
as the one representation-specific exception; the individual-groups export
can stack panes that are not on screen, so a pane the policy has never
framed takes the focused pane's window from the fields; the grouping
preview pane keeps its own Home scheme (separate dialog, later adoption);
fit range is untouched (independent of the view, seeds once from the full
extent). Phases 3 and 4 were run as one agent task (the `_apply_limits`
rewrite reaches every render path at once); Phase 5's agent was cut off by a
rate limit after finishing its diff and the lead committed it, plus a
follow-up making `axis_limits` optional in `PlotPanel.restore_state` like
the method's other keys (hand-built partial states are a legal input; the
schema migration completes real files).

## Agent rules (embedded verbatim in every phase prompt)

- **No defensive guards.** Do not add `hasattr`/`getattr(..., None)` on the
  panel's own attributes, `try/except` around our own code, re-entrancy or
  "in progress" flags, or `if x is None: return` where the design makes
  `None` impossible. Prevent the bad state by construction instead (e.g. an
  axis with no held value frames itself; there is no "initialized" latch).
  Existing `_has_mpl` checks stay; do not add new ones.
- **If you cannot see how to prevent a bad state by construction, stop.**
  Put the question under "Questions for the advisor" in your final report
  and leave that feature unimplemented. Do not add a guard "for now". The
  lead answers and resumes you with your context intact.
- **Delete, don't deprecate.** No compatibility shims, no `# removed`
  comments, no keeping an old method because a test used it. Rewrite the
  tests that pin old behaviour; never skip or xfail them.
- Work only on `feat/axis-limit-policy`. Commit once at the end of the
  phase with a conventional message (`refactor(plot): …`) ending in the
  Co-Authored-By trailer from CLAUDE.md. Do not push.
- Run tests only through `python tools/harness.py test -- <files>` (it
  re-execs into `.venv` and sets `QT_QPA_PLATFORM=offscreen`). Run
  `--tier fast` once before committing. Never run `validate`; the lead does.
- Docstrings yes; Sphinx docs and CHANGELOG no (Phase 6 owns them).
- Final report: files touched, what was deleted, test command + result
  summary, and "Questions for the advisor" (may be empty).

## Phases, agents and gates

Each phase is one subagent task on the feature branch, committed at the end
of the phase. The lead reviews every diff for conciseness (nothing kept that
the design makes unnecessary), robustness (no path where fields and axes can
disagree; no reframe not explained by a toggle, a first frame, or a quantity
change), and rule compliance, then runs the phase's tests before the next
phase starts.

### Phase 1 — `AxisLimitPolicy` + unit tests (Opus)

Files: `src/asymmetry/gui/widgets/axis_limits.py` (add the policy; leave
`FloatLimitField`/`AxisLimitControls` unchanged), new
`tests/gui/test_axis_limit_policy.py`.

Pure Python, no Qt. Implement the API in Design §1 exactly. Tests: first
frame from bounds; Auto follows changed bounds; Held ignores them; a typed
value holds; `record_gesture` marks only moved axes; quantity change refits
a Held axis once and stamps the quantity; `convert` changes `held` without a
refit; `None` bounds leave an axis unchanged; `reset` re-frames on the next
resolve; `state`/`restore` round-trip; unknown axis ids in `resolve` are
created Auto. Gate: that file + `--tier fast` green; lead API review before
Phase 2.

### Phase 2 — ALC panel adopts the policy (Sonnet)

Files: `src/asymmetry/gui/panels/alc_panel.py`, affected tests
(`tests/gui/test_rf_scan_panel_gui.py`, `tests/gui/test_integral_scan_gui.py`,
`tests/gui/test_alc_state_reset_on_new_project.py`).

Replace `_auto_x`/`_auto_y`, `_axis_target`, and the body of
`_apply_axis_limits` with one policy instance; `toggled` → `set_auto`; field
commit → `set_manual`; project reset → `policy.reset()`. Behaviour is
unchanged except that Auto defaults on (already true here). This is the
smallest caller and proves the API shape before the plot panel. Gate: ALC
tests green; lead confirms the panel got shorter, not longer.

### Phase 3 — Plot panel: single, overlay and frequency paths (Opus)

Files: `src/asymmetry/gui/panels/plot_panel.py` (`plot_dataset`,
`plot_datasets`, `_apply_limits`, auto/limit handlers, gestures, state),
`tests/gui/test_plot_panel.py`, `tests/gui/test_plot_panel_waterfall.py`.

Delete: `_limits_initialized`, `_limits_user_locked`, `_framed_identity`,
`_current_frame_identity`, `_reframe_if_content_changed`,
`_mark_frame_initialized`, `_apply_auto_limits_if_enabled`,
`_on_auto_x_button_clicked`/`_on_auto_y_button_clicked`,
`_clear_auto_limit_toggles`, `_connect_axis_limit_callbacks`,
`_disconnect_axis_limit_callbacks`, `_on_axis_limits_changed`,
`_sync_limits_from_axes`, `_syncing_limits_from_axes`, the limit sync in
`_on_canvas_draw_event`, the lock in `_on_canvas_button_release`,
`_last_draw_reframed`/`view_reframed_on_last_draw`, `clear(preserve_view_state)`,
the `schedule_viewport_refresh` plumbing for reframes, and the lock/latch
lines in `restore_state`.

Add: one `AxisLimitPolicy`; the one-pass render order (Design §3) in
`plot_dataset`/`plot_datasets`; `_auto_x_limits`/`_auto_y_limits` become
"set_auto + redraw" (their framing functions become the *bounds* suppliers —
`_default_x_limits` for x on both domains, so Auto X and first frame agree,
`_signal_y_limits_from_last_plot` for y); gesture capture/compare (Design
§4); quantity keys (Design §5) with the existing unit-switch code calling
`convert`; `get_state`/`restore_state` carry `policy.state()`;
`set_view_limits` → `set_manual` both; `clear()` blanks the canvas only and a
new `reset_view_limits()` calls `policy.reset()`.

Rewrite the tests that pin the old contract (first-paint-on-switch, zoom
lock, stale-latch, programmatic-vs-gesture, auto-toggle-reapply) to the new
one; keep every framing-function test (t0 clamp, saturation sentinels,
frequency peak framing) — those functions are unchanged. Add the four-step
reproduction from the investigation as a regression test. Gate: both test
files + `--tier fast` green; lead review; `validate` after this phase.

### Phase 4 — Stacked, grouped and MaxEnt paths (Opus)

Files: `src/asymmetry/gui/panels/plot_panel.py` (`plot_vector_subplots`,
`plot_grouped_time_domain_subplots`, `_plot_maxent_reconstruction_*`,
projection/fit-target switching), `tests/gui/test_plot_panel.py`,
`tests/gui/test_maxent_reconstruction_gui.py`, `tests/gui/test_vector_pane_collapse.py`.

Register one y axis per subplot (Design §6). Delete
`_y_limits_by_polarization`, `_cache_current_y_limits_for_axis`,
`_restore_y_limits_for_axis`, `_frame_subplot_axes_to_signal`, and the
y-cache handling in `_sync_y_controls_with_visible_axis`,
`_on_projection_selection_changed`, `set_fit_target_projection`, `get_state`/
`restore_state`. `_apply_limits` applies the resolved dict; the Y fields
mirror the fit-target subplot's axis; Auto Y sets every `"y:*"` axis. MaxEnt
supplies bounds and goes through the same resolve. Gate: the three test
files + `--tier fast`; lead review; `validate`.

### Phase 5 — Main window integration (Sonnet)

Files: `src/asymmetry/gui/mainwindow.py`, `tests/gui/test_mainwindow_additional.py`,
`tests/gui/test_project_schema.py` if the plot-state schema changes.

Delete the frequency preserved-limits capture/restore in
`_sync_frequency_plot_for_run` / `_render_frequency_spectra` and the
`preserve_x_limits` parameter and its two callers; the
`_limits_initialized` pokes and the QSettings limit save/restore in
`closeEvent`/startup; `clear(preserve_view_state=True)` calls become plain
`clear()`. Project close/new calls `reset_view_limits()` on both panels.
`_apply_view_mode` keeps using `set_view_limits`. Rewrite the frequency
preserve/reframe tests to the new contract. Gate: that test file (focused
classes) + `--tier fast`; lead review.

### Phase 6 — Guard rails, docs, changelog (Sonnet)

Files: `tools/harness.py`, `tests/tools/` (harness rule test), new
`tests/gui/test_axis_limit_contract.py`, `docs/reference/gui_usage.rst`,
`docs/reference/alc_mode.rst`, `docs/GUI_GUIDELINES.md`, `CHANGELOG.md`,
`docs/ARCHITECTURE.md` module map if it lists the deleted helpers.

Harness rule per Design §8 with an explicit allowlist, mirroring the
`FigureCanvasQTAgg` rule. Contract test: one scenario table (browse holds;
Auto Y follows a typed x; a horizontal zoom keeps Auto Y; Auto X leaves a
typed y alone; view-mode change refits only y; project restore holds; blank
canvas keeps holds) parametrised over single, overlay, stacked, grouped,
frequency and ALC. Docs: rewrite the "Axis limits" section around the
toggles; ALC and frequency pages point at it. Gate: `structural`, `docs`,
contract test green.

### Phase 7 — Lead: final review, validate, PR

`python tools/harness.py validate`, a last conciseness pass over the whole
diff (anything the design makes unnecessary that survived), update the
`docs/PLANS.md` entry, push, open the PR.

## Gates

After every phase the lead (advisor) reads the full diff before running
anything, answers the phase's "Questions for the advisor" (resuming the
same agent so it keeps its context), and only then runs the phase's tests.
A phase is accepted when: the diff deletes at least what its "Delete" list
names; no new guard of the kinds listed in the agent rules appears; the
fields, the axes and the policy cannot disagree on any path the phase
touches; and the tests named in the gate are green.
