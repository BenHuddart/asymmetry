# Shared-Foundations Audit — Behavior Changes

Every deliberate, user-visible behavior change made during this audit is logged
here (policy: **converge to best variant**). Each entry records **what** changed,
**where** (file/component + call sites), **which variant won**, and **why**.
This list is mirrored into the final PR description.

Format:

> ### <short title>
> - **What:** …
> - **Where:** …
> - **Winning variant / why:** …
> - **Phase:** …

---

<!-- entries appended below, newest last -->

### Axis-limit fields now clamp and commit on Return/Enter

- **What:** `plot_panel`'s X/Y axis-limit fields (and `alc_panel`'s ALC-plot
  X/Y axis-limit fields, which imported `plot_panel`'s class) now clamp
  out-of-range values to their validator range and commit on Return/Enter
  (and force a commit on focus-out for "Intermediate", not-yet-acceptable
  input). Previously they did neither: `setValue`/typed input could exceed
  any sane bound unclamped, and Return only committed when Qt's default
  `QLineEdit` judged the input `Acceptable` — an out-of-range or half-typed
  value could silently revert on the next external refresh.
- **Where:** `src/asymmetry/gui/panels/plot_panel.py` (`_x_min`/`_x_max`/
  `_y_min`/`_y_max` on `PlotPanel`) and `src/asymmetry/gui/panels/alc_panel.py`
  (`_x_min`/`_x_max`/`_y_min`/`_y_max` on the ALC panel). Both previously
  defined or imported their own `_FloatLimitField` (plot_panel's plain
  variant); both now construct
  `asymmetry.gui.widgets.axis_limits.FloatLimitField`, a new shared module.
  `src/asymmetry/gui/panels/fit_panel.py`'s fit-range min/max fields also
  migrated to the same class (no behavior change there — it was already the
  featured variant).
- **Winning variant / why:** `fit_panel._FloatLimitField` (the featured
  variant) won. It already had clamp-on-`setValue`/`setRange` and
  commit-on-Return/focus-out, added earlier to fix a "fit-range edits
  silently revert" regression (Round-10 #9); converging plot/ALC axis fields
  onto it removes the same class of silent-revert bug from the plot axis
  controls, which had no such fix.
- **Deliberately NOT converged — validator range stays per call site:** the
  two variants' `QDoubleValidator` ranges differ for a real reason (fit
  limits are physically bounded to roughly ±1000 µs/MHz; a plot/ALC axis
  limit can legitimately be much larger, e.g. thousands of MHz on a
  frequency axis) and the converged field *clamps* to its range, so
  collapsing the range too would silently clip legitimate axis limits.
  `plot_panel`/`alc_panel` call sites explicitly pass
  `value_range=(-1e6, 1e6)` (their historical validator range); `fit_panel`
  call sites keep the class default `(-1000.0, 1000.0)`, with its existing
  domain-switch `setRange(-1_000_000.0, 1_000_000.0)` /
  `setRange(-1000.0, 1000.0)` calls (frequency vs. time domain) preserved
  unchanged. `plot_panel`/`alc_panel` construction sites also pass
  `maximum_width=None` to preserve their historical unbounded field width
  (only `fit_panel`'s default caps width at 88px).
- **Phase:** 1a (part 1 — the field only; `AxisLimitControls`/toolbar
  assembly is a separate follow-up).

### Fit-range numeric fields adopt right-alignment + fixed size policy

- **What:** the fit-range min/max numeric entry fields in the Fit dock now
  render **right-aligned** with a **Fixed** horizontal size policy (and an
  explicitly-disabled clear button). Previously the old
  `fit_panel._FloatLimitField` set none of these, so the fields inherited
  Qt's `QLineEdit` defaults: left-aligned text with an Expanding size policy.
  (Width was already effectively bounded before — the old fit field set
  `minimumWidth=56`/`maximumWidth=88` — so the visible delta is primarily the
  left→right text alignment.)
- **Where:** `SingleFitTab` and `GlobalFitTab` fit-range fields, via the
  converged `asymmetry.gui.widgets.axis_limits.FloatLimitField`
  (`__init__` sets `AlignRight`, `setClearButtonEnabled(False)`,
  `Fixed/Fixed` size policy for all call sites).
- **Winning variant / why:** the plot/ALC axis-field styling won here — a
  right-aligned numeric field is conventional and now makes *all* numeric
  limit fields in the app (plot axes, ALC axes, fit range) look consistent,
  which is a stated goal of this audit (uniform GUI elements across
  representations). No functional or data impact.
- **Also (nit, no user impact):** the shared X/Y row is now wrapped in an
  `AxisLimitControls` `QWidget` with `contentsMargins(0,0,0,0)` and added via
  `addWidget` instead of `addLayout`; inter-widget spacing (4), order,
  stretch, unit-label placement and Auto-button caps are unchanged, so any
  margin difference around the row is negligible.
- **Phase:** 1a (surfaced by the Review A gate).

### Closing a fit-wizard window mid-analysis now cancels the run

- **What:** closing a fit-wizard window while an analysis is still running now
  **cancels** the background run (cooperatively, via `TaskRunner.shutdown()`)
  and closes the window. Previously each window's `closeEvent` **hid** the
  window and let the analysis run to completion behind it (`hide()` +
  `event.ignore()`).
- **Where:** both wizard windows — `FitWizardWindow` (migrated in Phase 3B) and
  `GlobalFitWizardWindow` (Phase 3C) — now inherit
  `WizardWindowBase.closeEvent`, which calls `self._tasks.shutdown()` then
  `super().closeEvent()`. The old per-window hide-and-run `closeEvent` overrides
  are deleted.
- **Winning variant / why:** the hide-and-run behavior actually *violated* the
  AGENTS engineering invariant "hold strong references to live threads, and
  shut them down in `closeEvent`" — a closed-but-hidden window left an
  orphaned worker running. Converging both windows onto the shared
  `TaskRunner`-driven shutdown is both the best variant and the
  invariant-compliant one. No test pinned the old hide behavior.
  `TaskRunner.shutdown()` bounds the wait and hands any overrunning thread to
  the process-level reaper, so a long fit between cancel-polls cannot abort the
  process.
- **Phase:** 3 (base decision recorded in `wizard-base-design.md`; live for the
  single-fit window as of 3B, for the global window as of 3C).
