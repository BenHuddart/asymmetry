# Investigation: macOS 26 (Tahoe) Qt `QScreen` use-after-free crash

**Status: ROOT CAUSE FOUND AND FIXED (2026-07-09).** The crash is a
**PySide/shiboken object-lifetime bug in the application process**, not a
Qt/macOS defect. Caught live with guard malloc + an lldb breakpoint on
`QScreen::~QScreen`: the destructor fired mid-session from
`SbkDeallocWrapperCommon` inside `gc_collect_main` — Python's cyclic GC
reclaimed a shiboken wrapper and deleted the live C++ `QScreen` (a 16-byte
allocation, exactly `sizeof(QScreen)`), whose address matched the subsequent
`screenAt` → `virtualSiblings` fault address byte-for-byte. No `screenRemoved`
ever fired, so the freed screen stayed in `QGuiApplication::screens()`
(`~QScreen`'s "must go through handleScreenRemoved" check is a debug-only
assert). The exposing relationship: the `QWidget.screen()` / `QWindow.screen()`
bindings attach a shiboken **parent link** from the process-wide cached
`QScreen` wrapper to the receiver's wrapper (verified with `shiboken6.dump`;
same bug class as PYSIDE-3380), and the app called `.screen()` from transient
dialogs via `resize_to_available`. The one cause also explains the Windows
symptom (deleted screen → empty list → ``Cannot create window: no screens
available`` + fatal exit) and why minimal repros never faulted (they never
touch QScreen wrappers from Python). **Fix (`gui/screen_guard.py`):**
`screen_for()` resolves screens via `screenAt`/`primaryScreen` (no parent
link) and replaces every `.screen()` call site; `pin_screens()` holds strong
references to all screen wrappers so the GC can never reclaim one; a
`destroyed`-signal tripwire logs loudly (with Python stack) if a screen ever
dies mid-session. Verified live: the previously reliable two-peak FB↔FFT
repro no longer crashes. Upstream follow-up: file a **PYSIDE** report asking
for an ownership annotation on `QWidget/QWindow::screen()` (as was done for
`QAction::menu` in 6.11.2) — not the QTBUG drafted at the end of this note.

**Everything below this line predates the root-cause finding** and is kept as
the investigation record; its working hypotheses (startup in-place `NSScreen`
update, upstream-Qt-only fix) are superseded by the paragraph above.

**Historical status:** open — native stacks on macOS show a Qt `QScreen` use-after-free;
app-side mitigation reduces some triggers but FB↔FFT still finds new ones.
**Also reported on Windows** (similar crash under normal use) — so this is
**not** safely treated as Tahoe/Cocoa-only until a Windows dump confirms or
refutes the same `screenAt`/`QScreen` tail. This note is a self-contained brief
for a reviewer (human or agent) who also has the Asymmetry source.

This is **not** a bug in Asymmetry's analysis logic. On macOS the fault is inside
Qt DPI/screen resolution; Asymmetry densifies the trigger via widget
construction / show-hide / paint on hot paths (especially FB↔FFT). A matching
Windows report means the durable story may be broader Qt (or shared app
trigger density), not only macOS 26 display reconfiguration.

**Review (2026-07-09):** mechanism confirmed against Qt 6.11.1 sources; open
questions below answered; model/view trigger-reduction assessed; no matching
public QTBUG found — file one with the draft at the end of this note.

---

## One-line

A PySide6 (Qt Widgets) app segfaults on the main thread inside Qt's DPI resolution
(`QGuiApplication::screenAt` → `QScreen::virtualSiblings`/`QScreen::handle`),
dereferencing a **freed `QScreen`**. It reproduces on a clean single-display
machine with no third-party display tools, and is **not fixed by any Qt version
tried** (6.9, 6.10.3, 6.11.1).

## Environment

- MacBook Air (Mac14,2), Apple M2, **single built-in Liquid Retina display**
  (2560×1664, no external displays, mirroring off, no ProMotion on this model).
- macOS 26.5.2 (build 25F84), "Tahoe", release build.
- PySide6 / Qt: reproduced on **6.9.x, 6.10.3, and 6.11.1** (6.11 is Qt's
  macOS-26-supported release). Python 3.12, matplotlib Qt Agg backend.
- **Windows (v0.9.0 from source, 2026-07-09):** after a few FB↔FFT switches the
  GUI **stopped responding** (not a clean crash-to-desktop); process was
  force-closed. Capture artifacts:
  - `gui.log` (PowerShell redirect): only
    ``Cannot create window: no screens available`` — no faulthandler Python
    stack (consistent with a hang / forced kill, not a Python-handled SIGSEGV).
    A later share still pointed at this same 806-byte UTF-16 log (mtime
    unchanged) — relaunch with ``cmd.exe`` ``> … 2>&1`` so a new session
    overwrites it.
  - `python.exe.25360.dmp` (newer minidump): Windows 11 10.0.26200;
    exception ``0xC0000409`` (``STATUS_STACK_BUFFER_OVERRUN`` /
    ``__fastfail``) with ``ExceptionInformation[0] = 7``
    (``FAST_FAIL_FATAL_APP_EXIT``); faulting RIP in ``Qt6Core.dll``.
    Heuristic stack is Qt Gui/Widgets → Core (abort path), not a classic
    access-violation UAF. Together with the log line, this reads as Qt
    **fatally exiting because the screen list is empty / unusable**, not
    the macOS freed-``QScreen`` scribble pattern.
  - Prefer launching with `cmd.exe` `>` redirect (not PowerShell `*>`) so
    stderr stays UTF-8 and faulthandler can append if a real fault fires.

## Symptom

`EXC_BAD_ACCESS (SIGSEGV)` on `com.apple.main-thread`, intermittent, within
1–5 minutes of normal use. Faulting addresses are small non-null garbage
(`0x29a`, `0x351`, `0x873`, `0x4962f4`, `0x561c77`). Register `x9 =
0xaaaaaaaaaaaaaaaa` (freed-memory scribble) in every report ⇒ **use-after-free**,
not a null dereference.

## The invariant faulting tail (from 6 native `.ips` reports)

Every crash ends the same way:

```
QScreen::virtualSiblings() const      (or QScreen::handle() const)
QGuiApplication::screenAt(QPoint const&)
QWidget::screen() const
QWidget::metric(QPaintDevice::PaintDeviceMetric) const   ← Qt resolving DPI (PdmDpiX)
<something that needs a font / metric>
```

The `<something>` varies (which is why it first looked like heap corruption — the
fault lands at whatever next asks Qt for a screen/DPI):

- `QPainter::begin` from `QPainter(self)` in matplotlib's canvas `paintEvent`
- `QFont::QFont(font, paintDevice)` building a widget's default font
- `QWidgetPrivate::init` while constructing a `QWidget()` (a fit-table cell widget)
- `QScreen::handle()` via `QWindow::safeAreaMargins()` during `QLayout::activate()`
- `QTextDocumentLayout::doLayout` via `QTextEdit::append()` (a log line)

**Qt-internals reading.** `QWidget::screen()` returns the window's *associated*
`QScreen` directly, and only falls through to `QGuiApplication::screenAt()` when
that association is null/stale. `screenAt()` iterates
`QGuiApplication::screens()` calling `virtualSiblings()` on each — and one entry is
a **freed `QScreen`**, so it segfaults.

Confirmed in Qt 6.11.1 `qwidget.cpp`: `QWidget::metric()` always calls
`this->screen()` before handling `PdmDpiX` / `PdmDpiY`; there is no env-var
bypass of that call. `QCocoaScreen::virtualSiblings()` walks
`QGuiApplication::screens()` and dereferences each entry's `handle()`.

## Suspected trigger

On a stable single display the user never reconfigures, a `QScreen` should not be
freed. Working hypothesis: macOS 26 fires display **reconfiguration** callbacks on
its own (background display-parameter / EDR / brightness churn), Qt's Cocoa QPA
reconciles its `QScreen` objects via `QCocoaScreen::updateScreens()`, and Qt
6.9–6.11 can leave a dangling reference in the global `screens()` list (or on a
window). The user *action* (a button click that builds a widget, a repaint, a log
append) merely dereferences the already-dangling pointer. Some crashes occur
~1–2 min into a fresh process, so it is not sleep/wake.

**Nuance (do not overclaim):** Qt 6.11.1 `updateScreens()` prefers UUID-based
*in-place* updates of existing `QCocoaScreen` objects. A freed `QScreen` implies
add/remove (or list corruption), not a mere property update. True Tone /
automatic brightness / EDR headroom changes are a plausible *class* of macOS
events that can drive `NSApplicationDidChangeScreenParametersNotification` and
`CGDisplayReconfigurationCallback` — they are not yet proven as the specific
cause of a remove+add on a single built-in display. Confirm with
`QT_LOGGING_RULES=qt.qpa.screen.info=true` (or `.warning=true`) during a repro
window: look for "Adding" / "Updated" / "Primary screen changed" lines around
the crash.

## Ruled out (with evidence)

- **Asymmetry logic / the fit code.** The crash is purely in Qt DPI/screen
  internals; the app never calls `screenAt()`. The model-evaluation code is pure
  numpy (fresh arrays, no shared scratch) — see `CompositeModel.function` in
  [`src/asymmetry/core/fitting/composite.py`](../../src/asymmetry/core/fitting/composite.py)
  (~line 2113) and
  [`parameter_models.py`](../../src/asymmetry/core/fitting/parameter_models.py)
  (~line 1567). A concurrent fit worker was present in the first crash, but the
  model is pure and GIL-protected; later crashes are single-threaded with no
  worker running.
- **A data race / our own heap corruption.** faulthandler shows a single main
  thread; the "moving" crash site is just `screenAt` landing on the next DPI
  query, not a UAF in our memory.
- **Sleep/wake stale screen.** A crash reproduced 104 s into a fresh process.
- **A virtual-display tool (BetterDisplay).** It aggravated the frequency (its
  virtual-display churn makes `QScreen` teardown frequent), but the crash
  reproduces with it fully quit and never launched — single built-in display,
  fresh reboot.
- **Qt version.** Reproduced on 6.9, 6.10.3, and 6.11.1. No version fixes it.

## Attempted mitigation — CONFIRMED INSUFFICIENT

Source: [`src/asymmetry/gui/screen_guard.py`](../../src/asymmetry/gui/screen_guard.py),
wired in [`src/asymmetry/gui/app.py`](../../src/asymmetry/gui/app.py)
(`install_screen_change_guard`, called just after the `QApplication` is created).

On `QGuiApplication.screenAdded`/`screenRemoved`/`primaryScreenChanged`, it
re-anchors every top-level window whose `QScreen` is no longer in `screens()` to
`primaryScreen()` via `window.setScreen(...)`. Rationale: keep windows' associated
screen valid so `QWidget::screen()` short-circuits before the crashing `screenAt`.

**It still crashes.** The post-guard native report (`python3.12-...195804.ips`,
Qt 6.10.3) shows the surviving crash is the **widget-construction path**:

```
QAbstractButton::mouseReleaseEvent            ← domain (FB↔FFT) button click
 → clicked → Python slot → (emit signal) → Python slot
  → QWidget()                                 ← a fit-table cell widget being built
   → QWidgetPrivate::init → QFont(font, paintDevice) → QWidget::metric(PdmDpiX)
    → QWidget::screen() → QGuiApplication::screenAt() → virtualSiblings()   ← UAF
```

**Why the guard cannot cover this.** `QWidget::metric(PdmDpiX)` **unconditionally**
calls `QWidget::screen()`, and a widget under construction is not yet shown, has no
window handle, and has its parent attached only *later* in `init` — so `screen()`
has no association and goes straight to `screenAt()` on the corrupt global
`screens()` list. There is no shown-window association to keep valid. Parenting the
widget at construction does not help either: Qt resolves the widget's font in
`QWidgetPrivate::init` *before* the parent is attached.

The guard *does* still help the already-shown-widget paths (paints, `QTextEdit`
appends, layout resizes), so it is a partial hardening, not a fix. Keep it; do
not treat it as complete.

**Guard signal coverage gap.** The guard only runs when Qt emits
`screenAdded` / `screenRemoved` / `primaryScreenChanged`. If Tahoe mutates
display parameters without those signals, or if `screens()` already holds a
freed pointer when the next DPI query runs, Python cannot heal Qt's internal
list.

## Relevant Asymmetry source (for cross-study)

The recurring FB↔FFT crash comes from the domain switch rebuilding the fit
parameter table. The call chain (top of stack downward):

- Domain button click is wired in
  [`src/asymmetry/gui/mainwindow.py`](../../src/asymmetry/gui/mainwindow.py) ~1264
  → `_on_domain_button_clicked` (~9659)
  → `PlotWorkspacePanel.set_active_view`
  ([`src/asymmetry/gui/panels/plot_workspace_panel.py`](../../src/asymmetry/gui/panels/plot_workspace_panel.py) ~178)
  → `_on_plot_workspace_view_changed` (~9721) → `_on_plot_time_view_changed` (~9637)
  → `_update_fit_block_state` (~2837)
  → `FitPanel.set_domain`
  ([`src/asymmetry/gui/panels/fit/panel.py`](../../src/asymmetry/gui/panels/fit/panel.py) ~247)
  → `SingleFitTab.set_domain`
  ([`src/asymmetry/gui/panels/fit/single_tab.py`](../../src/asymmetry/gui/panels/fit/single_tab.py) ~364)
  → `_set_composite_model` (~585)
  → `FitParameterTable.populate`
  ([`src/asymmetry/gui/panels/fit/tab_base.py`](../../src/asymmetry/gui/panels/fit/tab_base.py) ~1350).
- **The exact construction that faults:** `populate` builds one row per model
  parameter; each row constructs parentless cell widgets —
  `fix_widget = QWidget()` at `tab_base.py:1387`, plus a `QComboBox` (Link) and
  `QPushButton` (Tie) via `setCellWidget` (~1409–1417), and surrounding `QFont`
  usage via `mono_font` in
  [`src/asymmetry/gui/styles/fonts.py`](../../src/asymmetry/gui/styles/fonts.py) ~105.
  A two-peak frequency model has more parameters than the time model, so the
  FFT→FB switch changes the row count and reliably rebuilds/relayouts the table —
  which is why "two-peak fit" was the most reliable repro.
- **`FitPanel.set_domain` always updates both tabs** (`panel.py` ~247–248), so
  [`GlobalFitTab._set_composite_model`](../../src/asymmetry/gui/panels/fit/global_tab.py)
  also rebuilds per-row Type `QComboBox` widgets on every domain switch even when
  the Single tab is visible.
- **The two Matplotlib canvases** (time + frequency) live in a `QStackedWidget` in
  `plot_workspace_panel.py`; canvases are built once by `create_canvas` in
  [`src/asymmetry/gui/widgets/mpl_canvas.py`](../../src/asymmetry/gui/widgets/mpl_canvas.py)
  (`FigureCanvasQTAgg`) and are **not** recreated on FB↔FFT — but domain switch
  still triggers paints (`draw_idle`). The paint-path variant of the crash is
  `FigureCanvasQTAgg.paintEvent` → `QPainter(self)`.
- The `QTextEdit.append` variant is the log panel appending a status line.

Asymmetry does nothing unusual with screens: the only `.screen()` calls are
read-only sizing helpers (`mainwindow.py` ~704, `gui/widgets/screen_sizing.py`
~38); there are no `screenChanged` handlers other than the new guard, and `grep -r
screenAt src/` is empty.

## Model/view trigger reduction (assessed, not landed)

A `QTableView` + `QAbstractTableModel` migration of `FitParameterTable` (Fix as
check-state, Link/Tie as edit-time delegates, no `setCellWidget`) was attempted
on branch `codex/fit-parameter-model-view`. As of 2026-07-09 that work is **not
in the repository**: the named worktree is gone (prunable), and the branch tip
matches `main` / release 0.9.0 with no model/view commits. Current
`FitParameterTable` is still a `QTableWidget` that builds ~3N persistent cell
widgets per `populate()`.

**Assessment:** that refactor would have been a **sensible reduction of one
high-density trigger** (the post-guard FB↔FFT stack), not a root-cause fix and
not the wrong layer. After eliminating persistent cell widgets, the same domain
switch would still hit:

| Remaining site | Path |
| --- | --- |
| Item fonts | `mono_font(11.0)` / `setFont` in `populate` |
| Table sizing | `_size_param_table_to_content` → header/row `sizeHint` |
| Global tab Type combos | `GlobalFitTab._set_composite_model` on every `set_domain` |
| Formula box | `FormulaBox.set_formula` / `refresh_height` |
| Plot paint | stacked canvas `draw_idle` → `FigureCanvasQTAgg.paintEvent` |
| Dock relayout | `_apply_inspector_for_domain` / `resizeDocks` |
| Log | `QTextEdit.append` |
| Edit-time delegates | `createEditor` still constructs widgets when editing |

Continued crashes after a real model/view change would be expected.

## Reviewer questions — answers

1. **Can Python/PySide reliably stop `metric → screen → screenAt`?**
   **No.** Qt 6.11.1 `QWidget::metric()` always calls `this->screen()` first.
   `QT_FONT_DPI` / `QT_ENABLE_HIGHDPI_SCALING=0` / `QT_SCALE_FACTOR` / `Qt::AA_*`
   may change *which* DPI value is used; they do not skip `screen()`. Forcing a
   screen-list rebuild from Python is bogus (no public heal API). Extra
   `QScreen` property hooks / event filters can at best re-anchor windows (what
   `screen_guard` already does); they cannot prevent `screenAt` from walking a
   freed pointer in `screens()`.

2. **Which macOS 26 event frees the `QScreen`?**
   Unconfirmed. Cocoa QPA listens to `CGDisplayReconfigurationCallback` and
   `NSApplicationDidChangeScreenParametersNotification` (also used for EDR
   headroom). Capture `qt.qpa.screen` logs during repro to see add/remove vs
   in-place update. The guard's add/remove/primary signals may not fire for
   every parameter-only notification.

3. **Known QTBUG / `libqcocoa` workaround?**
   No matching public QTBUG found for this exact
   `screenAt`/`virtualSiblings` UAF on Tahoe (searched 2026-07-09). Related
   Tahoe Qt crashes (OBS/FreeCAD accessibility / layer-backing UAFs) are a
   different stack. `UIDesignRequiresCompatibility=YES` is a Liquid Glass /
   style compatibility knob — worth a try for general Tahoe fragility, **not
   evidenced** as a fix for this fault. Closest historical Qt pattern is
   non-transactional / recursive screen updates; 6.11.1 already has a
   recursion guard in `QCocoaScreen::updateScreens()`.

4. **Is the realistic fix upstream?**
   **Yes.** Prioritise a solid QTBUG (draft below) + construction-density
   reduction as risk mitigation. Do not expect env vars or parenting tricks to
   close the hole.

## Live `qt.qpa.screen` capture (2026-07-09) — decisive

Captured with `QT_LOGGING_RULES='qt.qpa.screen=true'` + an instrumented
`screen_guard` (`ASYMMETRY_SCREEN_GUARD_DEBUG=1`) across three repro sessions.
Findings, all reproduced:

- **No screen churn at FB↔FFT.** The only `Adding`/`Removing` line is the initial
  startup enumeration; there are **zero** `qt.qpa.screen` events at the domain
  switch. This **refutes** the theory that the plot-canvas swap frees a `QScreen`.
- **The disturbance is an in-place update at startup.** macOS emits
  `Received screen parameter change notification` and Qt logs
  `Updated QCocoaScreen(0x…, <NSScreen: 0xA>)` → `…<NSScreen: 0xB>` — the
  `QCocoaScreen` wrapper pointer is kept while the backing `NSScreen` pointer is
  swapped. This is the state that leaves `screenAt` fragile.
- **Window re-anchoring cannot help — measured.** An extended guard that also
  hooked every per-`QScreen` property signal (`geometryChanged`, `…DotsPerInch…`,
  `virtualGeometryChanged`, …) *did* fire on the in-place update, but
  `reanchor_stale_windows` found **zero** stale windows across the whole session
  (`re-anchored 0`): every top-level window still reported a live `QScreen`. So
  there is nothing for Python to re-anchor; the guard extension was reverted as
  inert (kept only the removal-case guard).
- **Not True Tone / not automatic brightness.** With both **off**, the startup
  `NSScreen` swap and `screen parameter change` notification still fire, and
  FB↔FFT still crashes. There is no display-setting workaround.
- **Crash always lands at `screenAt`.** Captured Python detector sites this
  session: `GlobalFitTab._update_mode_ui` → `QTextEdit.setText`
  (`global_tab.py:4234`); and `GlobalFitTab._rebuild_group_nuisance_table` →
  parentless `QComboBox()` (`global_tab.py:4592`) via
  `MultiGroupFitWindow.set_dataset` (`mainwindow.py:2096`). C++ tail is
  `…QWidget::metric(PdmDpiX) → QWidget::screen() → QGuiApplication::screenAt →
  QScreen::virtualSiblings` in every case (`0xaa` scribble present in some, absent
  in others — consistent with freed native memory being reused by crash time).

**Conclusion:** the FB↔FFT crash is an upstream Qt/macOS-26 defect in in-place
`QCocoaScreen` updating; no app-side lever (window re-anchor, per-site deferral,
Qt version, display settings) closes it. Ship effort should go to the QTBUG and,
optionally, reducing detector density only as risk reduction — not as a fix.

## Reproduction narrowing + eliminations (2026-07-09, part 2)

Driven through the background capture loop (launch GUI with
`QT_LOGGING_RULES='qt.qpa.screen=true' PYTHONFAULTHANDLER=1` on a real display;
user reproduces; parse `screencap*.log` + newest `python3.12-*.ips`).

**Trigger narrowed to the two-peak frequency fit (user-confirmed).** No fit → FB↔FFT
stable; single-peak frequency fit → stable; **two-peak** frequency fit → crashes on
FB↔FFT. The base FFT view (2026-05-17) ran ~7 weeks with no crashes; multi-peak
frequency fitting shipped in **#228 (2026-07-08)**, the crash-onset date.

**But it is density, not a feature bug.** The crash is `FitParameterTable.populate`
constructing the Fix-cell `QWidget` (`tab_base.py:1387`) → `screenAt`. A two-peak
model has ~2× the parameter rows of single-peak, so it does ~2× the
`screenAt`-hitting parentless constructions per switch — enough to cross the
"reliably faults" threshold. #228 did not introduce a bug; it made a heavier switch
common. Reverting it would not truly fix the crash (single-peak would still fault,
just rarely).

**Eliminated this session (each a clean single-point test on clean v0.9.0):**
- **Reseed-on-switch is NOT the trigger.** Gating off
  `FitPanel._reseed_frequency_peaks_if_default` (the #228 multi-peak reseed path)
  did not change the crash.
- **Parenting the cell widget does NOT help.** `QWidget(self)` instead of
  `QWidget()` still faults at the same line: Qt resolves the widget's font
  (`QFont(font, paintDevice)` → `metric(PdmDpiX)` → `screen()` → `screenAt`) inside
  `QWidgetPrivate::init` *before* the parent is attached, so there is no valid
  parent-window screen to short-circuit through. No construction-side dodge exists.

**The key unsolved puzzle.** Three standalone minimal repros — (v1) a trivial window
+ ~1.4M parentless `QWidget`/`QComboBox`/`QLabel` constructions; (v2) + styled
multi-dock `QMainWindow`, scaled application font, and repeated second-top-level-
window create/destroy churn; (v3) + two `FigureCanvasQTAgg` in a `QStackedWidget`
swapped and drawn every tick — **all survived**. Same OS, same startup in-place
`NSScreen` swap, same construction pattern, yet the minrepros never fault while the
app faults after a few dozen constructions. So **the real app holds some state that
makes `screenAt` fragile which the minrepros do not reproduce** — that unidentified
fragility is the true root, and it cannot be cracked by more black-box experiments.

**Recommended next tool: macOS guard malloc under lldb on the real app** — faults at
the moment freed screen memory is accessed, with alloc/free stacks, revealing *what*
is freed and *who* frees it (possibly an app/PySide/matplotlib lifetime issue that
is app-addressable):

```
DYLD_INSERT_LIBRARIES=/usr/lib/libgmalloc.dylib MallocScribble=1 \
  lldb -o run -- .venv/bin/python -c "from asymmetry.gui.app import main; main()"
# reproduce the two-peak FB↔FFT crash, then at the lldb prompt: bt all
```

## App-side strategy

1. **Upstream is the durable fix** — file the QTBUG with native stacks, the
   `0xaa` scribble, single-display repro, Qt 6.9/6.10/6.11 matrix, and
   `qt.qpa.screen` logs.
2. **Keep `screen_guard`** for shown-window paths; document it as partial
   (already the case).
3. **Reduce construction density on hot paths:**
   - **Landed:** `FitParameterTable.populate` reuses Fix/Link/Tie cell widgets
     when the parameter-name list is unchanged (same-model Reset/restore/
     grouped refresh) — no new parentless `QWidget`/`QComboBox`/`QPushButton`
     on that path.
   - **Landed:** `FitPanel.set_domain` defers `GlobalFitTab` Type-combo
     rebuild while the Batch tab is hidden; flush on Batch entry only.
     Domain-switch state save uses a synthesised snapshot
     (`materialize=False`) and defers restore while Single is showing —
     earlier flush-on-`get_state()` undid the deferral and was the
     surviving FB↔FFT crash (`get_global_state` → `QComboBox()`).
   - **Landed:** `SingleFitTab` keeps separate time/frequency
     `FitParameterTable` instances and switches by height-collapse (both
     stay `visible=True`) after each domain has been populated once —
     avoids `QStackedWidget.setCurrentWidget` → `setVisible` →
     `QCursor::pos(QScreen*)` (post-deferral FB↔FFT crash) and avoids
     tearing down Link/Fix/Tie widgets.
   - **Landed:** Batch group-nuisance Type-combo rebuild
     (`_update_group_parameter_defaults`) is deferred while Single is
     showing; flush on Batch entry. Post-stack-fix FB↔FFT crash was
     `set_dataset` → `QComboBox()` in `_rebuild_group_nuisance_table`.
   - **Landed:** Single-tab carry-forward badge uses height-collapse
     instead of `show()`/`hide()` — post-group-deferral crash was
     `show_carry_forward_badge` → `setVisible` → `QCursor::pos(QScreen*)`.
   - **Still deferred:** full `QTableView` model/view migration; audit
     other hot-path `show()`/`hide()`/`setVisible` on FB↔FFT.
4. **Do not expect** env vars, parenting-at-construction, or forcing a screen
   rebuild from Python to close the hole.
5. **Product options:** optional macOS 26 warning; ship a patched Qt/`libqcocoa`
   only if Qt or the community produces a real fix; try
   `UIDesignRequiresCompatibility` only as an experiment, not as a claimed fix.

## How to reproduce / capture

Run the GUI from source with faulthandler and use it for a few minutes doing things
that build widgets / repaint / append log text (in particular: switch the plot
between FB Asymmetry and FFT with a two-peak frequency fit active):

```
PYTHONFAULTHANDLER=1 .venv/bin/python -X faulthandler -c "from asymmetry.gui.app import main; main()"
```

For Cocoa screen-update logging during a repro window:

```
QT_LOGGING_RULES='qt.qpa.screen.info=true' PYTHONFAULTHANDLER=1 \
  .venv/bin/python -X faulthandler -c "from asymmetry.gui.app import main; main()"
```

faulthandler prints the **Python** frames (where it trips). The **C++** faulting
frame + registers (where it actually faults, and the `0xaa` scribble) are in the
native report at `~/Library/Logs/DiagnosticReports/<proc>-*.ips` — parse the
faulting thread's `frames` and `threadState.x`.

---

## QTBUG draft (file upstream)

**Title:** macOS 26: `QGuiApplication::screenAt` / `QScreen::virtualSiblings` UAF
after background display reconfiguration (single built-in display)

**Component:** Qt Gui / QPA Cocoa (`QCocoaScreen`)

**Affects:** 6.9.x, 6.10.3, 6.11.1 on macOS 26.5.x (Tahoe)

**Summary:** Intermittent main-thread `EXC_BAD_ACCESS` with freed-memory scribble
`0xaaaaaaaaaaaaaaaa`. Invariant stack:

```
QScreen::virtualSiblings() / QScreen::handle()
QGuiApplication::screenAt
QWidget::screen
QWidget::metric(PdmDpiX)
```

Entry points vary (parentless `QWidget` construction, `QPainter` on a
`FigureCanvasQTAgg`, `QTextEdit::append`, layout activation). Reproduces on a
clean single built-in display with no third-party display tools and no user
display reconfiguration. App-side re-anchoring of top-level windows on
`screenAdded`/`screenRemoved`/`primaryScreenChanged` reduces shown-widget
crashes but does not cover parentless widget construction (font/DPI resolved
before parent attach).

**Ask:** Does `QCocoaScreen::updateScreens()` ever leave a freed `QScreen*` in
`QGuiApplication::screens()` after Tahoe background
`NSApplicationDidChangeScreenParametersNotification` / display reconfiguration
(including EDR headroom / brightness-driven parameter changes)? Is there a
known fix or recommended `libqcocoa` patch?

**Attachments to include when filing:** several `.ips` reports, faulthandler
Python stacks, `qt.qpa.screen` log excerpt from a repro window, hardware/OS/Qt
matrix above.
