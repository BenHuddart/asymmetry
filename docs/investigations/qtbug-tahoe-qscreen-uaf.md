# QTBUG draft — segfault in QGuiApplication::screenAt after in-place QCocoaScreen update on macOS 26

> **SUPERSEDED (2026-07-09) — do not file this QTBUG.** The root cause turned
> out to be a PySide/shiboken wrapper-lifetime bug (Python GC deleting the
> live C++ `QScreen` via the parent link created by the `QWidget.screen()` /
> `QWindow.screen()` bindings), not a Cocoa QPA defect. File a **PYSIDE**
> report instead; evidence and fix are in `tahoe-qscreen-uaf.md`. Kept for
> the environment matrix and attachment checklist.

Submit at <https://bugreports.qt.io> (Component: **QPA: Cocoa**). Fields below map
to the Jira form; the Description block is ready to paste.

- **Summary:** macOS 26: `QGuiApplication::screenAt()` dereferences stale screen
  memory (segfault) after Qt updates a `QCocoaScreen` in place on a display-parameter change
- **Type:** Bug
- **Component/s:** QPA: Cocoa
- **Affects Version/s:** 6.9, 6.10.3, 6.11.1
- **Platform:** macOS
- **Priority:** (reporter's discretion — crash under normal use)

---

## Description

### Summary

On macOS 26 ("Tahoe"), a Qt Widgets application intermittently segfaults on the
main thread inside `QGuiApplication::screenAt()` → `QScreen::virtualSiblings()`
(sometimes `QScreen::handle()`), reached from ordinary DPI resolution
(`QWidget::metric(PdmDpiX)` → `QWidget::screen()`) during widget/font
construction, painting, or text layout.

It occurs on a **single built-in display that is never reconfigured by the user**,
with True Tone and automatic brightness **off**. `qt.qpa.screen` logging shows the
precondition: shortly after launch macOS delivers a display-parameter change and
the Cocoa QPA performs an **in-place** `QCocoaScreen` update — the `QCocoaScreen`
wrapper pointer is retained while its backing `NSScreen` pointer is replaced.
After that, `screenAt()` can dereference freed/stale screen memory.

Reproduced on **Qt 6.9, 6.10.3 and 6.11.1** (via PySide6, Python 3.12). Not fixed
by the macOS-26-targeted 6.11 release.

### Environment

- MacBook Air (Mac14,2), Apple M2, single built-in Liquid Retina display
  (2560×1664 @ dpr 2, no external displays, mirroring off, no ProMotion).
- macOS 26.5.2 (build 25F84), release.
- Qt 6.9 / 6.10.3 / 6.11.1 (PySide6). True Tone OFF, automatic brightness OFF.

### Steps to reproduce

1. Launch a Qt Widgets app on macOS 26 with `QT_LOGGING_RULES='qt.qpa.screen=true'`.
2. Observe in the log, within the first second, an in-place screen update:

   ```
   qt.qpa.screen: Adding QCocoaScreen(0x…, "Built-in Retina Display", …, <NSScreen: 0xA>) as new primary screen
   qt.qpa.screen: Updated QCocoaScreen(0x…, "Built-in Retina Display", …, <NSScreen: 0xB>)   # NSScreen pointer changed
   qt.qpa.screen: Received screen parameter change notification
   qt.qpa.screen: Updated QCocoaScreen(0x…, "Built-in Retina Display", …, <NSScreen: 0xB>)
   ```

   The `QCocoaScreen` pointer (`0x…`) is unchanged across the `Updated` lines while
   the backing `<NSScreen: …>` pointer changes (`0xA` → `0xB`).
3. Use the app normally for 1–5 minutes doing work that constructs widgets /
   resolves DPI (build widgets, repaint, set text). Intermittently the process
   segfaults inside `QGuiApplication::screenAt`.

The crash is timing-dependent, so there is no single deterministic click; it
reproduces within minutes. A candidate standalone reproducer is below.

### Actual result

`EXC_BAD_ACCESS (SIGSEGV)` on the main thread. Native stack (representative;
observed at several call sites that all share the `screen()`→`screenAt` tail):

```
QScreen::virtualSiblings() const                       ← faults (bad access)
QGuiApplication::screenAt(QPoint const&)
QWidget::screen() const
QWidget::metric(QPaintDevice::PaintDeviceMetric) const  ← PdmDpiX
QFont::QFont(QFont const&, QPaintDevice const*)
QWidgetPrivate::init(QWidget*, QFlags<Qt::WindowType>)
QWidget::QWidget(QWidget*, QFlags<Qt::WindowType>)      ← a parentless QWidget/QComboBox construction
```

Other observed tails from the same defect:

- `QPainter::begin` → `initPainter` → `QFont(…, paintDevice)` → `metric` →
  `screen()` → `screenAt` (a `QPainter(widget)` in a `paintEvent`).
- `QTextEdit::append`/`setText` → `QTextDocumentLayout::doLayout` → `metric` →
  `screen()` → `screenAt`.
- `QLayout::activate` → `QWidgetPrivate::doResize` → `QWindow::safeAreaMargins` →
  `QScreen::handle()` (faults here instead of `virtualSiblings`).

Faulting addresses are small non-null garbage (e.g. `0x29a`, `0x351`, `0x561c77`);
some dumps show the freed-memory fill `x9 = 0xaaaaaaaaaaaaaaaa`, others show the
freed block already reused — both consistent with a use-after-free of screen
memory reached through `QGuiApplication::screens()` / a `QCocoaScreen`.

### Expected result

`QGuiApplication::screenAt()` must never dereference a freed screen. After an
in-place `QCocoaScreen` update on a display-parameter change, the screen list and
any per-`QScreen`/`NSScreen` state it reaches must remain valid (or `screenAt`
should safely return `nullptr`).

### Analysis / suspected cause

`QWidget::metric(PdmDpiX)` unconditionally calls `QWidget::screen()`, which falls
through to `QGuiApplication::screenAt()`; `QCocoaScreen::virtualSiblings()` /
`handle()` then walk `QGuiApplication::screens()` and dereference per-screen
state. The `qt.qpa.screen` log shows that on macOS 26 the single display's backing
`NSScreen` is swapped early in the process lifetime and `QCocoaScreen::updateScreens()`
handles it as an **in-place** update (retaining the `QCocoaScreen`). The crash
signature indicates that after this in-place update some reachable pointer (the
previous `NSScreen`, or `QCocoaScreen`-derived state) is left dangling. Notably:

- The application-level signals (`screenAdded`/`screenRemoved`/`primaryScreenChanged`)
  do **not** fire for the in-place update, and every top-level `QWindow` still
  reports a live `QScreen` — so this is not a window losing its association; the
  stale state is internal to the screen machinery reached by `screenAt`.

### What was ruled out (by the reporter)

- Not application logic (the app never calls `screenAt`; single-threaded crash).
- Not a third-party virtual-display tool (reproduces with none installed, single
  built-in display, fresh reboot).
- Not True Tone / automatic brightness (both off; the in-place `NSScreen` swap and
  crash still occur).
- Not Qt-version-specific (6.9, 6.10.3, 6.11.1 all affected).

### Minimal reproducer status — not yet isolated

A trivial standalone reproducer was **attempted and did NOT reproduce**: a single
`QMainWindow` plus a timer that constructs ~600 parentless widgets
(`QWidget`/`QComboBox`/`QLabel.setText`) per tick survived **~1.4 million
constructions over 2+ minutes without crashing, including after an in-place
`NSScreen` swap / `screen parameter change` fired** (confirmed via
`qt.qpa.screen` logging). So the in-place update plus raw parentless-construction
volume is **not sufficient**; the fault requires additional state present in the
full application — candidates: a complex styled multi-dock `QMainWindow`, a second
top-level window (the crash was seen via a secondary window's `set_dataset`),
application font scaling, and/or interleaving construction with layout/show-hide/
paint of a live window. The reliable reproducer today is the full application.

The attempted (insufficient) script, for reference:

```python
import sys
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QComboBox, QLabel, QMainWindow, QWidget

app = QApplication(sys.argv)
win = QMainWindow(); win.resize(400, 300); win.show()

def churn():
    junk = []
    for _ in range(200):
        junk.append(QWidget())
        junk.append(QComboBox())
        lbl = QLabel(); lbl.setText("x"); junk.append(lbl)

timer = QTimer(); timer.timeout.connect(churn); timer.start(25)
sys.exit(app.exec())
```

Run: `QT_LOGGING_RULES='qt.qpa.screen=true' PYTHONFAULTHANDLER=1 python -X faulthandler repro.py`.
If a maintainer can extend this (multi-window / styled docks / font scaling) into a
crashing case, that would isolate the trigger; otherwise attach the application
recording + logs below.

### Attachments to include when filing

- `qt.qpa.screen` log covering launch through the crash (shows the in-place
  `Updated QCocoaScreen` / `NSScreen` pointer swap).
- One or more native `.ips` crash reports (the `screenAt`→`virtualSiblings` tail
  and register state, incl. the `0xaa` scribble).
- The minimal reproducer script above.
