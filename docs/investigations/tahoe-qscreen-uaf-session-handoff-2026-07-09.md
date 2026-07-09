# Session handoff: FB↔FFT Qt screen crashes (2026-07-09)

> **SUPERSEDED (2026-07-09, later the same day):** root cause found and fixed
> — a PySide/shiboken wrapper-lifetime bug, not Qt/macOS; see the status
> banner in [`tahoe-qscreen-uaf.md`](tahoe-qscreen-uaf.md). The whack-a-mole
> deferral WIP described below was dropped (preserved in a git stash on
> branch `harden-tahoe-qscreen-uaf`); the shipped fix lives on
> `fix-qscreen-pyside-gc`. Kept as the historical record.

**Audience:** another agent continuing this work.  
**Branch:** `harden-tahoe-qscreen-uaf` (uncommitted WIP — do not assume clean).  
**Canonical investigation:** [`tahoe-qscreen-uaf.md`](tahoe-qscreen-uaf.md)  
**User pause:** stop whack-a-mole on individual call sites until a broader strategy is agreed.

---

## Executive summary

1. **macOS (Tahoe):** main-thread segfault in Qt DPI/screen resolution — freed `QScreen` (`0xaa…` scribble) via `screenAt` / `virtualSiblings`. Upstream Qt/macOS defect; Asymmetry densifies triggers.
2. **Windows (v0.9.0 from source):** same user action (FB↔FFT a few times) → GUI **hangs / stops responding**; force-close. Dump shows Qt **`FAST_FAIL_FATAL_APP_EXIT`** in `Qt6Core.dll`, consistent with log line **`Cannot create window: no screens available`**. Same *family* (screen/QPA failure on FB↔FFT), **not** the same native failure mode as macOS UAF.
3. **App-side mitigations** on this branch reduce some triggers but each fix exposes the **next** FB↔FFT side effect that still constructs widgets or calls `setVisible`. Pattern is clear: densifying Qt screen/DPI work on the domain-button path.

---

## What is broken (product)

Switching plot domain **FB Asymmetry ↔ FFT** (fit panel Single tab typically showing) intermittently:

| Platform | Symptom | Native signature |
|---|---|---|
| macOS 26.5.x | Crash to desktop (SIGSEGV) | `QWidget::metric` / `QComboBox()` / `setVisible` → `QCursor::pos(QScreen*)` / `screenAt` → freed `QScreen` |
| Windows 11 (10.0.26200) | Hang / not responding; kill process | `0xC0000409` `__fastfail` / `FAST_FAIL_FATAL_APP_EXIT` in `Qt6Core.dll`; stderr: `Cannot create window: no screens available` |

Repro focus: FB↔FFT clicks (often with data loaded; multi-group window may exist). Capture with faulthandler + Qt screen logging (see investigation doc).

---

## Root cause framing (do not overclaim)

- **Not** an Asymmetry analysis/fit-math bug.
- **macOS:** strong evidence of Qt `QScreen` use-after-free after display-parameter churn (even single built-in display).
- **Windows:** evidence of empty/unusable screen list → Qt fatal exit / hang. **Not proven** identical to macOS UAF.
- **Shared actionable factor:** Asymmetry does a lot of widget construction / show-hide / table rebuild on the domain-switch path, which queries screen/DPI heavily.

---

## Whack-a-mole sequence this session (macOS captures)

Each “fix” removed one trigger; the next FB↔FFT session crashed elsewhere:

| # | Python site | Trigger | Mitigation landed (WIP on branch) |
|---|---|---|---|
| 1 | `FitPanel.set_domain` → `get_global_state` → `flush_deferred_domain_rebuild` → Batch Type `QComboBox()` | Parentless combo construction | `get_state(materialize=False)` on domain save; synthesised default snapshot; defer restore while Single showing |
| 2 | `SingleFitTab.set_domain` → `QStackedWidget.setCurrentWidget` | `setVisible` → `QCursor::pos(QScreen*)` | Replaced stack with **height-collapse** (both tables stay `visible=True`) |
| 3 | `FitPanel.set_dataset` → `GlobalFitTab._rebuild_group_nuisance_table` | Group-nuisance `QComboBox()` | `set_current_dataset(..., defer_group_rebuild=True)` while Single showing; flush on Batch entry |
| 4 | `show_carry_forward_badge` → `QFrame.show()` | `setVisible` → cursor/screen | Badge show/hide → **height-collapse** |
| 5 | **Paused here:** `MainWindow._on_plot_workspace_domain_changed` → `MultiGroupFitWindow.set_dataset` → same `_rebuild_group_nuisance_table` `QComboBox()` | Bypasses FitPanel deferral | **Not fixed** — next obvious site |

**Lesson:** patching individual sites will keep moving the crash. Prefer **one** policy: on FB↔FFT / domain button, do **no** Batch / multi-group / badge visibility / table rebuild work until those surfaces are actually shown (or idle).

---

## Code already changed (branch WIP — uncommitted)

Key files:

- `src/asymmetry/gui/panels/fit/single_tab.py` — per-domain `FitParameterTable`s; height-collapse swap; badge height-collapse
- `src/asymmetry/gui/panels/fit/panel.py` — Batch domain deferral; `materialize=False` snapshot; deferred global restore; defer group rebuild from `set_dataset` when Single active
- `src/asymmetry/gui/panels/fit/global_tab.py` — `_domain_table_stale`, `_group_table_stale`, `_pending_restore_state`, `flush_deferred_domain_rebuild`, synthesised default state
- `src/asymmetry/gui/panels/fit/tab_base.py` — incremental `FitParameterTable.populate` (reuse Fix/Link/Tie when name list unchanged)
- `src/asymmetry/gui/screen_guard.py` / `app.py` — earlier screen re-anchor (partial; does not cover parentless construction)
- Tests: `tests/gui/test_fit_panel_tabs.py`, `test_fit_parameter_table.py`, `test_single_fit_carry_forward_badge_gui.py`
- Docs: `docs/investigations/tahoe-qscreen-uaf.md`, `CHANGELOG.md` `[Unreleased]`, `PLANS.md`

**Not committed.** Also dirty: `uv.lock`, various docs (`INDEX`, `QUALITY`, …). User did not ask to commit.

**Known next hole:** `mainwindow.py` ~2095–2098 still calls `_multi_group_fit_window.set_dataset(...)` on domain change → rebuilds group-nuisance combos without deferral.

---

## Windows artifacts (user OneDrive)

Paths (may sync locally):

- `/Users/bhuddart/Library/CloudStorage/OneDrive-Nexus365/gui.log` — **stale** 806 B UTF-16 PowerShell capture; only `Cannot create window: no screens available`. Not a useful faulthandler log.
- `/Users/bhuddart/Library/CloudStorage/OneDrive-Nexus365/python.exe.25360.dmp` — useful minidump: Win11 26200; `0xC0000409` / `FAST_FAIL_FATAL_APP_EXIT`; RIP in `Qt6Core.dll`; stack Qt Widgets/Gui → Core.

**Capture tip for next Windows run:** use `cmd.exe`, not PowerShell `*>`:

```bat
cd /d C:\Users\benhu\Source\asymmetry
mkdir %TEMP%\asymmetry-crash-capture 2>nul
set PYTHONFAULTHANDLER=1
set PYTHONUNBUFFERED=1
set QT_LOGGING_RULES=qt.qpa.screen.info=true
.venv\Scripts\python.exe -X faulthandler -c "from asymmetry.gui.app import main; main()" > %TEMP%\asymmetry-crash-capture\gui.log 2>&1
```

macOS capture (used this session):

```bash
QT_LOGGING_RULES='qt.qpa.screen.info=true' PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 \
  .venv/bin/python -X faulthandler -c "from asymmetry.gui.app import main; main()" \
  > /tmp/asymmetry-crash-capture/gui.log 2>&1
```

---

## Recommended next strategies (pick one; don’t resume site-by-site)

**A. Domain-switch quarantine (preferred app-side)**  
On `_on_domain_button_clicked` / `_on_plot_workspace_domain_changed`:

- Update plot workspace + fit **domain labels / Single table swap only**.
- Do **not** call: `MultiGroupFitWindow.set_dataset`, Batch `set_current_dataset` rebuilds, badge expand if it can wait, any `flush` that builds combos.
- Mark Batch / multi-group / group-nuisance **stale**; flush only when that window/tab becomes visible (or on idle timer after domain switch settles).

**B. Upstream**  
File QTBUG with macOS `.ips` + Windows dump + “no screens available” (draft already in `tahoe-qscreen-uaf.md`). Windows dump weakens “Tahoe-only” title — frame as cross-platform screen-list / QPA failure under widget churn.

**C. Model/view**  
Still deferred; reduces combo density but does not stop `setVisible` / screen queries on other widgets.

**Do not claim** CHANGELOG “fixed Tahoe crash” — only risk reduction.

---

## Validation state

- Focused fit-panel / badge tests were green after last badge + group-deferral changes.
- Full `harness validate` once failed on flaky `test_log_panel_caps_block_count_on_unbounded_growth` under xdist worker crash; that test passed alone afterward.
- Re-run `python tools/harness.py validate` before any commit/PR.

---

## Explicit non-goals for the next agent (unless user asks)

- Do not keep fixing one `QComboBox()` / `show()` site at a time without a quarantine design.
- Do not edit the Cursor plan file for the per-domain table swap.
- Do not commit/push unless the user asks.
- Do not cut a release.

---

## One-paragraph blurb (pasteable)

Asymmetry FB↔FFT densifies Qt screen/DPI work. On macOS 26 this trips a Qt `QScreen` UAF (`screenAt` / freed scribble); on Windows 11 the same action empties/breaks the screen list and Qt fatal-exits (`Cannot create window: no screens available`, dump `FAST_FAIL_FATAL_APP_EXIT` in `Qt6Core`). Branch `harden-tahoe-qscreen-uaf` has partial deferrals (Batch Type combos, group-nuisance via FitPanel, Single per-domain tables without `QStackedWidget` visibility, badge height-collapse) but the crash/hang moves to the next site—latest unfixed: `MultiGroupFitWindow.set_dataset` on domain change. User paused whack-a-mole; next step should quarantine all Batch/multi-group/widget rebuilds off the domain-button path, plus upstream QTBUG with both OS artifacts. Details: `docs/investigations/tahoe-qscreen-uaf.md` and this handoff.
