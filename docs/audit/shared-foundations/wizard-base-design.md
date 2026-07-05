# Phase 3 Design Note — `WizardWindowBase`

**Audit:** shared-foundations · **Branch:** `audit/shared-foundations`
**Deliverable of the next step:** `src/asymmetry/gui/windows/wizard_base.py`, implemented by a Sonnet agent from this contract.
**Scope of this note:** the shared base that unifies the two fit-wizard windows. It is design only — no source was modified.

> **2026-07-05 update (wizard-ui-refresh):** the base now also owns the styled
> chrome — a `wizardHeaderBand` QFrame (title + context chips via
> `set_context_chips()` + muted status line) above a body layout of
> `[controls_row, content]`; `self._central_layout` refers to that body layout,
> so content sits at index 1. The global wizard no longer uses the default
> tab path: it overrides `_build_central()` with a Setup → Running → Result
> `QStackedWidget` (mirroring the single wizard), its parameter-setup modal is
> an embedded setup-page section, and its separate log window is an inline
> `LogPanel`. The driver/lifecycle contract below is unchanged.

## Orchestrator decisions (BINDING — resolve the two open behavior choices)

The implementer MUST follow these; they resolve the "open behavior choice" notes below.

1. **Failure reporting — NO new dialog (keep parity).** Both windows are
   status-label-only on failure today; there is no "better variant" to converge
   to, so adding a `QMessageBox.warning` would be a *new* behavior, not a
   convergence. The base's error slot must NOT show a dialog — it clears busy,
   runs the request-id guard, and delegates to `_on_analysis_failed(message)`
   (default = set status label). Drop the `QMessageBox` from §"Error handling".
   No BEHAVIOR-CHANGES entry needed for error reporting.

2. **closeEvent — ACCEPT cancel-on-close via `shutdown()` (best variant + invariant-compliant).**
   The current hide-and-run-while-busy behavior actually *violates* the AGENTS
   invariant "hold strong references to live threads, and shut them down in
   `closeEvent`". Converging both windows onto `closeEvent → self._tasks.shutdown()
   → super().closeEvent()` is both the best variant and the invariant-compliant
   one, and it is test-safe (no test pins the hide behavior). **This IS a
   deliberate, user-visible change — log it in BEHAVIOR-CHANGES.md** (what:
   closing a wizard mid-analysis now cancels the run instead of hiding the window
   and letting it finish; where: both wizard windows' closeEvent via the base;
   why: converges to the AGENTS thread-shutdown invariant).

## Overview

Two non-modal wizard windows duplicate the same skeleton:

- `src/asymmetry/gui/windows/fit_wizard_window.py` — `FitWizardWindow` (~937 lines), single-dataset. Tabs: Fingerprint → Portfolio → Compare → Apply. One analysis entry point (`_start_analysis`).
- `src/asymmetry/gui/windows/global_fit_wizard_window.py` — `GlobalFitWizardWindow` (~1553 lines), series. Tabs: Overview → Portfolio → Screening → Optimized → Roles → Apply. Two analysis entry points (`_start_analysis` "screening", `_start_selected_optimisation` "optimize"), a merge step, a separate `AnalysisLogWindow`, and a `single_fit_precomputed` side-channel.

Both hand-roll: a manual `QThread`/worker lifecycle (single: `fit_wizard_window.py:207-238`, `345-347`; global: `_launch_worker` at `733-768`, `_cleanup_analysis_thread` at `835-837`), a busy-state toggle (`_set_busy`), a hidden progress bar + label, a request-id staleness guard (`if request_id != self._analysis_request_id`), an error→status-text path, a result-cache signature (`_cached_signature` / `_cached_log_text`), a `set_cached_recommendation(...)` re-open path, and a `closeEvent` that hides-and-ignores while busy.

**The base owns the mechanism; subclasses own the analysis.** The base drives one shared `TaskRunner`, the progress UI, request-id staleness, error dialogs, the signature-cache plumbing, and closeEvent/cancel. Subclasses build their tabs, build their worker task, and populate their result tabs.

The design is constrained by one hard requirement: it **must not preclude** the pending, unmerged `feat/fit-wizard-scope` re-port (11 commits, ~7,600 lines), which adds a Scope tab, extra worker inputs, and cooperative cancel to *both* windows. Every base hook below is shaped so that re-port needs no edit to the base. See "feat/fit-wizard-scope accommodation".

## Class shape & ownership

```python
class WizardWindowBase(QMainWindow):
    """Shared skeleton for guided fit-wizard windows.

    Subclasses supply their tabs, their worker task, and their result
    population. The base owns the TaskRunner, progress UI, request-id
    staleness, error handling, the signature cache, and closeEvent/cancel.
    """
```

**Base class is `QMainWindow`**, matching both current windows (`fit_wizard_window.py:75`, `global_fit_wizard_window.py:311`). Not `QDialog`: both use `self.statusBar()` (`fit_wizard_window.py:862`, `global_fit_wizard_window.py:1348`) and are non-modal top-level windows. `GlobalFitWizardParameterSetupDialog` is a real `QDialog` but is a child dialog, not a wizard window — it stays in the subclass, untouched.

**The base OWNS (constructs and manages):**

| Member | Purpose |
|---|---|
| `self._tasks: TaskRunner` | one runner, parented to `self` (`TaskRunner(self)`). Drives every analysis. |
| `self._heading_label: QLabel` | bold heading; subclass sets text. |
| `self._status_label: QLabel` | word-wrapped status line. |
| `self._controls_row: QHBoxLayout` | the controls row; subclasses append their own buttons/combos into it. |
| `self._progress_label: QLabel` | hidden-when-idle busy label. |
| `self._progress_bar: QProgressBar` | indeterminate (`setRange(0, 0)`), hidden-when-idle, `maximumWidth(220)`. |
| `self._tabs: QTabWidget` | the tab container (populated by the subclass hook, see below). |
| `self._analysis_request_id: int` | monotonic staleness token. Bumped by the base on every `set_analysis_context`-style reset and on every analysis start. |
| `self._analysis_in_progress: bool` | busy flag. |
| `self._cached_signature: dict \| None` | last-analysis signature (see Result caching). |
| `self._cached_log_text: str` | last-analysis log text. |

**Subclasses OWN (declare and populate):** their result data (`self._recommendation`, `self._dataset`/`self._datasets`, `self._selected_key`, metric combos, per-window tables/plots/text widgets), plus everything scope-related in the re-port. Subclasses never touch `self._tasks` directly — they go through the base's `_run_analysis(...)` helper.

**Signals** stay declared on the subclasses (`apply_assessment_requested`, `analysis_cached`, and the global-only `parameter_setup_applied` / `single_fit_recommendations_generated`). Their shapes differ per window, so they are not lifted to the base. The base only *emits nothing on its own*; it calls a subclass hook that emits `analysis_cached` (see Result caching).

## Template-method hooks

The base calls these; the subclass implements them. Any hook with a trivial default is marked; the rest are abstract (`raise NotImplementedError`).

| Hook | Signature | When the base calls it | Contract |
|---|---|---|---|
| `_build_tabs()` | `(self) -> None` | Once, from `__init__`, after `self._tabs` exists and after `self._controls_row` exists. | Subclass creates its tab `QWidget`s and `self._tabs.addTab(...)` them **in display order**, then calls its per-tab `_build_*` methods. This is the tab-construction hook — a subclass may insert a tab *before* the analysis tabs here without the base caring (see scope accommodation). |
| `_create_worker_task(request_id)` | `(self, request_id: int) -> Callable[[TaskWorker], object]` | From `_run_analysis(...)`, on the GUI thread, immediately before submission. | Subclass returns a **closure** that runs its core analysis and returns a plain result object. The closure receives the `TaskWorker` and MAY call `worker.progress.emit(cur, total, msg)` and `worker.is_cancelled()`. The subclass captures `request_id` and its own inputs (dataset(s), model, metric, mode, scope, …) in the closure — the base never sees the input signature. This is the open-ended-worker-inputs hook. |
| `_populate_results(result)` | `(self, result: object) -> None` | On the GUI thread, from the base's finished-slot, only when `request_id` is current and `result` is non-`None`. | Subclass stores the result on itself, refreshes its tables/plots/status, and (if it caches) emits `analysis_cached`. Replaces today's `_populate_from_recommendation()` bodies. |
| `_analysis_signature()` | `(self) -> dict` | From `_run_analysis(...)` before submission (to stamp `_cached_signature`) and available to `_populate_results` for the `analysis_cached` emit. | Subclass returns a JSON-ish dict describing the inputs. Verbatim move of the existing `_analysis_signature()` bodies (`fit_wizard_window.py:328-337`, `global_fit_wizard_window.py:876-886`). |
| `_reset_result_state()` | `(self) -> None` | From the base's context-reset path and from `_run_analysis` before a fresh run. | Subclass clears its result members and calls its `_set_empty_state()`. Default: no-op is *not* acceptable — abstract. |
| `_on_analysis_failed(message)` | `(self, message: str) -> None` | From the base's error-slot, when `request_id` is current, **after** the base has shown the shared error dialog and cleared busy. | Optional override (default sets `self._status_label`). The global window overrides it to keep `self._recommendation` and re-populate on optimize-mode failure (`global_fit_wizard_window.py:815-824`). Error *mode* is carried in the result-less error path via a subclass-held field, see Error handling. |
| `_on_progress(current, total, message)` | `(self, current: int, total: int, message: str) -> None` | On the GUI thread, for each `worker.progress` emit, only when `request_id` is current. | Optional override. Default updates `self._progress_label`. The global window overrides it to append to its `AnalysisLogWindow` and refresh `_cached_log_text` (today's `_append_progress_log`, `global_fit_wizard_window.py:847-854`). |
| `_should_serve_cache()` | `(self) -> bool` | From `_run_analysis`, before submitting, **only if the subclass opts into short-circuiting.** Default returns `False`. | The global window's `_start_analysis` short-circuits when `self._cached_signature == signature and self._recommendation is not None` (`global_fit_wizard_window.py:659-663`). The single window does NOT (it always recomputes). To preserve both, the short-circuit is a subclass opt-in, not baked into the base. See Result caching. |

Notes on the two entry points: the base exposes **one** `_run_analysis(build_task, *, is_progress_streamed=…)` internal that both subclass entry points funnel through. The single window's `_start_analysis` and the global window's `_start_analysis`/`_start_selected_optimisation` become thin subclass methods that (a) do their preconditions and any dialog prompts, (b) set any mode field, then (c) call `self._run_analysis(...)`. The base does not model "screening vs optimize" — that lives entirely in the subclass's task closure and result object.

## Thread lifecycle — TaskRunner mapping (decision: replace both manual QThreads)

**Decision: both windows migrate onto `TaskRunner` (`src/asymmetry/gui/tasks.py:187-307`).** This matches the PR #68 migration pattern and deletes ~60 lines of hand-rolled `QThread` wiring per window. `TaskRunner` already solves everything the manual code solves and several things it gets subtly wrong (Windows GC-of-running-thread aborts, the orphan-reaper bounded-wait path). There is no wizard requirement TaskRunner cannot express — the four apparent gaps below all have clean mappings.

### The AGENTS bare-lambda invariant — how the base avoids it

> "Never connect a worker signal to a bare lambda/partial that touches widgets — with no receiver QObject the slot runs on the worker thread; route through a GUI-thread QObject method instead."

`TaskRunner` satisfies this **for free**. Its `start(...)` routes every caller callback — `on_finished`, `on_error`, `on_cancelled`, `on_progress` — through `_TaskRelay` (`tasks.py:145-185`, wired at `227-239`), a `QObject` parented to the runner and therefore living on the GUI thread. Delivery is queued, so the base's callbacks may safely touch widgets. The base's rule for the implementer: **pass the base's own bound methods** (`self._handle_finished`, `self._handle_error`, `self._handle_progress`, `self._handle_cancelled`) as the `on_*` callbacks — never a lambda that closes over a widget. The only closure allowed is the *task function* itself (`_create_worker_task`'s return), which runs on the worker thread by design and must touch **no** widgets (it returns a plain result object).

### How the base drives TaskRunner

`_run_analysis` (base, GUI thread):

1. Bump `self._analysis_request_id`; snapshot `request_id = self._analysis_request_id`.
2. `self._cached_signature = copy.deepcopy(self._analysis_signature())`.
3. `self._set_busy(True)`; `self._reset_result_state()`.
4. `task = self._create_worker_task(request_id)`.
5. Submit:

   ```python
   self._tasks.start(
       task,
       on_finished=self._make_finished_slot(request_id),
       on_error=self._make_error_slot(request_id),
       on_cancelled=self._make_cancelled_slot(request_id),
       on_progress=self._make_progress_slot(request_id),
       cancel_exceptions=self._cancel_exceptions(),   # subclass hook, default ()
   )
   ```

**Request-id staleness** (critical — pins the "context changed mid-run" tests at `fit_wizard_window.py:260-266`, `290-301`; `global_fit_wizard_window.py:776-778`, `812-814`): `TaskRunner` callbacks carry **no** worker identity, so the base captures `request_id` in each slot via a small closure factory (`_make_finished_slot(request_id)` returns a bound method wrapper, or uses `functools.partial` on a base method — but the *receiver is the base QObject*, so it is still queued/GUI-thread-safe; `partial` here is fine because it wraps a bound method, not a bare function). Each slot first checks `if request_id != self._analysis_request_id: return` (after clearing busy), exactly as today. This replaces the old `finished = Signal(int, object)` request-id-in-payload scheme without changing behavior.

**`_make_finished_slot(request_id)(result)`** (GUI thread):
- clear busy; if stale → return (with the "context changed" status text the tests expect).
- if `result` is not the expected type → status "unexpected result", return (mirrors `fit_wizard_window.py:267-271`).
- else `self._populate_results(result)`.

**`_make_progress_slot(request_id)(cur, total, msg)`**: if current, `self._on_progress(cur, total, msg)`.

**`_make_error_slot(request_id)(message)`**: if current, `self._set_busy(False)`, show shared error dialog, `self._on_analysis_failed(message)`.

**`_make_cancelled_slot(request_id)()`**: if current, `self._set_busy(False)`, set "Analysis cancelled." status. (Today's windows have no cancel path; the re-port adds one — see accommodation.)

### The four apparent gaps and their mappings

1. **Global's synchronous test-seam.** `_launch_worker` runs the worker **inline** (`worker.run()` on the GUI thread) when the core builders are monkeypatched (`global_fit_wizard_window.py:738-753`). TaskRunner has no inline path and we do **not** re-add one. Consequences and the required test edit are in "What each subclass keeps vs. moves" and "Risks". The migration goes **always-async**; the completion observable changes from `window._analysis_thread is None` to `window._tasks.active_count == 0` (or `window._analysis_in_progress is False`). This is a mechanical, required test edit (three call sites: `test_fit_wizard_window.py:179`, `:307`; `test_global_fit_wizard_window.py:386`).

2. **Global worker's extra signals.** The global worker emits `single_fit_precomputed(int, object)` (`global_fit_wizard_window.py:86`, `189-193`), a 2-arg `progress(int, str)` (`:85`), and `finished(int, str, object)` carrying `mode` (`:83`). TaskRunner's signal set is fixed (`progress(int,int,str)` / `finished(object)` / `error(str)` / `cancelled()`). Mapping:
   - **`single_fit_precomputed`** fires exactly once, immediately before `finished` (`:189-197`) — it is not truly mid-run. **Fold it into the returned result object** (e.g. `GlobalAnalysisResult(recommendation=…, mode=…, updated_single_fits={…})`). The base's finished-slot hands the whole object to `_populate_results`, which applies the single-fit update and emits `single_fit_recommendations_generated` there. Lossless.
   - **`mode`** rides in the same result object. `_populate_results` branches on `result.mode` to run `merge_global_fit_wizard_recommendations` for optimize (`:784-788`).
   - **2-arg → 3-arg progress.** The global task closure calls `worker.progress.emit(0, 0, msg)` (indeterminate) and the global `_on_progress` override ignores the counts and appends `msg` to its log window. Verbatim behavior, adapted signature.

3. **closeEvent policy divergence.** See closeEvent/cancel section — decided there.

4. **Cache-hit short-circuit divergence.** See Result caching — decided there (subclass opt-in).

## Progress UI

The base builds one progress pair (`_progress_bar` indeterminate + `_progress_label`), both hidden when idle, inside `_controls_row`. `_set_busy(busy)` (base) toggles their visibility and the busy flag; it also calls a subclass hook `_update_action_enablement(busy)` so each window can enable/disable its own buttons/combos (single: Start/Refresh/metric/nav; global: Build/Optimize/metric). The current `_set_busy` bodies split cleanly: the shared visibility/flag lines move to the base; the per-window button lines move into `_update_action_enablement`.

Subclasses report progress through `worker.progress.emit(cur, total, msg)` inside their task closure. The base relays it (queued, GUI thread) to `_on_progress`. Default `_on_progress` writes `self._progress_label`; the global window overrides to stream into its `AnalysisLogWindow` (keeps the `test_global_fit_wizard_window_shows_progress_log` behavior at `test_global_fit_wizard_window.py:438-466`).

## Result caching

The signature-cache is shared plumbing; the signature *content* is subclass-supplied.

- **`self._cached_signature` / `self._cached_log_text`** live on the base.
- **`_analysis_signature()`** (subclass hook) supplies the dict. Single: `{run_number, model}`. Global: `{run_numbers, model, types, values, bounds}`. Verbatim moves.
- **On analysis start**, the base stamps `self._cached_signature = copy.deepcopy(self._analysis_signature())` before submitting (as both windows do today).
- **On finish**, `_populate_results` emits `analysis_cached(recommendation, self.current_log_text(), copy.deepcopy(self._cached_signature))` — the base provides `current_log_text()` (returns `self._cached_log_text`; global overrides to first pull from its log window, `:943-946`). This keeps the payload the `test_fit_wizard_window_emits_cached_analysis_payload` test asserts (`test_fit_wizard_window.py:312-345`).
- **`set_cached_recommendation(...)`** (re-open-from-cache) stays a subclass method — its argument type differs per window (`FitWizardRecommendation` vs `GlobalFitWizardRecommendation`, and global takes an extra `status_text`). It sets `_cached_signature` / `_cached_log_text` / `_recommendation` and calls the subclass populate path. The base offers a protected helper `_store_cached_signature(signature)` (`copy.deepcopy` if `isinstance(dict)` else `None`) so both windows share that one line. Pins `test_fit_wizard_window_accepts_cached_recommendation` (`:348-363`) and `test_global_fit_wizard_window_apply_recommended_emits_assessment` (`:418-435`).
- **Same-signature-serves-cache short-circuit** (global only, `:659-663`): exposed as the `_should_serve_cache()` opt-in hook. Default `False`. The single window does **not** override it (preserves its always-recompute behavior). The global window overrides it to `self._cached_signature == self._analysis_signature() and self._recommendation is not None`; when `True`, its `_start_analysis` re-populates from the existing recommendation and returns without submitting — the base offers `_serve_cached_result()` for the re-populate-and-return branch, but the *decision* is the subclass's. This is untested today, so leaving it subclass-local risks nothing.

## Error handling

The base owns the worker-exception → dialog path. Today neither window shows a dialog — they write to the status label (single: `:304`; global: `:819`). **Behavior change (log in `BEHAVIOR-CHANGES.md`): the base additionally shows a shared `QMessageBox.warning(self, "<Wizard> Analysis Failed", message)`** and *then* delegates to `_on_analysis_failed(message)` for the status-label text (which subclasses already own). If Review B2 prefers status-label-only parity, drop the dialog and keep only the delegated `_on_analysis_failed` — the hook structure is unchanged either way; note this as the one open behavior choice. The request-id staleness guard runs first, so a stale error never dialogs.

The global window's error path needs the **mode** to decide whether to null `self._recommendation` (screening) or keep it (optimize) (`:816-824`). Since the base's error slot carries only `message`, the global subclass stashes the in-flight mode on itself when it starts an analysis (it already sets `self._analysis_mode`, `:701`, `:738`) and reads it inside its `_on_analysis_failed` override. No base change needed.

## closeEvent / cancel

**Today:** both windows override `closeEvent` to **hide-and-ignore while busy** (`fit_wizard_window.py:349-354`; `global_fit_wizard_window.py:1431-1436`) — they never cancel; the analysis runs to completion behind a hidden window.

**Decision: the base's `closeEvent` calls `self._tasks.shutdown()` then `super().closeEvent(event)`** — the documented TaskRunner pattern (`tasks.py:204`, `268-307`). `shutdown()` cooperatively cancels every live worker, quits/waits with a bounded timeout, and retires any thread that overruns to the process-level reaper (so a long numpy call between cancel-polls cannot abort the process). This is a **behavior change** (close now cancels instead of hiding-and-running) and is logged in `BEHAVIOR-CHANGES.md`.

Safety of the change, verified: **no test asserts the hide-while-busy behavior.** Grep of both full test files shows the only `isVisible`/`hide` references are on the global `_log_window` (`test_global_fit_wizard_window.py:462`, `:501`, `:518`), unrelated to `closeEvent`. So the switch is test-safe.

If Review B2 wants to *retain* an explicit "confirm close during a run" UX, that belongs in a subclass `closeEvent` override that calls the base after confirming — but the default, and the recommendation, is cancel-on-close via `shutdown()`. State clearly which wins if both are present: **an overriding subclass `closeEvent` wins and must call the base's cancel/shutdown helper itself.**

**Mid-analysis cancel** (no cancel control exists today, but the re-port adds one): the base exposes `_cancel_current_analysis()` which cancels the live worker(s) via the runner. Because TaskRunner returns a `TaskWorker` handle from `start(...)`, the base keeps the handle of the most recent submission (`self._current_worker`) and `_cancel_current_analysis()` calls `self._current_worker.cancel()`. The base's `on_cancelled` slot then clears busy and sets the cancelled status. Subclass cancel buttons call `self._cancel_current_analysis()`.

## feat/fit-wizard-scope accommodation

The pending branch (inspected read-only via `git diff main..feat/fit-wizard-scope`) adds the same three things to both windows. The base accommodates each **without needing an edit**:

1. **Scope tab inserted before the analysis tabs.** The branch prepends a `WizardScopeSelector` tab and renumbers the rest (single: `"1. Scope"` … `"5. Apply"`; global: `"1. Scope"` … `"7. Apply"`). Because tab construction is the subclass's `_build_tabs()` hook — the base only guarantees `self._tabs` and `self._controls_row` exist first — a subclass adds/inserts any tab (including one at index 0) with zero base changes. The base never assumes a fixed tab count or index.

2. **Extra worker inputs.** The branch threads a `scope` dict into each worker (`scope=scope` in the diff) plus multiplet/peak-seed inputs. Because the base takes an opaque **task closure** from `_create_worker_task(request_id)` and never names the worker's inputs, the subclass captures `scope` (and anything else) in the closure. `_analysis_signature()` gains a `"scope"` key on the subclass side; the base's cache plumbing is signature-content-agnostic, so it just works. The branch's legacy-signature handling (`signature.get("scope")` defaulting to `None`) lives in the subclass `set_cached_recommendation`.

3. **Cooperative cancel coexisting with progress/cancel wiring.** The branch's worker uses a `threading.Event` (`self._cancel_event`) polled by the fit engine via `cancel_callback`, and raises `FitCancelledError` (single) / checks between builder phases (global); a window-level **Cancel** button in `controls_row` calls `worker.request_cancel()`. Mapping onto the base:
   - `self._cancel_event.is_set()` → **`worker.is_cancelled()`** (TaskWorker's polled flag, `tasks.py:131-132`), passed as the `cancel_callback` inside the task closure.
   - the bespoke `except FitCancelledError` → **`cancel_exceptions=(FitCancelledError,)`** on `_tasks.start(...)`, surfaced via the `_cancel_exceptions()` subclass hook; TaskWorker then emits `cancelled` and the base's cancelled-slot handles UI.
   - the Cancel button → `controls_row` is base-owned and subclass-appendable; the button's slot calls the base's `_cancel_current_analysis()`.

   The task's original wording ("per-analysis Cancel buttons on the scope tree") is a superset of what the branch actually ships (one window-level Cancel button). The contract satisfies both: **the base exposes a single cancel entry point (`_cancel_current_analysis()`) that any subclass control — one button, or many tree buttons — can call, wherever it lives.** TaskRunner does not preclude the re-port's cancel; it expresses it more cleanly.

## What each subclass keeps vs. what moves to the base

### Moves to `WizardWindowBase`
- `TaskRunner` ownership; all `QThread`/worker lifecycle (deletes `_analysis_thread`, `_analysis_worker`, `_cleanup_analysis_thread`, `_launch_worker`'s thread wiring).
- Heading + status labels, `_controls_row`, progress bar + label, `_tabs` container.
- `_analysis_request_id` bump/compare; the request-id staleness guards.
- `_set_busy` visibility/flag half; the finished/error/progress/cancelled relay slots.
- Shared error dialog; `current_log_text()` default; `_store_cached_signature`; `closeEvent` → `shutdown()`; `_cancel_current_analysis()`.

### `FitWizardWindow` keeps (subclass)
- `self._dataset`, `self._current_model`, `self._recommendation`, `self._selected_key`; the 4 tab builders and every `_populate_*` / `_update_*` / plot method; `_metric_combo` + `_on_metric_changed`; nav buttons + `_go_previous_tab`/`_go_next_tab`/`_update_navigation_buttons`; `_apply_*` handlers; `_show_metric_info`/`_show_residual_info`; `_analysis_signature` (returns `{run_number, model}`); `set_cached_recommendation`; `set_analysis_context`.
- Implements: `_build_tabs`, `_create_worker_task` (closure calling `build_fit_wizard_recommendation`), `_populate_results` (= today's `_populate_from_recommendation` + `analysis_cached` emit), `_analysis_signature`, `_reset_result_state`, `_update_action_enablement`. Does **not** override `_should_serve_cache` (stays always-recompute), `_on_progress` (label default is fine), or `_cancel_exceptions` (until the re-port).

### `GlobalFitWizardWindow` keeps (subclass)
- All series state (`_datasets`, `_current_parameter_types/_values`, `_parameter_bounds`, `_recommendation`, `_selected_key`, `_screening_selected_keys`, `_running_template_keys`, `_analysis_mode`, `_single_fit_recommendations_by_run`); the 6 tab builders and every `_populate_*`/`_update_*`; `AnalysisLogWindow` + `_show_log_window` + `_append_progress_log` (now the `_on_progress` override); `GlobalFitWizardParameterSetupDialog` + `_prompt_parameter_setup` + `_apply_parameter_setup`; metric combo + rerank; both apply handlers; the merge-on-optimize logic; `_analysis_signature` (5-key); `set_cached_recommendation`; `set_analysis_context`.
- Implements: `_build_tabs`, `_create_worker_task` (closure that runs the screening/optimize builders per `mode`, folds `single_fit_precomputed` and `mode` into a result object), `_populate_results` (applies single-fit update, emits `single_fit_recommendations_generated`, branches on `mode` to merge, then repopulates), `_analysis_signature`, `_reset_result_state`, `_update_action_enablement`, `_on_progress` (→ log window), `_on_analysis_failed` (mode-aware), `_should_serve_cache` (True on matching signature). Its two entry points (`_start_analysis`, `_start_selected_optimisation`) stay as subclass methods that do preconditions/dialogs, set `_analysis_mode`, then call `self._run_analysis(...)`.

## Risks / open questions for the implementer & Review B2

1. **Test edits are mandatory, not incidental.** Removing `_analysis_thread` breaks the completion observable at three sites (`test_fit_wizard_window.py:179`, `:307`; `test_global_fit_wizard_window.py:386`). Rewrite them to `window._tasks.active_count == 0` / `window._analysis_in_progress is False`. These are the only such couplings found; grep confirms no other `_analysis_thread`/`_analysis_worker`/`_launch_worker` references in either full test file.

2. **The synchronous test-seam is gone.** Global tests monkeypatch the core builders and today rely on `_launch_worker` running inline. Going always-async, those tests already wrap starts in `wait_for(...)` (verified across both full test files — every `_start_analysis`/`_start_selected_optimisation` call is followed by `wait_for`). So an always-async migration keeps them green *provided* the monkeypatched builder still runs on the worker thread and `wait_for` pumps the event loop. Implementer must confirm `wait_for` drives `qapp.processEvents()` (it does in the existing helper). **Do not re-introduce an inline path** — it would violate the never-run-long-work-on-GUI-thread invariant and defeat the migration.

3. **Error-dialog behavior change.** The base adds a `QMessageBox.warning` on failure where today there is none. If Review B2 wants strict parity, make the dialog opt-in (subclass flag) or drop it and keep status-label-only. Flagged as the single open behavior choice; must land in `BEHAVIOR-CHANGES.md` either way.

4. **closeEvent behavior change.** Close now cancels (via `shutdown()`) instead of hiding-and-running. Test-safe (no assertion pins the old behavior) but user-visible; log it. Review B2 should confirm the shutdown timeout (default 10 s) and the reaper handoff are acceptable for a wizard whose optimize pass can be long.

5. **`request_id` capture correctness.** The base captures `request_id` per submission in slot closures. The implementer must ensure the closure captures the *value* at submit time (bind as a default arg or via `functools.partial(self._handle_finished, request_id)`), not `self._analysis_request_id` read late — otherwise every stale-guard passes and the "context changed mid-run" tests fail. This is the single highest-risk line in the migration.

6. **Global cache short-circuit stays subclass-local and untested.** If Review B2 wants it pinned, add a test; the design does not add one.

7. **`current_worker` handle lifetime.** The base holds `self._current_worker` for `_cancel_current_analysis()`. TaskRunner already keeps strong refs to live `(thread, worker)` pairs (`tasks.py:209`, `254`), so the base's handle is a convenience, not the keep-alive — but the implementer must null it in the finished/error/cancelled slots to avoid pointing at a `deleteLater`'d worker (call `.cancel()` is safe even post-terminal per `shutdown`'s `RuntimeError` guard at `:272-276`, but reading other attrs is not).
