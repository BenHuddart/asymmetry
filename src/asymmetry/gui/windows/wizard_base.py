"""Shared skeleton for guided fit-wizard windows.

See ``docs/audit/shared-foundations/wizard-base-design.md`` for the full
contract this class implements. In short: the base owns the mechanism (one
``TaskRunner``, the progress UI, request-id staleness, error handling, the
signature cache, and closeEvent/cancel); subclasses own the analysis (their
tabs, their worker task, and how they populate their result tabs).
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from functools import partial

from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from asymmetry.gui.tasks import TaskRunner, TaskWorker


class WizardWindowBase(QMainWindow):
    """Shared skeleton for guided fit-wizard windows.

    Subclasses supply their tabs, their worker task, and their result
    population. The base owns the TaskRunner, progress UI, request-id
    staleness, error handling, the signature cache, and closeEvent/cancel.

    The content region beneath the heading/status/controls chrome comes from
    the ``_build_central()`` hook. Its default creates ``self._tabs`` and
    calls the ``_build_tabs()`` hook, exactly as before; a subclass may
    instead override ``_build_central()`` to return its own content widget
    (e.g. a QStackedWidget) — on that path ``self._tabs`` stays ``None``,
    ``_build_tabs()`` is never called and need not be implemented. The
    surrounding chrome stays base-owned on both paths.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._tasks = TaskRunner(self)
        self._analysis_request_id = 0
        self._analysis_in_progress = False
        self._cached_signature: dict | None = None
        self._cached_log_text: str = ""
        self._current_worker: TaskWorker | None = None

        self._heading_label = QLabel()
        self._heading_label.setStyleSheet("font-weight: bold;")

        self._status_label = QLabel()
        self._status_label.setWordWrap(True)

        self._controls_row = QHBoxLayout()

        self._progress_label = QLabel()
        self._progress_label.setVisible(False)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setMaximumWidth(220)
        self._progress_bar.setVisible(False)

        self._controls_row.addWidget(self._progress_label)
        self._controls_row.addWidget(self._progress_bar)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(self._heading_label)
        layout.addWidget(self._status_label)
        layout.addLayout(self._controls_row)
        self.setCentralWidget(central)
        #: The central QVBoxLayout ([heading, status, controls, <content>]).
        #: Exposed so a subclass can append trailing rows (e.g. a nav row)
        #: beneath the content region from its _build_tabs() hook.
        self._central_layout = layout

        #: The tab container. Assigned a QTabWidget only by the default
        #: _build_central(); stays None when a subclass overrides the content
        #: region, so base and subclass code must not assume it exists.
        self._tabs: QTabWidget | None = None

        # Content-region construction runs after _controls_row and
        # _central_layout exist, so the hook may append its own
        # controls/tabs/rows safely. The default _build_central attaches the
        # tab widget itself (it must be in the layout before _build_tabs runs,
        # so _build_tabs can insert widgets relative to it and append trailing
        # rows beneath it); an override that merely returns its widget is
        # attached here.
        content = self._build_central()
        if layout.indexOf(content) == -1:
            layout.addWidget(content)

    # ------------------------------------------------------------------
    # Template-method hooks — abstract, subclasses must implement.
    # ------------------------------------------------------------------

    def _build_tabs(self) -> None:
        # Required only on the default _build_central() path; a subclass that
        # overrides _build_central() supplies its own content widget and is
        # not required to implement this hook.
        raise NotImplementedError

    def _create_worker_task(self, request_id: int) -> Callable[[TaskWorker], object]:
        raise NotImplementedError

    def _populate_results(self, result: object) -> None:
        raise NotImplementedError

    def _analysis_signature(self) -> dict:
        raise NotImplementedError

    def _reset_result_state(self) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Template-method hooks — optional, with sensible defaults.
    # ------------------------------------------------------------------

    def _build_central(self) -> QWidget:
        """Build and return the content region beneath the base chrome.

        Default: create ``self._tabs``, attach it to ``self._central_layout``
        (so ``_build_tabs()`` can insert widgets relative to it and append
        trailing rows beneath it), run the ``_build_tabs()`` hook, and return
        the tab widget. An override returns its own content widget instead;
        ``__init__`` attaches the returned widget to the central layout if
        the hook did not attach it already. On the override path
        ``self._tabs`` stays ``None`` and ``_build_tabs()`` is never called.
        """
        self._tabs = QTabWidget()
        self._central_layout.addWidget(self._tabs)
        self._build_tabs()
        return self._tabs

    def _on_analysis_failed(self, message: str) -> None:
        self._status_label.setText(message)

    def _on_progress(self, current: int, total: int, message: str) -> None:
        self._progress_label.setText(message)

    def _should_serve_cache(self) -> bool:
        return False

    def _update_action_enablement(self, busy: bool) -> None:
        pass

    def _cancel_exceptions(self) -> tuple[type[BaseException], ...]:
        return ()

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------

    def _run_analysis(self) -> None:
        # Cooperatively cancel any still-live prior worker before starting a new
        # one. The old per-window code serialized by joining the prior QThread
        # (thread.wait()); TaskRunner deliberately does not block the GUI, so we
        # request cancel instead. A prior run whose context changed mid-flight
        # (set_analysis_context clears busy but not _current_worker) is otherwise
        # only dropped by the staleness guard — this stops it wasting cycles once
        # the analysis honours cancel_callback. Its terminal signal is dropped by
        # the staleness guard regardless.
        if self._current_worker is not None:
            self._current_worker.cancel()

        self._analysis_request_id += 1
        request_id = self._analysis_request_id

        self._cached_signature = copy.deepcopy(self._analysis_signature())
        self._set_busy(True)
        self._reset_result_state()

        task = self._create_worker_task(request_id)

        # Bind request_id now via functools.partial on a bound method of this
        # base QObject — the receiver is a GUI-thread QObject, so delivery
        # stays queued/thread-safe. Reading self._analysis_request_id late
        # inside the callback would defeat the staleness guard.
        self._current_worker = self._tasks.start(
            task,
            on_finished=partial(self._handle_finished, request_id),
            on_error=partial(self._handle_error, request_id),
            on_progress=partial(self._handle_progress, request_id),
            on_cancelled=partial(self._handle_cancelled, request_id),
            cancel_exceptions=self._cancel_exceptions(),
        )

    # ------------------------------------------------------------------
    # Relay slots (GUI thread, via TaskRunner's queued relay)
    # ------------------------------------------------------------------

    def _handle_finished(self, request_id: int, result: object) -> None:
        self._set_busy(False)
        if request_id != self._analysis_request_id:
            return
        if result is not None:
            self._populate_results(result)
        self._current_worker = None

    def _handle_error(self, request_id: int, message: str) -> None:
        # Clear busy BEFORE the staleness guard, mirroring _handle_finished: a
        # window whose context changed mid-run (set_analysis_context bumps the
        # id and leaves busy True, delegating the clear to the terminal slot)
        # would otherwise stay soft-locked when the stale worker errors.
        self._set_busy(False)
        if request_id != self._analysis_request_id:
            return
        self._on_analysis_failed(message)
        self._current_worker = None

    def _handle_progress(self, request_id: int, current: int, total: int, message: str) -> None:
        if request_id != self._analysis_request_id:
            return
        self._on_progress(current, total, message)

    def _handle_cancelled(self, request_id: int) -> None:
        # Clear busy before the staleness guard (see _handle_error).
        self._set_busy(False)
        if request_id != self._analysis_request_id:
            return
        self._status_label.setText("Analysis cancelled.")
        self._current_worker = None

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _set_busy(self, busy: bool) -> None:
        self._analysis_in_progress = busy
        self._progress_bar.setVisible(busy)
        self._progress_label.setVisible(busy)
        self._update_action_enablement(busy)

    def current_log_text(self) -> str:
        return self._cached_log_text

    def _store_cached_signature(self, signature: object) -> None:
        self._cached_signature = copy.deepcopy(signature) if isinstance(signature, dict) else None

    def _cancel_current_analysis(self) -> None:
        if self._current_worker is not None:
            self._current_worker.cancel()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._tasks.shutdown()
        super().closeEvent(event)
