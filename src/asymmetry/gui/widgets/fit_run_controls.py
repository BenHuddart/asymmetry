"""Shared run-control widgets (Stop/Cancel button + optional progress bar).

Converges the near-identical Stop-button construction that grew up
independently in ``fit_panel.py``'s ``SingleFitTab`` and ``GlobalFitTab``
(same button, same hide-until-busy default, distinct tooltips), and offers
the same building blocks to ``MaxEntPanel``'s Cancel-button-plus-progress-bar
footer where the shape fits cleanly.

This is a plain holder, not a composite ``QWidget``: it builds the button
(and, optionally, the progress label/bar) and exposes them as attributes so
each caller can parent them into its own layout at the exact grid/box
position it already uses, and keep driving them with its own busy/enabled
policy. The two adopted call sites differ in that policy — the fit tabs
toggle the Stop button's *visibility* (it swaps in for the Fit button),
while MaxEnt toggles the Cancel button's *enabled* state (it sits fixed in
the footer) — so no single ``set_busy`` policy is imposed here; callers keep
their own ``_set_fit_busy``/``_set_busy`` methods, driving the widgets built
by this class instead of ones they construct inline.
"""

from __future__ import annotations

import html
from collections.abc import Callable

from PySide6.QtWidgets import QLabel, QProgressBar, QPushButton


class FitRunControls:
    """Builder/holder for a Stop-or-Cancel button and optional progress UI.

    Parameters
    ----------
    button_label:
        Text for the run-control button (``"Stop"`` for the fit tabs,
        ``"Cancel"`` for MaxEnt).
    tooltip:
        Tooltip text for the button. Empty string leaves the default (no
        tooltip set).
    on_cancel:
        Optional GUI-thread slot connected to ``button.clicked``. Pass
        ``None`` to wire the connection externally (as ``MainWindow`` does
        for MaxEnt's Cancel button via ``hasattr``/``.clicked.connect``).
    hidden:
        Whether the button starts hidden (the fit-tab style: it swaps in for
        the Fit button once a worker-based run starts). Defaults to ``True``.
        MaxEnt's Cancel button instead starts visible-but-disabled, so a
        caller with that shape passes ``hidden=False``.
    with_progress:
        When ``True``, also builds ``.progress_label`` (word-wrapped
        ``QLabel``) and ``.progress_bar`` (``QProgressBar``), matching
        MaxEnt's current footer. When ``False`` (default), both attributes
        are ``None``.
    parent:
        Optional parent widget passed through to the constructed widgets.
    """

    def __init__(
        self,
        *,
        button_label: str = "Stop",
        tooltip: str = "",
        on_cancel: Callable[[], None] | None = None,
        hidden: bool = True,
        with_progress: bool = False,
        parent=None,
    ) -> None:
        self.button = QPushButton(button_label, parent)
        if tooltip:
            self.button.setToolTip(tooltip)
        if on_cancel is not None:
            self.button.clicked.connect(on_cancel)
        if hidden:
            self.button.hide()

        if with_progress:
            self.progress_label = QLabel("", parent)
            self.progress_label.setWordWrap(True)
            self.progress_bar = QProgressBar(parent)
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(False)
        else:
            self.progress_label = None
            self.progress_bar = None

    def set_indeterminate(self, on: bool) -> None:
        """Switch the progress bar between indeterminate (busy) and 0/1 (idle)."""
        if self.progress_bar is None:
            return
        if on:
            self.progress_bar.setRange(0, 0)
        else:
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(0)

    def set_progress(self, current: int, total: int, message: str) -> None:
        """Update the progress bar/label from a worker-reported (current, total, message)."""
        if self.progress_bar is None or self.progress_label is None:
            return
        resolved_total = max(1, int(total))
        resolved_current = max(0, min(int(current), resolved_total))
        self.progress_bar.setRange(0, resolved_total)
        self.progress_bar.setValue(resolved_current)
        self.progress_label.setText(html.escape(str(message)))
