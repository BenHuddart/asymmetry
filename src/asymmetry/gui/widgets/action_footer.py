"""The pinned action footer that sits below a panel's scroll area.

Every control panel ends the same way: a thin divider, an optional read-only
hint, a primary action button (plus occasional secondaries or a custom control
cluster), a status line, and — for long-running panels — a progress row. This
grew up copy-pasted (Fourier's ``_build_action_footer``, MaxEnt's Cancel +
progress footer). ``ActionFooter`` is the one composed widget for it.

It imposes no layout policy on its parent beyond being added *after* the scroll
area: it is a plain ``QWidget`` with an internal vertical stack. Buttons are
returned from the ``add_*`` methods so the caller wires their ``clicked``
signals; the primary button carries the accent treatment from
:func:`~asymmetry.gui.styles.widgets.build_primary_button_qss`.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.typography import status_font
from asymmetry.gui.styles.widgets import build_primary_button_qss


class ActionFooter(QWidget):
    """Pinned footer: divider, hint, action buttons, status, and progress row."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 6, 0, 0)
        layout.setSpacing(4)

        # ── Top divider ─────────────────────────────────────────────────────
        divider = QFrame(self)
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setFrameShadow(QFrame.Shadow.Plain)
        divider.setStyleSheet(f"color: {tokens.BORDER};")
        layout.addWidget(divider)

        # ── Optional hint (muted, word-wrapped) ─────────────────────────────
        self._hint_label = QLabel("", self)
        self._hint_label.setWordWrap(True)
        self._hint_label.setStyleSheet(f"QLabel {{ color: {tokens.TEXT_MUTED}; }}")
        self._hint_label.hide()
        layout.addWidget(self._hint_label)

        # ── Button row (buttons + custom clusters, stacked vertically) ──────
        self._button_container = QWidget(self)
        self._button_layout = QVBoxLayout(self._button_container)
        self._button_layout.setContentsMargins(0, 0, 0, 0)
        self._button_layout.setSpacing(4)
        layout.addWidget(self._button_container)

        # ── Status line (rich text, word-wrapped) ───────────────────────────
        self._status_label = QLabel("", self)
        self._status_label.setFont(status_font())
        self._status_label.setWordWrap(True)
        self._status_label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self._status_label)

        # ── Progress row (label + bar), hidden by default ───────────────────
        self._progress_row = QWidget(self)
        progress_layout = QHBoxLayout(self._progress_row)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.setSpacing(6)
        self._progress_label = QLabel("", self._progress_row)
        self._progress_label.setFont(status_font())
        self._progress_bar = QProgressBar(self._progress_row)
        # Indeterminate (busy) style — the panel footers show activity, not a
        # determinate count, matching FitRunControls' idle default (0..1).
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setTextVisible(False)
        progress_layout.addWidget(self._progress_label)
        progress_layout.addWidget(self._progress_bar, 1)
        self._progress_row.hide()
        layout.addWidget(self._progress_row)

    # ── Buttons / custom clusters ─────────────────────────────────────────────

    def add_primary(self, text: str) -> QPushButton:
        """Add and return the accent-styled primary action button."""
        button = QPushButton(text, self._button_container)
        button.setStyleSheet(build_primary_button_qss())
        self._button_layout.addWidget(button)
        return button

    def add_secondary(self, text: str) -> QPushButton:
        """Add and return a secondary (default-styled) action button."""
        button = QPushButton(text, self._button_container)
        self._button_layout.addWidget(button)
        return button

    def add_widget(self, widget: QWidget) -> None:
        """Add a custom control cluster (e.g. a +1/+5/+25 stepper) to the button area."""
        self._button_layout.addWidget(widget)

    # ── Hint ──────────────────────────────────────────────────────────────────

    def set_hint(self, text: str | None) -> None:
        """Set (or clear, with ``None``/empty) the muted hint above the buttons."""
        if text:
            self._hint_label.setText(str(text))
            self._hint_label.show()
        else:
            self._hint_label.clear()
            self._hint_label.hide()

    # ── Status ────────────────────────────────────────────────────────────────

    def set_status(self, html_or_text: str) -> None:
        """Set the status line — accepts plain text or rich-text chip HTML."""
        self._status_label.setText(str(html_or_text))

    def clear_status(self) -> None:
        """Clear the status line."""
        self._status_label.clear()

    # ── Progress ──────────────────────────────────────────────────────────────

    def show_progress(
        self,
        text: str = "",
        *,
        current: int | None = None,
        total: int | None = None,
    ) -> None:
        """Show the progress row with optional *text*.

        With no *current*/*total*, the bar is indeterminate (busy) — the
        original behaviour. Passing both switches it to a determinate 0..total
        bar at *current* (e.g. MaxEnt's per-cycle progress), clamped to
        ``[0, total]``; passing only one of the two is treated as "not given"
        and falls back to indeterminate.
        """
        self._progress_label.setText(str(text))
        if current is not None and total is not None:
            resolved_total = max(1, int(total))
            resolved_current = max(0, min(int(current), resolved_total))
            self._progress_bar.setRange(0, resolved_total)
            self._progress_bar.setValue(resolved_current)
        else:
            self._progress_bar.setRange(0, 0)
        self._progress_row.show()

    def hide_progress(self) -> None:
        """Hide the progress row and reset it back to indeterminate."""
        self._progress_row.hide()
        self._progress_label.clear()
        self._progress_bar.setRange(0, 0)
