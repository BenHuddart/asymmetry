"""Compact per-range card for the trend-fit "range cards" redesign.

Replaces the wide row-of-buttons layout in the trend model-fit dialog with one
small card per fit range: a colour swatch matching the range's plot span, a
title, a fit-status chip, and its bounds. The card that is currently active
additionally shows a primary "Run Fit" button and dedicated "Edit Model" /
"Remove" buttons.

The card is a pure view — it renders a plain :class:`RangeCardView` handed to
it by the dialog and emits signals on user interaction. It owns no fit/range
state itself and imports nothing from ``asymmetry.core`` or the dialog
modules, so it can be constructed and tested in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QMouseEvent, QPainter
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.widgets import (
    build_primary_button_qss,
    widest_button_width,
)

#: Fixed size (px) of the colour-swatch square on line 1.
_SWATCH_SIZE = 12

#: objectName so the active/unselected stylesheet targets only the surface frame
#: (not its child labels/buttons, which a bare ``QFrame { … }`` rule would also
#: hit through descendant inheritance of the background).
_SURFACE_OBJECT_NAME = "rangeCardSurface"


class _CardSurface(QFrame):
    """The card's styled background surface.

    A ``QFrame`` (not a ``QPushButton``): a frame sizes to its child layout, so
    the two content rows lay out at their natural height. A ``QPushButton`` sizes
    to its own (empty) text hint and collapses the rows into a ~single-line
    button, which squashed the card. The active/unselected highlight is applied
    as a ``QFrame#rangeCardSurface`` stylesheet (see :meth:`RangeCard.set_active`)
    rather than via ``style_group_state_button`` (which targets a ``QPushButton``
    selector). Mouse presses on the card background (child labels do not consume
    them) are repurposed to emit the card's ``selected`` signal.
    """

    def __init__(self, on_pressed, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName(_SURFACE_OBJECT_NAME)
        self._on_pressed = on_pressed

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 — Qt override
        if event.button() == Qt.MouseButton.LeftButton:
            self._on_pressed()
        super().mousePressEvent(event)


class _ElidingLabel(QLabel):
    """A QLabel that elides its text to the CURRENT width at paint time.

    The card formula is long; a plain QLabel would force the card wide, and the
    previous manual elide-on-set_state over-truncated before the freshly-rebuilt
    card had a width. Eliding in paintEvent is always correct for the current
    width and needs no layout-timing juggling."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._full_text = ""
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

    def set_full_text(self, text: str) -> None:
        self._full_text = text or ""
        self.setToolTip(self._full_text)
        self.update()

    def full_text(self) -> str:
        return self._full_text

    def minimumSizeHint(self) -> QSize:  # noqa: N802 — Qt override
        # Don't let the long formula dictate a large minimum width.
        return QSize(0, self.fontMetrics().height())

    def paintEvent(self, event) -> None:  # noqa: N802 — Qt override
        painter = QPainter(self)
        elided = self.fontMetrics().elidedText(
            self._full_text, Qt.TextElideMode.ElideRight, self.width()
        )
        painter.drawText(
            self.rect(),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            elided,
        )


@dataclass(frozen=True)
class RangeCardView:
    """Plain render payload for one :class:`RangeCard`.

    The dialog owns every range/fit concept; this is the only thing it hands
    to the card. All strings arrive pre-formatted (bounds text, chip HTML,
    tooltips) so the card never needs to know how a range or a fit result is
    shaped.
    """

    idx: int
    title: str  # "Range 1" (base) or "" / formula-only (cross-group)
    swatch_color: str  # hex; matches the range's plot span colour
    bounds_text: str  # "[12-88 K]" or "[12-40] u [55-88] K" (windowed)
    formula: str  # elided in the card body, full in tooltip
    status: Literal["not_run", "success", "failed", "running"]
    status_chip_html: str  # rich-text chip (dialog builds via fit_quality_chip_html)
    status_tooltip: str  # chi^2 detail on success; result message on failure
    can_remove: bool  # False in cross-group / when only one range remains
    show_run: bool  # True only for the active card


class RangeCard(QFrame):
    """One fit-range card: swatch + title + status chip + bounds (+ actions when active)."""

    #: Card clicked/activated (press anywhere on the card body).
    selected = Signal(int)
    #: Primary "Run Fit" button pressed (active card only).
    run_requested = Signal(int)
    #: "Edit Model" button pressed.
    edit_model_requested = Signal(int)
    #: "Remove" button pressed.
    remove_requested = Signal(int)

    def __init__(self, idx: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._idx = idx
        self._view: RangeCardView | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._surface = _CardSurface(self._emit_selected, self)
        outer.addWidget(self._surface)

        surface_layout = QVBoxLayout(self._surface)
        surface_layout.setContentsMargins(8, 6, 8, 6)
        surface_layout.setSpacing(4)

        # ── Line 1: swatch + title + status chip + bounds ──────────────────
        line1 = QHBoxLayout()
        line1.setContentsMargins(0, 0, 0, 0)
        line1.setSpacing(6)

        self._swatch = QLabel(self._surface)
        self._swatch.setFixedSize(_SWATCH_SIZE, _SWATCH_SIZE)
        line1.addWidget(self._swatch)

        self._title_label = QLabel("", self._surface)
        self._title_label.setStyleSheet(f"color: {tokens.TEXT}; font-weight: 600;")
        line1.addWidget(self._title_label)

        self._chip_label = QLabel("", self._surface)
        self._chip_label.setTextFormat(Qt.TextFormat.RichText)
        line1.addWidget(self._chip_label)

        line1.addStretch(1)

        self._bounds_label = QLabel("", self._surface)
        self._bounds_label.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
        line1.addWidget(self._bounds_label)

        surface_layout.addLayout(line1)

        # ── Line 2: formula + action buttons (active card only) ────────────
        self._line2 = QWidget(self._surface)
        line2_layout = QHBoxLayout(self._line2)
        line2_layout.setContentsMargins(0, 0, 0, 0)
        line2_layout.setSpacing(6)

        self._formula_label = _ElidingLabel(self._line2)
        line2_layout.addWidget(self._formula_label, 1)

        # Order: primary -> secondary -> destructive.
        self._run_button = QPushButton("Run Fit", self._line2)
        self._run_button.setStyleSheet(build_primary_button_qss())
        self._run_button.clicked.connect(self._on_run_clicked)
        line2_layout.addWidget(self._run_button)

        self._edit_model_button = QPushButton("Edit Model", self._line2)
        self._edit_model_button.clicked.connect(self._on_edit_model_triggered)
        line2_layout.addWidget(self._edit_model_button)

        self._remove_button = QPushButton("Remove", self._line2)
        self._remove_button.clicked.connect(self._on_remove_triggered)
        line2_layout.addWidget(self._remove_button)

        surface_layout.addWidget(self._line2)
        self._line2.setVisible(False)

        self.set_active(False)

    # ── Public API ───────────────────────────────────────────────────────────

    def set_state(self, view: RangeCardView) -> None:
        """Fully repaint the card from a plain :class:`RangeCardView`."""
        self._view = view
        self._idx = view.idx

        self._swatch.setStyleSheet(
            f"background-color: {view.swatch_color}; border: 1px solid {tokens.BORDER};"
            " border-radius: 2px;"
        )
        self._title_label.setText(view.title)

        self._chip_label.setText(view.status_chip_html)
        self._chip_label.setToolTip(view.status_tooltip)

        self._bounds_label.setText(view.bounds_text)

        self.setToolTip(view.formula)
        self._formula_label.set_full_text(view.formula)

        self._remove_button.setVisible(view.can_remove)

        self._line2.setVisible(view.show_run)
        self._run_button.setFixedWidth(widest_button_width(self._run_button, "Run Fit", "Fitting…"))

    def set_active(self, active: bool) -> None:
        """Apply the active/unselected highlight to the card surface.

        Mirrors the BENCH group-state treatment (ACCENT_SOFT fill + 2px ACCENT
        border when active) but as a ``QFrame#rangeCardSurface`` rule so it works
        on the frame surface. Purely visual — the dialog sets ``show_run`` on the
        view to appear/disappear line 2; this never touches line-2 visibility.
        """
        if active:
            self._surface.setStyleSheet(
                f"QFrame#{_SURFACE_OBJECT_NAME} {{ border: 2px solid {tokens.ACCENT};"
                f" background: {tokens.ACCENT_SOFT}; border-radius: 4px; }}"
            )
        else:
            self._surface.setStyleSheet(
                f"QFrame#{_SURFACE_OBJECT_NAME} {{ border: 1px solid {tokens.BORDER};"
                f" background: {tokens.SURFACE}; border-radius: 4px; }}"
            )

    def set_enabled(self, enabled: bool) -> None:
        """Enable/disable the card's action buttons (Run Fit / Edit Model / Remove).

        Used by the dialog's fit-busy bookkeeping to lock the active card's
        actions while a fit runs. Only the interactive controls are toggled —
        the surface stays clickable so the user can still see/select the card;
        the disabled action buttons are what actually prevent a second run.
        """
        self._run_button.setEnabled(enabled)
        self._edit_model_button.setEnabled(enabled)
        self._remove_button.setEnabled(enabled)

    # ── Mouse handling: click-to-select on the card body ────────────────────

    def _emit_selected(self) -> None:
        self.selected.emit(self._idx)

    # ── Signal relays ────────────────────────────────────────────────────────

    def _on_run_clicked(self) -> None:
        self.run_requested.emit(self._idx)

    def _on_edit_model_triggered(self) -> None:
        self.edit_model_requested.emit(self._idx)

    def _on_remove_triggered(self) -> None:
        self.remove_requested.emit(self._idx)
