"""Compact per-range card for the trend-fit "range cards" redesign.

Replaces the wide row-of-buttons layout in the trend model-fit dialog with one
small card per fit range: a colour swatch matching the range's plot span, a
title, a fit-status chip, and its bounds. The card that is currently active
additionally shows a primary "Run Fit" button and a "..." overflow menu
(Edit Model / Exclude region.../Remove).

The card is a pure view — it renders a plain :class:`RangeCardView` handed to
it by the dialog and emits signals on user interaction. It owns no fit/range
state itself and imports nothing from ``asymmetry.core`` or the dialog
modules, so it can be constructed and tested in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFontMetrics, QMouseEvent
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.widgets import (
    build_primary_button_qss,
    style_group_state_button,
    widest_button_width,
)

#: Fixed size (px) of the colour-swatch square on line 1.
_SWATCH_SIZE = 12


class _CardSurface(QPushButton):
    """Flat, non-focusable QPushButton used purely as the card's styled background.

    ``style_group_state_button`` (gui/styles/widgets.py) paints via a
    ``QPushButton { ... }`` QSS selector, so the widget it is applied to must
    actually be a QPushButton for the rule to take effect — a QFrame's
    stylesheet would silently no-op against that selector. This surface is
    never meaningfully "clicked" as a button: Qt still delivers a mouse press
    on empty card background to this widget (real child buttons — Run Fit,
    the overflow button — sit on top and consume their own presses first), so
    its ``mousePressEvent`` is repurposed to emit the card's ``selected``
    signal instead of a button click.
    """

    def __init__(self, on_pressed, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFlat(True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._on_pressed = on_pressed

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 — Qt override
        if event.button() == Qt.MouseButton.LeftButton:
            self._on_pressed()
        super().mousePressEvent(event)


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
    #: Overflow menu -> "Edit Model".
    edit_model_requested = Signal(int)
    #: Overflow menu -> "Exclude region..." (numeric exclude/add-window path).
    exclude_requested = Signal(int)
    #: Overflow menu -> "Remove".
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
        line1.addWidget(self._title_label)

        self._chip_label = QLabel("", self._surface)
        self._chip_label.setTextFormat(Qt.TextFormat.RichText)
        line1.addWidget(self._chip_label)

        line1.addStretch(1)

        self._bounds_label = QLabel("", self._surface)
        self._bounds_label.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
        line1.addWidget(self._bounds_label)

        surface_layout.addLayout(line1)

        # ── Line 2: formula + Run Fit + overflow (active card only) ────────
        self._line2 = QWidget(self._surface)
        line2_layout = QHBoxLayout(self._line2)
        line2_layout.setContentsMargins(0, 0, 0, 0)
        line2_layout.setSpacing(6)

        self._formula_label = QLabel("", self._line2)
        self._formula_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        line2_layout.addWidget(self._formula_label, 1)

        self._run_button = QPushButton("Run Fit", self._line2)
        self._run_button.setStyleSheet(build_primary_button_qss())
        self._run_button.clicked.connect(self._on_run_clicked)
        line2_layout.addWidget(self._run_button)

        self._overflow_button = QPushButton("⋯", self._line2)  # midline horizontal ellipsis
        self._overflow_button.setFixedWidth(28)
        self._overflow_menu = QMenu(self._overflow_button)
        self._act_edit_model = self._overflow_menu.addAction("Edit Model")
        self._act_edit_model.triggered.connect(self._on_edit_model_triggered)
        self._act_exclude = self._overflow_menu.addAction("Exclude region…")
        self._act_exclude.triggered.connect(self._on_exclude_triggered)
        self._act_remove = self._overflow_menu.addAction("Remove")
        self._act_remove.triggered.connect(self._on_remove_triggered)
        self._overflow_button.setMenu(self._overflow_menu)
        line2_layout.addWidget(self._overflow_button)

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
        self._formula_label.setToolTip(view.formula)
        self._update_formula_elision()

        self._act_remove.setVisible(view.can_remove)

        self._line2.setVisible(view.show_run)
        self._run_button.setFixedWidth(widest_button_width(self._run_button, "Run Fit", "Fitting…"))

    def set_active(self, active: bool) -> None:
        """Apply the shared active/unselected treatment to the card surface.

        Purely visual — the dialog is responsible for setting ``show_run`` on
        the view so line 2 appears/disappears; this method never touches
        line-2 visibility itself.
        """
        state: Literal["active", "unselected"] = "active" if active else "unselected"
        style_group_state_button(self._surface, state)

    def set_enabled(self, enabled: bool) -> None:
        """Enable/disable the card's action controls (Run Fit + overflow).

        Used by the dialog's fit-busy bookkeeping to lock the active card's
        actions while a fit runs. Only the interactive controls are toggled —
        the surface stays clickable so the user can still see/select the card;
        the disabled Run Fit + overflow are what actually prevent a second run.
        """
        self._run_button.setEnabled(enabled)
        self._overflow_button.setEnabled(enabled)

    # ── Formula elision ──────────────────────────────────────────────────────

    def _update_formula_elision(self) -> None:
        if self._view is None:
            return
        metrics = QFontMetrics(self._formula_label.font())
        available = max(0, self._formula_label.width())
        if available <= 0:
            available = max(0, self.width() - 160)
        elided = metrics.elidedText(self._view.formula, Qt.TextElideMode.ElideRight, available)
        self._formula_label.setText(elided)

    def resizeEvent(self, event) -> None:  # noqa: N802 — Qt override
        super().resizeEvent(event)
        self._update_formula_elision()

    # ── Mouse handling: click-to-select on the card body ────────────────────

    def _emit_selected(self) -> None:
        self.selected.emit(self._idx)

    # ── Signal relays ────────────────────────────────────────────────────────

    def _on_run_clicked(self) -> None:
        self.run_requested.emit(self._idx)

    def _on_edit_model_triggered(self) -> None:
        self.edit_model_requested.emit(self._idx)

    def _on_exclude_triggered(self) -> None:
        self.exclude_requested.emit(self._idx)

    def _on_remove_triggered(self) -> None:
        self.remove_requested.emit(self._idx)
