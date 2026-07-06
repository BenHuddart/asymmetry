"""Multi-select chip bar for choosing which asymmetry projections to show.

An EMU vector-polarization grouping (and, in time, transverse-field dual
grouping) exposes several asymmetry *projections* of the same run. This bar
gives one toggle chip per declared projection, tinted with the projection's
fixed identity colour, so the user can show any subset as stacked subplots.

Design (see ``docs/porting/unified-asymmetry-projections/``):

* Multi-select with a **floor of one** — the last selected chip will not
  release, because an empty selection would mean zero subplots.
* A lightweight **"all" action** (a verb, not a toggle) selects every
  projection in one click and greys out when all are already selected.
* The whole bar **hides when fewer than two projections** exist; an ordinary
  single-pair grouping needs no selector.

The chip tint is *projection identity* and is deliberately separate from a data
trace colour (which encodes run identity in RG mode). This widget owns no plot
state — it emits :attr:`selection_changed` and the plot panel renders the rest.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QWidget,
)

from asymmetry.gui.styles import tokens

_DEFAULT_TINT = tokens.ACCENT


class ProjectionChipBar(QWidget):
    """A row of toggle chips selecting which projections are shown as subplots."""

    #: Emitted with the ordered ``list[str]`` of currently selected labels.
    selection_changed = Signal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._projections: list[dict] = []
        self._chips: dict[str, QPushButton] = {}
        self._suppress = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(6)

        self._label = QLabel("Projection:")
        layout.addWidget(self._label)

        # The chips live inside a horizontally-scrollable strip so a wide
        # projection set (vector ALL mode with many groups) scrolls in place
        # rather than pushing the plot's Pan/Zoom controls off-screen at narrow
        # widths (the 1280px 13-inch case with the left dock open). The strip
        # shrinks under layout pressure (small minimum) and only shows a
        # scrollbar when the chips exceed the available width.
        self._chip_host = QWidget()
        self._chip_row = QHBoxLayout(self._chip_host)
        self._chip_row.setContentsMargins(0, 0, 0, 0)
        self._chip_row.setSpacing(6)

        self._chip_scroll = QScrollArea()
        self._chip_scroll.setWidget(self._chip_host)
        self._chip_scroll.setWidgetResizable(True)
        self._chip_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._chip_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._chip_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # Preferred-but-shrinkable: claim the chips' natural width when there is
        # room, yield (and scroll) when there is not, so neighbouring controls
        # keep priority.
        self._chip_scroll.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        # Keep the strip transparent so the toolbar surface shows through. Set
        # the scroll area, its viewport, and the chip host directly rather than
        # via a depth-specific descendant selector (the viewport is an anonymous
        # widget, so a `QScrollArea > QWidget > QWidget` rule is fragile).
        self._chip_scroll.setStyleSheet("background: transparent; border: none;")
        self._chip_scroll.viewport().setStyleSheet("background: transparent;")
        self._chip_host.setStyleSheet("background: transparent;")
        layout.addWidget(self._chip_scroll)

        self._all_btn = QToolButton()
        self._all_btn.setText("all")
        self._all_btn.setAutoRaise(True)
        self._all_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._all_btn.setToolTip("Show all projections")
        self._all_btn.clicked.connect(self._on_all_clicked)
        layout.addWidget(self._all_btn)

        layout.addStretch()
        self.hide()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_projections(self, projections: list[dict], selected: list[str] | None = None) -> None:
        """Set chips from ``projections`` (each ``{"label", "tint"?}``).

        ``selected`` chooses the checked chips; when omitted, a selection for
        labels that persist across the update is preserved, else every
        projection starts selected (the old "All" default). The chip widgets are
        only torn down and rebuilt when the label/tint set actually changes — a
        no-op update (the common case on every plot refresh) keeps the existing
        chips, so a click does not destroy the chip the user just pressed. The
        bar is shown only when at least two projections exist. Does not emit
        :attr:`selection_changed` — the caller drives the initial render.
        """
        prior = self.selected_labels()
        new_specs = [dict(p) for p in projections if p.get("label")]
        new_sig = [(str(p["label"]), str(p.get("tint") or "")) for p in new_specs]
        current_sig = [(str(p["label"]), str(p.get("tint") or "")) for p in self._projections]
        self._projections = new_specs
        if new_sig != current_sig:
            self._rebuild_chips()

        source = selected if selected is not None else prior
        keep = [lbl for lbl in source if lbl in self._chips]
        if not keep:
            keep = list(self._chips)
        self._apply_selection(keep, emit=False)
        self.setVisible(len(self._chips) >= 2)
        self._update_all_button()

    def selected_labels(self) -> list[str]:
        """Return selected labels in declaration order."""
        return [label for label, chip in self._chips.items() if chip.isChecked()]

    def set_selected(self, labels: list[str]) -> None:
        """Set the checked chips (floor of one), without emitting a change."""
        wanted = [label for label in self._ordered_labels() if label in set(labels)]
        if not wanted and self._chips:
            wanted = [next(iter(self._chips))]
        self._apply_selection(wanted, emit=False)
        self._update_all_button()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ordered_labels(self) -> list[str]:
        return [str(p["label"]) for p in self._projections]

    def _rebuild_chips(self) -> None:
        for chip in self._chips.values():
            self._chip_row.removeWidget(chip)
            chip.deleteLater()
        self._chips = {}
        for proj in self._projections:
            label = str(proj["label"])
            tint = str(proj.get("tint") or _DEFAULT_TINT)
            chip = QPushButton(label)
            chip.setCheckable(True)
            chip.setCursor(Qt.CursorShape.PointingHandCursor)
            chip.setStyleSheet(self._chip_qss(tint))
            chip.toggled.connect(self._on_chip_toggled)
            self._chip_row.addWidget(chip)
            self._chips[label] = chip

    def _apply_selection(self, labels: list[str], *, emit: bool) -> None:
        wanted = set(labels)
        self._suppress = True
        try:
            for label, chip in self._chips.items():
                chip.setChecked(label in wanted)
        finally:
            self._suppress = False
        if emit:
            self.selection_changed.emit(self.selected_labels())

    def _on_chip_toggled(self, checked: bool) -> None:
        if self._suppress:
            return
        # Floor of one: a toggle that empties the selection is vetoed by
        # re-checking the chip the user just released.
        if not checked and not self.selected_labels():
            sender = self.sender()
            if isinstance(sender, QPushButton):
                self._suppress = True
                sender.setChecked(True)
                self._suppress = False
            return
        self._update_all_button()
        self.selection_changed.emit(self.selected_labels())

    def _on_all_clicked(self) -> None:
        if len(self.selected_labels()) == len(self._chips):
            return
        self._apply_selection(self._ordered_labels(), emit=False)
        self._update_all_button()
        self.selection_changed.emit(self.selected_labels())

    def _update_all_button(self) -> None:
        all_selected = bool(self._chips) and len(self.selected_labels()) == len(self._chips)
        self._all_btn.setEnabled(not all_selected)

    def _chip_qss(self, tint: str) -> str:
        return (
            "QPushButton {"
            f" border: 1px solid {tint};"
            f" color: {tint};"
            " background: transparent;"
            " border-radius: 9px;"
            " padding: 2px 10px;"
            "}"
            f"QPushButton:checked {{ background: {tint}; color: {tokens.WHITE}; }}"
        )
