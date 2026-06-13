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
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QToolButton, QWidget

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

        self._chip_row = QHBoxLayout()
        self._chip_row.setSpacing(6)
        layout.addLayout(self._chip_row)

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

    def set_projections(self, projections: list[dict]) -> None:
        """Rebuild chips from ``projections`` (each ``{"label", "tint"?}``).

        Selection for labels that persist across the rebuild is preserved;
        otherwise every projection starts selected (the old "All" default).
        The bar is shown only when at least two projections exist. Does not
        emit :attr:`selection_changed` — the caller drives the initial render.
        """
        prior = self.selected_labels()
        self._projections = [dict(p) for p in projections if p.get("label")]
        self._rebuild_chips()

        keep = [lbl for lbl in prior if lbl in self._chips]
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
            f"QPushButton:checked {{ background: {tint}; color: #ffffff; }}"
        )
