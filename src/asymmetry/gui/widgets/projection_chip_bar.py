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
* The bar never sets the plot toolbar's minimum width. It shows full chip
  labels when they fit, short labels (``Top-Bottom`` → ``T–B``) when they do
  not, and below that **folds** into one menu button summarising the selection.
* When the projections include the transverse pair (P_x and P_y) a **Lab |
  Rotating** switch sits left of the chips (``Lab | Rot`` when short, a
  "Frame" section of the fold menu when folded). Rotating relabels the pair's
  chips P′_x, P′_y; the chips keep their lab identity (selection, memory, tint).

The chip tint is *projection identity* and is deliberately separate from a data
trace colour (which encodes run identity in RG mode). This widget owns no plot
state — it emits :attr:`selection_changed` and the plot panel renders the rest.
"""

from __future__ import annotations

from PySide6.QtCore import QSignalBlocker, QSize, Qt, Signal
from PySide6.QtGui import QActionGroup
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLayout,
    QMenu,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QWidget,
)

from asymmetry.core.transform.rotating_frame import ROTATED_LABELS
from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.widgets import build_segmented_cell_qss, build_segmented_container_qss

_DEFAULT_TINT = tokens.ACCENT

#: Presentations from widest to narrowest; the bar shows the first that fits.
_PRESENTATIONS = ("full", "short", "folded")

#: Longest projection name the folded button shows; the menu keeps full names.
_FOLD_LABEL_CHARS = 8

#: The frame switch's segment texts by presentation (folded: in the fold menu).
_FRAME_TEXTS = {"full": ("Lab", "Rotating"), "short": ("Lab", "Rot")}
#: The folded button's prefix while the rotating frame is shown.
_ROTATING_FOLD_PREFIX = "Rot · "


def short_projection_label(label: str) -> str:
    """Return the compact chip text: ``Top-Bottom`` → ``T–B``; ``P_x`` stays ``P_x``."""
    parts = label.split("-")
    return "–".join(part[:1] for part in parts) if len(parts) > 1 else label


def _chip_qss(tint: str) -> str:
    return (
        "QPushButton {"
        f" border: 1px solid {tint};"
        f" color: {tint};"
        " background: transparent;"
        " border-radius: 9px;"
        " padding: 2px 10px;"
        "}"
        f"QPushButton:checked {{ background: {tint}; color: {tokens.WHITE}; }}"
        "QPushButton::menu-indicator { image: none; }"
    )


def _fold_label(label: str) -> str:
    """Short label clipped to :data:`_FOLD_LABEL_CHARS`, so the folded width is bounded."""
    short = short_projection_label(label)
    return short if len(short) <= _FOLD_LABEL_CHARS else short[: _FOLD_LABEL_CHARS - 1] + "…"


class ProjectionChipBar(QWidget):
    """A row of toggle chips selecting which projections are shown as subplots."""

    #: Emitted with the ordered ``list[str]`` of currently selected labels.
    selection_changed = Signal(list)
    #: Emitted with ``True`` when the user switches to the rotating frame, ``False`` to Lab.
    frame_changed = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._projections: list[dict] = []
        self._chips: dict[str, QPushButton] = {}
        self._suppress = False
        self._presentation = "full"

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(6)
        # sizeHint/minimumSizeHint below own the bar's width range; the default
        # constraint would pin the minimum to the visible chips and undo the fold.
        layout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        # Leading stretch: spare width sits left of the chips, keeping them
        # beside the toolbar's Fits/Pan/Zoom group rather than drifting away.
        layout.addStretch()

        self._frame_switch = QFrame()
        self._frame_switch.setStyleSheet(build_segmented_container_qss())
        cells = QHBoxLayout(self._frame_switch)
        cells.setContentsMargins(0, 0, 0, 0)
        cells.setSpacing(0)
        self._frame_group = QButtonGroup(self)
        self._lab_btn, self._rotating_btn = (QPushButton(text) for text in _FRAME_TEXTS["full"])
        for index, cell in enumerate((self._lab_btn, self._rotating_btn)):
            cell.setCheckable(True)
            cell.setCursor(Qt.CursorShape.PointingHandCursor)
            cell.setStyleSheet(
                build_segmented_cell_qss(first=index == 0, last=index == 1, padding_h=6)
            )
            self._frame_group.addButton(cell)
            cells.addWidget(cell)
        self._lab_btn.setChecked(True)
        self._lab_btn.setToolTip("Lab frame")
        self._rotating_btn.setToolTip("Rotating frame")
        self._frame_group.buttonToggled.connect(self._on_frame_toggled)
        self._frame_switch.hide()
        layout.addWidget(self._frame_switch)

        self._all_btn = QToolButton()
        self._all_btn.setText("all")
        self._all_btn.setAutoRaise(True)
        self._all_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._all_btn.setToolTip("Show all projections")
        self._all_btn.clicked.connect(self._on_all_clicked)
        layout.addWidget(self._all_btn)

        self._fold_menu = QMenu(self)
        self._fold_menu.aboutToShow.connect(self._rebuild_fold_menu)
        self._fold_btn = QPushButton()
        self._fold_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._fold_btn.setToolTip("Choose which projections are shown")
        self._fold_btn.setMenu(self._fold_menu)
        self._fold_btn.hide()
        layout.addWidget(self._fold_btn)

        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
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

    def selected_labels(self) -> list[str]:
        """Return selected labels in declaration order."""
        return [label for label, chip in self._chips.items() if chip.isChecked()]

    def set_selected(self, labels: list[str]) -> None:
        """Set the checked chips (floor of one), without emitting a change."""
        wanted = [label for label in self._ordered_labels() if label in set(labels)]
        if not wanted and self._chips:
            wanted = [next(iter(self._chips))]
        self._apply_selection(wanted, emit=False)

    def presentation(self) -> str:
        """Return ``"full"``, ``"short"`` or ``"folded"`` — what the bar shows now."""
        return self._presentation

    def frame_available(self) -> bool:
        """True when the projections include the transverse pair a frame rotates."""
        return set(ROTATED_LABELS) <= set(self._chips)

    def is_rotating(self) -> bool:
        """True when the rotating frame is chosen and the projections allow it."""
        return self.frame_available() and self._rotating_btn.isChecked()

    def rotating_chosen(self) -> bool:
        """The switch's own position, kept while browsing runs without the pair."""
        return self._rotating_btn.isChecked()

    def set_rotating(self, rotating: bool) -> None:
        """Move the switch without emitting :attr:`frame_changed` (a restore)."""
        with QSignalBlocker(self._frame_group):
            (self._rotating_btn if rotating else self._lab_btn).setChecked(True)
        self._refit()
        self._sync_summary()

    def display_label(self, label: str) -> str:
        """The name *label*'s chip shows: its rotated name while rotating."""
        return ROTATED_LABELS.get(label, label) if self.is_rotating() else label

    # ------------------------------------------------------------------
    # Width negotiation
    # ------------------------------------------------------------------

    def sizeHint(self) -> QSize:  # noqa: N802 — Qt override
        return QSize(self._presentation_width("full"), super().sizeHint().height())

    def minimumSizeHint(self) -> QSize:  # noqa: N802 — Qt override
        return QSize(self._presentation_width("folded"), super().minimumSizeHint().height())

    def resizeEvent(self, event) -> None:  # noqa: N802 — Qt override
        super().resizeEvent(event)
        self._present(self._fitting_presentation(event.size().width()))

    def _fitting_presentation(self, width: int) -> str:
        return next(
            p for p in _PRESENTATIONS if p == "folded" or self._presentation_width(p) <= width
        )

    def _presentation_width(self, presentation: str) -> int:
        layout = self.layout()
        margins = layout.contentsMargins().left() + layout.contentsMargins().right()
        if presentation == "folded":
            return margins + self._fold_width()
        texts = [self._chip_text(label, presentation) for label in self._chips]
        chips = sum(self._chip_width(text) for text in texts)
        spacing = layout.spacing() * len(texts)
        switch = (
            self._switch_width(presentation) + layout.spacing() if self.frame_available() else 0
        )
        return margins + switch + chips + spacing + self._all_btn.sizeHint().width()

    def _chip_text(self, label: str, presentation: str) -> str:
        shown = self.display_label(label)
        return shown if presentation == "full" else short_projection_label(shown)

    def _switch_width(self, presentation: str) -> int:
        """Width of the Lab | Rotating switch showing *presentation*'s texts."""
        widths = []
        for cell, text in zip(
            (self._lab_btn, self._rotating_btn), _FRAME_TEXTS[presentation], strict=True
        ):
            cell.ensurePolished()
            metrics = cell.fontMetrics()
            flags = Qt.TextFlag.TextShowMnemonic
            chrome = cell.sizeHint().width() - metrics.size(flags, cell.text()).width()
            widths.append(chrome + metrics.size(flags, text).width())
        return sum(widths)

    def _chip_width(self, text: str) -> int:
        """Width of a chip showing *text*: its QSS chrome plus the text width.

        Text is measured with ``QFontMetrics.size`` — what ``QPushButton.sizeHint``
        uses — so the derived chrome is the same whichever chip text it came from.
        """
        chip = next(iter(self._chips.values()))
        chip.ensurePolished()
        metrics = chip.fontMetrics()
        flags = Qt.TextFlag.TextShowMnemonic
        chrome = chip.sizeHint().width() - metrics.size(flags, chip.text()).width()
        return chrome + metrics.size(flags, text).width()

    def _fold_width(self) -> int:
        """Width of the fold button holding its widest possible summary."""
        if not self._chips:
            return 0
        count = len(self._chips)
        candidates = [_fold_label(label) for label in self._chips]
        candidates += [f"{count} of {count}", f"All {count}"]
        if self.frame_available():
            candidates += [_ROTATING_FOLD_PREFIX + _fold_label(ROTATED_LABELS["P_x"])]
            candidates += [f"{_ROTATING_FOLD_PREFIX}{count} of {count}"]
        return max(self._chip_width(f"{text} ▾") for text in candidates)

    def _refit(self) -> None:
        """Re-measure after the chip texts or the switch changed, and present what fits."""
        self.updateGeometry()
        self._present(self._fitting_presentation(self.width()))

    def _present(self, presentation: str) -> None:
        self._presentation = presentation
        folded = presentation == "folded"
        for label, chip in self._chips.items():
            chip.setText(self._chip_text(label, presentation))
            chip.setToolTip(self.display_label(label))
            chip.setVisible(not folded)
        if not folded:
            self._lab_btn.setText(_FRAME_TEXTS[presentation][0])
            self._rotating_btn.setText(_FRAME_TEXTS[presentation][1])
        self._frame_switch.setVisible(self.frame_available() and not folded)
        self._all_btn.setVisible(not folded)
        self._fold_btn.setVisible(folded)
        self._fold_btn.setFixedWidth(self._fold_width())

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ordered_labels(self) -> list[str]:
        return [str(p["label"]) for p in self._projections]

    def _rebuild_chips(self) -> None:
        layout = self.layout()
        for chip in self._chips.values():
            layout.removeWidget(chip)
            chip.deleteLater()
        self._chips = {}
        for index, proj in enumerate(self._projections):
            label = str(proj["label"])
            chip = QPushButton(label)
            chip.setCheckable(True)
            chip.setCursor(Qt.CursorShape.PointingHandCursor)
            chip.setToolTip(label)
            chip.setStyleSheet(_chip_qss(str(proj.get("tint") or _DEFAULT_TINT)))
            chip.toggled.connect(self._on_chip_toggled)
            # After the stretch and the frame switch.
            layout.insertWidget(index + 2, chip)
            self._chips[label] = chip
        self.updateGeometry()
        if self._chips:
            self._present(self._fitting_presentation(self.width()))

    def _apply_selection(self, labels: list[str], *, emit: bool) -> None:
        wanted = set(labels)
        self._suppress = True
        try:
            for label, chip in self._chips.items():
                chip.setChecked(label in wanted)
        finally:
            self._suppress = False
        self._sync_summary()
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
        self._sync_summary()
        self.selection_changed.emit(self.selected_labels())

    def _on_frame_toggled(self, button: QPushButton, checked: bool) -> None:
        if not checked:
            return
        self._refit()
        self._sync_summary()
        self.frame_changed.emit(button is self._rotating_btn)

    def _on_all_clicked(self) -> None:
        if len(self.selected_labels()) == len(self._chips):
            return
        self._apply_selection(self._ordered_labels(), emit=True)

    def _sync_summary(self) -> None:
        """Mirror the selection into the "all" action and the fold button."""
        selected = self.selected_labels()
        count = len(self._chips)
        self._all_btn.setEnabled(bool(self._chips) and len(selected) < count)
        if len(selected) == 1:
            text = _fold_label(self.display_label(selected[0]))
            tint = next(
                str(p.get("tint") or _DEFAULT_TINT)
                for p in self._projections
                if p["label"] == selected[0]
            )
            qss = _chip_qss(tint) + f"QPushButton {{ background: {tint}; color: {tokens.WHITE}; }}"
        else:
            text = f"All {count}" if len(selected) == count else f"{len(selected)} of {count}"
            qss = _chip_qss(tokens.BORDER_STRONG) + f"QPushButton {{ color: {tokens.TEXT}; }}"
        if self.is_rotating():
            text = _ROTATING_FOLD_PREFIX + text
        self._fold_btn.setText(f"{text} ▾")
        self._fold_btn.setStyleSheet(qss)

    def _rebuild_fold_menu(self) -> None:
        """Fill the fold menu on ``aboutToShow``: the frame, then one entry per chip."""
        menu = self._fold_menu
        menu.clear()
        if self.frame_available():
            menu.addAction("Frame").setEnabled(False)
            frames = QActionGroup(menu)
            for text, cell in (("Lab", self._lab_btn), ("Rotating", self._rotating_btn)):
                action = menu.addAction(text)
                action.setCheckable(True)
                action.setChecked(cell.isChecked())
                frames.addAction(action)
                # The switch stays the one source of truth, as the chips do below.
                action.triggered.connect(cell.click)
            menu.addSeparator()
        header = menu.addAction("Projections shown")
        header.setEnabled(False)
        for label, chip in self._chips.items():
            action = menu.addAction(self.display_label(label))
            action.setCheckable(True)
            action.setChecked(chip.isChecked())
            # The chip stays the one source of truth, so the floor-of-one veto
            # and the emit follow the same path as a click on the chip itself.
            action.triggered.connect(chip.toggle)
        menu.addSeparator()
        show_all = menu.addAction("Show all")
        show_all.setEnabled(self._all_btn.isEnabled())
        show_all.triggered.connect(self._on_all_clicked)
