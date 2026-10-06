"""The rotating-frame bar: the frame's setup and run fields above the plot.

Shown while the projection chips show the rotating frame
(docs/plans/rotating-frame-projection.md, D3–D6):

- **Setup** fields (ν_RF, B₁ axis, φ_RF, sense) are shared by the displayed
  runs; an edit is written to every one of them, and a field they disagree on
  reads "mixed" (D4).
- **Run** fields (b_x, b_y, a_y/a_x) belong to the run selected in the Data
  Browser.
- Each field wears its provenance: a default is grey italic, a typed value
  carries ✎, an estimate looks plain (D5).
- Until every displayed run has a frame only ν_RF can be entered; nothing is
  estimated for the user (D6).

The bar owns no frames: the host pushes them with :meth:`show_frames` and
writes what :attr:`field_edited` reports. Narrow panels wrap the bar to two
rows, and below that fold the run fields into a "Run <n> ▾" popover.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial

from PySide6.QtCore import QLocale, QSignalBlocker, QSize, Qt, Signal
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.fourier.units import FieldUnit, convert
from asymmetry.core.transform.rotating_frame import (
    SETUP_FIELDS,
    B1Axis,
    Provenance,
    RotatingFrame,
)
from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.metrics import field_width_for
from asymmetry.gui.styles.widgets import (
    build_primary_button_qss,
    build_segmented_button_qss,
    build_segmented_cell_qss,
    build_segmented_container_qss,
    make_section_header,
)

__all__ = [
    "FRAME_METADATA_KEY",
    "FREQUENCY_UNITS",
    "RotatingFrameBar",
    "frame_badge_text",
    "rotated_ylabel",
]

#: Metadata key under which a rotated projection's dataset carries its frame.
FRAME_METADATA_KEY = "rotating_frame"

#: Placeholder of a shared field the displayed runs disagree on (D4).
MIXED = "mixed"

#: The hint shown until every displayed run has a frame (D6).
FIRST_SWITCH_HINT = (
    "Enter ν_RF (the generator frequency) and the B₁ axis. Auto-detect… proposes "
    "the rest and changes nothing until you apply."
)

#: The values a field shows before the run has a frame: ``typed_frequency``'s defaults.
_DEFAULTS = RotatingFrame.typed_frequency(1.0)

#: The units ν_RF can be typed in, and their toggle labels.
FREQUENCY_UNITS = {FieldUnit.MHZ: "MHz", FieldUnit.GAUSS: "G"}


@dataclass(frozen=True)
class _NumberField:
    label: str
    chars: int
    bottom: float
    top: float
    fmt: str
    suffix: str
    tooltip: str


#: The typed number fields; ν_RF's suffix is the MHz | G toggle instead.
_NUMBER_FIELDS = {
    "frequency_mhz": _NumberField(
        "ν_RF", 8, 1e-9, 1e9, "{:.6g}", "", "Generator frequency ν_RF of the RF field"
    ),
    "rf_phase_deg": _NumberField(
        "φ_RF", 5, -1e5, 1e5, "{:.1f}", "°", "Phase of the RF drive at t0"
    ),
    "baseline_x": _NumberField(
        "b_x", 5, -1e3, 1e3, "{:.3g}", "%", "Baseline of P_x left after alpha (%)"
    ),
    "baseline_y": _NumberField(
        "b_y", 5, -1e3, 1e3, "{:.3g}", "%", "Baseline of P_y left after alpha (%)"
    ),
    "gain": _NumberField(
        "a_y/a_x", 5, 1e-9, 1e3, "{:.4g}", "", "Transverse gain: the amplitude of P_y over P_x"
    ),
}

_SENSE_TEXT = {1: "↺ +1", -1: "↻ −1"}

#: The bar's presentations from widest to narrowest; it shows the first that fits.
_PRESENTATIONS = ("wide", "wrapped", "folded")


def _signed(value: int) -> str:
    return "+1" if value > 0 else "−1"


#: The badge's parts: frame field, its symbol, and how a shared value reads.
_BADGE_PARTS = (
    ("frequency_mhz", "ν", lambda v: f"ν = {v:.6g} MHz"),
    ("b1_axis", "B₁", lambda v: f"B₁ ∥ {v}′"),
    ("rf_phase_deg", "φ_RF", lambda v: f"φ_RF = {v:.1f}°"),
    ("sense", "s", lambda v: f"s = {_signed(v)}"),
)


def frame_badge_text(frames: Sequence[RotatingFrame]) -> str:
    """The plot badge naming the frame the rotated curves were drawn in."""
    parts = ["Rotating frame"]
    for name, symbol, render in _BADGE_PARTS:
        values = {getattr(frame, name) for frame in frames}
        parts.append(render(values.pop()) if len(values) == 1 else f"{symbol} {MIXED}")
    return " · ".join(parts)


def rotated_ylabel(label: str, frames: Sequence[RotatingFrame]) -> str:
    """The y label of a rotated projection, naming its relation to B₁ when shared."""
    component = label.removeprefix("P′_")
    axes = {str(frame.b1_axis) for frame in frames}
    relation = ""
    if len(axes) == 1:
        relation = r"\parallel B_1" if axes == {component} else r"\perp B_1"
    return rf"$a_0 P'_{{{component}}}(t)\ {relation}$ (%)"


def _segmented(texts: Sequence[str]) -> tuple[QFrame, QButtonGroup, list[QPushButton]]:
    """A joined segmented control of checkable cells, the app's toolbar style."""
    frame = QFrame()
    frame.setStyleSheet(build_segmented_container_qss())
    row = QHBoxLayout(frame)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(0)
    group = QButtonGroup(frame)
    cells = []
    for index, text in enumerate(texts):
        cell = QPushButton(text)
        cell.setCheckable(True)
        cell.setCursor(Qt.CursorShape.PointingHandCursor)
        cell.setStyleSheet(
            build_segmented_cell_qss(first=index == 0, last=index == len(texts) - 1, padding_h=6)
        )
        group.addButton(cell, index)
        row.addWidget(cell)
        cells.append(cell)
    return frame, group, cells


def _muted(text: str) -> QLabel:
    label = QLabel(text)
    label.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
    return label


def _row(*widgets: QWidget, spacing: int = 3) -> QWidget:
    box = QWidget()
    layout = QHBoxLayout(box)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(spacing)
    for widget in widgets:
        layout.addWidget(widget)
    return box


def _shared_provenance(sources: set[Provenance]) -> Provenance:
    """A shared field is typed if any run typed it, a default only if all are."""
    if Provenance.TYPED in sources:
        return Provenance.TYPED
    return Provenance.DEFAULT if sources == {Provenance.DEFAULT} else Provenance.ESTIMATED


class RotatingFrameBar(QWidget):
    """Setup and run fields of the displayed runs' rotating frames."""

    #: ``(run numbers, field, value)``: the user set *field* on those runs.
    field_edited = Signal(list, str, object)
    auto_detect_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("rotatingFrameBar")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            f"QWidget#rotatingFrameBar {{ background-color: {tokens.ACCENT_SOFT};"
            f" border-bottom: 1px solid {tokens.BORDER}; }}"
        )
        self._frames: dict[int, RotatingFrame | None] = {}
        self._selected_run: int | None = None
        self._unit = FieldUnit.MHZ
        self._presentation = "wide"

        self._edits: dict[str, QLineEdit] = {}
        self._markers: dict[str, QLabel] = {}
        cells: dict[str, QWidget] = {}
        for name, spec in _NUMBER_FIELDS.items():
            edit = QLineEdit()
            validator = QDoubleValidator(spec.bottom, spec.top, 9, edit)
            validator.setLocale(QLocale.c())
            validator.setNotation(QDoubleValidator.Notation.StandardNotation)
            edit.setValidator(validator)
            edit.setMinimumWidth(field_width_for(spec.chars, edit))
            edit.setMaximumWidth(field_width_for(spec.chars + 2, edit))
            edit.setToolTip(spec.tooltip)
            edit.editingFinished.connect(partial(self._on_number_edited, name))
            self._edits[name] = edit
            parts = [QLabel(spec.label), edit] + ([_muted(spec.suffix)] if spec.suffix else [])
            cells[name] = _row(*parts, self._marker(name))

        self._unit_toggle, self._unit_group, unit_cells = _segmented(list(FREQUENCY_UNITS.values()))
        unit_cells[0].setChecked(True)
        self._unit_group.idClicked.connect(self._on_unit_clicked)

        self._b1_toggle, self._b1_group, self._b1_cells = _segmented(["x", "y"])
        self._b1_group.idClicked.connect(self._on_b1_clicked)
        b1_cell = _row(QLabel("B₁ ∥"), self._b1_toggle, self._marker("b1_axis"))

        self._sense_btn = QPushButton()
        self._sense_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._sense_btn.setToolTip("Rotation sense of P_x + iP_y")
        self._sense_btn.clicked.connect(self._on_sense_clicked)
        sense_cell = _row(self._sense_btn, self._marker("sense"))

        frequency = cells["frequency_mhz"]
        frequency.layout().insertWidget(2, self._unit_toggle)
        self._setup_box = _row(
            make_section_header("Setup"),
            frequency,
            b1_cell,
            cells["rf_phase_deg"],
            sense_cell,
            spacing=10,
        )
        self._run_header = make_section_header("Run")
        self._run_box = _row(cells["baseline_x"], cells["baseline_y"], cells["gain"], spacing=10)
        self._divider = QFrame()
        self._divider.setFrameShape(QFrame.Shape.VLine)
        self._divider.setStyleSheet(f"color: {tokens.BORDER_STRONG};")

        self._run_btn = QPushButton()
        self._run_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._run_btn.setToolTip("This run's baselines and gain")
        self._run_btn.setStyleSheet(build_segmented_button_qss(padding_h=8))
        self._run_btn.clicked.connect(self._open_run_popover)
        self._run_popover = QFrame(self, Qt.WindowType.Popup)
        self._run_popover.setObjectName("runFramePopover")
        self._run_popover.setStyleSheet(
            f"QFrame#runFramePopover {{ background: {tokens.SURFACE};"
            f" border: 1px solid {tokens.BORDER}; }}"
        )
        self._popover_layout = QVBoxLayout(self._run_popover)
        self._popover_layout.setContentsMargins(10, 8, 10, 8)

        self._detect_btn = QPushButton("Auto-detect…")
        self._detect_btn.setStyleSheet(build_primary_button_qss())
        self._detect_btn.setToolTip(
            "Estimate the sense, φ_RF, baselines and gain from the displayed runs over "
            "the visible window; nothing changes until you apply."
        )
        self._detect_btn.clicked.connect(self.auto_detect_requested)

        self._hint = _muted(FIRST_SWITCH_HINT)
        self._hint.setWordWrap(True)

        self._row1 = QHBoxLayout()
        self._row2 = QHBoxLayout()
        for row in (self._row1, self._row2):
            row.setSpacing(10)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 5, 10, 5)
        outer.setSpacing(4)
        # sizeHint/minimumSizeHint own the width range, as in ProjectionChipBar.
        outer.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        outer.addLayout(self._row1)
        outer.addLayout(self._row2)
        outer.addWidget(self._hint)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self._present("wide")

    def _marker(self, name: str) -> QLabel:
        marker = _muted("✎")
        marker.setToolTip("Typed by you — Auto-detect never proposes over it")
        marker.hide()
        self._markers[name] = marker
        return marker

    # ── data ───────────────────────────────────────────────────────────────

    def show_frames(
        self, frames: Mapping[int, RotatingFrame | None], selected_run: int | None
    ) -> None:
        """Show the displayed runs' frames (``None``: no frame yet) and the selected run's."""
        self._frames = dict(frames)
        self._selected_run = selected_run
        self._refresh()

    def runs(self) -> list[int]:
        """The displayed runs whose frames the bar shows."""
        return list(self._frames)

    def frequency_unit(self) -> FieldUnit:
        return self._unit

    def set_frequency_unit(self, unit: FieldUnit) -> None:
        self._unit = unit
        with QSignalBlocker(self._unit_group):
            self._unit_group.button(list(FREQUENCY_UNITS).index(unit)).setChecked(True)
        self._refresh()

    def presentation(self) -> str:
        """``"wide"``, ``"wrapped"`` or ``"folded"`` — what the bar shows now."""
        return self._presentation

    def _refresh(self) -> None:
        frames = [frame for frame in self._frames.values() if frame is not None]
        complete = bool(self._frames) and len(frames) == len(self._frames)
        selected = self._frames.get(self._selected_run)
        for name in SETUP_FIELDS:
            values = {getattr(frame, name) for frame in frames}
            sources = {frame.provenance[name] for frame in frames}
            if not complete:
                shown = None if name == "frequency_mhz" else getattr(_DEFAULTS, name)
                self._show(name, shown, Provenance.DEFAULT, enabled=name == "frequency_mhz")
            else:
                shown = next(iter(values)) if len(values) == 1 else None
                self._show(name, shown, _shared_provenance(sources), enabled=True)
        required = not complete
        self._edits["frequency_mhz"].setPlaceholderText("required" if required else MIXED)
        if required:
            self._edits["frequency_mhz"].setStyleSheet(
                f"QLineEdit {{ border: 1px solid {tokens.ACCENT}; border-radius: 3px; }}"
            )
        for name in ("baseline_x", "baseline_y", "gain"):
            frame = selected if selected is not None else _DEFAULTS
            source = Provenance.DEFAULT if selected is None else frame.provenance[name]
            self._show(name, getattr(frame, name), source, enabled=selected is not None)
        run_text = "—" if self._selected_run is None else str(self._selected_run)
        self._run_header.setText(f"RUN {run_text}")
        self._run_btn.setText(f"Run {run_text} ▾")
        shared = {(frame.frequency_mhz, frame.b1_axis) for frame in frames}
        self._detect_btn.setEnabled(complete and len(shared) == 1)
        self._hint.setVisible(required)
        self._refit()

    def _show(self, name: str, value: object, provenance: Provenance, *, enabled: bool) -> None:
        """Show one field's value (``None``: mixed or empty) in its provenance style."""
        default = provenance is Provenance.DEFAULT
        self._markers[name].setVisible(provenance is Provenance.TYPED)
        dim = f"color: {tokens.TEXT_DIM}; font-style: italic;" if default else ""
        if name in self._edits:
            edit = self._edits[name]
            if value is not None and name == "frequency_mhz":
                value = float(convert(value, FieldUnit.MHZ, self._unit))
            edit.setText("" if value is None else _NUMBER_FIELDS[name].fmt.format(value))
            edit.setPlaceholderText(MIXED)
            edit.setStyleSheet(f"QLineEdit {{ {dim} }}" if dim else "")
            edit.setEnabled(enabled)
        elif name == "b1_axis":
            self._b1_group.setExclusive(False)
            for cell, axis in zip(self._b1_cells, B1Axis, strict=True):
                cell.setChecked(value == axis)
                cell.setStyleSheet(
                    build_segmented_cell_qss(
                        first=axis is B1Axis.X, last=axis is B1Axis.Y, padding_h=6
                    )
                    + (f" QPushButton:checked {{ {dim} }}" if dim else "")
                )
            self._b1_group.setExclusive(True)
            self._b1_toggle.setEnabled(enabled)
        else:
            self._sense_btn.setText(_SENSE_TEXT[value] if value is not None else f"↻ {MIXED}")
            self._sense_btn.setStyleSheet(
                build_segmented_button_qss(padding_h=6)
                + (f" QPushButton {{ {dim} }}" if dim else "")
            )
            self._sense_btn.setEnabled(enabled)

    # ── edits ──────────────────────────────────────────────────────────────

    def _targets(self, name: str) -> list[int]:
        """Setup fields go to every displayed run; run fields to the selected one."""
        return list(self._frames) if name in SETUP_FIELDS else [self._selected_run]

    def _on_number_edited(self, name: str) -> None:
        edit = self._edits[name]
        if not edit.isModified():
            return
        edit.setModified(False)
        value = float(edit.text())
        if name == "frequency_mhz":
            value = float(convert(value, self._unit, FieldUnit.MHZ))
        self.field_edited.emit(self._targets(name), name, value)

    def _on_b1_clicked(self, index: int) -> None:
        self.field_edited.emit(self._targets("b1_axis"), "b1_axis", list(B1Axis)[index])

    def _on_sense_clicked(self) -> None:
        senses = {frame.sense for frame in self._frames.values() if frame is not None}
        sense = -senses.pop() if len(senses) == 1 else -1
        self.field_edited.emit(self._targets("sense"), "sense", sense)

    def _on_unit_clicked(self, index: int) -> None:
        self._unit = list(FREQUENCY_UNITS)[index]
        self._refresh()

    def _open_run_popover(self) -> None:
        self._run_popover.adjustSize()
        anchor = self._run_btn.mapToGlobal(self._run_btn.rect().bottomLeft())
        self._run_popover.move(anchor)
        self._run_popover.show()

    # ── width negotiation ─────────────────────────────────────────────────

    def _presentation_width(self, presentation: str) -> int:
        margins = self.layout().contentsMargins()
        spacing = self._row1.spacing()
        setup = self._setup_box.sizeHint().width()
        run = self._run_header.sizeHint().width() + spacing + self._run_box.sizeHint().width()
        detect = self._button_width(self._detect_btn, "Auto-detect…")
        if presentation == "wide":
            content = setup + run + detect + self._divider.sizeHint().width() + 4 * spacing
        elif presentation == "wrapped":
            content = max(setup, run + spacing + detect)
        else:
            run_btn = self._button_width(self._run_btn, self._run_btn.text())
            content = max(
                setup, run_btn + spacing + self._button_width(self._detect_btn, "Detect…")
            )
        return margins.left() + margins.right() + content

    @staticmethod
    def _button_width(button: QPushButton, text: str) -> int:
        button.ensurePolished()
        metrics = button.fontMetrics()
        chrome = button.sizeHint().width() - metrics.horizontalAdvance(button.text())
        return chrome + metrics.horizontalAdvance(text)

    def sizeHint(self) -> QSize:  # noqa: N802 — Qt override
        return QSize(self._presentation_width("wide"), super().sizeHint().height())

    def minimumSizeHint(self) -> QSize:  # noqa: N802 — Qt override
        return QSize(self._presentation_width("folded"), super().minimumSizeHint().height())

    def resizeEvent(self, event) -> None:  # noqa: N802 — Qt override
        super().resizeEvent(event)
        self._present(self._fitting_presentation(event.size().width()))

    def _fitting_presentation(self, width: int) -> str:
        return next(
            p for p in _PRESENTATIONS if p == "folded" or self._presentation_width(p) <= width
        )

    def _refit(self) -> None:
        self.updateGeometry()
        self._present(self._fitting_presentation(self.width()))

    def _present(self, presentation: str) -> None:
        """Lay the parts out on one row, two rows, or two rows with the run fields folded."""
        self._presentation = presentation
        folded = presentation == "folded"
        run_parts: list[QWidget | None] = (
            [self._run_btn] if folded else [self._run_header, self._run_box]
        )
        # ``None`` is the stretch that pushes Auto-detect… to the right edge.
        if presentation == "wide":
            rows = [[self._setup_box, self._divider, *run_parts, None, self._detect_btn], []]
        else:
            rows = [[self._setup_box, None], [*run_parts, None, self._detect_btn]]
        for row, parts in zip((self._row1, self._row2), rows, strict=True):
            while row.count():
                row.takeAt(0)
            for part in parts:
                if part is None:
                    row.addStretch()
                else:
                    row.addWidget(part)
        if folded:
            self._popover_layout.addWidget(self._run_box)
        self._divider.setVisible(presentation == "wide")
        self._run_header.setVisible(not folded)
        self._run_btn.setVisible(folded)
        self._run_box.setVisible(True)
        self._detect_btn.setText("Detect…" if folded else "Auto-detect…")
        self.updateGeometry()
