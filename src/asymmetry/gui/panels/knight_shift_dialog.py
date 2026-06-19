"""Dialog for configuring the Knight-shift conversion of fitted frequencies."""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.fitting.knight_shift import (
    REFERENCE_APPLIED_FIELD,
    REFERENCE_COMPONENT,
    KnightShiftConfig,
    KnightShiftUnit,
)
from asymmetry.core.fitting.parameters import get_param_info

_UNIT_CHOICES = [
    ("Auto (ppm / %)", KnightShiftUnit.AUTO),
    ("ppm", KnightShiftUnit.PPM),
    ("Percent (%)", KnightShiftUnit.PERCENT),
    ("Fraction", KnightShiftUnit.FRACTION),
]


def _component_label(name: str) -> str:
    """Readable label for a frequency component, e.g. 'frequency_2  (f₂)'."""
    return f"{name}  ({get_param_info(name).unicode_label(include_unit=False)})"


class KnightShiftDialog(QDialog):
    """Choose the reference, components and unit for the Knight-shift conversion."""

    def __init__(
        self,
        *,
        available_components: Sequence[str],
        config: KnightShiftConfig,
        crossing_count: int = 0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Knight Shift")
        self.setMinimumWidth(420)
        self._components = [str(c) for c in available_components]
        self._result: KnightShiftConfig | None = None

        root = QVBoxLayout(self)

        intro = QLabel(
            "Convert fitted precession frequencies to the Knight shift K = (ν − ν_ref) / ν_ref."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        # Reference mode --------------------------------------------------
        ref_box = QGroupBox("Reference (ν_ref)")
        ref_layout = QVBoxLayout(ref_box)
        self._applied_radio = QRadioButton("Applied field  (γ_µ · B, no reference line needed)")
        self._component_radio = QRadioButton("Designated component:")
        self._ref_group = QButtonGroup(self)
        self._ref_group.addButton(self._applied_radio)
        self._ref_group.addButton(self._component_radio)
        ref_layout.addWidget(self._applied_radio)

        component_row = QFormLayout()
        self._reference_combo = QComboBox()
        for name in self._components:
            self._reference_combo.addItem(_component_label(name), userData=name)
        component_row.addRow(self._component_radio, self._reference_combo)
        ref_layout.addLayout(component_row)
        root.addWidget(ref_box)

        # Components to convert -------------------------------------------
        comp_box = QGroupBox("Components to convert (none ticked → all)")
        comp_layout = QVBoxLayout(comp_box)
        self._component_list = QListWidget()
        self._component_list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        for name in self._components:
            item = QListWidgetItem(_component_label(name))
            item.setData(Qt.ItemDataRole.UserRole, name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self._component_list.addItem(item)
        comp_layout.addWidget(self._component_list)
        root.addWidget(comp_box)

        # Unit ------------------------------------------------------------
        unit_form = QFormLayout()
        self._unit_combo = QComboBox()
        for label, unit in _UNIT_CHOICES:
            self._unit_combo.addItem(label, userData=unit)
        unit_form.addRow("Display unit:", self._unit_combo)
        root.addLayout(unit_form)

        # Crossing summary + status --------------------------------------
        if crossing_count:
            note = QLabel(
                f"⚠ {crossing_count} component crossing(s) detected along the scan — "
                "K traces follow the raw component labels and may swap identity there."
            )
            note.setWordWrap(True)
            root.addWidget(note)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        root.addWidget(self._status)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        # The OK button enables the conversion; offer an explicit "Disable" path too.
        self._disable_button = self._buttons.addButton(
            "Turn off", QDialogButtonBox.ButtonRole.DestructiveRole
        )
        self._disable_button.clicked.connect(self._on_disable)
        self._buttons.accepted.connect(self._on_accept)
        self._buttons.rejected.connect(self.reject)
        root.addWidget(self._buttons)

        self._applied_radio.toggled.connect(self._sync_enabled)
        self._component_radio.toggled.connect(self._sync_enabled)

        self._apply_initial(config)
        self._sync_enabled()

    # -- state ----------------------------------------------------------
    def _apply_initial(self, config: KnightShiftConfig) -> None:
        if config.reference_mode == REFERENCE_COMPONENT and self._components:
            self._component_radio.setChecked(True)
            idx = self._reference_combo.findData(config.reference_component)
            if idx >= 0:
                self._reference_combo.setCurrentIndex(idx)
        else:
            self._applied_radio.setChecked(True)
        unit_idx = self._unit_combo.findData(config.unit)
        self._unit_combo.setCurrentIndex(unit_idx if unit_idx >= 0 else 0)
        selected = set(config.components)
        for i in range(self._component_list.count()):
            item = self._component_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) in selected:
                item.setCheckState(Qt.CheckState.Checked)

    def _sync_enabled(self) -> None:
        component_mode = self._component_radio.isChecked()
        self._reference_combo.setEnabled(component_mode)
        ok = bool(self._components) and (not component_mode or self._reference_combo.count() > 0)
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(ok)
        if not self._components:
            self._status.setText("No oscillation-frequency components found in this series.")
        else:
            self._status.setText("")

    def _checked_components(self) -> tuple[str, ...]:
        names: list[str] = []
        for i in range(self._component_list.count()):
            item = self._component_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                names.append(str(item.data(Qt.ItemDataRole.UserRole)))
        return tuple(names)

    def _build_config(self, *, enabled: bool) -> KnightShiftConfig:
        component_mode = self._component_radio.isChecked()
        return KnightShiftConfig(
            enabled=enabled,
            reference_mode=REFERENCE_COMPONENT if component_mode else REFERENCE_APPLIED_FIELD,
            reference_component=(
                str(self._reference_combo.currentData()) if component_mode else None
            ),
            unit=self._unit_combo.currentData(),
            components=self._checked_components(),
        )

    def _on_accept(self) -> None:
        self._result = self._build_config(enabled=True)
        self.accept()

    def _on_disable(self) -> None:
        self._result = self._build_config(enabled=False)
        self.accept()

    def knight_shift_config(self) -> KnightShiftConfig | None:
        """Return the configuration when the dialog was accepted, else None."""
        return self._result
