"""Period → red/green/ignore mapping dialog for multi-period runs.

WiMDA's ``PeriodMappingUnit`` as a matrix: one row per period showing its
label and good frames, with a three-way Red/Green/Ignore choice. Defaults
follow WiMDA — first period red, second green, the rest ignored. The dialog
only collects the mapping; building the combined dataset happens in the
main window through :func:`asymmetry.core.io.periods.combine_mapped_periods`.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QLabel,
    QMessageBox,
    QRadioButton,
    QVBoxLayout,
)

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.io.periods import normalise_period_mapping
from asymmetry.gui.styles import tokens

_TARGETS = ("red", "green", "ignore")
_TARGET_COLOURS = {"red": tokens.ACCENT_RED, "green": tokens.OK, "ignore": ""}


class PeriodMappingDialog(QDialog):
    """Collect a ``{period_number: "red"|"green"|"ignore"}`` mapping."""

    def __init__(self, period_datasets: list[MuonDataset], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Map Periods")
        self.mapping: dict[int, str] | None = None
        self._n_periods = len(period_datasets)
        self._choices: dict[int, dict[str, QRadioButton]] = {}

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "Sum arbitrary subsets of this run's periods into the red and "
                "green sets. Ignored periods are left out entirely."
            )
        )
        grid = QGridLayout()
        grid.addWidget(QLabel("Period"), 0, 0)
        grid.addWidget(QLabel("Good frames"), 0, 1)
        for column, target in enumerate(_TARGETS, start=2):
            grid.addWidget(QLabel(target.capitalize()), 0, column)

        existing = {}
        for ds in period_datasets:
            metadata = ds.metadata or {}
            mapping = metadata.get("period_mapping")
            if isinstance(mapping, dict):
                existing = {int(k): str(v) for k, v in mapping.items()}
                break

        for row, ds in enumerate(period_datasets, start=1):
            metadata = ds.metadata or {}
            period = int(metadata.get("period_number", row))
            grouping = ds.run.grouping if ds.run is not None else {}
            frames = grouping.get("good_frames") if isinstance(grouping, dict) else None
            grid.addWidget(QLabel(f"{period}"), row, 0)
            grid.addWidget(
                QLabel(f"{frames:g}" if isinstance(frames, (int, float)) else "–"), row, 1
            )
            group = QButtonGroup(self)
            buttons: dict[str, QRadioButton] = {}
            default = existing.get(period) or (
                "red" if period == 1 else "green" if period == 2 else "ignore"
            )
            for column, target in enumerate(_TARGETS, start=2):
                button = QRadioButton()
                colour = _TARGET_COLOURS[target]
                if colour:
                    button.setStyleSheet(f"color: {colour};")
                group.addButton(button)
                grid.addWidget(button, row, column)
                buttons[target] = button
            buttons.get(default, buttons["ignore"]).setChecked(True)
            self._choices[period] = buttons
        layout.addLayout(grid)

        buttons_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons_box.accepted.connect(self._on_accept)
        buttons_box.rejected.connect(self.reject)
        layout.addWidget(buttons_box)

    def current_mapping(self) -> dict[int, str]:
        return {
            period: next(target for target, btn in buttons.items() if btn.isChecked())
            for period, buttons in self._choices.items()
        }

    def _on_accept(self) -> None:
        mapping = self.current_mapping()
        try:
            self.mapping = normalise_period_mapping(mapping, self._n_periods)
        except ValueError as exc:
            QMessageBox.warning(self, "Map Periods", str(exc))
            return
        self.accept()
