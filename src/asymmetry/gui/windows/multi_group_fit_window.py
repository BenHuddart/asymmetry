"""Dock-ready grouped time-domain fitting widget for one active dataset."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.gui.panels.fit_panel import GlobalFitTab


class MultiGroupFitWindow(QWidget):
    """Grouped time-domain fitting surface used inside the main fit dock."""

    grouped_fit_completed = Signal(object, object)
    grouped_preview_requested = Signal(object, object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._fit_tab = GlobalFitTab(self, allowed_modes=("grouped",))
        self._fit_tab.grouped_fit_completed.connect(self.grouped_fit_completed.emit)
        self._fit_tab.grouped_preview_requested.connect(self.grouped_preview_requested.emit)
        layout.addWidget(self._fit_tab)
        self._run_label = ""

    def set_dataset(self, dataset: MuonDataset | None) -> None:
        """Update the active grouped-fit dataset shown by the widget."""
        self._fit_tab.set_current_dataset(dataset)
        if dataset is None:
            self._run_label = ""
            return
        self._run_label = str(getattr(dataset, "run_label", dataset.run_number))

    def set_fit_blocked(self, blocked: bool, reason: str = "") -> None:
        """Apply fit blocking rules from the main window context."""
        self._fit_tab.set_fit_blocked(blocked, reason)

    def dock_title(self) -> str:
        """Return the preferred fit-dock title for the current grouped dataset."""
        if self._run_label:
            return f"Multi-Group Fit — {self._run_label}"
        return "Multi-Group Fit"

    def grouped_fit_formula_string(self) -> str | None:
        """Return the active grouped-fit formula string, if available."""
        model = getattr(self._fit_tab, "_composite_model", None)
        if model is None:
            return None
        try:
            return str(model.formula_string())
        except Exception:
            return None

    def get_state(self) -> dict:
        """Return serialisable grouped-fit state for project persistence."""
        return self._fit_tab.get_state()

    def restore_state(self, state: dict) -> None:
        """Restore grouped-fit state from project persistence."""
        if not isinstance(state, dict):
            return
        self._fit_tab.restore_state(state)
