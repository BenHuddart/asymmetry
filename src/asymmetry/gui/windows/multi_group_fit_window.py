"""Dock-ready grouped time-domain fitting widget.

Mirrors the F-B asymmetry fit panel's structure: a **Single** tab that fits the
active run's detector groups jointly (one dataset) and a **Batch** tab that fits
a series across the selected runs. Both surfaces are grouped
:class:`~asymmetry.gui.panels.fit_panel.GlobalFitTab` instances; they differ only
in their member set (Single → the active run; Batch → the selection).
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.gui.panels.fit_panel import GlobalFitTab

#: Fit-target choices (label, mode key) shown in the count-domain selector.
_FIT_TARGETS: tuple[tuple[str, str], ...] = (
    ("All groups", "all"),
    ("Forward + Backward (free α)", "fb"),
    ("Single group", "single"),
)
_FIT_COSTS: tuple[tuple[str, str], ...] = (("Poisson", "poisson"), ("Gaussian √N", "gaussian"))
_SINGLE_SIDES: tuple[tuple[str, str], ...] = (("Forward", "forward"), ("Backward", "backward"))


class MultiGroupFitWindow(QWidget):
    """Grouped time-domain fitting surface used inside the main fit dock."""

    grouped_fit_completed = Signal(object, object)
    grouped_preview_requested = Signal(object, object)
    fit_range_edit_committed = Signal(float, float)
    count_fit_completed = Signal(object, object)  # (dataset, count-fit result)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self._build_target_controls())

        self._tabs = QTabWidget()
        # Single = the active run's multi-group fit; Batch = the multi-run series.
        self._single_fit_tab = GlobalFitTab(self, member_kind="groups")
        self._batch_fit_tab = GlobalFitTab(self, member_kind="groups")
        for tab in (self._single_fit_tab, self._batch_fit_tab):
            tab.grouped_fit_completed.connect(self.grouped_fit_completed.emit)
            tab.grouped_preview_requested.connect(self.grouped_preview_requested.emit)
            tab.fit_range_edit_committed.connect(self.fit_range_edit_committed.emit)
            tab.count_fit_completed.connect(self.count_fit_completed.emit)
        self._tabs.addTab(self._single_fit_tab, "Single")
        self._tabs.addTab(self._batch_fit_tab, "Batch")
        layout.addWidget(self._tabs)
        self._run_label = ""
        self._sync_count_fit_target()

    def _build_target_controls(self) -> QWidget:
        """Build the count-domain fit-target / cost / side selector row."""
        box = QGroupBox("Fit target")
        form = QFormLayout(box)
        form.setContentsMargins(8, 4, 8, 4)

        self._target_combo = QComboBox()
        for label, _key in _FIT_TARGETS:
            self._target_combo.addItem(label)
        self._target_combo.currentIndexChanged.connect(self._sync_count_fit_target)

        self._cost_combo = QComboBox()
        for label, _key in _FIT_COSTS:
            self._cost_combo.addItem(label)
        self._cost_combo.currentIndexChanged.connect(self._sync_count_fit_target)

        self._side_combo = QComboBox()
        for label, _key in _SINGLE_SIDES:
            self._side_combo.addItem(label)
        self._side_combo.currentIndexChanged.connect(self._sync_count_fit_target)

        form.addRow(QLabel("Target"), self._target_combo)
        form.addRow(QLabel("Cost"), self._cost_combo)
        self._side_label = QLabel("Single group")
        form.addRow(self._side_label, self._side_combo)
        return box

    def _sync_count_fit_target(self, *_args) -> None:
        """Push the selector state down to both grouped surfaces."""
        mode = _FIT_TARGETS[self._target_combo.currentIndex()][1]
        cost = _FIT_COSTS[self._cost_combo.currentIndex()][1]
        side = _SINGLE_SIDES[self._side_combo.currentIndex()][1]
        single = mode == "single"
        self._side_combo.setEnabled(single)
        self._side_label.setEnabled(single)
        self._cost_combo.setEnabled(mode != "all")
        for tab in (self._single_fit_tab, self._batch_fit_tab):
            tab.set_count_fit_mode(mode)
            tab.set_count_fit_cost(cost)
            tab.set_count_single_side(side)

    def _grouped_tabs(self) -> tuple[GlobalFitTab, GlobalFitTab]:
        return (self._single_fit_tab, self._batch_fit_tab)

    def _active_tab(self) -> GlobalFitTab:
        current = self._tabs.currentWidget()
        return current if isinstance(current, GlobalFitTab) else self._single_fit_tab

    def set_dataset(self, dataset: MuonDataset | None) -> None:
        """Update the active grouped-fit dataset shown by both surfaces."""
        for tab in self._grouped_tabs():
            tab.set_current_dataset(dataset)
        if dataset is None:
            self._run_label = ""
            return
        self._run_label = str(getattr(dataset, "run_label", dataset.run_number))

    def set_member_datasets(self, datasets: list[MuonDataset]) -> None:
        """Set the member runs for the Batch grouped surface (the series)."""
        self._batch_fit_tab.set_member_datasets(datasets)

    def get_grouped_state(self) -> dict:
        """Return the grouped-fit classification from the active surface."""
        return self._active_tab().get_grouped_state()

    def grouped_simulate_seed_for_run(self, run_number: int) -> dict | None:
        """Return a cached multi-group simulate seed for a run, if available.

        Seeds the Generate Synthetic Run dialog's per-group amplitude/phase
        table from this run's last converged grouped time-domain fit.
        """
        for tab in self._grouped_tabs():
            seed = tab.grouped_simulate_seed_for_run(run_number)
            if seed is not None:
                return seed
        return None

    def set_fit_range_display(self, x_min: float | None, x_max: float | None) -> None:
        """Update fit-range spinboxes on both surfaces to match the plot range."""
        for tab in self._grouped_tabs():
            tab.set_fit_range_display(x_min, x_max)

    def set_fit_blocked(self, blocked: bool, reason: str = "") -> None:
        """Apply fit blocking rules from the main window context to both surfaces."""
        for tab in self._grouped_tabs():
            tab.set_fit_blocked(blocked, reason)

    def dock_title(self) -> str:
        """Return the preferred fit-dock title for the current grouped dataset."""
        if self._run_label:
            return f"Multi-Group Fit — {self._run_label}"
        return "Multi-Group Fit"

    def grouped_fit_formula_string(self) -> str | None:
        """Return the active grouped-fit formula string, if available."""
        model = getattr(self._active_tab(), "_composite_model", None)
        if model is None:
            return None
        try:
            return str(model.formula_string())
        except Exception:
            return None

    def get_state(self) -> dict:
        """Return serialisable grouped-fit state (both surfaces) for persistence."""
        return {
            "single": self._single_fit_tab.get_state(),
            "batch": self._batch_fit_tab.get_state(),
        }

    def restore_state(self, state: dict) -> None:
        """Restore grouped-fit state from project persistence.

        Accepts the new ``{single, batch}`` shape; a legacy single-surface state
        dict is applied to both surfaces for backward compatibility.
        """
        if not isinstance(state, dict):
            return
        if "single" in state or "batch" in state:
            if isinstance(state.get("single"), dict):
                self._single_fit_tab.restore_state(state["single"])
            if isinstance(state.get("batch"), dict):
                self._batch_fit_tab.restore_state(state["batch"])
        else:
            self._single_fit_tab.restore_state(state)
            self._batch_fit_tab.restore_state(state)
