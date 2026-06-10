"""Generate-synthetic-run dialog (File → Generate Synthetic Run…).

Front end for :func:`asymmetry.core.simulate.simulate_run`: pick a loaded run
as the instrument template, choose a time-domain model (reusing the fit
function builder), set parameter values — seeded from the run's current fit
when one exists — an event budget, an optional flat background and a fixed
RNG seed, then Generate. The result is emitted via :attr:`run_generated` and
appears in the Data Browser badged as synthetic; Save as NeXus… writes it as
a loadable file through :func:`asymmetry.core.io.nexus_writer.write_nexus_v1`.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.data.dataset import MuonDataset, Run
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.domain_library import default_model_for_domain
from asymmetry.core.io.nexus_writer import write_nexus_v1
from asymmetry.core.simulate import BUILTIN_TEMPLATES, build_builtin_template, simulate_run
from asymmetry.gui.panels.fit_function_builder import FitFunctionBuilderDialog

#: Synthetic runs are numbered from here, clear of real ISIS/PSI run series.
_SYNTHETIC_RUN_SERIES = 90001


class DegradeStatisticsDialog(QDialog):
    """Factor + seed prompt for the Data Browser's Degrade Statistics action."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Degrade Statistics")

        layout = QVBoxLayout(self)
        hint = QLabel(
            "Resample every detector histogram to a different statistics "
            "level. A factor below 1 thins the recorded counts exactly as a "
            "shorter measurement would; a factor above 1 extrapolates and "
            "slightly over-disperses."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        form = QFormLayout()
        self._factor_spin = QDoubleSpinBox()
        self._factor_spin.setRange(0.001, 1000.0)
        self._factor_spin.setDecimals(3)
        self._factor_spin.setSingleStep(0.05)
        self._factor_spin.setValue(0.5)
        form.addRow("Factor:", self._factor_spin)

        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(0, 2**31 - 1)
        self._seed_spin.setValue(0)
        form.addRow("Seed:", self._seed_spin)
        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def factor(self) -> float:
        return float(self._factor_spin.value())

    def seed(self) -> int:
        return int(self._seed_spin.value())


class SimulateDialog(QDialog):
    """Modal dialog generating synthetic runs from a loaded-run template."""

    #: Emitted with the generated :class:`Run` each time Generate succeeds.
    run_generated = Signal(object)

    def __init__(
        self,
        datasets: list[MuonDataset],
        *,
        parent: QWidget | None = None,
        preselected_run: int | None = None,
        fit_state_provider: Callable[[int], dict | None] | None = None,
        run_number_allocator: Callable[[], int] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Generate Synthetic Run")
        self._fit_state_provider = fit_state_provider
        self._run_number_allocator = run_number_allocator
        self._templates = [ds for ds in datasets if ds.run is not None and ds.run.histograms]
        self._model: CompositeModel = default_model_for_domain("time")
        self._param_values: dict[str, float] = dict(self._model.param_defaults)
        self._last_run: Run | None = None
        self._allocated: set[int] = set()

        layout = QVBoxLayout(self)

        form = QFormLayout()
        self._template_combo = QComboBox()
        for ds in self._templates:
            title = str(ds.metadata.get("title", "")).strip()
            label = f"{ds.run_label} — {title}" if title else str(ds.run_label)
            self._template_combo.addItem(label, ds.run_number)
        # Built-in idealised instruments stand in when no run is loaded (and
        # are always offered as a teaching baseline). Their combo data is the
        # string registry key, distinguishing them from a loaded run number.
        for key, template in BUILTIN_TEMPLATES.items():
            self._template_combo.addItem(f"Built-in: {template.label}", key)
        form.addRow("Template run:", self._template_combo)

        model_row = QHBoxLayout()
        self._model_label = QLabel()
        self._model_label.setWordWrap(True)
        edit_model = QPushButton("Edit Model…")
        edit_model.clicked.connect(self._on_edit_model)
        model_row.addWidget(self._model_label, stretch=1)
        model_row.addWidget(edit_model)
        form.addRow("Model A(t):", model_row)
        layout.addLayout(form)

        self._param_table = QTableWidget(0, 3)
        self._param_table.setHorizontalHeaderLabels(["Parameter", "Value", "Unit"])
        self._param_table.horizontalHeader().setStretchLastSection(True)
        self._param_table.verticalHeader().setVisible(False)
        layout.addWidget(self._param_table)

        controls = QFormLayout()
        self._events_spin = QDoubleSpinBox()
        self._events_spin.setRange(0.01, 1.0e5)
        self._events_spin.setDecimals(2)
        self._events_spin.setSuffix(" MEv")
        self._events_spin.setValue(10.0)
        controls.addRow("Total events:", self._events_spin)

        self._background_spin = QDoubleSpinBox()
        self._background_spin.setRange(0.0, 1.0e6)
        self._background_spin.setDecimals(3)
        self._background_spin.setValue(0.0)
        controls.addRow("Background (counts/bin/detector):", self._background_spin)

        seed_row = QHBoxLayout()
        self._seed_check = QCheckBox("Fixed seed")
        self._seed_check.setChecked(True)
        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(0, 2**31 - 1)
        self._seed_spin.setValue(0)
        self._seed_check.toggled.connect(self._seed_spin.setEnabled)
        seed_row.addWidget(self._seed_check)
        seed_row.addWidget(self._seed_spin, stretch=1)
        controls.addRow("Reproducibility:", seed_row)
        layout.addLayout(controls)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        buttons = QDialogButtonBox()
        self._generate_button = buttons.addButton(
            "Generate", QDialogButtonBox.ButtonRole.ActionRole
        )
        self._save_button = buttons.addButton(
            "Save as NeXus…", QDialogButtonBox.ButtonRole.ActionRole
        )
        close_button = buttons.addButton(QDialogButtonBox.StandardButton.Close)
        self._generate_button.clicked.connect(self._on_generate)
        self._save_button.clicked.connect(self._on_save_nexus)
        self._save_button.setEnabled(False)
        close_button.clicked.connect(self.reject)
        layout.addWidget(buttons)

        self._template_combo.currentIndexChanged.connect(self._on_template_changed)
        if preselected_run is not None:
            index = self._template_combo.findData(preselected_run)
            if index >= 0:
                self._template_combo.setCurrentIndex(index)
        # Built-in templates are always available, so Generate is never blocked
        # purely by the absence of a loaded run.
        self._generate_button.setEnabled(self._template_combo.count() > 0)
        self._on_template_changed()

    # ------------------------------------------------------------------
    # Template / model / parameters
    # ------------------------------------------------------------------

    def _current_template(self) -> Run | None:
        data = self._template_combo.currentData()
        if isinstance(data, str):
            return build_builtin_template(data)
        for ds in self._templates:
            if ds.run_number == data and ds.run is not None:
                return ds.run
        return None

    def _on_template_changed(self) -> None:
        template = self._current_template()
        if template is None:
            return
        # Keep values the user typed but has not generated with yet.
        self._param_values.update(self._table_parameters())
        data = self._template_combo.currentData()
        if isinstance(data, str):
            # Built-in instrument: seed the event budget and background from
            # its teaching-sensible defaults (the run has zero counts), and
            # skip fit seeding (a built-in carries no fit state).
            spec = BUILTIN_TEMPLATES[data]
            self._events_spin.setValue(max(0.01, round(spec.default_total_events / 1.0e6, 2)))
            self._background_spin.setValue(spec.default_background_per_bin)
        else:
            # Default the event budget to the template's realised statistics.
            total_counts = sum(float(h.counts.sum()) for h in template.histograms)
            if total_counts > 0:
                self._events_spin.setValue(max(0.01, round(total_counts / 1.0e6, 2)))
            self._seed_from_fit(template)
        self._refresh_model_view()

    def _seed_from_fit(self, template: Run) -> None:
        """Adopt the template run's current fit model and values when present."""
        if self._fit_state_provider is None:
            return
        state = self._fit_state_provider(template.run_number)
        if not isinstance(state, dict):
            return
        model_payload = state.get("composite_model")
        if not isinstance(model_payload, dict):
            return
        try:
            model = CompositeModel.from_dict(model_payload)
        except Exception:
            return
        values = dict(model.param_defaults)
        for entry in state.get("parameters", []):
            if isinstance(entry, dict) and entry.get("name") in values:
                try:
                    values[str(entry["name"])] = float(entry.get("value", 0.0))
                except (TypeError, ValueError):
                    continue
        self._model = model
        self._param_values = values

    def _on_edit_model(self) -> None:
        # Capture in-table edits first — the table rebuild below would
        # otherwise silently revert values typed since the last Generate.
        self._param_values.update(self._table_parameters())
        dialog = FitFunctionBuilderDialog(self, initial_model=self._model, domain="time")
        if dialog.exec():
            model = dialog.get_composite_model()
            if model is not None:
                self._model = model
                merged = dict(model.param_defaults)
                merged.update({k: v for k, v in self._param_values.items() if k in merged})
                self._param_values = merged
                self._refresh_model_view()

    def _refresh_model_view(self) -> None:
        self._model_label.setText(self._model.formula_string())
        params = list(self._model.param_names)
        self._param_table.setRowCount(len(params))
        for row, name in enumerate(params):
            info = self._model.param_info.get(name)
            name_item = QTableWidgetItem(name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._param_table.setItem(row, 0, name_item)

            value = self._param_values.get(name, self._model.param_defaults.get(name, 0.0))
            self._param_table.setItem(row, 1, QTableWidgetItem(f"{value:g}"))

            unit_item = QTableWidgetItem(str(getattr(info, "unit", "") or ""))
            unit_item.setFlags(unit_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._param_table.setItem(row, 2, unit_item)

    def _table_parameters(self) -> dict[str, float]:
        values: dict[str, float] = {}
        for row in range(self._param_table.rowCount()):
            name_item = self._param_table.item(row, 0)
            value_item = self._param_table.item(row, 1)
            if name_item is None or value_item is None:
                continue
            try:
                values[name_item.text()] = float(value_item.text())
            except ValueError:
                values[name_item.text()] = float(
                    self._model.param_defaults.get(name_item.text(), 0.0)
                )
        return values

    # ------------------------------------------------------------------
    # Generate / save
    # ------------------------------------------------------------------

    def _next_run_number(self) -> int:
        if self._run_number_allocator is not None:
            return int(self._run_number_allocator())
        existing = {ds.run_number for ds in self._templates} | self._allocated
        number = _SYNTHETIC_RUN_SERIES
        while number in existing:
            number += 1
        self._allocated.add(number)
        return number

    def _on_generate(self) -> None:
        template = self._current_template()
        if template is None:
            QMessageBox.information(
                self, "Generate Synthetic Run", "Load a run to act as the template first."
            )
            return
        self._param_values = self._table_parameters()
        seed = self._seed_spin.value() if self._seed_check.isChecked() else None
        if seed is None:
            import secrets

            seed = secrets.randbelow(2**31 - 1)
            self._seed_spin.setValue(seed)
        run_number = self._next_run_number()
        try:
            run = simulate_run(
                template,
                self._model,
                self._param_values,
                total_events=self._events_spin.value() * 1.0e6,
                seed=int(seed),
                background_per_bin=self._background_spin.value(),
                run_number=run_number,
            )
        except (TypeError, ValueError) as exc:
            QMessageBox.warning(self, "Generate Synthetic Run", str(exc))
            self._allocated.discard(run_number)
            return
        self._last_run = run
        self._save_button.setEnabled(True)
        self._status_label.setText(
            f"Generated SIM {run.run_number} from run {template.run_number} "
            f"({self._events_spin.value():g} MEv, seed {seed})."
        )
        self.run_generated.emit(run)

    def _on_save_nexus(self) -> None:
        if self._last_run is None:
            return
        path, _selected = QFileDialog.getSaveFileName(
            self,
            "Save Synthetic Run as NeXus",
            f"SIM{self._last_run.run_number}.nxs",
            "NeXus files (*.nxs)",
        )
        if not path:
            return
        try:
            write_nexus_v1(self._last_run, path)
        except (OSError, ValueError, ImportError) as exc:
            QMessageBox.warning(self, "Save Synthetic Run", str(exc))
            return
        self._status_label.setText(f"Saved SIM {self._last_run.run_number} to {path}.")
