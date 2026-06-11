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
from asymmetry.core.fitting.grouped_time_domain import normalize_to_grouped_contract
from asymmetry.core.io.nexus_writer import write_nexus_v1
from asymmetry.core.simulate import (
    BUILTIN_TEMPLATES,
    GroupSignalSpec,
    PeriodSpec,
    build_builtin_template,
    simulate_count_run,
    simulate_multi_group_run,
    simulate_run,
    simulate_two_period_run,
    total_events_of,
)
from asymmetry.gui.panels.fit_function_builder import FitFunctionBuilderDialog

#: Synthetic runs are numbered from here, clear of real ISIS/PSI run series.
_SYNTHETIC_RUN_SERIES = 90001


class _SimulateDialogBase(QDialog):
    """Shared run-numbering, seed and NeXus-save plumbing for simulate dialogs.

    :class:`SimulateDialog` (forward/backward) and :class:`MultiGroupSimulateDialog`
    differ only in their template/model/parameter surface; the event-budget,
    seed and save mechanics live here so a fix lands once. Subclasses build
    their own widgets (including ``_seed_check``/``_seed_spin`` and
    ``_status_label``) and set ``_last_run`` on a successful Generate.
    """

    #: Emitted with the generated :class:`Run` each time Generate succeeds.
    run_generated = Signal(object)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        run_number_allocator: Callable[[], int] | None = None,
        run_number_releaser: Callable[[int], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self._run_number_allocator = run_number_allocator
        self._run_number_releaser = run_number_releaser
        self._allocated: set[int] = set()
        self._last_run: Run | None = None

    def _extra_reserved(self) -> set[int]:
        """Run numbers to avoid in the internal fallback (e.g. loaded runs)."""
        return set()

    def _next_run_number(self) -> int:
        if self._run_number_allocator is not None:
            number = int(self._run_number_allocator())
        else:
            existing = self._extra_reserved() | self._allocated
            number = _SYNTHETIC_RUN_SERIES
            while number in existing:
                number += 1
        self._allocated.add(number)
        return number

    def _release_run_number(self, number: int) -> None:
        """Return a reserved number after a failed Generate (no leaked gaps)."""
        self._allocated.discard(number)
        if self._run_number_releaser is not None:
            self._run_number_releaser(number)

    def _resolve_seed(self) -> int:
        """Read the seed widgets, drawing a fresh random seed when unfixed."""
        if self._seed_check.isChecked():
            return int(self._seed_spin.value())
        import secrets

        seed = secrets.randbelow(2**31 - 1)
        self._seed_spin.setValue(seed)
        return seed

    def _save_generated_as_nexus(self, error_title: str) -> None:
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
            QMessageBox.warning(self, error_title, str(exc))
            return
        self._status_label.setText(f"Saved SIM {self._last_run.run_number} to {path}.")


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


class SimulateDialog(_SimulateDialogBase):
    """Modal dialog generating synthetic runs from a loaded-run template."""

    def __init__(
        self,
        datasets: list[MuonDataset],
        *,
        parent: QWidget | None = None,
        preselected_run: int | None = None,
        fit_state_provider: Callable[[int], dict | None] | None = None,
        run_number_allocator: Callable[[], int] | None = None,
        run_number_releaser: Callable[[int], None] | None = None,
    ) -> None:
        super().__init__(
            parent,
            run_number_allocator=run_number_allocator,
            run_number_releaser=run_number_releaser,
        )
        self.setWindowTitle("Generate Synthetic Run")
        self._fit_state_provider = fit_state_provider
        self._templates = [ds for ds in datasets if ds.run is not None and ds.run.histograms]
        self._model: CompositeModel = default_model_for_domain("time")
        self._param_values: dict[str, float] = dict(self._model.param_defaults)
        self._builtin_cache: dict[str, Run] = {}

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
        self._mode_combo = QComboBox()
        # (label, userData) — userData is the generation mode key.
        self._mode_combo.addItem("Forward/backward asymmetry", "fb")
        self._mode_combo.setItemData(
            0, "±a(t) on the F/B groups — the standard asymmetry run.", Qt.ItemDataRole.ToolTipRole
        )
        self._mode_combo.addItem("Count histograms (single-group)", "count")
        self._mode_combo.setItemData(
            1,
            "+a(t) on every group as independent single-histogram counts, for "
            "the count-domain fit modes.",
            Qt.ItemDataRole.ToolTipRole,
        )
        self._mode_combo.addItem("Two-period (red/green)", "two_period")
        self._mode_combo.setItemData(
            2,
            "Two period histograms (red = full signal, green scaled) for the "
            "red/green reduction and G∓R combination.",
            Qt.ItemDataRole.ToolTipRole,
        )
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        controls.addRow("Generation:", self._mode_combo)

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

        self._green_amp_spin = QDoubleSpinBox()
        self._green_amp_spin.setRange(0.0, 2.0)
        self._green_amp_spin.setDecimals(2)
        self._green_amp_spin.setSingleStep(0.1)
        self._green_amp_spin.setValue(0.0)
        self._green_amp_spin.setToolTip(
            "Green-period signal as a fraction of the model amplitude. 0 makes "
            "green a flat reference (the usual light-off / RF-off period), so "
            "G−R recovers the red signal."
        )
        self._green_amp_row_label = QLabel("Green amplitude (×):")
        controls.addRow(self._green_amp_row_label, self._green_amp_spin)

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
        self._on_mode_changed()

    # ------------------------------------------------------------------
    # Template / model / parameters
    # ------------------------------------------------------------------

    def _current_template(self) -> Run | None:
        data = self._template_combo.currentData()
        if isinstance(data, str):
            # Built-in templates are immutable per key — build once and cache,
            # rather than re-allocating zero arrays on every combo change.
            run = self._builtin_cache.get(data)
            if run is None:
                run = build_builtin_template(data)
                self._builtin_cache[data] = run
            return run
        for ds in self._templates:
            if ds.run_number == data and ds.run is not None:
                return ds.run
        return None

    def _on_mode_changed(self) -> None:
        """Show the green-amplitude control only for the two-period mode."""
        two_period = str(self._mode_combo.currentData()) == "two_period"
        self._green_amp_row_label.setVisible(two_period)
        self._green_amp_spin.setVisible(two_period)

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
            total_counts = total_events_of(template)
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

    def _extra_reserved(self) -> set[int]:
        return {ds.run_number for ds in self._templates}

    def _on_generate(self) -> None:
        template = self._current_template()
        if template is None:
            QMessageBox.information(
                self, "Generate Synthetic Run", "Load a run to act as the template first."
            )
            return
        self._param_values = self._table_parameters()
        seed = self._resolve_seed()
        events = self._events_spin.value() * 1.0e6
        background = self._background_spin.value()
        mode = str(self._mode_combo.currentData())
        run_number = self._next_run_number()
        try:
            if mode == "count":
                run = simulate_count_run(
                    template,
                    self._model,
                    self._param_values,
                    total_events=events,
                    seed=seed,
                    background_per_bin=background,
                    run_number=run_number,
                )
            elif mode == "two_period":
                run = simulate_two_period_run(
                    template,
                    self._two_period_specs(),
                    total_events=events,
                    seed=seed,
                    background_per_bin=background,
                    run_number=run_number,
                )
            else:
                run = simulate_run(
                    template,
                    self._model,
                    self._param_values,
                    total_events=events,
                    seed=seed,
                    background_per_bin=background,
                    run_number=run_number,
                )
        except (TypeError, ValueError) as exc:
            QMessageBox.warning(self, "Generate Synthetic Run", str(exc))
            self._release_run_number(run_number)
            return
        self._last_run = run
        self._save_button.setEnabled(True)
        mode_note = {"count": " count histograms", "two_period": " two-period"}.get(mode, "")
        self._status_label.setText(
            f"Generated SIM {run.run_number}{mode_note} from run {template.run_number} "
            f"({self._events_spin.value():g} MEv, seed {seed})."
        )
        self.run_generated.emit(run)

    def _two_period_specs(self) -> list[PeriodSpec]:
        """Build red/green :class:`PeriodSpec`\\ s from the model and green scale.

        Both periods share the model and parameters; green carries the
        green-amplitude factor as its ``scale`` (0 → a flat reference period),
        so G−R recovers the red signal. Using ``scale`` keeps the model's
        provenance intact instead of wrapping it in an opaque closure.
        """
        params = dict(self._param_values)
        green_factor = float(self._green_amp_spin.value())
        return [
            PeriodSpec(self._model, params, label="red"),
            PeriodSpec(self._model, params, scale=green_factor, label="green"),
        ]

    def _on_save_nexus(self) -> None:
        self._save_generated_as_nexus("Save Synthetic Run")


def _template_group_ids(template: Run) -> list[int]:
    """Sorted integer group ids of a template's grouping."""
    groups = template.grouping.get("groups", {}) if isinstance(template.grouping, dict) else {}
    ids: list[int] = []
    for key in groups:
        try:
            ids.append(int(key))
        except (TypeError, ValueError):
            continue
    return sorted(ids)


class MultiGroupSimulateDialog(_SimulateDialogBase):
    """Simulate a run with a distinct amplitude/phase per detector group.

    The multi-group counterpart of :class:`SimulateDialog`: a shared normalised
    polarisation model plus a per-group amplitude/phase/N₀ table (seeded from a
    grouped time-domain fit when one is supplied, otherwise from the template's
    groups), driving
    :func:`asymmetry.core.simulate.simulate_multi_group_run`. Used to synthesise
    a transverse-field ring whose groups differ in phase.
    """

    def __init__(
        self,
        template: Run,
        *,
        parent: QWidget | None = None,
        seed: dict | None = None,
        run_number_allocator: Callable[[], int] | None = None,
        run_number_releaser: Callable[[int], None] | None = None,
    ) -> None:
        super().__init__(
            parent,
            run_number_allocator=run_number_allocator,
            run_number_releaser=run_number_releaser,
        )
        self.setWindowTitle("Generate Multi-Group Run")
        self._template = template

        if seed and isinstance(seed.get("model"), dict):
            self._model = CompositeModel.from_dict(seed["model"])
            self._base_values = dict(seed.get("base_parameters", {}))
        else:
            # A multi-group ring needs a phase-capable polarisation; the
            # per-group amplitude owns the scale, so normalise the model.
            self._model = CompositeModel(["Oscillatory"])
            self._base_values = dict(self._model.param_defaults)
        self._normalize_base()

        layout = QVBoxLayout(self)

        model_row = QHBoxLayout()
        self._model_label = QLabel(self._model.formula_string())
        self._model_label.setWordWrap(True)
        edit_model = QPushButton("Edit Model…")
        edit_model.clicked.connect(self._on_edit_model)
        model_row.addWidget(QLabel("Polarisation P(t):"))
        model_row.addWidget(self._model_label, stretch=1)
        model_row.addWidget(edit_model)
        layout.addLayout(model_row)

        layout.addWidget(QLabel("Per-group signal (amplitude is fractional; phase in radians):"))
        self._group_table = QTableWidget(0, 4)
        self._group_table.setHorizontalHeaderLabels(["Group", "Amplitude", "Phase (rad)", "N₀"])
        self._group_table.verticalHeader().setVisible(False)
        layout.addWidget(self._group_table)
        self._seed_group_table(seed)

        controls = QFormLayout()
        self._events_spin = QDoubleSpinBox()
        self._events_spin.setRange(0.01, 1.0e5)
        self._events_spin.setDecimals(2)
        self._events_spin.setSuffix(" MEv")
        self._events_spin.setValue(40.0)
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

    def _seed_group_table(self, seed: dict | None) -> None:
        group_ids = _template_group_ids(self._template)
        specs: list[dict] = []
        if seed and isinstance(seed.get("specs"), list):
            specs = [s for s in seed["specs"] if int(s.get("group_id", -1)) in set(group_ids)]
        if not specs:
            # Spread phases evenly so the default ring is visibly multi-phase.
            import math

            n = max(1, len(group_ids))
            specs = [
                {
                    "group_id": gid,
                    "amplitude": 0.2,
                    "relative_phase": 2.0 * math.pi * index / n,
                    "n0_weight": 1.0,
                }
                for index, gid in enumerate(group_ids)
            ]
        self._group_table.setRowCount(len(specs))
        for row, spec in enumerate(specs):
            gid_item = QTableWidgetItem(str(spec.get("group_id")))
            gid_item.setFlags(gid_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            gid_item.setData(Qt.ItemDataRole.UserRole, int(spec.get("group_id")))
            self._group_table.setItem(row, 0, gid_item)
            self._group_table.setItem(
                row, 1, QTableWidgetItem(f"{float(spec.get('amplitude', 0.2)):g}")
            )
            self._group_table.setItem(
                row, 2, QTableWidgetItem(f"{float(spec.get('relative_phase', 0.0)):g}")
            )
            self._group_table.setItem(
                row, 3, QTableWidgetItem(f"{float(spec.get('n0_weight', 1.0)):g}")
            )

    def _normalize_base(self) -> None:
        """Force the shared model to the normalised-polarisation contract.

        The per-group amplitude owns the overall scale and the per-group N₀ the
        background, so the shared model must be a unit-amplitude, zero-baseline
        polarisation — delegated to the single core definition of the grouped
        contract.
        """
        self._base_values = normalize_to_grouped_contract(
            list(getattr(self._model, "param_names", [])), self._base_values
        )

    def _on_edit_model(self) -> None:
        dialog = FitFunctionBuilderDialog(self, initial_model=self._model, domain="time")
        if dialog.exec():
            model = dialog.get_composite_model()
            if model is not None:
                self._model = model
                merged = dict(model.param_defaults)
                merged.update({k: v for k, v in self._base_values.items() if k in merged})
                self._base_values = merged
                self._normalize_base()
                self._model_label.setText(self._model.formula_string())

    def _specs_from_table(self) -> list[GroupSignalSpec]:
        specs: list[GroupSignalSpec] = []
        for row in range(self._group_table.rowCount()):
            gid_item = self._group_table.item(row, 0)
            if gid_item is None:
                continue
            gid = int(gid_item.data(Qt.ItemDataRole.UserRole))

            def _value(column: int, default: float) -> float:
                item = self._group_table.item(row, column)
                try:
                    return float(item.text())
                except (TypeError, ValueError, AttributeError):
                    return default

            specs.append(
                GroupSignalSpec(
                    group_id=gid,
                    amplitude=_value(1, 0.2),
                    relative_phase=_value(2, 0.0),
                    n0_weight=_value(3, 1.0),
                )
            )
        return specs

    def _on_generate(self) -> None:
        seed = self._resolve_seed()
        run_number = self._next_run_number()
        try:
            run = simulate_multi_group_run(
                self._template,
                self._model,
                self._specs_from_table(),
                total_events=self._events_spin.value() * 1.0e6,
                seed=seed,
                base_parameters=self._base_values,
                background_per_bin=self._background_spin.value(),
                run_number=run_number,
            )
        except (TypeError, ValueError) as exc:
            QMessageBox.warning(self, "Generate Multi-Group Run", str(exc))
            self._release_run_number(run_number)
            return
        self._last_run = run
        self._save_button.setEnabled(True)
        self._status_label.setText(
            f"Generated SIM {run.run_number} ({len(run.histograms)} detectors, "
            f"{self._events_spin.value():g} MEv, seed {seed})."
        )
        self.run_generated.emit(run)

    def _on_save_nexus(self) -> None:
        self._save_generated_as_nexus("Save Multi-Group Run")
