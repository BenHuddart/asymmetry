"""GUI tests for the simulate dialog and the Data Browser degrade action.

Verification-plan §5 of docs/porting/simulate-mode/verification-plan.md.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("h5py")
from PySide6.QtWidgets import QApplication

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.io import load
from asymmetry.core.simulate import reduce_run_to_dataset, simulate_run
from asymmetry.gui.panels.data_browser import DataBrowserPanel
from asymmetry.gui.windows.simulate_dialog import (
    DegradeStatisticsDialog,
    MultiGroupSimulateDialog,
    SimulateDialog,
)


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


N_BINS = 600
BIN_WIDTH = 0.016
T0_BIN = 30


def _template_run(run_number: int = 1234) -> Run:
    histograms = [
        Histogram(
            counts=np.zeros(N_BINS),
            bin_width=BIN_WIDTH,
            t0_bin=T0_BIN,
            good_bin_start=T0_BIN + 5,
            good_bin_end=N_BINS - 5,
        )
        for _ in range(2)
    ]
    grouping = {
        "groups": {1: [1], 2: [2]},
        "forward_group": 1,
        "backward_group": 2,
        "alpha": 1.0,
        "t0_bin": T0_BIN,
        "first_good_bin": T0_BIN + 5,
        "last_good_bin": N_BINS - 5,
        "good_frames": 1000.0,
    }
    return Run(
        run_number=run_number,
        histograms=histograms,
        metadata={"title": "Template", "temperature": 5.0, "field": 0.0},
        grouping=grouping,
    )


def _template_dataset(run_number: int = 1234) -> MuonDataset:
    run = _template_run(run_number)
    t = np.arange(50) * BIN_WIDTH
    return MuonDataset(
        time=t,
        asymmetry=np.zeros_like(t),
        error=np.ones_like(t),
        metadata={"run_number": run_number, "title": "Template"},
        run=run,
    )


class TestSimulateDialog:
    def test_generate_emits_badged_synthetic_run(self, qapp) -> None:
        dataset = _template_dataset()
        dialog = SimulateDialog([dataset], preselected_run=1234)
        generated: list[Run] = []
        dialog.run_generated.connect(generated.append)

        dialog._events_spin.setValue(1.0)
        dialog._seed_spin.setValue(7)
        dialog._on_generate()

        assert len(generated) == 1
        run = generated[0]
        assert run.metadata["synthetic"] is True
        assert run.metadata["simulation"]["seed"] == 7
        assert run.metadata["simulation"]["template_run_number"] == 1234
        assert run.run_number >= 90001
        assert dialog._save_button.isEnabled()

    def test_seeds_model_and_values_from_fit_state(self, qapp) -> None:
        model = CompositeModel(["Exponential"])
        fitted_values = dict(model.param_defaults)
        first = next(iter(fitted_values))
        fitted_values[first] = 17.25
        state = {
            "composite_model": model.to_dict(),
            "parameters": [{"name": name, "value": value} for name, value in fitted_values.items()],
        }

        dialog = SimulateDialog(
            [_template_dataset()],
            preselected_run=1234,
            fit_state_provider=lambda rn: state if rn == 1234 else None,
        )
        assert dialog._model.formula_string() == model.formula_string()
        assert dialog._table_parameters()[first] == pytest.approx(17.25)

    def test_run_number_allocator_used(self, qapp) -> None:
        dialog = SimulateDialog(
            [_template_dataset()],
            preselected_run=1234,
            run_number_allocator=lambda: 90555,
        )
        generated: list[Run] = []
        dialog.run_generated.connect(generated.append)
        dialog._events_spin.setValue(0.5)
        dialog._on_generate()
        assert generated[0].run_number == 90555

    def test_edit_model_preserves_typed_values(self, qapp, monkeypatch) -> None:
        """Values typed into the table survive Edit Model… (cancel and accept)."""
        from asymmetry.gui.panels.fit_function_builder import FitFunctionBuilderDialog

        dialog = SimulateDialog([_template_dataset()], preselected_run=1234)
        name = dialog._param_table.item(0, 0).text()
        dialog._param_table.item(0, 1).setText("33.5")

        monkeypatch.setattr(FitFunctionBuilderDialog, "exec", lambda self: 0)
        dialog._on_edit_model()  # cancelled builder
        assert dialog._table_parameters()[name] == pytest.approx(33.5)

        monkeypatch.setattr(FitFunctionBuilderDialog, "exec", lambda self: 1)
        monkeypatch.setattr(
            FitFunctionBuilderDialog,
            "get_composite_model",
            lambda self, _model=dialog._model: _model,
        )
        dialog._on_edit_model()  # accepted builder, same model
        assert dialog._table_parameters()[name] == pytest.approx(33.5)

    def test_save_as_nexus_round_trips(self, qapp, tmp_path, monkeypatch) -> None:
        dialog = SimulateDialog([_template_dataset()], preselected_run=1234)
        dialog._events_spin.setValue(1.0)
        dialog._on_generate()

        target = tmp_path / "sim.nxs"
        monkeypatch.setattr(
            "asymmetry.gui.windows.simulate_dialog.QFileDialog.getSaveFileName",
            staticmethod(lambda *args, **kwargs: (str(target), "NeXus files (*.nxs)")),
        )
        dialog._on_save_nexus()

        assert target.exists()
        reloaded = load(target)
        assert reloaded.run is not None
        for original, again in zip(
            dialog._last_run.histograms, reloaded.run.histograms, strict=True
        ):
            assert np.array_equal(again.counts, original.counts)


class TestBuiltinTemplateDialog:
    """The dialog offers built-in instruments and works with no run loaded."""

    def test_builtins_present_with_no_loaded_run(self, qapp) -> None:
        from asymmetry.core.simulate import BUILTIN_TEMPLATES

        dialog = SimulateDialog([])
        # Every built-in is offered even though no run was loaded.
        offered = {
            dialog._template_combo.itemData(i) for i in range(dialog._template_combo.count())
        }
        for key in BUILTIN_TEMPLATES:
            assert key in offered
        # Generate is not blocked by the absence of a loaded run.
        assert dialog._generate_button.isEnabled()

    def test_generate_from_builtin(self, qapp) -> None:
        dialog = SimulateDialog([], run_number_allocator=lambda: 90001)
        index = dialog._template_combo.findData("ideal_pulsed_fb")
        assert index >= 0
        dialog._template_combo.setCurrentIndex(index)
        generated: list[Run] = []
        dialog.run_generated.connect(generated.append)
        dialog._on_generate()

        assert len(generated) == 1
        run = generated[0]
        assert run.metadata["synthetic"] is True
        assert len(run.histograms) == 64

    def test_continuous_seeds_background_default(self, qapp) -> None:
        dialog = SimulateDialog([])
        index = dialog._template_combo.findData("ideal_continuous_fb")
        dialog._template_combo.setCurrentIndex(index)
        # The continuous template seeds a non-zero flat background.
        assert dialog._background_spin.value() > 0.0


def _ring_template_run(n_groups: int = 4) -> Run:
    histograms = [
        Histogram(
            counts=np.zeros(N_BINS),
            bin_width=BIN_WIDTH,
            t0_bin=T0_BIN,
            good_bin_start=T0_BIN,
            good_bin_end=N_BINS - 5,
        )
        for _ in range(n_groups)
    ]
    grouping = {
        "groups": {gid: [gid] for gid in range(1, n_groups + 1)},
        "forward_group": 1,
        "backward_group": 3,
        "alpha": 1.0,
        "t0_bin": T0_BIN,
        "first_good_bin": T0_BIN,
        "last_good_bin": N_BINS - 5,
    }
    return Run(
        run_number=2200,
        histograms=histograms,
        metadata={"title": "Ring", "field": 200.0},
        grouping=grouping,
    )


class TestMultiGroupSimulateDialog:
    def test_default_table_one_row_per_group(self, qapp) -> None:
        dialog = MultiGroupSimulateDialog(_ring_template_run(4))
        assert dialog._group_table.rowCount() == 4

    def test_generate_emits_multi_group_run(self, qapp) -> None:
        dialog = MultiGroupSimulateDialog(_ring_template_run(4), run_number_allocator=lambda: 90010)
        generated: list[Run] = []
        dialog.run_generated.connect(generated.append)
        dialog._events_spin.setValue(5.0)
        dialog._on_generate()
        assert len(generated) == 1
        run = generated[0]
        assert run.run_number == 90010
        assert len(run.histograms) == 4
        assert run.metadata["simulation"]["multi_group"] is True

    def test_seeded_from_grouped_fit(self, qapp) -> None:
        model = CompositeModel(["Oscillatory"])
        seed = {
            "model": model.to_dict(),
            "base_parameters": dict(model.param_defaults),
            "specs": [
                {"group_id": 1, "amplitude": 0.21, "relative_phase": 0.0, "n0_weight": 1.0},
                {"group_id": 2, "amplitude": 0.19, "relative_phase": 1.57, "n0_weight": 1.2},
                {"group_id": 3, "amplitude": 0.20, "relative_phase": 3.14, "n0_weight": 0.9},
                {"group_id": 4, "amplitude": 0.18, "relative_phase": 4.71, "n0_weight": 1.0},
            ],
        }
        dialog = MultiGroupSimulateDialog(_ring_template_run(4), seed=seed)
        specs = dialog._specs_from_table()
        by_id = {s.group_id: s for s in specs}
        assert by_id[2].amplitude == pytest.approx(0.19)
        assert by_id[2].relative_phase == pytest.approx(1.57)
        assert by_id[2].n0_weight == pytest.approx(1.2)


class TestDegradeAction:
    def test_apply_degrade_adds_derived_run(self, qapp) -> None:
        browser = DataBrowserPanel()
        source = simulate_run(
            _template_run(),
            lambda t: np.zeros_like(t),
            total_events=2.0e6,
            seed=1,
            run_number=2000,
        )
        browser.add_dataset(reduce_run_to_dataset(source))
        before = [h.counts.copy() for h in source.histograms]

        derived = browser.apply_degrade_statistics(2000, 0.5, seed=3)
        assert derived is not None
        assert derived.run_number >= 90001
        assert derived.metadata["degraded"]["factor"] == 0.5
        assert derived.run_number in {ds.run_number for ds in browser.all_datasets()}
        # Source untouched.
        for hist, original in zip(source.histograms, before, strict=True):
            assert np.array_equal(hist.counts, original)
        browser.deleteLater()

    def test_apply_degrade_without_histograms_returns_none(self, qapp, monkeypatch) -> None:
        warnings: list[str] = []
        monkeypatch.setattr(
            "asymmetry.gui.panels.data_browser.QMessageBox.warning",
            staticmethod(lambda _parent, _title, text, *a, **k: warnings.append(text)),
        )
        browser = DataBrowserPanel()
        t = np.arange(10, dtype=float)
        browser.add_dataset(
            MuonDataset(
                time=t,
                asymmetry=np.zeros_like(t),
                error=np.ones_like(t),
                metadata={"run_number": 3000, "title": "curve only"},
            )
        )
        assert browser.apply_degrade_statistics(3000, 0.5, seed=1) is None
        assert warnings and "histograms" in warnings[0]
        browser.deleteLater()

    def test_badge_tooltip_for_synthetic_and_degraded(self, qapp) -> None:
        browser = DataBrowserPanel()
        synthetic = simulate_run(
            _template_run(),
            lambda t: np.zeros_like(t),
            total_events=1.0e6,
            seed=2,
            run_number=90010,
        )
        browser.add_dataset(reduce_run_to_dataset(synthetic))

        row_texts = {}
        for row in range(browser._table.rowCount()):
            item = browser._table.item(row, 0)
            row_texts[item.text().strip()] = item

        sim_item = row_texts["SIM 90010"]
        assert "Synthetic run" in sim_item.toolTip()
        assert "seed 2" in sim_item.toolTip()
        browser.deleteLater()

    def test_reloaded_synthetic_nexus_stays_badged(self, qapp, tmp_path) -> None:
        """A saved-and-reopened synthetic run keeps its provenance badge."""
        from asymmetry.core.io.nexus_writer import write_nexus_v1

        run = simulate_run(
            _template_run(),
            lambda t: np.zeros_like(t),
            total_events=1.0e6,
            seed=5,
            run_number=90020,
        )
        path = tmp_path / "sim.nxs"
        write_nexus_v1(run, path)
        reloaded = load(path)

        browser = DataBrowserPanel()
        browser.add_dataset(reloaded)
        item = next(
            browser._table.item(row, 0)
            for row in range(browser._table.rowCount())
            if browser._table.item(row, 0).text().strip() == "90020"
        )
        assert "Synthetic run" in item.toolTip()
        assert "reloaded from file" in item.toolTip()
        browser.deleteLater()

    def test_degrade_dialog_defaults(self, qapp) -> None:
        dialog = DegradeStatisticsDialog()
        assert dialog.factor() == pytest.approx(0.5)
        assert dialog.seed() == 0
        dialog.deleteLater()
