"""Focused tests for SingleFitTab and GlobalFitTab logic."""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QComboBox

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.engine import FitResult
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.gui.panels import fit_panel as fit_panel_module
from asymmetry.gui.panels.fit_panel import FitPanel, GlobalFitTab, SingleFitTab


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def dataset() -> MuonDataset:
    t = np.linspace(0.0, 4.0, 80)
    a = 0.2 * np.exp(-0.4 * t)
    e = np.full_like(t, 0.01)
    return MuonDataset(time=t, asymmetry=a, error=e, metadata={"run_number": 101})


def test_single_fit_requires_dataset(qapp: QApplication) -> None:
    tab = SingleFitTab()
    tab._current_dataset = None
    tab._run_fit()
    assert "No dataset selected" in tab._result_label.text()


def test_fit_panel_forwards_single_tab_preview(
    qapp: QApplication, dataset: MuonDataset
) -> None:
    panel = FitPanel()
    panel.set_dataset(dataset)

    emitted = {}
    panel.preview_requested.connect(
        lambda result, curve, components: emitted.update(
            {"result": result, "curve": curve, "components": components}
        )
    )

    panel._single_tab._on_preview()

    assert "curve" in emitted
    assert len(emitted["curve"][0]) == 500


def test_single_fit_invalid_value_shows_error(qapp: QApplication, dataset: MuonDataset) -> None:
    tab = SingleFitTab()
    tab.set_dataset(dataset)

    # Corrupt first value cell.
    tab._param_table.item(0, 1).setText("not-a-number")
    tab._run_fit()

    assert "Invalid value" in tab._result_label.text()


def test_single_fit_success_emits_and_updates_table(
    qapp: QApplication, dataset: MuonDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    tab = SingleFitTab()
    tab.set_dataset(dataset)

    model = tab._composite_model

    fitted = ParameterSet(
        [Parameter(name=p, value=float(i + 1)) for i, p in enumerate(model.param_names)]
    )
    result = FitResult(
        success=True,
        chi_squared=10.0,
        reduced_chi_squared=0.5,
        parameters=fitted,
        uncertainties={p: 0.01 for p in model.param_names},
    )

    tab._fit_engine = SimpleNamespace(fit=lambda *_args, **_kwargs: result)

    emitted = {}
    tab.fit_completed.connect(lambda res, curve: emitted.update({"res": res, "curve": curve}))

    tab._run_fit()

    assert "Fit failed" not in tab._result_label.text()
    assert "χ²" in tab._result_label.text()
    assert emitted["res"].success is True
    assert len(emitted["curve"][0]) == 500


def test_single_fit_uses_dataset_object_it_was_given(
    qapp: QApplication, dataset: MuonDataset
) -> None:
    rebinned = MuonDataset(
        time=dataset.time[::4],
        asymmetry=dataset.asymmetry[::4],
        error=dataset.error[::4],
        metadata=dict(dataset.metadata),
    )

    tab = SingleFitTab()
    tab.set_dataset(rebinned)

    model = tab._composite_model

    captured = {}

    def _fit(captured_dataset, model_fn, parameters):
        captured["dataset"] = captured_dataset
        captured["model_fn"] = model_fn
        captured["n_points"] = len(captured_dataset.time)
        return FitResult(
            success=True,
            chi_squared=1.0,
            reduced_chi_squared=0.1,
            parameters=ParameterSet(
                [Parameter(name=p, value=float(i + 1)) for i, p in enumerate(model.param_names)]
            ),
            uncertainties={p: 0.01 for p in model.param_names},
        )

    tab._fit_engine = SimpleNamespace(fit=_fit)

    tab._run_fit()

    assert captured["dataset"] is rebinned
    assert captured["n_points"] == len(rebinned.time)


def test_global_tab_set_datasets_states(qapp: QApplication, dataset: MuonDataset) -> None:
    tab = GlobalFitTab()

    tab.set_datasets([])
    assert tab._fit_btn.isEnabled() is False
    assert "No datasets selected" in tab._result_text.toPlainText()

    tab.set_datasets([dataset])
    assert tab._fit_btn.isEnabled() is False
    assert "requires at least 2 datasets" in tab._result_text.toPlainText()

    d2 = MuonDataset(dataset.time, dataset.asymmetry, dataset.error, {"run_number": 102})
    tab.set_datasets([dataset, d2])
    assert tab._fit_btn.isEnabled() is True


def test_global_fit_rejects_non_finite_value(qapp: QApplication, dataset: MuonDataset) -> None:
    tab = GlobalFitTab()
    d2 = MuonDataset(dataset.time, dataset.asymmetry, dataset.error, {"run_number": 102})
    tab.set_datasets([dataset, d2])

    # Make first parameter non-finite.
    tab._param_table.item(0, 1).setText("nan")
    tab._run_global_fit()

    assert "must be finite" in tab._result_text.toPlainText()


def test_global_fit_rejects_invalid_bounds(qapp: QApplication, dataset: MuonDataset) -> None:
    tab = GlobalFitTab()
    d2 = MuonDataset(dataset.time, dataset.asymmetry, dataset.error, {"run_number": 102})
    tab.set_datasets([dataset, d2])

    tab._param_table.item(0, 3).setText("2, 1")
    tab._run_global_fit()

    assert "invalid bounds" in tab._result_text.toPlainText()


def test_global_fit_finished_success_emits(qapp: QApplication, dataset: MuonDataset) -> None:
    tab = GlobalFitTab()
    d2 = MuonDataset(dataset.time, dataset.asymmetry, dataset.error, {"run_number": 102})
    tab._datasets = [dataset, d2]

    model = tab._composite_model
    tab._current_model = model
    tab._current_global_params = [model.param_names[0]]

    pset = ParameterSet([Parameter(name=p, value=1.0) for p in model.param_names])
    result = FitResult(
        success=True,
        chi_squared=5.0,
        reduced_chi_squared=0.5,
        parameters=pset,
        uncertainties={model.param_names[0]: 0.1},
    )
    fitted_global = ParameterSet([Parameter(name=model.param_names[0], value=1.0)])

    emitted = {}
    tab.global_fit_completed.connect(lambda res, glob: emitted.update({"res": res, "glob": glob}))

    tab._on_fit_finished({101: result, 102: result}, fitted_global)

    assert "Global Fit Successful" in tab._result_text.toHtml()
    assert set(emitted["res"]) == {101, 102}


def test_global_fit_finished_failure_lists_failed_runs(qapp: QApplication, dataset: MuonDataset) -> None:
    tab = GlobalFitTab()
    tab._current_model = tab._composite_model
    tab._current_global_params = []
    fail = FitResult(success=False, message="x")

    tab._on_fit_finished({101: fail}, ParameterSet())
    assert "Global fit failed" in tab._result_text.toPlainText()


def test_global_fit_error_sets_message(qapp: QApplication) -> None:
    tab = GlobalFitTab()
    tab._fit_btn.setEnabled(False)
    tab._on_fit_error("boom")
    assert tab._fit_btn.isEnabled() is True
    assert "boom" in tab._result_text.toPlainText()


def test_global_fit_parses_type_combo_defaults(qapp: QApplication) -> None:
    tab = GlobalFitTab()
    # First row defaults to Global, subsequent rows Local.
    c0 = tab._param_table.cellWidget(0, 2)
    c1 = tab._param_table.cellWidget(1, 2) if tab._param_table.rowCount() > 1 else None
    assert isinstance(c0, QComboBox)
    assert c0.currentText() == "Global"
    if isinstance(c1, QComboBox):
        assert c1.currentText() == "Local"


def test_single_tab_default_model_includes_background(qapp: QApplication) -> None:
    tab = SingleFitTab()
    assert tab._composite_model.component_names == ["Exponential", "Constant"]
    assert "A_bg" in tab._composite_model.param_names


def test_global_tab_default_model_includes_background(qapp: QApplication) -> None:
    tab = GlobalFitTab()
    assert tab._composite_model.component_names == ["Exponential", "Constant"]
    assert "A_bg" in tab._composite_model.param_names


def test_single_edit_function_updates_parameter_rows(qapp: QApplication) -> None:
    tab = SingleFitTab()
    tab._set_composite_model(CompositeModel(["Exponential", "Constant"], operators=["+"]))
    assert tab._param_table.rowCount() == 3


def test_global_edit_function_updates_parameter_rows(qapp: QApplication) -> None:
    tab = GlobalFitTab()
    tab._set_composite_model(CompositeModel(["Gaussian", "Constant"], operators=["+"]))
    assert tab._param_table.rowCount() == 3


def test_global_tab_inherits_model_and_average_values_from_single_fits(
    qapp: QApplication,
    dataset: MuonDataset,
) -> None:
    tab = GlobalFitTab()
    d2 = MuonDataset(dataset.time, dataset.asymmetry, dataset.error, {"run_number": 102})
    tab.set_datasets([dataset, d2])

    model = CompositeModel(["Gaussian", "Constant"], operators=["+"])
    p1 = ParameterSet([Parameter("A_1", 0.20), Parameter("sigma", 1.1), Parameter("A_bg", 0.01)])
    p2 = ParameterSet([Parameter("A_1", 0.30), Parameter("sigma", 1.5), Parameter("A_bg", 0.03)])
    r1 = FitResult(success=True, parameters=p1)
    r2 = FitResult(success=True, parameters=p2)

    tab.register_single_fit_seed(101, model, r1)
    tab.register_single_fit_seed(102, model, r2)

    assert tab._composite_model.to_dict() == model.to_dict()

    value_by_name = {}
    for row in range(tab._param_table.rowCount()):
        name_item = tab._param_table.item(row, 0)
        value_item = tab._param_table.item(row, 1)
        assert name_item is not None
        assert value_item is not None
        pname = name_item.data(Qt.ItemDataRole.UserRole)
        value_by_name[pname] = float(value_item.text())

    assert value_by_name["A_1"] == pytest.approx(0.25)
    assert value_by_name["sigma"] == pytest.approx(1.3)
    assert value_by_name["A_bg"] == pytest.approx(0.02)


def test_global_fit_uses_inherited_local_values_per_run(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tab = GlobalFitTab()
    d2 = MuonDataset(dataset.time, dataset.asymmetry, dataset.error, {"run_number": 102})
    tab.set_datasets([dataset, d2])

    model = tab._composite_model
    p1 = ParameterSet([Parameter("A_1", 0.22), Parameter("Lambda", 0.40), Parameter("A_bg", 0.01)])
    p2 = ParameterSet([Parameter("A_1", 0.30), Parameter("Lambda", 0.85), Parameter("A_bg", 0.02)])
    tab.register_single_fit_seed(101, model, FitResult(success=True, parameters=p1))
    tab.register_single_fit_seed(102, model, FitResult(success=True, parameters=p2))

    # Enforce classification for this test case.
    row_by_name = {}
    for row in range(tab._param_table.rowCount()):
        name_item = tab._param_table.item(row, 0)
        assert name_item is not None
        row_by_name[name_item.data(Qt.ItemDataRole.UserRole)] = row

    global_combo = tab._param_table.cellWidget(row_by_name["A_1"], 2)
    local_combo = tab._param_table.cellWidget(row_by_name["Lambda"], 2)
    fixed_combo = tab._param_table.cellWidget(row_by_name["A_bg"], 2)
    assert isinstance(global_combo, QComboBox)
    assert isinstance(local_combo, QComboBox)
    assert isinstance(fixed_combo, QComboBox)
    global_combo.setCurrentText("Global")
    local_combo.setCurrentText("Local")
    fixed_combo.setCurrentText("Fixed")

    captured: dict[str, object] = {}

    class _DummySignal:
        def connect(self, *_args, **_kwargs):
            return None

    class _FakeThread:
        def __init__(self):
            self.started = _DummySignal()
            self.finished = _DummySignal()

        def start(self):
            return None

        def quit(self):
            return None

        def wait(self):
            return None

        def deleteLater(self):
            return None

    class _FakeWorker:
        def __init__(
            self,
            _fit_engine,
            _datasets,
            _model_fn,
            _global_params,
            _local_params,
            initial_params,
        ):
            captured["initial_params"] = initial_params
            self.finished = _DummySignal()
            self.error = _DummySignal()

        def moveToThread(self, _thread):
            return None

        def run(self):
            return None

        def deleteLater(self):
            return None

    monkeypatch.setattr(fit_panel_module, "QThread", _FakeThread)
    monkeypatch.setattr(fit_panel_module, "GlobalFitWorker", _FakeWorker)

    tab._run_global_fit()

    initial_params = captured["initial_params"]
    pset_101 = initial_params[101]
    pset_102 = initial_params[102]

    assert pset_101["Lambda"].value == pytest.approx(0.40)
    assert pset_102["Lambda"].value == pytest.approx(0.85)
    # Global/fixed parameters are seeded from per-run averages.
    assert pset_101["A_1"].value == pytest.approx(0.26)
    assert pset_102["A_1"].value == pytest.approx(0.26)
    assert pset_101["A_bg"].value == pytest.approx(0.015)
    assert pset_102["A_bg"].value == pytest.approx(0.015)


def test_fit_panel_restores_single_fit_state_per_dataset(qapp: QApplication, dataset: MuonDataset) -> None:
    panel = FitPanel()
    d1 = dataset
    d2 = MuonDataset(dataset.time, dataset.asymmetry, dataset.error, {"run_number": 102})

    panel.set_dataset(d1)
    panel._single_tab._result_label.setText("fit for run 101")
    panel._single_tab._param_table.item(0, 1).setText("0.123")

    panel.set_dataset(d2)
    panel._single_tab._result_label.setText("fit for run 102")
    panel._single_tab._param_table.item(0, 1).setText("0.456")

    panel.set_dataset(d1)
    assert "fit for run 101" in panel._single_tab._result_label.text()
    assert float(panel._single_tab._param_table.item(0, 1).text()) == pytest.approx(0.123)

    panel.set_dataset(d2)
    assert "fit for run 102" in panel._single_tab._result_label.text()
    assert float(panel._single_tab._param_table.item(0, 1).text()) == pytest.approx(0.456)


def test_fit_panel_single_state_roundtrip_preserves_per_run_states(
    qapp: QApplication,
    dataset: MuonDataset,
) -> None:
    panel = FitPanel()
    d1 = dataset
    d2 = MuonDataset(dataset.time, dataset.asymmetry, dataset.error, {"run_number": 102})

    panel.set_dataset(d1)
    panel._single_tab._result_label.setText("saved fit 101")
    panel.set_dataset(d2)
    panel._single_tab._result_label.setText("saved fit 102")

    saved = panel.get_single_state()
    assert isinstance(saved.get("states_by_run"), dict)
    assert "101" in saved["states_by_run"]
    assert "102" in saved["states_by_run"]

    restored = FitPanel()
    restored.set_dataset(d1)
    restored.restore_single_state(saved)
    assert "saved fit 101" in restored._single_tab._result_label.text()

    restored.set_dataset(d2)
    assert "saved fit 102" in restored._single_tab._result_label.text()


def test_fit_panel_global_fit_results_seed_single_state_per_run(
    qapp: QApplication,
    dataset: MuonDataset,
) -> None:
    panel = FitPanel()
    d1 = dataset
    d2 = MuonDataset(dataset.time, dataset.asymmetry, dataset.error, {"run_number": 102})

    panel.set_dataset(d1)

    model = panel._global_tab._composite_model
    pnames = model.param_names

    def _fit_result(values: list[float]) -> FitResult:
        params = ParameterSet(
            [Parameter(name=name, value=value) for name, value in zip(pnames, values, strict=False)]
        )
        return FitResult(
            success=True,
            chi_squared=2.0,
            reduced_chi_squared=1.0,
            parameters=params,
            uncertainties={name: 0.01 for name in pnames},
        )

    results = {
        101: (_fit_result([0.11, 0.22, 0.33]), (np.array([0.0, 1.0]), np.array([0.2, 0.1])), []),
        102: (_fit_result([0.44, 0.55, 0.66]), (np.array([0.0, 1.0]), np.array([0.2, 0.1])), []),
    }
    panel.register_global_fit_results(results)

    assert "Global fit" in panel._single_tab._result_label.text()
    assert float(panel._single_tab._param_table.item(0, 1).text()) == pytest.approx(0.11)

    panel.set_dataset(d2)
    assert "Global fit" in panel._single_tab._result_label.text()
    assert float(panel._single_tab._param_table.item(0, 1).text()) == pytest.approx(0.44)

    saved = panel.get_single_state()
    assert "101" in saved.get("states_by_run", {})
    assert "102" in saved.get("states_by_run", {})


def test_fit_panel_share_single_function_state_to_other_runs(
    qapp: QApplication,
    dataset: MuonDataset,
) -> None:
    panel = FitPanel()
    d1 = dataset
    d2 = MuonDataset(dataset.time, dataset.asymmetry, dataset.error, {"run_number": 102})
    d3 = MuonDataset(dataset.time, dataset.asymmetry, dataset.error, {"run_number": 103})

    panel.set_dataset(d1)
    panel._single_tab._param_table.item(0, 1).setText("0.777")
    panel._single_tab._result_label.setText("source fit result")

    copied = panel.share_single_function_state(101, [102, 103])
    assert copied == 2

    panel.set_dataset(d2)
    assert float(panel._single_tab._param_table.item(0, 1).text()) == pytest.approx(0.777)
    assert panel._single_tab._result_label.text() == "No fit performed yet"

    panel.set_dataset(d3)
    assert float(panel._single_tab._param_table.item(0, 1).text()) == pytest.approx(0.777)
    assert panel._single_tab._result_label.text() == "No fit performed yet"


def test_fit_panel_clear_fits_for_runs_removes_cached_fit_state(
    qapp: QApplication,
    dataset: MuonDataset,
) -> None:
    panel = FitPanel()
    d1 = dataset
    d2 = MuonDataset(dataset.time, dataset.asymmetry, dataset.error, {"run_number": 102})

    panel.set_dataset(d1)
    panel._single_tab._result_label.setText("fit for run 101")
    panel._single_state_by_run[101] = panel._single_tab.get_state()

    panel.set_dataset(d2)
    panel._single_tab._result_label.setText("fit for run 102")
    panel._single_state_by_run[102] = panel._single_tab.get_state()

    panel._global_tab._single_fit_seed_by_run[101] = {"model": {}, "values": {"A": 0.1}}
    panel._global_tab._single_fit_seed_by_run[102] = {"model": {}, "values": {"A": 0.2}}

    cleared = panel.clear_fits_for_runs([101])

    assert cleared == 1
    assert 101 not in panel._single_state_by_run
    assert 101 not in panel._global_tab._single_fit_seed_by_run
    assert 102 in panel._single_state_by_run
    assert 102 in panel._global_tab._single_fit_seed_by_run
