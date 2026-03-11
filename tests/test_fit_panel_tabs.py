"""Focused tests for SingleFitTab and GlobalFitTab logic."""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QComboBox

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.engine import FitResult
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.gui.panels.fit_panel import GlobalFitTab, SingleFitTab


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
