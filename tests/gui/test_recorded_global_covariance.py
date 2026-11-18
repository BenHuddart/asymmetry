"""Recorded global-fit series retain covariance for derived errors through projects."""

import json

import numpy as np
import pytest

from asymmetry.core import computation
from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting import FitEngine, Parameter, ParameterSet
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.composite_parameters import CompositeParameterDefinition
from asymmetry.core.project.schema import load_project, save_project

pytestmark = pytest.mark.gui


@pytest.mark.parametrize("case", ["valid", "bound", "singular", "historical"])
def test_launched_global_series_retains_member_cross_terms_after_project_reopen(
    qapp, tmp_path, case
):
    pytest.importorskip("PySide6")
    from asymmetry.gui.mainwindow import MainWindow
    from asymmetry.gui.panels.fit.global_tab import FitLaunch
    from docs.screenshots.scenarios._base import auto_dismiss_modals
    from tests._qt_helpers import wait_for

    windows = []
    with auto_dismiss_modals():
        try:
            window = MainWindow()
            windows.append(window)
            window._on_fit()
            t = np.linspace(0.02, 8, 90)
            rng = np.random.default_rng(41824)
            datasets = [
                MuonDataset(
                    time=t,
                    asymmetry=(18 + run) * np.exp(-0.35 * t) + 0.5 + rng.normal(0, 0.08, len(t)),
                    error=np.full(len(t), 0.08),
                    metadata={"run_number": run, "temperature": float(run), "field": 0.0},
                )
                for run in (1, 2)
            ]
            for ds in datasets:
                window._data_browser.add_dataset(ds)
            window._fit_panel.set_datasets(datasets)
            tab = window._fit_panel._global_tab
            window._fit_panel._tabs.setCurrentWidget(tab)
            model = CompositeModel(["Exponential", "Constant"])
            tab._set_composite_model(model, seed_values={"A_1": "20", "Lambda": ".3", "A_bg": ".2"})
            for row, name in enumerate(model.param_names):
                tab._param_table.cellWidget(row, 2).setCurrentText(
                    "Local" if name == "A_1" else "Global"
                )
            initial = {
                run: ParameterSet(
                    [
                        Parameter(
                            "A_1",
                            8 if case == "bound" else 20,
                            min=0,
                            max=10 if case == "bound" else 40,
                        ),
                        Parameter(
                            "Lambda",
                            0 if case == "singular" else 0.3,
                            min=0,
                            max=2,
                            fixed=case == "singular",
                        ),
                        Parameter("A_bg", 0.2, min=-3, max=3),
                    ]
                )
                for run in (1, 2)
            }
            results, shared = FitEngine().global_fit(
                datasets,
                model.function,
                ["Lambda", "A_bg"],
                ["A_1"],
                initial,
                strategy="profiled",
            )
            if case == "historical":
                for result in results.values():
                    result.covariance = None
                    result.covariance_accurate = False
                    result.uncertainties = {}
                    result.computation_revision = None
            assert all(result.success for result in results.values())
            assert all(
                result.covariance_accurate == (case == "valid") for result in results.values()
            )
            window._on_global_fit_started()
            tab._emit_global_fit_success(
                launch=FitLaunch(
                    model=model, global_params=("Lambda", "A_bg"), datasets=tuple(datasets)
                ),
                results_dict=results,
                successful=results,
                fitted_global=shared,
                detail_html=tab._batch_detail_html(results, results, {}, ["Lambda", "A_bg"], ""),
            )
            panel = window._fit_parameters_panel
            wait_for(lambda: len(panel._rows) == 2, qapp)
            panel._composite_parameters = [
                CompositeParameterDefinition("A_total", "A_1+A_bg"),
                CompositeParameterDefinition("lambda_offset", "Lambda+1"),
            ]
            panel._apply_composite_parameters_to_rows(panel._rows, panel._composite_parameters)
            panel._refresh_after_composite_change(preferred_selected=["A_total"])
            expected = {}
            for run, result in results.items():
                if case != "valid":
                    expected[run] = float("nan")
                    assert result.covariance is None
                    continue
                gradient = np.array(
                    [float(name in {"A_1", "A_bg"}) for name in result.covariance_parameters]
                )
                expected[run] = float(np.sqrt(gradient @ result.covariance @ gradient))
                diagonal_only = np.sqrt(
                    result.uncertainties["A_1"] ** 2 + result.uncertainties["A_bg"] ** 2
                )
                assert not np.isclose(expected[run], diagonal_only, rtol=0.01)
            for row in panel._rows:
                assert row.values["A_total"] == pytest.approx(
                    sum(row.values[name] for name in ("A_1", "A_bg"))
                )
                assert row.errors["A_total"] == pytest.approx(
                    expected[row.run_number], rel=1e-10, nan_ok=True
                )
                assert (row.covariance is not None) == (case == "valid")
                if case == "singular":
                    assert row.errors["Lambda"] == 0.0
                    assert row.errors["lambda_offset"] == 0.0
            path = tmp_path / "global-covariance.asymp"
            save_project(window.collect_project_state(), path, backup=False)
            json.loads(path.read_text(), parse_constant=lambda value: pytest.fail(value))
            restored = MainWindow()
            windows.append(restored)
            restored.restore_project_state(load_project(path), str(path))
            restored_panel = restored._fit_parameters_panel
            wait_for(lambda: len(restored_panel._rows) == 2, qapp)
            for row in restored_panel._rows:
                assert row.values["A_total"] == pytest.approx(
                    sum(row.values[name] for name in ("A_1", "A_bg"))
                )
                assert row.warnings == results[row.run_number].warnings
                if case == "singular":
                    assert row.errors["lambda_offset"] == 0.0
                assert (row.covariance is not None) == (case == "valid")
                assert row.errors["A_total"] == pytest.approx(
                    expected[row.run_number], rel=1e-10, nan_ok=True
                )
                assert row.computation_revision == (
                    None if case == "historical" else computation.COMPUTATION_REVISION
                )
        finally:
            for window in reversed(windows):
                window._dirty = False
                window.close()
                window.deleteLater()
