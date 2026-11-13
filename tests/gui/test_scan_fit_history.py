"""Actual-open preservation of scalar-scan numerical history and attribution."""

from copy import deepcopy

import numpy as np
import pytest
from PySide6.QtWidgets import QTableWidgetItem

import asymmetry.gui.mainwindow as mw
from asymmetry.core import computation
from asymmetry.core.project import load_project, save_project
from asymmetry.core.project.json_values import encode_non_finite
from asymmetry.gui.mainwindow import MainWindow
from tests.gui.test_integral_scan_gui import _ds, _set_regions
from tests.gui.test_rf_scan_panel_gui import _rf_datasets

pytestmark = pytest.mark.gui


def build_fitted_scan(window, kind, *, source_root=None):
    """Owned Gaussian dip or RF doublet with independently specified parameters."""
    window._plot_workspace.set_active_view("integral_scan")
    if kind == "rf":
        datasets = _rf_datasets()
        window._alc_fit_panel._rf_difference_check.setChecked(True)
    else:
        datasets = []
        for i, field in enumerate(np.linspace(50.0, 350.0, 31)):
            a = 0.1 - 0.05 * np.exp(-0.5 * ((field - 200.0) / 20.0) ** 2)
            datasets.append(_ds(11 + i, 100 * (1 + a), 100 * (1 - a), float(field)))
    if source_root is not None:
        from asymmetry.core.io import load
        from asymmetry.core.io.nexus_writer import write_nexus_v1
        from asymmetry.core.transform.preparation import histogram_observation

        loaded = []
        for dataset in datasets:
            path = source_root / f"owned-{dataset.run_number}.nxs"
            write_nexus_v1(dataset.run, path)
            current = load(str(path))
            current.run.grouping.update(
                {
                    key: dataset.run.grouping[key]
                    for key in (
                        "groups",
                        "forward_group",
                        "backward_group",
                        "alpha",
                        "first_good_bin",
                        "last_good_bin",
                        "bunching_factor",
                    )
                }
            )
            current.histogram_observation = histogram_observation(current.run)
            loaded.append(current)
            window._data_browser.add_dataset(current)
        datasets = loaded
    window._fit_panel.set_datasets(datasets)
    window._on_scan_requested()
    if kind == "rf":
        window._on_fit_rf()
        assert window._alc_rf_fitted
    else:
        _set_regions(window._alc_scan_view, [(50.123456, 125.0), (275.0, 350.0)])
        window._on_fit_baseline()
        if kind == "peaks":
            window._alc_scan_view._add_peak("Gaussian")
            window._on_fit_peaks()
    window._sync_alc_series_extra()
    return window._project_model.batch(window._alc_scan_series_id)


def forbid_optimizers(monkeypatch):
    for name in ("fit_scan_baseline", "fit_scan_peaks", "fit_rf_resonance"):
        monkeypatch.setattr(mw, name, lambda *a, **kw: pytest.fail("Implicit scan optimizer call"))


@pytest.mark.parametrize("revision", [None, "pre-current", computation.COMPUTATION_REVISION])
@pytest.mark.parametrize("kind", ["baseline", "peaks", "rf"])
def test_reopen_does_not_refit_history(revision, kind, tmp_path, qapp, monkeypatch):
    monkeypatch.setattr(computation, "COMPUTATION_REVISION", revision)
    original = MainWindow()
    series = build_fitted_scan(original, kind)
    captured = deepcopy(series.to_dict())
    view = original._alc_scan_view
    baseline_curve = deepcopy(view._baseline_curve)
    fit_curve = deepcopy(view._fit_curve)
    readouts = (view._peaks_results.text(), view._rf_results.text())
    records = view.fit_history.records
    result = records[-1].fit_result()
    # Independent analytic inputs establish capture/display scales and positions.
    if kind == "baseline":
        np.testing.assert_allclose(records[0].prediction, 10.0, atol=0.002)
    elif kind == "peaks":
        assert result.parameters["B0"].value == pytest.approx(200.0, abs=0.03)
        assert result.parameters["Bwid"].value == pytest.approx(20.0, abs=0.05)
        assert result.parameters["f"].value == pytest.approx(-5.0, abs=0.02)
        assert "B₀ = 200" in readouts[0]
    else:
        assert result.parameters["A_mu"].value == pytest.approx(515.0, abs=0.02)
        assert result.parameters["A_p"].value == pytest.approx(124.0, abs=0.02)
        assert "515.00" in readouts[1]
    project = tmp_path / "owned-history.asymp"
    save_project(original.collect_project_state(), project)
    captured = load_project(project)["batches"][0]
    original._clear_dirty()
    original.close()
    monkeypatch.setattr(computation, "COMPUTATION_REVISION", "new-execution")
    forbid_optimizers(monkeypatch)
    for _ in range(2):
        reopened = MainWindow()
        reopened._open_project_file(str(project))
        restored = reopened._project_model.batch(series.batch_id)
        assert restored is not None
        assert restored.to_dict()["results_by_run"] == captured["results_by_run"]
        assert restored.extra.get("computation_revision") == revision
        assert restored.extra["fit_history"] == captured["extra"]["fit_history"]
        restored_view = reopened._alc_scan_view
        assert (restored_view._peaks_results.text(), restored_view._rf_results.text()) == readouts
        for expected, actual in (
            (baseline_curve, restored_view._baseline_curve),
            (fit_curve, restored_view._fit_curve),
        ):
            if expected is None:
                assert actual is None
            else:
                np.testing.assert_array_equal(actual, expected)
        save_project(reopened.collect_project_state(), project)
        reopened._clear_dirty()
        reopened.close()


@pytest.mark.parametrize("kind", ["baseline", "peaks", "rf"])
def test_legacy_settings_require_explicit_fit(kind, tmp_path, qapp, monkeypatch):
    original = MainWindow()
    series = build_fitted_scan(original, kind)
    state = original.collect_project_state()
    state["schema_version"] = 39
    extra = state["batches"][0]["extra"]
    extra.pop("fit_history")
    extra["baseline_fitted"] = kind != "rf"
    extra["peaks_fitted"] = kind in {"peaks", "rf"}
    extra["rf_fitted"] = kind == "rf"
    retained = deepcopy(extra)
    project = tmp_path / "legacy.asymp"
    save_project(state, project)
    original._clear_dirty()
    original.close()
    with monkeypatch.context() as guard:
        forbid_optimizers(guard)
        for _ in range(2):
            restored = MainWindow()
            restored._open_project_file(str(project))
            view = restored._alc_scan_view
            assert not view.fit_history.active
            assert not view.fit_history.records
            assert kind in view.fit_history.missing
            assert "no captured numerical result" in view._fit_history_status.text()
            assert view._fit_curve is None
            assert restored._alc_baseline is None
            for key in ("peaks", "regions", "rf_nu", "rf_a_mu", "rf_a_p"):
                assert view.analysis_state()[key] == retained[key]
            save_project(restored.collect_project_state(), project)
            restored._clear_dirty()
            restored.close()
    restored = MainWindow()
    restored._open_project_file(str(project))
    if kind == "rf":
        restored._on_fit_rf()
    else:
        restored._on_fit_baseline()
        if kind == "peaks":
            restored._on_fit_peaks()
    assert kind in restored._alc_scan_view.fit_history.active
    assert not restored._alc_scan_view.fit_history.missing
    assert restored._alc_scan_view.fit_history.legacy_settings == retained
    assert restored._project_model.batch(series.batch_id).results_by_run == series.results_by_run
    restored._clear_dirty()
    restored.close()


@pytest.mark.parametrize("kind", ["baseline", "peaks", "rf"])
def test_explicit_refit_appends_execution(kind, tmp_path, qapp, monkeypatch):
    original = MainWindow()
    build_fitted_scan(original, kind)
    project = tmp_path / "history.asymp"
    save_project(original.collect_project_state(), project)
    original._clear_dirty()
    original.close()
    restored = MainWindow()
    restored._open_project_file(str(project))
    history = restored._alc_scan_view.fit_history
    previous = deepcopy(history.to_dict()["records"])
    monkeypatch.setattr(computation, "COMPUTATION_REVISION", "explicit-refit")
    {
        "baseline": restored._on_fit_baseline,
        "peaks": restored._on_fit_peaks,
        "rf": restored._on_fit_rf,
    }[kind]()
    assert history.to_dict()["records"][:-1] == previous
    assert history.records[-1].result["computation_revision"] == "explicit-refit"
    assert history.records[-1].support["computation_revision"] != "explicit-refit"
    restored._clear_dirty()
    restored.close()


@pytest.mark.parametrize(
    "edit",
    [
        "axis",
        "derivative",
        "exclude",
        "region",
        "model",
        "peak",
        "rf_nu",
        "rf_mu",
        "rf_p",
        "rf_axis",
        "rf_derivative",
        "rf_exclude",
    ],
)
def test_settings_detach_history(edit, tmp_path, qapp, monkeypatch):
    window = MainWindow()
    kind = "rf" if edit.startswith("rf_") else "peaks"
    build_fitted_scan(window, kind)
    view = window._alc_scan_view
    assert kind in view.fit_history.active
    captured = deepcopy(view.fit_history.to_dict()["records"])
    option = edit.removeprefix("rf_")
    forbid_optimizers(monkeypatch)
    if option == "axis":
        view._x_combo.setCurrentIndex(2)
    elif option == "derivative":
        view._derivative_check.setChecked(True)
    elif option == "exclude":
        window._on_alc_point_toggled(window._alc_scan_points[0]["run"])
    elif option == "region":
        view._regions_table.setItem(0, 0, QTableWidgetItem("60"))
    elif option == "model":
        view._baseline_model_combo.setCurrentText("Constant")
    elif option == "peak":
        view._peaks_table.setItem(0, 1, QTableWidgetItem("210"))
    else:
        spin = {"rf_nu": view._rf_nu_spin, "rf_mu": view._rf_a_mu_spin, "rf_p": view._rf_a_p_spin}[
            edit
        ]
        spin.setValue(spin.value() + 1.0)
    assert kind not in view.fit_history.active
    assert view.fit_history.to_dict()["records"] == captured
    assert view._fit_curve is None
    project = tmp_path / "edited.asymp"
    save_project(window.collect_project_state(), project)
    window._clear_dirty()
    window.close()
    reopened = MainWindow()
    reopened._open_project_file(str(project))
    assert kind not in reopened._alc_scan_view.fit_history.active
    assert encode_non_finite(
        reopened._alc_scan_view.fit_history.to_dict()["records"]
    ) == encode_non_finite(captured)
    assert reopened._alc_scan_view._fit_curve is None
    reopened._clear_dirty()
    reopened.close()


@pytest.mark.parametrize(
    "choice", ["missing", "Open History Only", "Accept Replacement", "relocate"]
)
def test_scan_history_survives_source_decisions(choice, tmp_path, qapp, monkeypatch):
    import h5py
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    from asymmetry.core.project import load_prepared_project, verify_project_sources

    original = MainWindow()
    series = build_fitted_scan(original, "peaks", source_root=tmp_path)
    project = tmp_path / "sources.asymp"
    save_project(original.collect_project_state(), project)
    before = load_project(project)["batches"][0]
    original._clear_dirty()
    original.close()
    path = tmp_path / "owned-11.nxs"
    if choice == "missing":
        path.unlink()
    elif choice == "relocate":
        folder = tmp_path / "relocated"
        folder.mkdir()
        path.rename(folder / path.name)
    else:
        with h5py.File(path, "r+") as handle:
            # Own synthetic acquisition: retain the file's readable layout.
            counts = handle["run/histogram_data_1/counts"]
            counts[0, :] = counts[0, :] + 50

    def decide(box):
        if box.windowTitle() == "Data Files Not Found":
            return (
                QMessageBox.StandardButton.Yes
                if choice == "relocate"
                else QMessageBox.StandardButton.No
            )
        assert box.windowTitle() == "Source Content Changed"
        box._chosen = next(button for button in box.buttons() if button.text() == choice)
        return 0

    monkeypatch.setattr(QMessageBox, "exec", decide)
    monkeypatch.setattr(QMessageBox, "clickedButton", lambda box: box._chosen)
    monkeypatch.setattr(QFileDialog, "exec", lambda dialog: QFileDialog.DialogCode.Accepted)
    monkeypatch.setattr(QFileDialog, "selectedFiles", lambda dialog: [str(tmp_path / "relocated")])
    forbid_optimizers(monkeypatch)
    reopened = MainWindow()
    reopened._open_project_file(str(project))
    for _ in range(2):
        restored = reopened._project_model.batch(series.batch_id)
        assert restored is not None
        assert restored.to_dict()["results_by_run"] == before["results_by_run"]
        assert restored.extra["fit_history"] == before["extra"]["fit_history"]
        assert reopened._alc_scan_view._fit_curve is not None
        save_project(reopened.collect_project_state(), project)
        reopened._open_project_file(str(project))
    if choice == "Accept Replacement":
        facts = verify_project_sources(load_prepared_project(project), project).facts
        assert any(f.source.owner.active and f.status == "matched" for f in facts)
        assert any(not f.source.owner.active and f.status == "changed" for f in facts), [
            (f.source.owner, f.status, f.source.expected) for f in facts
        ]
    reopened._clear_dirty()
    reopened.close()


@pytest.mark.parametrize("kind", ["baseline", "peaks", "rf"])
def test_changed_captured_points_do_not_apply_saved_fit(kind, tmp_path, qapp, monkeypatch):
    window = MainWindow()
    build_fitted_scan(window, kind)
    state = window.collect_project_state()
    batch = state["batches"][0]
    first = next(iter(batch["results_by_run"].values()))
    first["parameters"][window._SCAN_QUANTITY] += 1.0
    history = deepcopy(batch["extra"]["fit_history"])
    project = tmp_path / "changed-points.asymp"
    save_project(state, project)
    window._clear_dirty()
    window.close()
    forbid_optimizers(monkeypatch)
    reopened = MainWindow()
    reopened._open_project_file(str(project))
    view = reopened._alc_scan_view
    assert not view.fit_history.active
    assert encode_non_finite(view.fit_history.to_dict()["records"]) == encode_non_finite(
        history["records"]
    )
    assert view._fit_curve is None
    assert view._baseline_curve is None
    reopened._clear_dirty()
    reopened.close()


@pytest.mark.parametrize(
    "target",
    ["prediction", "covariance_names", "parameters", "covariance_shape", "statistic", "error"],
)
def test_malformed_fit_rejected_before_application(target, tmp_path, qapp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from asymmetry.core.project import ProjectValidationError, prepare_project

    window = MainWindow()
    build_fitted_scan(window, "peaks")
    state = window.collect_project_state()
    record = state["batches"][0]["extra"]["fit_history"]["records"][-1]
    if target == "prediction":
        record["prediction"].pop()
    elif target == "covariance_names":
        record["covariance"][0][0] = "not_a_fitted_parameter"
    elif target == "parameters":
        record["result"]["parameters"][0].pop("value")
    elif target == "statistic":
        record["result"]["reduced_chi_squared"] = "not a number"
    elif target == "error":
        record["result"]["uncertainties"]["B0"] = "not a number"
    else:
        record["covariance"][1].pop()
    with pytest.raises(ProjectValidationError, match="extra.fit_history"):
        prepare_project(state)
    series = window._project_model.batch(window._alc_scan_series_id)
    curve = window._alc_scan_view._fit_curve
    path = tmp_path / "malformed.asymp"
    save_project(state, path)
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: errors.append(str(args[-1])))
    window._mark_dirty()
    window._open_project_file(str(path))
    assert errors
    assert window._dirty
    assert window._project_model.batch(window._alc_scan_series_id) is series
    assert window._alc_scan_view._fit_curve is curve
    window._clear_dirty()
    window.close()


def test_new_project_discards_baseline_dependency(qapp, monkeypatch):
    window = MainWindow()
    build_fitted_scan(window, "peaks")
    assert window._alc_baseline is not None
    window._on_new_project()
    assert window._alc_baseline is None
    assert window._alc_scan_series_id is None
    assert not window._alc_scan_view.fit_history.records
    window._alc_scan_view._add_peak("Gaussian")
    forbid_optimizers(monkeypatch)
    window._on_fit_peaks()
    assert "Fit a baseline first" in window._log_panel.to_plain_text()
    window._clear_dirty()
    window.close()
