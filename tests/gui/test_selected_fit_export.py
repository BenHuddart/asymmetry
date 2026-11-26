"""Text and GLE exports follow the plot's fit visibility per run."""

import numpy as np
import pytest
from PySide6.QtWidgets import QFileDialog, QMessageBox

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.gui.panels.plot_panel import SINGLE_FIT_ID, PlotPanel

pytestmark = pytest.mark.gui

_SERIES = "series"
_MODELS = {SINGLE_FIT_ID: "Model A", _SERIES: "Model B"}
_VALUES = {(11, SINGLE_FIT_ID): 1, (11, _SERIES): 2, (22, _SERIES): 4}
_TIME = np.arange(7, dtype=float)


@pytest.fixture
def fit_plot(qapp):
    panel = PlotPanel()
    datasets = [
        MuonDataset(_TIME, _TIME / run, np.full_like(_TIME, 0.1), {"run_number": run})
        for run in (11, 22)
    ]
    for fit_id, model in _MODELS.items():
        panel.set_global_fits(
            {
                run: (_TIME, np.full_like(_TIME, value), "Shared label", [], None, model)
                for (run, stored_id), value in _VALUES.items()
                if stored_id == fit_id
            },
            fit_id=fit_id,
        )
    panel.set_active_fit_id(_SERIES)
    yield panel, datasets
    panel.close()
    panel.deleteLater()


def assert_curves(paths, selections, time):
    by_model = {(run, _MODELS[fit_id]): fit_id for run, fit_id in selections}
    assert len(paths) == len(by_model)
    for path in paths:
        header = dict(
            line[2:].split(": ", 1) for line in path.read_text().splitlines() if ": " in line
        )
        run = int(header["run_number"])
        fit_id = by_model.pop((run, header["fit_function"]))
        assert header["fit_id"] == fit_id
        assert header["fit_label"] == "Shared label"
        np.testing.assert_array_equal(
            np.loadtxt(path, comments="!", ndmin=2),
            np.column_stack((time, np.full_like(time, _VALUES[run, fit_id]))),
        )
    assert not by_model


@pytest.mark.parametrize("content", ["fit", "data_fit"])
@pytest.mark.parametrize("shown", [[], [SINGLE_FIT_ID], [_SERIES], [SINGLE_FIT_ID, _SERIES]])
def test_text_exports_selected_fits(fit_plot, tmp_path, monkeypatch, content, shown):
    panel, datasets = fit_plot
    panel.plot_dataset(datasets[0])
    panel.set_shown_fits(11, shown)
    panel._x_min.setValue(2.0)
    panel._x_max.setValue(5.0)
    target = tmp_path / "run.dat"
    monkeypatch.setattr(panel, "_prompt_text_export_options", lambda: (content, True))
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_a, **_k: (str(target), ""))
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda _parent, _title, text: warnings.append(text))
    monkeypatch.setattr(panel, "_show_export_result_dialog", lambda *_a: None)

    panel.export_plotted_data_as_text()

    paths = sorted(tmp_path.glob("*.fit"))
    assert_curves(paths, [(11, fit_id) for fit_id in shown], _TIME[2:6])
    if len(shown) == 1:
        assert paths == [target.with_suffix(".fit")]
    elif len(shown) == 2:
        assert [path.name for path in paths] == ["run__shared_label.fit", "run__shared_label_2.fit"]
    assert sorted(tmp_path.glob("*.dat")) == ([target] if content == "data_fit" else [])
    if content == "data_fit":
        np.testing.assert_array_equal(np.loadtxt(target, comments="!")[:, 0], _TIME[2:6])
    if content == "fit" and not shown:
        assert warnings and "no files" in warnings[0].lower()
    else:
        assert not warnings


def test_multi_run_text_and_gle_preserve_fit_identity(fit_plot, tmp_path, monkeypatch):
    glp = pytest.importorskip("gleplot")
    from asymmetry.gui.utils.gle_export import extract_gle_data_dependencies

    panel, datasets = fit_plot
    panel.plot_datasets(datasets)
    panel.set_shown_fits(11, [SINGLE_FIT_ID, _SERIES])
    panel.set_shown_fits(22, [_SERIES])
    monkeypatch.setattr(panel, "_prompt_text_export_options", lambda: ("data_fit", False))
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_a, **_k: str(tmp_path))
    monkeypatch.setattr(panel, "_show_export_result_dialog", lambda *_a: None)

    panel.export_plotted_data_as_text()

    assert_curves(list(tmp_path.glob("*.fit")), _VALUES, _TIME)
    assert len(list(tmp_path.glob("*.dat"))) == 2

    gle_folder = tmp_path / "gle"
    gle_folder.mkdir()
    script = gle_folder / "comparison.gle"
    panel._gle_snapshot(panel._collect_export_payloads()).build(glp, script, gle_folder)

    assert_curves(list(gle_folder.glob("*.fit")), _VALUES, _TIME)
    dependencies = extract_gle_data_dependencies(script)
    assert len(dependencies) == len(set(dependencies)) == 5
    fit_data = [
        np.loadtxt(gle_folder / name, skiprows=1) for name in dependencies if "_fit" in name
    ]
    assert [list(rows[:, 1]) for rows in fit_data] == [[value] * len(_TIME) for value in (2, 1, 4)]

    panel.set_shown_fits(11, [])
    panel.set_shown_fits(22, [])
    empty_folder = tmp_path / "hidden"
    empty_folder.mkdir()
    script = empty_folder / "comparison.gle"
    panel._gle_snapshot(panel._collect_export_payloads()).build(glp, script, empty_folder)

    assert len(extract_gle_data_dependencies(script)) == 2
    assert list(empty_folder.glob("*.fit")) == []


def test_vector_export_uses_the_mixed_run_fit_slot(fit_plot):
    panel, (plain, other) = fit_plot
    run = plain.run_number
    projection = MuonDataset(
        plain.time,
        plain.asymmetry,
        plain.error,
        {"run_number": run, "grouping": {"vector_axis": "P_y"}},
    )
    panel.set_global_fits(
        {run: (_TIME, np.full_like(_TIME, 3), "Projection only", [], None, "Projected", "P_y")},
        fit_id=_SERIES,
    )
    panel._current_polarization_axis = "ALL"
    panel.plot_vector_subplots({"P_y": [plain, projection], "P_z": [other]})
    lines = [
        line
        for line, _, _ in panel._time_view.fits
        if line.axes is panel._subplot_axes_by_polarization["P_y"]
    ]

    groups = dict(panel._export_axis_groups())
    for payload, line, model, value in zip(
        groups["P_y"], lines, ("Model B", "Projected"), (2, 3), strict=True
    ):
        (fit,) = payload["fits"]
        assert fit["metadata"]["fit_function"] == model
        np.testing.assert_array_equal(fit["y"], line.get_ydata())
        np.testing.assert_array_equal(fit["y"], np.full_like(_TIME, value))


def test_gle_reserves_normalised_fit_names_across_runs(fit_plot, tmp_path):
    glp = pytest.importorskip("gleplot")
    from asymmetry.gui.utils.gle_export import extract_gle_data_dependencies

    panel, datasets = fit_plot
    for dataset, label in zip(datasets, ("A", "A Main"), strict=True):
        dataset.metadata["run_label"] = label
    panel.set_global_fits(
        {
            run: (_TIME, np.full_like(_TIME, value), label)
            for run, label, value in ((11, "Main Model", 3), (22, "Model", 4))
        },
        fit_id=_SERIES,
    )
    panel.set_global_fits(
        {run: (_TIME, np.full_like(_TIME, value), "Other") for run, value in ((11, 5), (22, 6))}
    )
    panel.plot_datasets(datasets)
    for dataset in datasets:
        panel.set_shown_fits(dataset.run_number, [_SERIES, SINGLE_FIT_ID])

    script = tmp_path / "fits.gle"
    panel._gle_snapshot(panel._collect_export_payloads()).build(glp, script, tmp_path)

    names = extract_gle_data_dependencies(script)
    fits, data = names[:4], names[4:]
    assert len(names) == len(set(names)) == 6
    assert fits[2] == "a_main_main_model_fit_2.dat"
    for name, y in zip(fits, (3, 5, 4, 6), strict=True):
        np.testing.assert_array_equal(np.loadtxt(tmp_path / name, skiprows=1)[:, 1], y)
    for name, dataset in zip(data, datasets, strict=True):
        np.testing.assert_allclose(
            np.loadtxt(tmp_path / name, comments="!")[:, 1], dataset.asymmetry
        )
