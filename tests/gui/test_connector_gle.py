"""Full-source connector exports through gleplot and the native GLE compiler."""

import shutil
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QSize
from PySide6.QtPdf import QPdfDocument

from asymmetry.gui.utils.export import compile_gle
from asymmetry.gui.utils.plot_gle import PlotGleSnapshot

glp = pytest.importorskip("gleplot")
pytestmark = pytest.mark.gui


def connector_snapshot(*, stacked=False, offset=False):
    time = np.arange(12, dtype=float) * 0.1
    time[9:] += 0.4
    values = np.cos(2 * np.pi * 4 * time)
    error = np.full_like(time, 0.05)
    error[4] = np.nan
    payloads = [
        {
            "label": f"Run {run}",
            "data": {"t": time, "y": values + run, "err": error},
            "connector_low_mask": np.arange(12) < 2,
            "fit": {"t": time, "y": values + run, "label": "Model"},
            "waterfall_offset": float(run * 3) if offset else 0.0,
        }
        for run in (1, 2)
    ]
    axes = (("P_x", payloads, ""), ("P_y", payloads, "")) if stacked else ((None, payloads, ""),)
    return PlotGleSnapshot(
        axes=axes,
        x_limits=(-0.1, 1.7),
        y_limits={key: (-1.0, 12.0) for key, _, _ in axes},
        stacked=stacked,
        connect_points=True,
    )


def test_connected_export_keeps_gaps_confidence_fit_identity_and_raw_sidecars(tmp_path):
    snapshot = connector_snapshot(offset=True)
    path = tmp_path / "connectors.gle"
    snapshot.build(glp, path, tmp_path)
    script = path.read_text()
    assert "smooth" not in script
    assert 'key "Fit: Run 1 - Model"' in script
    assert 'key "Fit: Run 2 - Model"' in script
    guides = [line for line in script.splitlines() if " line " in line and " lstyle " in line]
    assert len(guides) == 4
    assert all("marker FCIRCLE" in line and " lstyle 3" in line for line in guides)
    assert sum('key "Run ' in line for line in guides) == 2
    points = [line for line in script.splitlines() if " err " in line]
    assert len(points) == 4
    assert sum("color GRAY" in line for line in points) == 2
    assert all(" line " not in line for line in points)
    main = np.genfromtxt(tmp_path / "run_1_main_guide_black.dat", skip_header=1, missing_values="*")
    np.testing.assert_allclose(
        main[np.isfinite(main[:, 0]), 0], [0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 1.3, 1.4, 1.5]
    )
    assert np.isnan(main[:, 0]).sum() == 2
    for _key, payloads, _title in snapshot.axes:
        for run, payload in enumerate(payloads, 1):
            raw = np.loadtxt(tmp_path / f"run_{run}_main.dat", comments="!")
            np.testing.assert_allclose(raw[:, :3], np.column_stack(list(payload["data"].values())))
    assert "+3" in script and "+6" in script


@pytest.mark.skipif(shutil.which("gle") is None, reason="GLE compiler not installed")
@pytest.mark.parametrize("layout", ["overlay", "waterfall", "stacked"])
@pytest.mark.parametrize("output_format", ["pdf", "eps", "svg"])
def test_connected_exports_compile_and_render(qapp, tmp_path, layout, output_format):
    snapshot = connector_snapshot(stacked=layout == "stacked", offset=layout == "waterfall")
    path = tmp_path / "connectors.gle"
    build = snapshot.build(glp, path, tmp_path)
    result = compile_gle(shutil.which("gle"), path, output_format, cwd=tmp_path, cairo=build.cairo)
    assert result.returncode == 0
    assert path.with_suffix("." + output_format).stat().st_size > 0
    if output_format == "pdf":
        document = QPdfDocument()
        document.load(str(path.with_suffix(".pdf")))
        assert document.pageCount() == 1
        image = document.render(0, QSize(1100, 850))
        assert not image.isNull()
        image.save(str(Path(tmp_path) / "preview.png"))


@pytest.mark.skipif(shutil.which("gle") is None, reason="GLE compiler not installed")
@pytest.mark.parametrize("layout", ["single", "overlay", "waterfall", "projections", "groups"])
def test_main_window_plot_snapshots_export_connected_fits(qapp, tmp_path, layout):
    from asymmetry.core.data.dataset import MuonDataset
    from asymmetry.gui.panels.plot_panel import PlotPanel

    panel = PlotPanel()
    time = np.linspace(0, 0.8, 121)
    dense_time = np.linspace(0, 0.8, 3001)
    datasets = [
        MuonDataset(
            time,
            np.cos(2 * np.pi * 18.2 * time + run),
            np.full_like(time, 0.06),
            metadata={"run_number": run},
        )
        for run in (1, 2)
    ]
    try:
        panel.set_connect_points_enabled(True)
        for run in (1, 2):
            panel._fit_curves[run] = (
                dense_time,
                np.cos(2 * np.pi * 18.2 * dense_time + run),
                "Oscillatory",
            )
        if layout == "single":
            panel.plot_dataset(datasets[0])
        elif layout == "projections":
            panel._current_polarization_axis = "ALL"
            panel.plot_vector_subplots({"P_x": datasets, "P_y": datasets})
        elif layout == "groups":
            panel._current_time_view_mode = "groups"
            panel.plot_grouped_time_domain_subplots(datasets)
        else:
            panel.set_waterfall_enabled(layout == "waterfall")
            panel.plot_datasets(datasets)
        panel.set_view_limits(0, 0.2, -1.3, 1.3)
        path = tmp_path / "main_plot.gle"
        snapshot = panel._gle_snapshot(panel._collect_export_payloads())
        build = snapshot.build(glp, path, tmp_path)
        compile_gle(shutil.which("gle"), path, "pdf", cwd=tmp_path, cairo=build.cairo)
        script = path.read_text()
        assert "Fit:" in script and " lstyle 3" in script
        assert "smooth" not in script
        document = QPdfDocument()
        document.load(str(path.with_suffix(".pdf")))
        assert document.pageCount() == 1
        assert not document.render(0, QSize(1100, 850)).isNull()
    finally:
        panel.close()
