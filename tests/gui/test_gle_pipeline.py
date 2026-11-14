"""Exercise real data-only exporters and their asynchronous preview lifecycle."""

# ruff: noqa: E402

from __future__ import annotations

import shutil
import subprocess
import threading
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")
glp = pytest.importorskip("gleplot")
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QFileDialog, QWidget

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.parameter_models import CrossGroupFitResult, ParameterGroupData
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel, _FitRow
from asymmetry.gui.panels.plot_panel import PlotPanel
from asymmetry.gui.utils import gle_export
from asymmetry.gui.utils.gle_figure import gle_figure
from asymmetry.gui.windows.gle_preview import GlePlotPreview, close_gle_previews
from asymmetry.gui.windows.global_parameter_fit_window import GlobalParameterFitWindow
from tests._qt_helpers import wait_for
from tests.gui.gle_helpers import write_compile_outputs

pytestmark = pytest.mark.gui


@pytest.fixture
def destination(monkeypatch, tmp_path):
    path = tmp_path / "figure.gleplot"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(path), ""))
    monkeypatch.setattr(gle_export, "get_gle_executable", lambda: None)
    return path


def spectrum():
    time = np.linspace(0, 8, 80)
    return MuonDataset(
        time=time,
        asymmetry=np.exp(-time),
        error=np.full_like(time, 0.01),
        metadata={"run_number": 1, "title": "Synthetic spectrum"},
    )


def completed(qapp, owner):
    preview = owner.findChild(GlePlotPreview)
    assert preview is not None
    wait_for(lambda: not preview.tasks.active_count, qapp)
    assert not preview.status.text().startswith("Export failed"), preview.status.text()
    return preview


def test_worker_snapshot_isolated_from_widget_and_array_edits(qapp, destination, monkeypatch):
    panel = PlotPanel()
    dataset = spectrum()
    panel.plot_dataset(dataset)
    snapshot = panel._gle_snapshot(panel._collect_export_payloads())
    entered, release = threading.Event(), threading.Event()
    observed = []
    original_save = glp.Figure.savefig

    def save(figure, *args, **kwargs):
        observed.append(QThread.currentThread() == qapp.thread())
        return original_save(figure, *args, **kwargs)

    def build(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return snapshot.build(*args, **kwargs)

    monkeypatch.setattr(glp.Figure, "savefig", save)
    try:
        gle_export.run_gle_export(
            panel,
            dialog_title="Export",
            default_name="figure",
            output_format="pdf",
            prepare=lambda: build,
        )
        wait_for(entered.is_set, qapp)
        dataset.asymmetry[:] = 900
        panel._x_max.setValue(999)
        release.set()
        preview = completed(qapp, panel)
        assert observed == [False]
        script = (destination / "figure.gle").read_text()
        assert "max 999" not in script
        rows = np.loadtxt(next(destination.glob("*.dat")), comments="!")
        assert np.max(rows[:, 1]) <= 1
        assert "saved" in preview.status.text()
    finally:
        release.set()
        close_gle_previews(panel)
        panel.close()


def test_parameter_snapshot_retains_rows(qapp, tmp_path):
    panel = FitParametersPanel()
    panel._rows = [_FitRow(1, "one", 100.0, 10.0, {"Lambda": 0.2}, {"Lambda": 0.01})]
    panel._varying_params = ["Lambda"]
    snapshot = panel._gle_snapshot()
    panel._rows[0].values["Lambda"] = 900
    try:
        snapshot.build(glp, tmp_path / "trend.gle", tmp_path)
        table = np.loadtxt(tmp_path / "trend.dat", comments="!")
        assert table[3] == pytest.approx(0.2)
    finally:
        panel.close()


def test_global_snapshot_retains_local_parameters(qapp, tmp_path):
    window = GlobalParameterFitWindow()
    window._groups = [
        ParameterGroupData(
            "one", "One", np.array([1.0, 2.0]), np.array([0.1, 0.2]), np.array([0.01, 0.01]), 20.0
        )
    ]
    window._result = CrossGroupFitResult(
        success=True,
        chi_squared=1,
        reduced_chi_squared=1,
        local_parameters={"one": ParameterSet([Parameter("c", value=0.2)])},
    )
    window._parameter_name = "Lambda"
    snapshot = window._gle_snapshot()
    window._result.local_parameters["one"]["c"].value = 900
    try:
        snapshot.build_local(glp, tmp_path / "local.gle", tmp_path)
        table = np.loadtxt(tmp_path / "local_local_parameters.dat", comments="!")
        assert 0.2 in table
        snapshot.build_fits(glp, tmp_path / "fits.gle", tmp_path)
    finally:
        window.close()


def test_independent_figures_do_not_touch_global_state():
    original = glp.gcf()
    first, second = gle_figure(glp), gle_figure(glp)
    first.graph.smooth_curves = not second.graph.smooth_curves
    assert first.graph is not second.graph
    assert first.graph.smooth_curves != second.graph.smooth_curves
    assert glp.gcf() is original


@pytest.mark.parametrize("device", ["pdf", "eps"])
def test_real_generation_compiles_once_and_zoom_never_recompiles(
    qapp, destination, monkeypatch, device
):
    panel = PlotPanel()
    panel.plot_dataset(spectrum())
    counts = {"save": 0, "compile": 0}
    original = glp.Figure.savefig

    def save(figure, *args, **kwargs):
        counts["save"] += 1
        return original(figure, *args, **kwargs)

    def compile(executable, script, devices, **kwargs):
        counts["compile"] += 1
        assert devices == (("eps", "pdf") if device == "eps" else ("pdf",))
        write_compile_outputs(script, devices)

    monkeypatch.setattr(glp.Figure, "savefig", save)
    monkeypatch.setattr(gle_export, "get_gle_executable", lambda: "gle")
    monkeypatch.setattr(gle_export, "compile_gle", compile)
    snapshot = panel._gle_snapshot(panel._collect_export_payloads())
    try:
        gle_export.run_gle_export(
            panel,
            dialog_title="Export",
            default_name="figure",
            output_format=device,
            prepare=lambda: snapshot.build,
        )
        preview = completed(qapp, panel)
        assert preview.document.pageCount() == 1
        preview.zoom_in()
        preview.zoom_out()
        preview.fit()
        preview.resize(preview.width() + 1, preview.height() + 1)
        qapp.processEvents()
        assert counts == {"save": 1, "compile": 1}
        assert (destination / "figure.pdf").exists()
    finally:
        close_gle_previews(panel)
        panel.close()


def test_active_destination_is_exclusive_across_owners(qapp, destination):
    owners = [QWidget(), QWidget()]
    entered, release = threading.Event(), threading.Event()

    def build(glp, script, directory, **kwargs):
        entered.set()
        assert release.wait(5)
        script.write_text("size 8 6\n")
        return gle_export.GleExportBuild([script])

    try:
        for owner in owners:
            gle_export.run_gle_export(
                owner,
                dialog_title="Export",
                default_name="figure",
                output_format="pdf",
                prepare=lambda: build,
            )
            wait_for(entered.is_set, qapp)
        assert len(owners[0].findChildren(GlePlotPreview)) == 1
        assert not owners[1].findChildren(GlePlotPreview)
    finally:
        release.set()
        completed(qapp, owners[0])
        for owner in owners:
            close_gle_previews(owner)
            owner.close()


@pytest.mark.parametrize("failure", ["stderr", "timeout", "missing_output"])
def test_compiler_failure_never_displays_stale_pdf(qapp, destination, monkeypatch, failure):
    destination.mkdir()
    (destination / "figure.pdf").write_bytes(b"stale")
    owner = QWidget()

    def build(glp, script, directory, **kwargs):
        script.write_text("size 8 6\n")
        return gle_export.GleExportBuild([script])

    def compile(*args, **kwargs):
        if failure == "stderr":
            raise subprocess.CalledProcessError(1, ["gle"], stderr="bad font")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(["gle"], 120)

    monkeypatch.setattr(gle_export, "get_gle_executable", lambda: "gle")
    monkeypatch.setattr(gle_export, "compile_gle", compile)
    try:
        gle_export.run_gle_export(
            owner,
            dialog_title="Export",
            default_name="figure",
            output_format="pdf",
            prepare=lambda: build,
        )
        preview = owner.findChild(GlePlotPreview)
        wait_for(lambda: not preview.tasks.active_count, qapp)
        assert preview.status.text().startswith("Export failed")
        assert preview.document.pageCount() == 0
        assert not (destination / "figure.pdf").exists()
        if failure == "stderr":
            assert "bad font" in preview.status.text()
    finally:
        close_gle_previews(owner)
        owner.close()


@pytest.mark.skipif(shutil.which("gle") is None, reason="GLE compiler not installed")
def test_real_compiler_produces_eps_and_vector_pdf(qapp, tmp_path):
    from PySide6.QtPdf import QPdfDocument

    from asymmetry.gui.utils.export import compile_gle

    figure = gle_figure(glp)
    axes = figure.add_subplot(111)
    axes.plot([1, 2, 3], [2, 1, 3])
    script = tmp_path / "real.gle"
    figure.savefig(str(script))
    compile_gle(shutil.which("gle"), script, ("eps", "pdf"), cwd=tmp_path)
    assert script.with_suffix(".eps").stat().st_size > 0
    document = QPdfDocument()
    document.load(str(script.with_suffix(".pdf")))
    assert document.pageCount() == 1


@pytest.mark.parametrize("surface", ["plot", "parameters", "global_fits", "global_local"])
def test_every_public_export_uses_worker_and_vector_preview(
    qapp, destination, monkeypatch, surface
):
    import builtins

    original_import = builtins.__import__

    def import_without_editor(name, *args, **kwargs):
        assert not name.startswith("gleplot.gui")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_editor)

    def forbidden(*args, **kwargs):
        raise AssertionError("Workers must not use gleplot current-figure helpers")

    monkeypatch.setattr(glp, "figure", forbidden)
    monkeypatch.setattr(glp, "subplots", forbidden)
    calls = []

    def compile(executable, script, devices, **kwargs):
        calls.append(QThread.currentThread() == qapp.thread())
        write_compile_outputs(script, devices)

    monkeypatch.setattr(gle_export, "get_gle_executable", lambda: "gle")
    monkeypatch.setattr(gle_export, "compile_gle", compile)
    if surface == "plot":
        owner = PlotPanel()
        owner.plot_dataset(spectrum())
        action = owner.export_plots_to_gle
    elif surface == "parameters":
        owner = FitParametersPanel()
        owner._rows = [_FitRow(1, "one", 100, 10, {"Lambda": 0.2}, {"Lambda": 0.01})]
        owner._varying_params = ["Lambda"]

        def action():
            owner._export_gle("pdf")
    else:
        owner = GlobalParameterFitWindow()
        owner._groups = [
            ParameterGroupData(
                "one",
                "One",
                np.array([1.0, 2.0]),
                np.array([0.1, 0.2]),
                np.array([0.01, 0.01]),
                20.0,
            )
        ]
        owner._parameter_name = "Lambda"
        owner._result = CrossGroupFitResult(
            success=True,
            chi_squared=1,
            reduced_chi_squared=1,
            local_parameters={"one": ParameterSet([Parameter("c", value=0.2)])},
        )
        action = (
            owner._export_fit_subplot_gle
            if surface == "global_fits"
            else owner._export_local_parameters_gle
        )
    try:
        action()
        preview = completed(qapp, owner)
        assert preview.document.pageCount() == 1
        assert calls == [False]
    finally:
        close_gle_previews(owner)
        owner.close()


def test_close_cancels_worker_without_blocking_gui(qapp, destination, monkeypatch):
    from PySide6.QtCore import QTimer

    from asymmetry.gui.tasks import TaskCancelledError

    owner = QWidget()
    entered = threading.Event()
    ticks = []
    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start(1)

    def build(glp, script, directory, *, cancelled):
        entered.set()
        while not cancelled():
            threading.Event().wait(0.005)
        raise TaskCancelledError

    try:
        gle_export.run_gle_export(
            owner,
            dialog_title="Export",
            default_name="figure",
            output_format="pdf",
            prepare=lambda: build,
        )
        wait_for(lambda: entered.is_set() and len(ticks) >= 3, qapp)
        preview = owner.findChild(GlePlotPreview)
        preview.close()
        assert not preview.tasks.active_count
    finally:
        timer.stop()
        close_gle_previews(owner)
        owner.close()


def test_snapshot_generation_cooperatively_cancels(qapp, tmp_path):
    from asymmetry.gui.tasks import TaskCancelledError

    panel = PlotPanel()
    panel.plot_dataset(spectrum())
    snapshot = panel._gle_snapshot(panel._collect_export_payloads())
    polls = []

    def cancelled():
        polls.append(1)
        return len(polls) > 4

    try:
        with pytest.raises(TaskCancelledError):
            snapshot.build(glp, tmp_path / "cancelled.gle", tmp_path, cancelled=cancelled)
        assert len(polls) == 5
    finally:
        panel.close()


def test_mainwindow_and_shell_shutdown_cancel_previews(qapp, destination):
    from asymmetry.gui.mainwindow import MainWindow
    from asymmetry.gui.shell import ProjectShell

    for shell_mode in (False, True):
        shell = ProjectShell() if shell_mode else None
        page = shell.add_project() if shell is not None else MainWindow()
        entered = threading.Event()

        def build(glp, script, directory, *, cancelled):
            from asymmetry.gui.tasks import TaskCancelledError

            entered.set()
            while not cancelled():
                threading.Event().wait(0.005)
            raise TaskCancelledError

        try:
            gle_export.run_gle_export(
                page._plot_panel,
                dialog_title="Export",
                default_name="figure",
                output_format="pdf",
                prepare=lambda: build,
            )
            wait_for(entered.is_set, qapp)
            preview = page.findChild(GlePlotPreview)
            assert preview is not None
            if shell is None:
                page.close()
            else:
                shell.close_page(page)
            assert not preview.tasks.active_count
        finally:
            if shell is not None:
                shell.close()
            else:
                page.close()


def test_reexport_keeps_existing_preview_snapshot(qapp, destination, monkeypatch):
    owner = QWidget()

    def build(glp, script, directory, **kwargs):
        script.write_text("size 8 6\n")
        return gle_export.GleExportBuild([script])

    calls = []

    def compile(executable, script, devices, **kwargs):
        calls.append(1)
        write_compile_outputs(script, devices)
        if len(calls) == 2:
            pdf = Path(script).with_suffix(".pdf")
            pdf.write_bytes(pdf.read_bytes().replace(b"200 150", b"400 300"))

    monkeypatch.setattr(gle_export, "get_gle_executable", lambda: "gle")
    monkeypatch.setattr(gle_export, "compile_gle", compile)
    try:
        gle_export.run_gle_export(
            owner,
            dialog_title="Export",
            default_name="figure",
            output_format="pdf",
            prepare=lambda: build,
        )
        original = completed(qapp, owner)
        size = original.document.pagePointSize(0)
        gle_export.run_gle_export(
            owner,
            dialog_title="Export",
            default_name="figure",
            output_format="pdf",
            prepare=lambda: build,
        )
        previews = owner.findChildren(GlePlotPreview)
        wait_for(lambda: all(not p.tasks.active_count for p in previews), qapp)
        assert len(previews) == 2
        assert original.document.pagePointSize(0) == size
        assert previews[-1].document.pagePointSize(0) != size
        assert len(calls) == 2
    finally:
        close_gle_previews(owner)
        owner.close()


def test_empty_trailing_projection_keeps_time_axis_label(qapp, tmp_path, monkeypatch):
    from dataclasses import replace

    panel = PlotPanel()
    panel.plot_dataset(spectrum())
    snapshot = panel._gle_snapshot(panel._collect_export_payloads())
    snapshot = replace(snapshot, axes=(*snapshot.axes, (None, [], "")), stacked=True)
    labels = []
    original = glp.Axes.set_xlabel

    def label(axis, text):
        labels.append(text)
        return original(axis, text)

    monkeypatch.setattr(glp.Axes, "set_xlabel", label)
    try:
        snapshot.build(glp, tmp_path / "trailing.gle", tmp_path)
        assert "Time (µs)" in labels
    finally:
        panel.close()
