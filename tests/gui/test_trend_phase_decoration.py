"""Fit Parameters panel: phase swatches, plot colour, band and boundaries (D4).

Covers Phase D4 of ``docs/plans/global-wizard-transitions.md``: a series bound
to a phase group (Global Fit Wizard transitions, D1) carries a
:class:`~asymmetry.gui.panels.fit_parameters_panel.PhaseDecoration` through
``load_representation_series`` — a swatch on its pill, its phase colour on the
plotted trace (single and overlay modes), and a shaded range + boundary lines
on the active series' plot (and the matching gleplot band/lines on export).  A
plain series (no phase) is unaffected. The last test builds a real phase group
through :class:`~asymmetry.core.representation.project_model.ProjectModel` and
checks that :meth:`~asymmetry.gui.mainwindow.MainWindow._refresh_trend_panel`
derives the decoration from it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.colors as mcolors
import pytest

pytestmark = [pytest.mark.gui]

pyside6 = pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # type: ignore

from asymmetry.core.fitting.axis_transforms import RECIPROCAL, AxisTransform
from asymmetry.core.representation import FitSeries, RepresentationType
from asymmetry.core.representation.group import PhaseSpec
from asymmetry.gui.export_paths import resolve_gle_export_paths
from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel, PhaseDecoration
from asymmetry.gui.utils.phase_colors import phase_color

_PHASE = PhaseDecoration(
    color="#2F4DA0",
    color_dark="#8FA3E8",
    ordinal=1,
    name="Phase I",
    axis_key="temperature",
    range=(1.8, 10.0),
    lower=None,
    upper=(13.0, 1.5),
)

_T_AXIS_TEXT = "\U0001d447 (K)"  # matches the combo item exactly ("𝑇 (K)")


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _row(run: int, temperature: float, sigma: float) -> dict:
    return {
        "run_number": run,
        "run_label": str(run),
        "field": 0.0,
        "temperature": temperature,
        "values": {"sigma": sigma},
        "errors": {"sigma": 0.02},
    }


@pytest.fixture
def panel(qapp: QApplication) -> FitParametersPanel:
    """A panel with one plain series and one phase-owned series over temperature."""
    w = FitParametersPanel()
    plain_rows = [_row(10, 2.0, 0.30), _row(11, 6.0, 0.32), _row(12, 9.0, 0.34)]
    phase_rows = [_row(20, 2.0, 0.50), _row(21, 6.0, 0.55), _row(22, 9.0, 0.60)]
    w.load_representation_series(
        [("s-plain", "Plain series", plain_rows), ("s-phase", "Phase I", phase_rows)],
        phase_by_id={"s-phase": _PHASE},
    )
    w._x_combo.setCurrentText(_T_AXIS_TEXT)
    return w


def test_swatch_icon_present_for_phase_series_absent_for_plain(
    panel: FitParametersPanel,
) -> None:
    assert not panel._group_button_map["s-phase"].icon().isNull()
    assert panel._group_button_map["s-plain"].icon().isNull()


def test_series_color_equals_phase_color_when_single(panel: FitParametersPanel) -> None:
    panel.select_series(["s-phase"])
    (series,) = panel._series_to_plot()
    assert series.color == _PHASE.color


def test_series_color_equals_phase_color_when_overlaid(panel: FitParametersPanel) -> None:
    panel.select_series(["s-plain", "s-phase"])
    colors = {s.group_id: s.color for s in panel._series_to_plot()}
    assert colors["s-phase"] == _PHASE.color
    # The plain series keeps the ordinary pill-order cycle colour.
    assert colors["s-plain"] != _PHASE.color


def _dashed_lines(ax, color: str) -> list:
    return [ln for ln in ax.lines if ln.get_linestyle() == "--" and ln.get_color() == color]


def _band_patches(ax, color: str, alpha: float) -> list:
    target = mcolors.to_rgba(color, alpha=alpha)
    return [
        p
        for p in ax.patches
        if tuple(round(c, 6) for c in p.get_facecolor()) == tuple(round(c, 6) for c in target)
    ]


def test_draw_single_series_shows_band_and_boundary_for_phase_series(
    panel: FitParametersPanel,
) -> None:
    panel.select_series(["s-phase"])
    panel._draw_plot()
    (ax,) = panel._figure.axes
    # One range span (alpha 0.12) plus one boundary (only "upper" is set) — a
    # dashed line at its estimate and a fainter (alpha 0.08) uncertainty span.
    assert len(_band_patches(ax, _PHASE.color, 0.12)) == 1
    assert len(_band_patches(ax, _PHASE.color, 0.08)) == 1
    assert len(_dashed_lines(ax, _PHASE.color)) == 1


def test_draw_single_series_no_band_for_plain_series(panel: FitParametersPanel) -> None:
    panel.select_series(["s-plain"])
    panel._draw_plot()
    (ax,) = panel._figure.axes
    assert _band_patches(ax, _PHASE.color, 0.12) == []
    assert _dashed_lines(ax, _PHASE.color) == []


def test_draw_multi_series_bands_only_the_active_phase_series(
    panel: FitParametersPanel,
) -> None:
    # Active = "s-phase" (first argument); overlaid with the plain series.
    panel.select_series(["s-phase", "s-plain"])
    panel._draw_plot()
    (ax,) = panel._figure.axes
    assert len(_band_patches(ax, _PHASE.color, 0.12)) == 1
    assert len(_dashed_lines(ax, _PHASE.color)) == 1

    # Flip which series is active: now nothing is banded, even though the
    # phase series is still on the plot (overlaid, not active).
    panel.select_series(["s-plain", "s-phase"])
    panel._draw_plot()
    (ax2,) = panel._figure.axes
    assert _band_patches(ax2, _PHASE.color, 0.12) == []


def test_log_x_suppresses_the_band(panel: FitParametersPanel) -> None:
    panel.select_series(["s-phase"])
    panel._log_x_check.setChecked(True)
    panel._draw_plot()
    (ax,) = panel._figure.axes
    assert _band_patches(ax, _PHASE.color, 0.12) == []
    assert _dashed_lines(ax, _PHASE.color) == []


def test_axis_transform_suppresses_the_band(panel: FitParametersPanel) -> None:
    panel.select_series(["s-phase"])
    panel._x_transform = AxisTransform.preset(RECIPROCAL)
    panel._draw_plot()
    (ax,) = panel._figure.axes
    assert _band_patches(ax, _PHASE.color, 0.12) == []


def test_wrong_axis_suppresses_the_band(panel: FitParametersPanel) -> None:
    """The phase's axis_key is "temperature"; plotting vs field must not band."""
    panel.select_series(["s-phase"])
    panel._x_combo.setCurrentText("\U0001d435 (G)")  # "𝐵 (G)"
    panel._draw_plot()
    (ax,) = panel._figure.axes
    assert _band_patches(ax, _PHASE.color, 0.12) == []


# ── GLE export ───────────────────────────────────────────────────────────────


class _FakeGleAxis:
    def __init__(self) -> None:
        self.errorbar_calls: list[dict] = []
        self.fill_between_calls: list[dict] = []
        self.plot_calls: list[dict] = []

    def errorbar_from_file(self, *args, **kwargs) -> None:
        self.errorbar_calls.append({"args": args, "kwargs": kwargs})

    def line_from_file(self, *_args, **_kwargs) -> None:
        return

    def fill_between(self, xs, lower, upper, **kwargs) -> None:
        self.fill_between_calls.append({"xs": xs, "lower": lower, "upper": upper, **kwargs})

    def plot(self, xs, ys, **kwargs) -> None:
        self.plot_calls.append({"xs": xs, "ys": ys, **kwargs})

    def set_xlabel(self, *_a, **_k) -> None:
        return

    def set_ylabel(self, *_a, **_k) -> None:
        return

    def set_xscale(self, *_a, **_k) -> None:
        return

    def set_yscale(self, *_a, **_k) -> None:
        return

    def set_ylim(self, *_a, **_k) -> None:
        return

    def legend(self, *_a, **_k) -> None:
        return

    def text(self, *_a, **_k) -> None:
        return


class _FakeGleFigure:
    def __init__(self, axis: _FakeGleAxis) -> None:
        self._axis = axis
        self.saved_paths: list[str] = []

    def add_subplot(self, *_args, **_kwargs) -> _FakeGleAxis:
        return self._axis

    def savefig(self, path: str, **_kwargs) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.saved_paths.append(str(out))
        out.write_text("! fake gle", encoding="utf-8")


def test_gle_export_writes_the_phase_band_and_boundary_line(
    panel: FitParametersPanel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    panel.select_series(["s-phase"])

    axis = _FakeGleAxis()
    fig = _FakeGleFigure(axis)
    fake_glp = SimpleNamespace(
        Axes=type(
            "FakeAxes",
            (),
            {
                "errorbar_from_file": staticmethod(lambda *a, **k: None),
                "line_from_file": staticmethod(lambda *a, **k: None),
            },
        ),
        figure=lambda **_kwargs: fig,
    )
    monkeypatch.setitem(sys.modules, "gleplot", fake_glp)

    out_gle = tmp_path / "phase.gleplot"
    gle_path, export_dir = resolve_gle_export_paths(out_gle, folder=True)
    export_dir.mkdir(parents=True, exist_ok=True)
    build = panel._build_gle_export(fake_glp, gle_path, export_dir)
    assert build is not None

    # The lone y-parameter's data-point colour is the phase colour (mirrors
    # the on-screen single-series substitution).
    assert axis.errorbar_calls[0]["kwargs"]["color"] == _PHASE.color
    # One filled range band plus one filled boundary-uncertainty band.
    assert len(axis.fill_between_calls) == 2
    assert all(call["color"] == _PHASE.color for call in axis.fill_between_calls)
    # One dashed boundary line.
    (line_call,) = axis.plot_calls
    assert line_call["color"] == _PHASE.color
    assert line_call["linestyle"] == "--"


def test_gle_export_omits_band_for_plain_series(
    panel: FitParametersPanel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    panel.select_series(["s-plain"])

    axis = _FakeGleAxis()
    fig = _FakeGleFigure(axis)
    fake_glp = SimpleNamespace(
        Axes=type(
            "FakeAxes",
            (),
            {
                "errorbar_from_file": staticmethod(lambda *a, **k: None),
                "line_from_file": staticmethod(lambda *a, **k: None),
            },
        ),
        figure=lambda **_kwargs: fig,
    )
    monkeypatch.setitem(sys.modules, "gleplot", fake_glp)

    out_gle = tmp_path / "plain.gleplot"
    gle_path, export_dir = resolve_gle_export_paths(out_gle, folder=True)
    export_dir.mkdir(parents=True, exist_ok=True)
    build = panel._build_gle_export(fake_glp, gle_path, export_dir)
    assert build is not None
    assert axis.fill_between_calls == []
    assert axis.plot_calls == []


# ── MainWindow seam: building the decoration from a real phase group ─────────


@pytest.fixture
def mw(qapp: QApplication):
    from asymmetry.gui.mainwindow import MainWindow

    return MainWindow()


def test_refresh_trend_panel_builds_phase_decoration_from_project_model(mw) -> None:
    parent = mw._project_model.create_data_group(
        "Zero-field scan", [10, 11, 12, 13, 14, 15], order_key="temperature"
    )
    (phase_id,) = mw._project_model.create_phase_groups(
        parent.group_id,
        [
            PhaseSpec(
                ordinal=1,
                name="Phase I",
                member_run_numbers=(10, 11, 12),
                phase_range=(1.8, 10.0),
                phase_boundaries={"lower": None, "upper": (13.0, 1.5)},
                phase_color=None,
                phase_provenance={},
            )
        ],
    )

    series = FitSeries(
        "batch-phase",
        RepresentationType.TIME_FB_ASYMMETRY,
        member_kind="runs",
        member_run_numbers=[10, 11, 12],
        group_id=phase_id,
        order_key="temperature",
        results_by_run={
            10: {
                "success": True,
                "parameters": {"sigma": 0.5},
                "uncertainties": {"sigma": 0.02},
                "temperature": 2.0,
                "field": 0.0,
            },
            11: {
                "success": True,
                "parameters": {"sigma": 0.55},
                "uncertainties": {"sigma": 0.02},
                "temperature": 6.0,
                "field": 0.0,
            },
            12: {
                "success": True,
                "parameters": {"sigma": 0.6},
                "uncertainties": {"sigma": 0.02},
                "temperature": 9.0,
                "field": 0.0,
            },
        },
    )
    mw._project_model.add_batch(series)
    mw._plot_workspace.set_active_view("fb_asymmetry")
    mw._refresh_trend_panel()

    gdata = mw._fit_parameters_panel._group_fit_results.get("batch-phase")
    assert gdata is not None
    phase = gdata.phase
    assert phase is not None
    assert phase.ordinal == 1
    assert phase.name == "Phase I"
    assert phase.axis_key == "temperature"
    assert phase.range == (1.8, 10.0)
    assert phase.lower is None
    assert phase.upper == (13.0, 1.5)
    assert phase.color == phase_color(1)


def test_refresh_trend_panel_leaves_plain_series_undecorated(mw) -> None:
    group = mw._project_model.create_data_group("A scan", [30, 31], order_key="field")
    series = FitSeries(
        "batch-plain",
        RepresentationType.TIME_FB_ASYMMETRY,
        member_kind="runs",
        member_run_numbers=[30, 31],
        group_id=group.group_id,
        order_key="field",
        results_by_run={
            30: {
                "success": True,
                "parameters": {"sigma": 0.5},
                "uncertainties": {"sigma": 0.02},
                "temperature": 2.0,
                "field": 100.0,
            },
            31: {
                "success": True,
                "parameters": {"sigma": 0.55},
                "uncertainties": {"sigma": 0.02},
                "temperature": 2.0,
                "field": 200.0,
            },
        },
    )
    mw._project_model.add_batch(series)
    mw._plot_workspace.set_active_view("fb_asymmetry")
    mw._refresh_trend_panel()

    gdata = mw._fit_parameters_panel._group_fit_results.get("batch-plain")
    assert gdata is not None
    assert gdata.phase is None
