"""Standalone tests for the SeriesFitCanvas widget."""

from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.fit_wizard import SelectionMetric
from asymmetry.core.fitting.model_comparison import (
    CandidateSummary,
    RunFit,
    summarise_candidates,
)
from asymmetry.gui.utils.series_colours import series_colours
from asymmetry.gui.widgets.series_fit_canvas import (
    DISPLAY_POINTS_PER_RUN,
    MAX_RESIDUAL_STRIPS,
    RESIDUAL_CLIP_SIGMA,
    SeriesFitCanvas,
    spread_labels,
)
from tests.core.test_model_comparison import _FIELDS, _RUNS, _assessment, _datasets

_LABELS = [f"{_FIELDS[run]:g} G" for run in _RUNS]
_AXIS = [_FIELDS[run] for run in _RUNS]


def _pair() -> tuple[CandidateSummary, CandidateSummary]:
    a, b = summarise_candidates(
        [
            _assessment("a", aicc=100.0, prescreen=True),
            _assessment("b", aicc=110.0, prescreen=True),
        ],
        _datasets(),
        SelectionMetric.AICC,
    )
    return a, b


def _canvas() -> SeriesFitCanvas:
    canvas = SeriesFitCanvas()
    canvas.resize(900, 700)
    return canvas


def _overlay(canvas: SeriesFitCanvas):
    return canvas.figure.axes[0]


def _markers(axes) -> list:
    return [line for line in axes.lines if line.get_linestyle() == "None"]


def _curves(axes, style: str) -> list:
    return [line for line in axes.lines if line.get_linestyle() == style]


def test_series_draws_one_marker_trace_per_run_in_graded_colours(qapp: QApplication) -> None:
    canvas = _canvas()
    canvas.set_series(_datasets(), _LABELS, _AXIS)
    markers = _markers(_overlay(canvas))
    assert [line.get_color() for line in markers] == series_colours(_AXIS)
    texts = [text.get_text() for text in _overlay(canvas).texts]
    assert texts == _LABELS


def test_host_colours_replace_the_axis_gradient(qapp: QApplication) -> None:
    canvas = _canvas()
    colours = ["#2f4da0", "#2aa198", "#4f7f1e"]
    canvas.set_series(_datasets(), _LABELS, _AXIS, colours)
    assert [line.get_color() for line in _markers(_overlay(canvas))] == colours


def test_series_rejects_mismatched_labels_or_colours(qapp: QApplication) -> None:
    canvas = _canvas()
    with pytest.raises(ValueError, match="one label"):
        canvas.set_series(_datasets(), _LABELS[:2], _AXIS)
    with pytest.raises(ValueError, match="one colour"):
        canvas.set_series(_datasets(), _LABELS, _AXIS, ["#000000"])


def test_markers_are_decimated_to_the_display_budget(qapp: QApplication) -> None:
    time = np.linspace(0.0, 10.0, 5000)
    dataset = MuonDataset(
        time=time,
        asymmetry=np.exp(-time),
        error=np.full_like(time, 0.1),
        metadata={"run_number": 1, "field": 0.0},
    )
    canvas = _canvas()
    canvas.set_series([dataset], ["0 G"], [0.0])
    (markers,) = _markers(_overlay(canvas))
    assert len(markers.get_xdata()) <= DISPLAY_POINTS_PER_RUN


def test_candidate_a_is_solid_and_b_dashed_per_run(qapp: QApplication) -> None:
    a, b = _pair()
    canvas = _canvas()
    canvas.set_series(_datasets(), _LABELS, _AXIS)
    canvas.set_curves(a, b)
    overlay = _overlay(canvas)
    assert len(_curves(overlay, "-")) == len(_RUNS)
    assert len(_curves(overlay, "--")) == len(_RUNS)
    canvas.set_curves(a, None)
    assert len(_curves(_overlay(canvas), "--")) == 0


def test_residual_strips_follow_candidate_a_and_the_toggle(qapp: QApplication) -> None:
    a, b = _pair()
    canvas = _canvas()
    canvas.set_series(_datasets(), _LABELS, _AXIS)
    assert len(canvas.figure.axes) == 1  # no candidate, no residuals
    canvas.set_curves(a, b)
    assert len(canvas.figure.axes) == 1 + len(_RUNS)
    canvas.set_residuals_visible(False)
    assert len(canvas.figure.axes) == 1


def test_strips_label_the_run_and_both_chi_squared(qapp: QApplication) -> None:
    a, b = _pair()
    canvas = _canvas()
    canvas.set_series(_datasets(), _LABELS, _AXIS)
    canvas.set_curves(a, b)
    first_strip = canvas.figure.axes[1]
    texts = [text.get_text() for text in first_strip.texts]
    chi = a.runs[0].reduced_chi_squared
    assert texts == [_LABELS[0], f"{chi:.2f} · {chi:.2f}"]
    headers = [text.get_text() for text in canvas.figure.texts]
    assert "χ²ᵣ A · B" in headers


def test_residuals_are_clipped_to_the_strip(qapp: QApplication) -> None:
    a, _b = _pair()
    run = a.runs[0]
    time, residual = run.residuals
    wild = replace(run, residuals=(time, np.where(np.arange(time.size) % 2, 50.0, -50.0)))
    canvas = _canvas()
    canvas.set_series(_datasets(), _LABELS, _AXIS)
    canvas.set_curves(replace(a, runs=(wild, *a.runs[1:])), None)
    (markers,) = _markers(canvas.figure.axes[1])
    assert np.max(np.abs(markers.get_ydata())) == RESIDUAL_CLIP_SIGMA


def test_a_run_the_candidate_never_fitted_has_no_curve(qapp: QApplication) -> None:
    a, _b = _pair()
    canvas = _canvas()
    canvas.set_series(_datasets(), _LABELS, _AXIS)
    canvas.set_curves(replace(a, runs=a.runs[1:]), None)
    assert len(_curves(_overlay(canvas), "-")) == len(_RUNS) - 1
    assert [text.get_text() for text in canvas.figure.axes[1].texts][1] == "—"


def _long_series(count: int) -> tuple[list[MuonDataset], CandidateSummary]:
    time = np.linspace(0.0, 8.0, 60)
    datasets = [
        MuonDataset(
            time=time,
            asymmetry=np.exp(-0.1 * index * time),
            error=np.full_like(time, 0.05),
            metadata={"run_number": 100 + index, "field": float(index)},
        )
        for index in range(count)
    ]
    runs = tuple(
        RunFit(
            100 + index,
            str(100 + index),
            float(index),
            (time, np.exp(-0.1 * index * time)),
            (time, np.zeros_like(time)),
            1.0,
        )
        for index in range(count)
    )
    a, _b = _pair()
    return datasets, replace(a, runs=runs)


def test_many_runs_cap_the_strips_and_note_the_rest(qapp: QApplication) -> None:
    datasets, summary = _long_series(15)
    labels = [f"{index} G" for index in range(15)]
    canvas = _canvas()
    canvas.set_series(datasets, labels, [float(index) for index in range(15)])
    canvas.set_curves(summary, None)
    strips = canvas.figure.axes[1:]
    assert len(strips) == MAX_RESIDUAL_STRIPS
    strip_labels = [strip.texts[0].get_text() for strip in strips]
    assert strip_labels[0] == labels[0] and strip_labels[-1] == labels[-1]
    assert f"+{15 - MAX_RESIDUAL_STRIPS} more runs" in [t.get_text() for t in canvas.figure.texts]


def test_canvas_grows_with_the_run_count(qapp: QApplication) -> None:
    heights = []
    for count in (2, 6, 12):
        datasets, summary = _long_series(count)
        canvas = _canvas()
        canvas.set_series(
            datasets, [str(i) for i in range(count)], [float(i) for i in range(count)]
        )
        canvas.set_curves(summary, None)
        heights.append(canvas.minimumSizeHint().height())
    assert heights[0] < heights[1] < heights[2]


def test_spread_labels_keeps_order_and_gap_within_range() -> None:
    placed = spread_labels([0.50, 0.51, 0.52, 0.10], gap=0.05, low=0.0, high=1.0)
    assert placed[3] == pytest.approx(0.10)
    ordered = [placed[i] for i in (3, 0, 1, 2)]
    assert all(b - a >= 0.05 - 1e-12 for a, b in zip(ordered, ordered[1:]))
    assert 0.0 <= min(placed) and max(placed) <= 1.0


def test_spread_labels_spaces_evenly_when_the_gap_cannot_fit() -> None:
    placed = spread_labels([0.5] * 5, gap=0.5, low=0.0, high=1.0)
    assert sorted(placed) == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
