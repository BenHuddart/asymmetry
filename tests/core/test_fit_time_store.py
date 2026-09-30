"""Tests for the per-machine store of measured wizard fit times."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.engine import FitResult
from asymmetry.core.fitting.fit_time_store import SAMPLES_KEPT, FitTimeStore
from asymmetry.core.fitting.fit_wizard import (
    CandidateAssessment,
    CandidateTemplate,
    FitTiming,
    screening_points,
)
from asymmetry.core.fitting.wizard_scope import UNTIMED, FitTimeEstimates


def _timed(components: list[str], seconds: float, points: int = 1000) -> CandidateAssessment:
    """A row fitted in this run: *components* joined by ``+``, taking *seconds* on *points*."""
    empty = np.array([], dtype=float)
    return CandidateAssessment(
        template=CandidateTemplate(
            key="+".join(components),
            title="",
            category="General",
            rationale="",
            model=CompositeModel(components, operators=["+"] * (len(components) - 1)),
        ),
        fit_result=FitResult(success=True, chi_squared=1.0, reduced_chi_squared=1.0),
        aic=1.0,
        aicc=1.0,
        bic=1.0,
        selected_score=1.0,
        residual_rms=1.0,
        runs_z_score=0.0,
        max_abs_autocorrelation=0.0,
        residual_fft_peak_snr=0.0,
        residual_gate_passed=True,
        residual_gate_reasons=(),
        bound_hits=(),
        fitted_time=empty,
        fitted_curve=empty,
        component_curves=(),
        timing=FitTiming(seconds, points),
    )


def _run(n_points: int) -> MuonDataset:
    time = np.linspace(0.0, 10.0, n_points)
    return MuonDataset(time=time, asymmetry=np.zeros(n_points), error=np.ones(n_points))


def test_a_run_adds_one_sample_per_component_the_median_over_its_templates() -> None:
    store = FitTimeStore()
    store.record(
        [
            _timed(["DynamicGaussianKT", "Constant"], 1.0),
            _timed(["DynamicGaussianKT", "Exponential", "Constant"], 3.0),
            _timed(["DynamicGaussianKT", "Gaussian", "Constant"], 8.0, points=2000),
            _timed(["Exponential", "Constant"], 0.2),
        ]
    )
    # Per 1000 points: 1.0, 3.0 and 4.0 for the KT templates; 3.0 and 0.2 with Exponential.
    assert store.samples == {
        "DynamicGaussianKT": (3.0,),
        "Exponential": (pytest.approx(1.6),),
        "Gaussian": (4.0,),
    }


def test_the_background_is_never_timed() -> None:
    store = FitTimeStore()
    store.record([_timed(["Constant"], 2.0), _timed(["Exponential", "Constant"], 0.1)])
    assert set(store.samples) == {"Exponential"}


def test_only_rows_fitted_in_the_run_are_recorded() -> None:
    restored = replace(_timed(["Exponential", "Constant"], 0.1), timing=None)
    with pytest.raises(ValueError, match="carries no timing"):
        FitTimeStore().record([restored])


def test_the_store_keeps_the_last_nine_runs_and_estimates_their_median() -> None:
    store = FitTimeStore()
    for seconds in range(1, 13):
        store.record([_timed(["Keren", "Constant"], float(seconds))])
    assert store.samples["Keren"] == tuple(float(s) for s in range(13 - SAMPLES_KEPT, 13))
    # The median of 4..12 s per 1000 points, on runs of 2000 points.
    assert store.estimates([_run(2000)]).seconds_per_run == {"Keren": 16.0}


def test_screening_points_is_the_length_after_the_cost_rebin() -> None:
    assert screening_points(_run(5000)) == 5000
    assert screening_points(_run(8192)) == 8192
    assert screening_points(_run(100_000)) == 100_000 // 12


def test_a_series_is_estimated_at_its_median_run_length() -> None:
    store = FitTimeStore({"Keren": [2.0]})
    runs = [_run(1000), _run(3000), _run(100_000)]
    assert store.estimates(runs).seconds_per_run == {"Keren": 6.0}


def test_no_runs_means_nothing_is_timed() -> None:
    assert FitTimeStore({"Keren": [2.0]}).estimates([]) is UNTIMED


def test_estimates_are_a_fit_time_judgement() -> None:
    estimates = FitTimeStore({"Keren": [2.0]}).estimates([_run(1000)])
    assert estimates == FitTimeEstimates({"Keren": 2.0})


def test_the_store_round_trips_through_its_file(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "fit_times.json"
    store = FitTimeStore({"Keren": [1.5, 2.5], "Oscillatory": [0.25]})
    store.save(path)
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "version": 1,
        "seconds_per_kpoint": {"Keren": [1.5, 2.5], "Oscillatory": [0.25]},
    }
    assert FitTimeStore.load(path).samples == store.samples


def test_an_absent_file_is_an_empty_store(tmp_path: Path) -> None:
    assert FitTimeStore.load(tmp_path / "fit_times.json").samples == {}


@pytest.mark.parametrize(
    "text",
    [
        "{not json",
        "[]",
        json.dumps({"version": 1}),
        json.dumps({"version": 2, "seconds_per_kpoint": {}}),
        json.dumps({"version": 1, "seconds_per_kpoint": {"Keren": []}}),
        json.dumps({"version": 1, "seconds_per_kpoint": {"Keren": [-1.0]}}),
        json.dumps({"version": 1, "seconds_per_kpoint": {"Keren": ["1.0"]}}),
        json.dumps({"version": 1, "seconds_per_kpoint": {"Keren": [True]}}),
        json.dumps({"version": 1, "seconds_per_kpoint": {"Keren": [1.0] * 10}}),
        '{"version": 1, "seconds_per_kpoint": {"Keren": [NaN]}}',
    ],
)
def test_a_malformed_file_raises_naming_its_path(tmp_path: Path, text: str) -> None:
    path = tmp_path / "fit_times.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="fit_times.json"):
        FitTimeStore.load(path)
