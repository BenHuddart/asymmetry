"""Tests for :mod:`asymmetry.core.workflow.trend_fit`."""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.workflow.series import TrendTable
from asymmetry.core.workflow.trend_fit import fit_trend

#: An EuO-like precession frequency: y0 (1 - T/Tc)^beta.
_Y0, _TC, _BETA = 30.0, 70.0, 0.35
_TEMPERATURES = np.arange(5.0, 70.0, 5.0)
#: A run near Tc whose fit the series flagged, carrying a wild value.
_FLAGGED_RUN = 199


def _frequency(temperature: float) -> float:
    return _Y0 * (1.0 - temperature / _TC) ** _BETA


def _trend() -> TrendTable:
    rows = [
        {
            "run": 100 + index,
            "x": float(temperature),
            "frequency": _frequency(temperature),
            "frequency_err": 0.05,
            "flags": [],
        }
        for index, temperature in enumerate(_TEMPERATURES)
    ]
    rows.append(
        {
            "run": _FLAGGED_RUN,
            "x": 68.0,
            "frequency": 25.0,
            "frequency_err": 0.05,
            "flags": ["spurious_reseeded"],
        }
    )
    return TrendTable(
        order_key="temperature",
        columns=["run", "x", "frequency", "frequency_err", "flags"],
        rows=rows,
    )


def test_a_flagged_row_enters_and_is_named() -> None:
    fit = fit_trend(_trend(), "frequency", "OrderParameter", fixed={"alpha": 1.0})

    assert _FLAGGED_RUN in fit.runs
    assert fit.flagged == [{"run": _FLAGGED_RUN, "flags": ["spurious_reseeded"]}]
    assert fit.excluded == []
    # The wild value pulls the fit off the curve the clean points trace ...
    assert fit.reduced_chi_squared > 100.0


def test_excluding_the_flagged_row_recovers_tc_and_beta() -> None:
    fit = fit_trend(
        _trend(), "frequency", "OrderParameter", fixed={"alpha": 1.0}, exclude=[_FLAGGED_RUN]
    )

    assert fit.success
    assert fit.excluded == [{"run": _FLAGGED_RUN, "reason": "excluded"}]
    assert fit.flagged == []
    assert fit.to_dict()["n_points"] == len(_TEMPERATURES)
    assert fit.parameters["Tc"] == pytest.approx(_TC, rel=1e-3)
    assert fit.parameters["beta"] == pytest.approx(_BETA, rel=1e-2)
    assert fit.parameters["y0"] == pytest.approx(_Y0, rel=1e-3)
    assert "alpha" in fit.fixed
    assert "alpha" not in fit.uncertainties


def test_excluding_runs_and_windowing_x_are_both_reported() -> None:
    fit = fit_trend(
        _trend(),
        "frequency",
        "OrderParameter",
        fixed={"alpha": 1.0},
        exclude=[100, _FLAGGED_RUN],
        x_max=50.0,
    )

    reasons = {entry["run"]: entry["reason"] for entry in fit.excluded}
    assert reasons[100] == "excluded"
    assert reasons[112] == "outside the x range"
    assert all(_TEMPERATURES[run - 100] <= 50.0 for run in fit.runs)
    assert fit.success


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"param": "Lambda"}, "no column 'Lambda'"),
        ({"param": "frequency_err"}, "no column 'frequency_err'"),
        ({"expression": "NoSuchModel"}, "NoSuchModel"),
        ({"fixed": {"gamma": 1.0}}, "gamma is not a parameter of 'OrderParameter'"),
        ({"initial": {"Tn": 60.0}}, "Tn is not a parameter of 'OrderParameter'"),
        ({"exclude": [5]}, "Run\\(s\\) 5 are not in the trend"),
        ({"x_min": 60.0}, "point\\(s\\) are left to fit"),
    ],
)
def test_bad_requests_are_refused_naming_the_problem(kwargs, match) -> None:
    arguments = {"param": "frequency", "expression": "OrderParameter"} | kwargs
    param = arguments.pop("param")
    expression = arguments.pop("expression")
    with pytest.raises(ValueError, match=match):
        fit_trend(_trend(), param, expression, **arguments)


def test_a_turning_point_is_found_only_when_both_ends_clear_it() -> None:
    from asymmetry.core.workflow.trend_fit import turning_point

    x = np.array([5.0, 40.0, 60.0, 100.0, 200.0])
    errors = np.full(5, 0.01)
    # A minimum at 40 K both ends rise well clear of.
    assert turning_point(x, np.array([0.10, 0.02, 0.05, 0.4, 2.2]), errors) == 40.0
    # Monotonic: none.
    assert turning_point(x, np.array([0.01, 0.02, 0.05, 0.4, 2.2]), errors) is None
    # An interior dip inside the errors: none.
    assert turning_point(x, np.array([0.03, 0.02, 0.05, 0.4, 2.2]), errors) is None


def test_the_fit_reports_its_span_and_units() -> None:
    fit = fit_trend(
        _trend(), "frequency", "OrderParameter", fixed={"alpha": 1.0}, exclude=[_FLAGGED_RUN]
    )
    assert fit.x_fitted == (float(_TEMPERATURES[0]), float(_TEMPERATURES[-1]))
    assert fit.units["Tc"] == "K"
    assert fit.turning_point is None
