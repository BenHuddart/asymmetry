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
_FLAGGED_RUN = "199"


def _frequency(temperature: float) -> float:
    return _Y0 * (1.0 - temperature / _TC) ** _BETA


def _trend() -> TrendTable:
    rows = [
        {
            "key": str(100 + index),
            "x": float(temperature),
            "frequency": _frequency(temperature),
            "frequency_err": 0.05,
            "flags": [],
        }
        for index, temperature in enumerate(_TEMPERATURES)
    ]
    rows.append(
        {
            "key": _FLAGGED_RUN,
            "x": 68.0,
            "frequency": 25.0,
            "frequency_err": 0.05,
            "flags": ["spurious_reseeded"],
        }
    )
    return TrendTable(
        order_key="temperature",
        columns=["key", "x", "frequency", "frequency_err", "flags"],
        rows=rows,
    )


def test_a_flagged_row_enters_and_is_named() -> None:
    fit = fit_trend(_trend(), "frequency", "OrderParameter", fixed={"alpha": 1.0})

    assert _FLAGGED_RUN in fit.keys
    assert fit.flagged == [{"key": _FLAGGED_RUN, "flags": ["spurious_reseeded"]}]
    assert fit.excluded == []
    # The wild value pulls the fit off the curve the clean points trace ...
    assert fit.reduced_chi_squared > 100.0


def test_excluding_the_flagged_row_recovers_tc_and_beta() -> None:
    fit = fit_trend(
        _trend(), "frequency", "OrderParameter", fixed={"alpha": 1.0}, exclude=[_FLAGGED_RUN]
    )

    assert fit.success
    assert fit.excluded == [{"key": _FLAGGED_RUN, "reason": "excluded"}]
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
        exclude=["100", _FLAGGED_RUN],
        x_max=50.0,
    )

    reasons = {entry["key"]: entry["reason"] for entry in fit.excluded}
    assert reasons["100"] == "excluded"
    assert reasons["112"] == "outside the x range"
    assert all(_TEMPERATURES[int(key) - 100] <= 50.0 for key in fit.keys)
    assert fit.success


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"param": "Lambda"}, "no column 'Lambda'"),
        ({"param": "frequency_err"}, "no column 'frequency_err'"),
        ({"expression": "NoSuchModel"}, "NoSuchModel"),
        ({"fixed": {"gamma": 1.0}}, "gamma is not a parameter of 'OrderParameter'"),
        ({"initial": {"Tn": 60.0}}, "Tn is not a parameter of 'OrderParameter'"),
        ({"exclude": ["5"]}, "Not in the trend: 5 \\(it holds 100, "),
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


def _stored_fit(**changes) -> dict:
    """A stored Linear trend fit as ``trend --model`` writes it."""
    return {
        "param": "Lambda",
        "expression": "Linear",
        "success": True,
        "parameters": {"m": 0.5, "b": 0.1},
        "uncertainties": {"m": 0.02, "b": 0.01},
        "reduced_chi_squared": 4.0,
        "params_at_bound": [],
    } | changes


def test_a_fit_trend_row_carries_the_scaled_error_and_the_fits_state() -> None:
    from asymmetry.core.workflow.series import ScanAxis
    from asymmetry.core.workflow.trend_fit import fit_trend_table

    fits = {
        "hot": _stored_fit(),
        "cold": _stored_fit(success=False, reduced_chi_squared=50.0),
        "warm": _stored_fit(params_at_bound=["m"], uncertainties={"b": 0.01}),
    }
    axis = ScanAxis("temperature", {"hot": 300.0, "cold": 200.0, "warm": 250.0})

    trend = fit_trend_table(fits, "m", axis)

    assert trend.columns == ["key", "x", "m", "m_err", "flags"]
    rows = {row["key"]: row for row in trend.rows}
    assert [row["key"] for row in trend.rows] == ["cold", "warm", "hot"]
    # sqrt(chi2_red) = 2 scales a converged fit's error; a failed fit's is left alone.
    assert rows["hot"]["m_err"] == pytest.approx(0.04)
    assert rows["cold"]["m_err"] == pytest.approx(0.02)
    assert rows["cold"]["flags"] == ["failed"]
    # A parameter the law held has no error, so the row enters no law.
    assert rows["warm"]["m_err"] is None
    assert rows["warm"]["flags"] == ["bound_pinned"]


def test_a_fit_trend_of_a_parameter_the_law_lacks_is_refused() -> None:
    from asymmetry.core.workflow.series import ScanAxis
    from asymmetry.core.workflow.trend_fit import fit_trend_table

    with pytest.raises(ValueError, match="The trend fit of a has no parameter 'Ea'"):
        fit_trend_table({"a": _stored_fit()}, "Ea", ScanAxis("temperature", {"a": 1.0}))
