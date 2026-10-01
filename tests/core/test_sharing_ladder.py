"""The sharing ladder (plan D3, D7–D9, D12) on synthetic series.

Each series is simulated from the repo's own components with muon-decay noise
and modelled on a case in ``docs/plans/global-wizard-trend-objective.md``.
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.engine import FitCancelledError, FitEngine, FitResult
from asymmetry.core.fitting.global_search.sharing_ladder import (
    ADEQUACY_SIGMA,
    SharingLadder,
    climb_sharing_ladder,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet

_MUON_LIFETIME_US = 2.197
_LIMITS = {"A_1": 100.0, "A_2": 100.0, "Lambda": 50.0, "sigma": 50.0, "Delta": 10.0, "nu": 100.0}


class _Series:
    """A simulated series and its independent per-run fits, runs numbered 1, 2, 3…"""

    def __init__(
        self,
        expression: str,
        truths: list[dict[str, float]],
        *,
        sigma: float,
        n_points: int = 400,
        t_max: float = 10.0,
        fixed: tuple[str, ...] = (),
        axis: list[float] | None = None,
        starts: list[dict[str, float]] | None = None,
    ) -> None:
        """``starts`` are where each run's independent fit begins: the truth by default."""
        self.model = CompositeModel.from_expression(expression)
        self.axis = axis or [float(run) for run in range(1, len(truths) + 1)]
        rng = np.random.default_rng(11)
        time = np.linspace(0.02, t_max, n_points)
        # Counting errors grow as the muons decay.
        error = sigma * np.exp(time / (2.0 * _MUON_LIFETIME_US))
        self.datasets: list[MuonDataset] = []
        self.base_by_run: dict[int, ParameterSet] = {}
        self.all_local: dict[int, FitResult] = {}
        for run, (truth, start) in enumerate(zip(truths, starts or truths, strict=True), start=1):
            dataset = MuonDataset(
                time=time,
                asymmetry=self.model.function(time, **truth) + rng.normal(0.0, error),
                error=error,
                metadata={"run_number": run},
            )
            base = ParameterSet(
                [
                    Parameter(
                        name,
                        start[name],
                        min=0.1 if name == "beta" else 0.0 if name in _LIMITS else -np.inf,
                        max=3.0 if name == "beta" else _LIMITS.get(name, np.inf),
                        fixed=name in fixed,
                    )
                    for name in self.model.param_names
                ]
            )
            self.datasets.append(dataset)
            self.base_by_run[run] = base
            self.all_local[run] = FitEngine().fit(dataset, self.model.function, base)

    def climb(self, **options) -> SharingLadder:
        return climb_sharing_ladder(
            self.datasets,
            self.model,
            all_local_results=self.all_local,
            base_by_run=self.base_by_run,
            axis_values=self.axis,
            **{"cancel_callback": lambda: False, **options},
        )


def _glassy_series(*, runs_with_lost_asymmetry: int) -> _Series:
    """Stretched exponential + constant, the rate falling by three decades (YMnAl).

    The five slowest runs cannot tell amplitude from background: their
    independent fits start ten units of asymmetry along that valley and stay
    there. The exponent truly varies. The first ``runs_with_lost_asymmetry``
    runs carry less amplitude.
    """
    rates = np.logspace(0.5, -2.5, 12)
    exponents = np.linspace(0.5, 1.0, 12)
    truths = [
        {
            "A_1": 13.0 if index < runs_with_lost_asymmetry else 20.0,
            "Lambda": rates[index],
            "beta": exponents[index],
            "A_bg": 5.0,
        }
        for index in range(12)
    ]
    valley = [
        {**truth, "A_1": truth["A_1"] + 10.0, "A_bg": truth["A_bg"] - 10.0} for truth in truths
    ]
    return _Series(
        "StretchedExponential + Constant", truths, sigma=0.15, starts=truths[:7] + valley[7:]
    )


def _hopping_series(
    *, runs: int = 10, sigma: float = 0.15, extra_amplitude: tuple[int, ...] = ()
) -> _Series:
    """Dynamic Gaussian KT + constant: one static width, a hop rate rising (copper).

    The runs in ``extra_amplitude`` carry 2.5 % more asymmetry (Re₆Zr).
    """
    hop_rates = np.logspace(-1.5, 0.4, runs)
    return _Series(
        "DynamicGaussianKT + Constant",
        [
            {
                "A_1": 18.0 * (1.025 if run in extra_amplitude else 1.0),
                "Delta": 0.39,
                "nu": hop_rates[run - 1],
                "B_L": 0.0,
                "A_bg": 4.0,
            }
            for run in range(1, runs + 1)
        ],
        sigma=sigma,
        n_points=300,
        t_max=12.0,
        fixed=("B_L",),
    )


def test_first_rung_is_the_all_local_fits_at_no_cost() -> None:
    series = _hopping_series()
    first = series.climb(max_further_parameters=0).rungs[0]

    assert first.shared == ()
    assert first.results_by_run == series.all_local
    assert first.series_cost == 0.0
    assert set(first.run_costs.values()) == {0.0}
    assert first.adequate
    # Four free parameters on each of ten runs; the fixed field is not one.
    assert first.free_parameter_count == 40
    assert set(first.trend.parameters) == {"A_1", "Delta", "nu", "A_bg"}


def test_degenerate_amplitude_and_background_are_shared_and_the_rate_trends_better() -> None:
    series = _glassy_series(runs_with_lost_asymmetry=0)
    # The slow runs' independent fits sit in an unphysical basin, with small errors.
    for run in range(8, 13):
        slow = series.all_local[run]
        assert slow.parameters["A_1"].value == pytest.approx(30.0, abs=0.5)
        assert slow.parameters["A_bg"].value == pytest.approx(-5.0, abs=0.5)
        assert slow.uncertainties["A_1"] < 0.5

    ladder = series.climb()
    all_local, background, amplitude, exponent, rate = ladder.rungs

    assert [rung.shared for rung in ladder.rungs] == [
        (),
        ("A_bg",),
        ("A_bg", "A_1"),
        ("A_bg", "A_1", "beta"),
        ("A_bg", "A_1", "Lambda"),
    ]
    assert background.adequate and amplitude.adequate
    assert abs(amplitude.series_cost) < 0.5
    assert background.results_by_run[12].parameters["A_1"].value == pytest.approx(20.0, abs=1.0)
    assert amplitude.results_by_run[12].parameters["A_1"].value == pytest.approx(20.0, abs=0.3)
    assert amplitude.results_by_run[12].parameters["A_bg"].value == pytest.approx(5.0, abs=0.3)
    # A shared parameter counts once: 48 − 11 − 11.
    assert amplitude.free_parameter_count == 26
    # The exponent truly varies and the rate spans decades: neither can be shared,
    # and the rate is tried from the rung below the exponent's.
    assert exponent.series_cost > ADEQUACY_SIGMA and not exponent.adequate
    assert rate.series_cost > ADEQUACY_SIGMA and not rate.adequate

    assert ladder.preselected is amplitude
    assert set(amplitude.trend.parameters) == {"Lambda", "beta"}
    assert (
        amplitude.trend.parameters["Lambda"].quality > all_local.trend.parameters["Lambda"].quality
    )
    assert amplitude.trend.ordering_key > all_local.trend.ordering_key


def test_amplitude_lost_in_an_end_block_is_reported_and_only_the_background_is_shared() -> None:
    ladder = _glassy_series(runs_with_lost_asymmetry=3).climb(max_further_parameters=1)
    _all_local, background, amplitude, exponent = ladder.rungs

    assert amplitude.shared == ("A_bg", "A_1")
    assert amplitude.offending_runs == (1, 2, 3)
    assert amplitude.amplitude_unshareable_runs == (1, 2, 3)
    assert amplitude.exempt_runs == ()
    assert not amplitude.adequate
    # The amplitude is released: the exponent is tried on top of the background alone.
    assert exponent.shared == ("A_bg", "beta")
    assert not exponent.adequate
    assert ladder.preselected is background


def test_shared_static_width_leaves_a_smooth_local_hop_rate() -> None:
    ladder = _hopping_series().climb()
    width, hop_rate = ladder.rungs[-2:]

    assert width.shared == ("A_bg", "A_1", "Delta")
    assert width.adequate
    assert ladder.preselected is width
    assert set(width.trend.parameters) == {"nu"}
    assert width.trend.parameters["nu"].quality > 0.95
    assert width.results_by_run[1].parameters["Delta"].value == pytest.approx(0.39, abs=0.01)

    assert hop_rate.shared == ("A_bg", "A_1", "Delta", "nu")
    assert hop_rate.series_cost > 100.0 and not hop_rate.adequate
    # Nothing is left local, so there is no trend to rank.
    assert not hop_rate.trend.is_trend


def test_two_isolated_anomalous_runs_are_exempt_from_the_shared_amplitude() -> None:
    ladder = _hopping_series(runs=14, sigma=0.09, extra_amplitude=(4, 10)).climb()
    amplitude, width, hop_rate = ladder.rungs[2:]

    assert amplitude.shared == ("A_bg", "A_1")
    assert amplitude.exempt_runs == (4, 10)
    assert amplitude.adequate
    assert amplitude.amplitude_unshareable_runs == ()
    values = {
        run: result.parameters["A_1"].value for run, result in amplitude.results_by_run.items()
    }
    assert {values[run] for run in values if run not in (4, 10)} == {values[1]}
    assert values[1] == pytest.approx(18.0, abs=0.1)
    assert values[4] == pytest.approx(18.45, abs=0.2)
    assert values[10] == pytest.approx(18.45, abs=0.2)
    # 56 − 13 for the background − 11 for an amplitude twelve runs share.
    assert amplitude.free_parameter_count == 32
    # A shared amplitude has no trace, exempt runs or not.
    assert "A_1" not in amplitude.trend.parameters

    # The rungs above inherit the exemption.
    assert width.exempt_runs == hop_rate.exempt_runs == (4, 10)
    assert ladder.preselected is width


def test_three_contiguous_interior_anomalous_runs_are_not_exempted() -> None:
    ladder = _hopping_series(runs=14, sigma=0.09, extra_amplitude=(6, 7, 8)).climb(
        max_further_parameters=1
    )
    amplitude, width = ladder.rungs[2:]

    assert amplitude.shared == ("A_bg", "A_1")
    assert {6, 7, 8} <= set(amplitude.offending_runs)
    assert amplitude.exempt_runs == ()
    assert amplitude.amplitude_unshareable_runs == ()
    assert not amplitude.adequate
    assert width.shared == ("A_bg", "Delta")
    assert ladder.preselected is width


def test_shared_rate_under_a_local_amplitude_is_flagged_hard_to_justify() -> None:
    # Two components trading volume fraction; the exponential's rate is constant.
    fraction = np.linspace(0.2, 0.8, 10)
    series = _Series(
        "Exponential + Gaussian + Constant",
        [
            {
                "A_1": 20.0 * fraction[index],
                "Lambda": 1.5,
                "A_2": 20.0 * (1.0 - fraction[index]),
                "sigma": 0.25 + 0.03 * index,
                "A_bg": 3.0,
            }
            for index in range(10)
        ],
        sigma=0.15,
    )
    ladder = series.climb()
    _all_local, background, amplitudes, rate, width = ladder.rungs

    assert amplitudes.shared == ("A_bg", "A_1", "A_2")
    assert not amplitudes.adequate
    assert background.hard_to_justify == amplitudes.hard_to_justify == ()

    assert rate.shared == ("A_bg", "Lambda")
    assert rate.adequate
    assert rate.hard_to_justify == ("Lambda",)
    # A flag, not a veto.
    assert ladder.preselected is rate

    assert width.shared == ("A_bg", "Lambda", "sigma")
    assert width.hard_to_justify == ("Lambda", "sigma")
    assert not width.adequate


def test_passes_that_disagree_are_named_on_the_rungs_that_keep_the_rate_local() -> None:
    # A coarse pass, then an infill pass whose rates sit 25 % higher.
    axis = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0]
    series = _Series(
        "Exponential + Constant",
        [
            {"A_1": 20.0, "Lambda": 0.01 * x * (1.0 if index < 6 else 1.25), "A_bg": 3.0}
            for index, x in enumerate(axis)
        ],
        sigma=0.15,
        axis=axis,
    )
    ladder = series.climb()

    assert list(ladder.rungs[0].results_by_run) == [1, 7, 2, 8, 3, 9, 4, 10, 5, 11, 6, 12]
    for rung in ladder.rungs[:3]:
        assert set(rung.pass_disagreements) == {"Lambda"}
        assert rung.pass_disagreements["Lambda"].passes == (
            (1, 2, 3, 4, 5, 6),
            (7, 8, 9, 10, 11, 12),
        )
    assert ladder.rungs[3].shared == ("A_bg", "A_1", "Lambda")
    assert ladder.rungs[3].pass_disagreements == {}


def test_budget_limits_the_further_parameters() -> None:
    series = _hopping_series()

    assert [rung.shared for rung in series.climb(max_further_parameters=0).rungs] == [
        (),
        ("A_bg",),
        ("A_bg", "A_1"),
    ]
    assert series.climb(max_further_parameters=1).rungs[-1].shared == ("A_bg", "A_1", "Delta")


def test_rung_that_does_not_converge_is_reported_and_the_ladder_goes_on_without_it() -> None:
    # Ten evaluations are too few for the background and amplitude rungs here,
    # and enough for the width's.
    ladder = _hopping_series().climb(max_calls=10)
    _all_local, background, amplitude, width, hop_rate = ladder.rungs

    assert not background.converged and not background.adequate
    assert amplitude.shared == ("A_1",)
    assert not amplitude.converged and not amplitude.adequate
    assert width.shared == ("Delta",)
    assert width.converged and width.adequate
    assert hop_rate.shared == ("Delta", "nu")
    assert ladder.preselected is width


def test_cancel_is_polled_before_each_rung() -> None:
    series = _hopping_series()
    polls = iter([False, True])

    with pytest.raises(FitCancelledError):
        series.climb(cancel_callback=lambda: next(polls, True))


def test_ladder_refuses_an_all_local_fit_that_did_not_converge() -> None:
    series = _hopping_series()
    series.all_local[3] = FitResult(success=False)

    with pytest.raises(ValueError, match=r"runs \[3\]"):
        series.climb()
