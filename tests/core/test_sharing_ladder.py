"""The sharing ladder (plan D3, D7–D9, D11–D12) on synthetic series.

Each series is simulated from the repo's own components with muon-decay noise
and modelled on a case in ``docs/plans/global-wizard-trend-objective.md``.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from asymmetry.core.fitting.engine import FitCancelledError, FitResult
from asymmetry.core.fitting.global_search.sharing_ladder import ADEQUACY_SIGMA
from tests.core.sharing_series import (
    TRANSITION,
    SimulatedSeries,
    glassy_series,
    hopping_series,
    two_line_series,
)


def test_first_rung_is_the_all_local_fits_at_no_cost() -> None:
    series = hopping_series()
    first = series.climb(further=0).rungs[0]

    assert first.shared == ()
    assert first.model is series.model
    assert first.results_by_run == series.all_local
    assert first.series_cost == 0.0
    assert set(first.run_costs.values()) == {0.0}
    assert first.adequate
    # Four free parameters on each of ten runs; the fixed field is not one.
    assert first.free_parameter_count == 40
    assert set(first.trend.parameters) == {"A_1", "Delta", "nu", "A_bg"}


def test_degenerate_amplitude_and_background_are_shared_and_the_rate_trends_better() -> None:
    series = glassy_series(runs_with_lost_asymmetry=0)
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
    ladder = glassy_series(runs_with_lost_asymmetry=3).climb(further=1)
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
    ladder = hopping_series().climb()
    width, hop_rate = ladder.rungs[-2:]

    assert width.shared == ("A_bg", "A_1", "Delta")
    assert width.adequate
    assert width.exempt_runs == ()
    assert ladder.preselected is width
    assert set(width.trend.parameters) == {"nu"}
    assert width.trend.parameters["nu"].quality > 0.95
    assert width.results_by_run[1].parameters["Delta"].value == pytest.approx(0.39, abs=0.01)

    assert hop_rate.shared == ("A_bg", "A_1", "Delta", "nu")
    assert hop_rate.series_cost > 100.0 and not hop_rate.adequate
    # Nothing is left local, so there is no trend to rank.
    assert not hop_rate.trend.is_trend


def test_two_isolated_anomalous_runs_are_exempt_from_the_shared_amplitude() -> None:
    ladder = hopping_series(runs=14, sigma=0.09, amplitude_scale={4: 1.025, 10: 1.025}).climb()
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


@pytest.mark.parametrize("runs", [2, 3])
def test_exemptions_always_leave_two_runs_to_share_the_amplitude(runs: int) -> None:
    # The amplitudes disagree far beyond the tolerance and nothing is a block of three.
    series = hopping_series(runs=runs, sigma=0.05, amplitude_scale={1: 1.3, 2: 0.8})
    amplitude = series.climb(further=0).rungs[2]

    assert amplitude.shared == ("A_bg", "A_1")
    assert len(amplitude.results_by_run) - len(amplitude.exempt_runs) >= 2
    assert not amplitude.adequate


@pytest.mark.parametrize("excess", [0.025, 0.04, 0.10, 0.30])
def test_isolated_anomalous_runs_are_exempt_however_strong_the_anomaly(excess: float) -> None:
    # At this noise even 2.5 % drags a plainly shared amplitude until other runs offend.
    series = hopping_series(
        runs=12, sigma=0.05, amplitude_scale={4: 1.0 + excess, 10: 1.0 + excess}
    )
    amplitude = series.climb(further=0).rungs[2]

    assert amplitude.exempt_runs == (4, 10)
    assert amplitude.adequate
    assert amplitude.amplitude_unshareable_runs == ()
    assert amplitude.results_by_run[1].parameters["A_1"].value == pytest.approx(18.0, abs=0.1)


def test_three_isolated_anomalous_runs_of_a_long_series_are_exempt() -> None:
    series = hopping_series(runs=38, sigma=0.09, amplitude_scale={5: 1.02, 17: 1.02, 30: 1.02})
    amplitude = series.climb(further=0).rungs[2]

    assert amplitude.exempt_runs == (5, 17, 30)
    assert amplitude.adequate
    # 152 − 37 for the background − 34 for an amplitude 35 runs share.
    assert amplitude.free_parameter_count == 81


def test_anomalous_runs_the_rung_below_cannot_place_are_found_by_their_cost() -> None:
    # No background, so the amplitude rung climbs from the all-local fits, and
    # these gave no covariance to judge an amplitude by.
    series = hopping_series(
        runs=14, sigma=0.09, amplitude_scale={4: 1.025, 10: 1.025}, background=False
    )
    series.all_local = {
        run: replace(result, covariance=None, covariance_parameters=[])
        for run, result in series.all_local.items()
    }
    amplitude = series.climb(further=0).rungs[1]

    assert amplitude.shared == ("A_1",)
    assert amplitude.exempt_runs == (4, 10)
    assert amplitude.adequate


@pytest.mark.parametrize("reduction", [0.03, 0.35])
@pytest.mark.parametrize("block", [(1, 2, 3, 4), (12, 13, 14)])
def test_end_block_with_less_amplitude_is_named_exactly(
    block: tuple[int, ...], reduction: float
) -> None:
    series = hopping_series(
        runs=14, sigma=0.09, amplitude_scale=dict.fromkeys(block, 1.0 - reduction)
    )
    amplitude = series.climb(further=0).rungs[2]

    assert amplitude.amplitude_unshareable_runs == block
    assert amplitude.exempt_runs == ()
    assert not amplitude.adequate


@pytest.mark.parametrize("excess", [0.025, 0.10])
def test_three_contiguous_interior_anomalous_runs_are_not_exempted(excess: float) -> None:
    ladder = hopping_series(
        runs=14, sigma=0.09, amplitude_scale=dict.fromkeys((6, 7, 8), 1.0 + excess)
    ).climb(further=1)
    amplitude, width = ladder.rungs[2:]

    assert amplitude.shared == ("A_bg", "A_1")
    assert {6, 7, 8} <= set(amplitude.offending_runs)
    assert amplitude.exempt_runs == ()
    assert amplitude.amplitude_unshareable_runs == ()
    assert not amplitude.adequate
    assert width.shared == ("A_bg", "Delta")
    assert ladder.preselected is width


@pytest.mark.parametrize(
    "amplitude_scale",
    [
        {run: 1.0 + 0.05 * (run - 1) / 13 for run in range(1, 15)},
        {run: 1.0 + 0.20 * (run - 1) / 13 for run in range(1, 15)},
        {run: 1.0 + 0.03 * np.sin(1.9 * run) for run in range(1, 15)},
    ],
    ids=["drift of 5 %", "drift of 20 %", "wander of 3 %"],
)
def test_amplitude_that_varies_across_the_series_is_neither_exempted_nor_an_end_block(
    amplitude_scale: dict[int, float],
) -> None:
    series = hopping_series(runs=14, sigma=0.09, amplitude_scale=amplitude_scale)
    background, amplitude = series.climb(further=0).rungs[1:]

    assert amplitude.exempt_runs == ()
    assert amplitude.amplitude_unshareable_runs == ()
    assert not amplitude.adequate
    # The amplitude stays local, where its trace is the finding.
    assert "A_1" in background.trend.parameters


def test_total_is_shared_and_the_fraction_trends_through_a_transition() -> None:
    series = two_line_series(TRANSITION)
    ladder = series.climb()
    _all_local, background, total, amplitudes, *further = ladder.rungs

    assert background.model is series.model
    assert total.model.component_expression_string() == (
        "(Oscillatory * Gaussian + Oscillatory * Gaussian){frac} + Constant"
    )
    assert total.shared == ("A_bg", "A_1")
    assert total.adequate and abs(total.series_cost) < 0.5
    # A total and a fraction stand for two amplitudes: 7 on each of 12 runs, less 11 + 11.
    assert total.free_parameter_count == 62
    assert set(total.results_by_run[1].parameters.names) == set(total.model.param_names)
    assert total.results_by_run[1].parameters["A_1"].value == pytest.approx(20.0, abs=0.1)
    fractions = [
        result.parameters["f_Oscillatory"].value for result in total.results_by_run.values()
    ]
    assert fractions == pytest.approx(list(TRANSITION), abs=0.01)
    assert total.trend.parameters["f_Oscillatory"].quality > 0.95
    # The shared total has no trace; the widths keep theirs under their grouped names.
    assert set(total.trend.parameters) == {
        "frequency_1",
        "f_Oscillatory",
        "sigma_1",
        "frequency_2",
        "sigma_2",
    }

    assert amplitudes.model is series.model
    assert amplitudes.shared == ("A_bg", "A_1", "A_3")
    assert amplitudes.series_cost > 100.0 and not amplitudes.adequate
    assert amplitudes.exempt_runs == () and amplitudes.amplitude_unshareable_runs == ()

    # Every further parameter is tried on top of the total, in its form.
    assert [rung.shared for rung in further] == [
        ("A_bg", "A_1", "frequency_1"),
        ("A_bg", "A_1", "frequency_1", "sigma_1"),
        ("A_bg", "A_1", "frequency_1", "frequency_2"),
        ("A_bg", "A_1", "frequency_1", "frequency_2", "sigma_2"),
    ]
    assert all(rung.model is total.model for rung in further)
    assert [rung.adequate for rung in further] == [True, False, True, False]
    # Each line's amplitude is local through its fraction, so its shared
    # frequency and width are flagged, the Gaussian factor's included.
    assert further[1].hard_to_justify == ("frequency_1", "sigma_1")
    assert ladder.preselected is further[2]


def test_constant_amplitudes_are_shared_outright_and_the_ladder_goes_on_from_there() -> None:
    series = two_line_series([0.6] * 12, fixed=("phase_1", "phase_3", "frequency_1", "frequency_3"))
    ladder = series.climb(further=1)
    _all_local, _background, total, amplitudes, width = ladder.rungs

    # The total can be shared too, and leaves a fraction that does not move.
    assert total.model.fraction_groups and total.adequate
    assert total.trend.parameters["f_Oscillatory"].quality < 0.5
    assert amplitudes.model is series.model
    assert amplitudes.shared == ("A_bg", "A_1", "A_3")
    assert amplitudes.adequate
    assert ladder.preselected is amplitudes
    # The later adequate rung is the foothold, in the template as given.
    assert width.model is series.model
    assert width.shared == ("A_bg", "A_1", "A_3", "sigma_2")
    assert width.hard_to_justify == ()
    assert not width.adequate


def test_lines_of_opposite_sign_have_no_shared_total_rung() -> None:
    series = two_line_series(TRANSITION, second_line_sign=-1.0)
    ladder = series.climb(further=2)
    _all_local, _background, amplitudes, frequency, width = ladder.rungs

    assert all(rung.model is series.model for rung in ladder.rungs)
    assert amplitudes.shared == ("A_bg", "A_1", "A_3")
    assert not amplitudes.adequate
    # In the template as given a shared width is flagged under the local
    # amplitude of the line it multiplies.
    assert frequency.shared == ("A_bg", "frequency_1")
    assert frequency.hard_to_justify == ("frequency_1",)
    assert width.shared == ("A_bg", "frequency_1", "sigma_2")
    assert width.hard_to_justify == ("frequency_1", "sigma_2")


def test_line_that_vanishes_is_a_fraction_at_its_limit_not_a_sign() -> None:
    # The second line is absent from the first three runs, where its fitted
    # amplitude falls either side of zero by less than its error.
    series = two_line_series(
        np.clip(1.25 - 0.1 * np.arange(12), 0.0, 1.0),
        fixed=("phase_1", "phase_3", "frequency_1", "frequency_3", "sigma_4"),
    )
    absent = [series.all_local[run].parameters["A_3"].value for run in (1, 2, 3)]
    assert min(absent) < 0.0 and max(np.abs(absent)) < 0.1

    total = series.climb(further=0).rungs[2]

    assert total.model.fraction_groups
    assert total.shared == ("A_bg", "A_1")
    assert total.adequate
    fractions = [
        result.parameters["f_Oscillatory"].value for result in total.results_by_run.values()
    ]
    assert fractions[:3] == pytest.approx([1.0, 1.0, 1.0], abs=0.005)
    assert fractions[-1] == pytest.approx(0.15, abs=0.01)


def test_shared_rate_under_a_local_fraction_is_flagged_hard_to_justify() -> None:
    # Two components trading volume fraction; the exponential's rate is constant.
    fraction = np.linspace(0.2, 0.8, 10)
    series = SimulatedSeries(
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
    _all_local, background, total, amplitudes, rate, width = ladder.rungs

    assert total.shared == ("A_bg", "A_1") and total.model.fraction_groups
    assert total.adequate
    assert amplitudes.shared == ("A_bg", "A_1", "A_2")
    assert not amplitudes.adequate
    assert background.hard_to_justify == total.hard_to_justify == amplitudes.hard_to_justify == ()

    assert rate.shared == ("A_bg", "A_1", "Lambda")
    assert rate.adequate
    assert rate.hard_to_justify == ("Lambda",)
    # A flag, not a veto.
    assert ladder.preselected is rate

    assert width.shared == ("A_bg", "A_1", "Lambda", "sigma")
    assert width.hard_to_justify == ("Lambda", "sigma")
    assert not width.adequate


def test_passes_that_disagree_are_named_on_the_rungs_that_keep_the_rate_local() -> None:
    # A coarse pass, then an infill pass whose rates sit 25 % higher.
    axis = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0]
    series = SimulatedSeries(
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


def test_budget_ends_the_climb_before_a_further_parameter() -> None:
    series = hopping_series()

    assert [rung.shared for rung in series.climb(further=0).rungs] == [
        (),
        ("A_bg",),
        ("A_bg", "A_1"),
    ]
    assert series.climb(further=1).rungs[-1].shared == ("A_bg", "A_1", "Delta")


def test_rung_that_does_not_converge_is_reported_and_the_ladder_goes_on_without_it() -> None:
    # Ten evaluations are too few for the background and amplitude rungs here,
    # and enough for the width's.
    ladder = hopping_series().climb(max_calls=10)
    _all_local, background, amplitude, width, hop_rate = ladder.rungs

    assert not background.converged and not background.adequate
    assert amplitude.shared == ("A_1",)
    assert not amplitude.converged and not amplitude.adequate
    assert width.shared == ("Delta",)
    assert width.converged and width.adequate
    assert hop_rate.shared == ("Delta", "nu")
    assert ladder.preselected is width


def test_cancel_is_polled_before_each_rung() -> None:
    series = hopping_series()
    polls = iter([False, True])

    with pytest.raises(FitCancelledError):
        series.climb(cancel_callback=lambda: next(polls, True))


def test_ladder_refuses_an_all_local_fit_that_did_not_converge() -> None:
    series = hopping_series()
    series.all_local[3] = FitResult(success=False)

    with pytest.raises(ValueError, match=r"runs \[3\]"):
        series.climb()
