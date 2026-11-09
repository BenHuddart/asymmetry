"""Selected-pair consumers derive balances without replacing the base fallback."""

from dataclasses import replace

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.fourier.fb_input import build_fb_signal_dataset
from asymmetry.core.fourier.spectrum import fourier_grouping_digest
from asymmetry.core.maxent import MaxEntConfig, build_maxent_input
from asymmetry.core.representation.time import TimeFBAsymmetry
from asymmetry.core.simulate import (
    PeriodSpec,
    degrade_run,
    reduce_run_to_dataset,
    simulate_double_pulse_run,
    simulate_run,
    simulate_two_period_run,
)
from asymmetry.core.transform.grouping import grouping_balances
from asymmetry.core.transform.integral import integrate_run


def pair_run(canonical=False):
    label = "P_x" if canonical else "FB"
    grouping = {
        "groups": {1: [1], 2: [2], 3: [3], 4: [4]},
        "projections": [
            {"label": label, "forward_group": 1, "backward_group": 2, "alpha": 0.75},
            {"label": "UD", "forward_group": 3, "backward_group": 4, "alpha": 1.1},
        ],
        "forward_group": 1,
        "backward_group": 2,
        "vector_axis": label,
        "alpha": 1.3,
        "beta": 1.4,
        "first_good_bin": 0,
        "last_good_bin": 63,
        "deadtime_correction": False,
        "background_correction": False,
    }
    if canonical:
        grouping["alpha_x"] = 0.75
    return Run(
        7110,
        [Histogram(np.full(64, value), 0.04) for value in (100, 80, 120, 60)],
        metadata={"field": 0.0},
        grouping=grouping,
    )


@pytest.mark.parametrize("canonical", [False, True])
@pytest.mark.parametrize(
    "consumer", ["balances", "reduce", "fourier", "degrade", "recipe", "integral"]
)
def test_active_pair_consumers_keep_the_analytic_balance(canonical, consumer):
    run = pair_run(canonical)
    if consumer == "balances":
        assert grouping_balances(run.grouping) == (0.75, 1.0)
    elif consumer == "integral":
        assert integrate_run(run)[0] == pytest.approx(0.25)
    else:
        data = (
            TimeFBAsymmetry().compute(run)[0]
            if consumer == "recipe"
            else build_fb_signal_dataset(run)
            if consumer == "fourier"
            else reduce_run_to_dataset(degrade_run(run, 1) if consumer == "degrade" else run)
        )
        np.testing.assert_allclose(data.asymmetry, 0.25 if consumer == "recipe" else 25.0)
    assert run.grouping["alpha"] == 1.3


def test_pair_identity_wins_over_stale_selection_text_and_keeps_missing_fallback():
    run = pair_run()
    run.grouping.update(forward_group=3, backward_group=4)
    assert grouping_balances(run.grouping) == (1.1, 1.0)
    run.grouping["projections"][1].pop("alpha")
    assert grouping_balances(run.grouping) == (1.3, 1.0)


def test_fb_fourier_digest_tracks_the_selected_projection_balance():
    run = pair_run()
    before = fourier_grouping_digest(run, "fb_asymmetry")
    run.grouping["projections"][0]["alpha"] = 0.8
    assert fourier_grouping_digest(run, "fb_asymmetry") != before


@pytest.mark.parametrize("selected,expected", [([1, 2], 0.75), ([3, 4], 1.1)])
def test_maxent_uses_the_explicitly_selected_pair_balance(selected, expected):
    result = build_maxent_input(
        pair_run(),
        MaxEntConfig(
            mode="zf_lf",
            selected_group_ids=selected,
            auto_window=False,
            n_spectrum_points=32,
            f_max_mhz=3,
        ),
    )
    assert result.zf_lf_alpha == expected


@pytest.mark.parametrize("kind", ["single", "double", "periods"])
def test_simulation_defaults_capture_the_selected_pair_alpha(kind):
    def zero(time):
        return np.zeros_like(time)

    run = pair_run()
    if kind == "single":
        simulated = simulate_run(run, zero, total_events=10000, seed=12)
    elif kind == "double":
        simulated = simulate_double_pulse_run(run, zero, total_events=10000, seed=12, dpsep_us=0)
    else:
        simulated = simulate_two_period_run(
            run, [PeriodSpec(zero), PeriodSpec(zero)], total_events=10000, seed=12
        )
    provenance = simulated.metadata["simulation"]
    assert (
        [p["alpha"] for p in provenance["periods"]] == [0.75, 0.75]
        if kind == "periods"
        else provenance["alpha"] == 0.75
    )


def test_selected_pair_does_not_change_scalar_legacy_balances():
    run = pair_run()
    run = replace(run, grouping={"alpha": 0.8, "beta": 1.2})
    assert grouping_balances(run.grouping) == (0.8, 1.2)
