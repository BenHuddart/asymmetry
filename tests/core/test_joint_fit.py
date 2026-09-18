"""Acceptance tests for the joint fit: several series, different models, shared columns.

A joint fit is the generalisation of ``global_fit`` from "one model over several
runs" to "several models over several series, coupled by named shared columns".
These tests pin the three things that generalisation must get right: the shared
column really is one fitted quantity (same value *and* same σ in both series'
results), a joint fit with nothing shared decomposes back into independent
global fits, and every malformed shared table raises rather than fitting
something subtly wrong.

The last test in the file is the regression gate on the other half of the
change: ``global_fit`` now goes through the joint builder, and its fitted
values, χ² and dof must be what they were before the builder learned about
blocks. The literals were captured from ``main`` before the change.
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.engine import FitCancelledError, FitEngine
from asymmetry.core.fitting.joint import (
    JointSeriesProblem,
    SharedParameter,
    fit_joint,
)
from asymmetry.core.fitting.models import MODELS
from asymmetry.core.fitting.parameters import AffineTie, Parameter, ParameterSet

#: Shared background both synthetic series sit on.
_A_BG_TRUE = 1.7


def _oscillating(t, A, freq, Lambda, A_bg):  # noqa: N803 (conventional muSR symbols)
    """The ordered phase: a damped precession on the shared background."""
    return A * np.cos(2.0 * np.pi * freq * np.asarray(t)) * np.exp(-Lambda * np.asarray(t)) + A_bg


def _relaxing(t, A, Lambda, A_bg):  # noqa: N803 (conventional muSR symbols)
    """The paramagnetic phase: a plain relaxation on the same background."""
    return A * np.exp(-Lambda * np.asarray(t)) + A_bg


def _dataset(run_number: int, time, values, sigma) -> MuonDataset:
    return MuonDataset(
        time=time,
        asymmetry=values,
        error=np.full_like(time, sigma),
        metadata={"run_number": run_number},
    )


def _oscillating_series(
    *, seed: int = 11, sigma: float = 0.3, n_points: int = 200
) -> JointSeriesProblem:
    rng = np.random.default_rng(seed)
    time = np.linspace(0.05, 6.0, n_points)
    datasets = []
    inits: dict[int, ParameterSet] = {}
    for run, (freq, lam) in zip((101, 102), ((1.1, 0.35), (1.4, 0.55)), strict=True):
        clean = _oscillating(time, A=18.0, freq=freq, Lambda=lam, A_bg=_A_BG_TRUE)
        datasets.append(_dataset(run, time, clean + rng.normal(0.0, sigma, time.size), sigma))
        params = ParameterSet()
        params.add(Parameter("A", 16.0, min=0.0))
        params.add(Parameter("A_bg", 0.0))
        params.add(Parameter("freq", freq, min=0.0))
        params.add(Parameter("Lambda", 0.4, min=0.0))
        inits[run] = params
    return JointSeriesProblem(
        key="ordered",
        datasets=datasets,
        model_fn=_oscillating,
        global_params=["A", "A_bg"],
        local_params=["freq", "Lambda"],
        initial_params=inits,
    )


def _relaxing_series(
    *, seed: int = 23, sigma: float = 0.3, n_points: int = 200
) -> JointSeriesProblem:
    rng = np.random.default_rng(seed)
    time = np.linspace(0.05, 6.0, n_points)
    datasets = []
    inits: dict[int, ParameterSet] = {}
    for run, lam in zip((201, 202), (0.25, 0.65), strict=True):
        clean = _relaxing(time, A=14.0, Lambda=lam, A_bg=_A_BG_TRUE)
        datasets.append(_dataset(run, time, clean + rng.normal(0.0, sigma, time.size), sigma))
        params = ParameterSet()
        params.add(Parameter("A", 13.0, min=0.0))
        params.add(Parameter("A_bg", 0.0))
        params.add(Parameter("Lambda", 0.4, min=0.0))
        inits[run] = params
    return JointSeriesProblem(
        key="paramagnetic",
        datasets=datasets,
        model_fn=_relaxing,
        global_params=["A", "A_bg"],
        local_params=["Lambda"],
        initial_params=inits,
    )


def _shared_background(value: float = 0.0) -> SharedParameter:
    return SharedParameter(
        name="A_bg_shared",
        members={"ordered": "A_bg", "paramagnetic": "A_bg"},
        value=value,
    )


# --------------------------------------------------------------------------- #
# The shared column really is one fitted quantity
# --------------------------------------------------------------------------- #


def test_two_models_sharing_a_background_recover_it_with_one_uncertainty() -> None:
    """The shared value is fitted once, and both series report the *same* σ for it."""
    problems = [_oscillating_series(), _relaxing_series()]
    result = fit_joint(problems, [_shared_background()])

    assert result.success
    shared = result.shared_parameters["A_bg_shared"].value
    sigma = result.shared_uncertainties["A_bg_shared"]
    assert shared == pytest.approx(_A_BG_TRUE, abs=5.0 * sigma)

    # Every run of every series carries the shared value under its own name...
    for key in ("ordered", "paramagnetic"):
        assert result.series_global_parameters[key]["A_bg"].value == pytest.approx(shared)
        for run_result in result.series_results[key].values():
            assert run_result.parameters["A_bg"].value == pytest.approx(shared)
            # ...and the one shared uncertainty, not a per-series re-estimate.
            assert run_result.uncertainties["A_bg"] == pytest.approx(sigma, rel=1e-12)

    # The non-shared globals stay per-series: two different amplitudes.
    assert result.series_global_parameters["ordered"]["A"].value == pytest.approx(18.0, abs=0.5)
    assert result.series_global_parameters["paramagnetic"]["A"].value == pytest.approx(
        14.0, abs=0.5
    )

    assert result.shared_covariance is not None
    assert result.shared_covariance.shape == (1, 1)
    assert float(np.sqrt(result.shared_covariance[0, 0])) == pytest.approx(sigma, rel=1e-9)

    # Combined bookkeeping: every point, every free column, counted once.
    total_points = sum(len(ds.time) for problem in problems for ds in problem.datasets)
    # 1 shared + (A, freq×2, Lambda×2) + (A, Lambda×2) = 1 + 5 + 3 columns.
    assert result.dof == total_points - 9
    assert result.reduced_chi_squared == pytest.approx(result.chi_squared / result.dof)
    assert set(result.series_reduced_chi_squared) == {"ordered", "paramagnetic"}


def test_an_empty_shared_table_reproduces_two_independent_global_fits() -> None:
    """With nothing shared the blocks are uncoupled, so each must fit as it would alone."""
    problems = [_oscillating_series(), _relaxing_series()]
    joint = fit_joint(problems, [])

    engine = FitEngine()
    for problem in problems:
        alone, alone_global = engine.global_fit(
            problem.datasets,
            problem.model_fn,
            problem.global_params,
            problem.local_params,
            problem.initial_params,
            strategy="least_squares",
        )
        # Two separate trust-region solves of the same objective: the minimum is
        # the same, the last few digits of the path to it are not.
        for pname in problem.global_params:
            assert joint.series_global_parameters[problem.key][pname].value == pytest.approx(
                alone_global[pname].value, rel=1e-4
            )
        for run, run_result in alone.items():
            packed = joint.series_results[problem.key][run]
            assert packed.chi_squared == pytest.approx(run_result.chi_squared, rel=1e-4)
            for pname in problem.local_params:
                assert packed.parameters[pname].value == pytest.approx(
                    run_result.parameters[pname].value, rel=1e-4
                )


def test_the_two_strategies_agree_on_the_shared_value() -> None:
    """Minuit over the whole vector and the trust region reach the same minimum."""
    problems = [_oscillating_series(), _relaxing_series()]
    sparse = fit_joint(problems, [_shared_background()], strategy="least_squares")
    minuit = fit_joint(problems, [_shared_background()], strategy="joint")

    sigma = sparse.shared_uncertainties["A_bg_shared"]
    assert minuit.shared_parameters["A_bg_shared"].value == pytest.approx(
        sparse.shared_parameters["A_bg_shared"].value, abs=0.05 * sigma
    )
    assert minuit.chi_squared == pytest.approx(sparse.chi_squared, abs=0.1)
    for key in ("ordered", "paramagnetic"):
        assert minuit.series_global_parameters[key]["A"].value == pytest.approx(
            sparse.series_global_parameters[key]["A"].value, rel=1e-3
        )


def test_a_run_that_pins_a_shared_member_keeps_its_own_value() -> None:
    """A pinned value is a measurement; the shared column must not overwrite it."""
    ordered = _oscillating_series()
    pinned = ordered.initial_params[101]
    pinned.add(Parameter("A_bg", 4.2, fixed=True))

    result = fit_joint([ordered, _relaxing_series()], [_shared_background()])

    assert result.series_results["ordered"][101].parameters["A_bg"].value == pytest.approx(4.2)
    assert result.series_results["ordered"][101].parameters["A_bg"].fixed
    # Every other run still reports the one fitted shared value.
    shared = result.shared_parameters["A_bg_shared"].value
    assert shared != pytest.approx(4.2)
    for run in (102, 201, 202):
        key = "ordered" if run == 102 else "paramagnetic"
        assert result.series_results[key][run].parameters["A_bg"].value == pytest.approx(shared)


def test_an_equality_link_group_inside_one_series_still_shares_one_column() -> None:
    """A WiMDA link group behaves exactly as it does in ``global_fit``."""
    ordered = _oscillating_series()
    linked = JointSeriesProblem(
        key=ordered.key,
        datasets=ordered.datasets,
        model_fn=ordered.model_fn,
        global_params=ordered.global_params,
        local_params=ordered.local_params,
        initial_params=ordered.initial_params,
        local_param_groups={"Lambda": {101: "both", 102: "both"}},
    )

    result = fit_joint([linked, _relaxing_series()], [_shared_background()])

    first = result.series_results["ordered"][101].parameters["Lambda"].value
    second = result.series_results["ordered"][102].parameters["Lambda"].value
    assert first == pytest.approx(second, rel=1e-12)
    # The paramagnetic series' locals stay independent.
    assert result.series_results["paramagnetic"][201].parameters["Lambda"].value != pytest.approx(
        result.series_results["paramagnetic"][202].parameters["Lambda"].value
    )


@pytest.mark.parametrize("strategy", ["least_squares", "joint"])
def test_a_cancel_callback_aborts_the_joint_fit(strategy: str) -> None:
    """Cancellation propagates out of the cost function; nothing partial is returned."""
    with pytest.raises(FitCancelledError):
        fit_joint(
            [_oscillating_series(), _relaxing_series()],
            [_shared_background()],
            strategy=strategy,
            cancel_callback=lambda: True,
        )


# --------------------------------------------------------------------------- #
# Everything malformed raises, with a message that says what to do
# --------------------------------------------------------------------------- #


def test_one_series_is_not_a_joint_fit() -> None:
    with pytest.raises(ValueError, match="at least two series"):
        fit_joint([_oscillating_series()], [])


def test_a_run_in_two_series_raises() -> None:
    ordered = _oscillating_series()
    overlapping = JointSeriesProblem(
        key="other",
        datasets=ordered.datasets,
        model_fn=ordered.model_fn,
        global_params=ordered.global_params,
        local_params=ordered.local_params,
        initial_params=ordered.initial_params,
    )
    with pytest.raises(ValueError, match="may not weigh the same data twice"):
        fit_joint([ordered, overlapping], [])


def test_sharing_a_non_global_parameter_raises() -> None:
    shared = SharedParameter(
        name="lambda_shared",
        members={"ordered": "Lambda", "paramagnetic": "Lambda"},
        value=0.4,
    )
    with pytest.raises(ValueError, match="not one of that series' Global parameters"):
        fit_joint([_oscillating_series(), _relaxing_series()], [shared])


def test_a_shared_parameter_with_one_member_raises() -> None:
    shared = SharedParameter(name="A_bg_shared", members={"ordered": "A_bg"}, value=0.0)
    with pytest.raises(ValueError, match="at least two series"):
        fit_joint([_oscillating_series(), _relaxing_series()], [shared])


def test_sharing_a_parameter_one_series_pins_everywhere_raises() -> None:
    relaxing = _relaxing_series()
    for params in relaxing.initial_params.values():
        params.add(Parameter("A_bg", 2.0, fixed=True))
    with pytest.raises(ValueError, match="pins on every run"):
        fit_joint([_oscillating_series(), relaxing], [_shared_background()])


def test_an_unknown_series_key_in_the_shared_table_raises() -> None:
    shared = SharedParameter(
        name="A_bg_shared",
        members={"ordered": "A_bg", "nowhere": "A_bg"},
        value=0.0,
    )
    with pytest.raises(ValueError, match="not\n?\\s*one of the joint fit's series"):
        fit_joint([_oscillating_series(), _relaxing_series()], [shared])


def test_an_affine_tie_anywhere_raises() -> None:
    ordered = _oscillating_series()
    ordered.initial_params[101]["Lambda"].tie = AffineTie(main="freq", scale=0.25)
    with pytest.raises(NotImplementedError, match="affine parameter ties"):
        fit_joint([ordered, _relaxing_series()], [_shared_background()])


def test_the_profiled_strategy_is_not_available_for_a_joint_fit() -> None:
    with pytest.raises(NotImplementedError, match="'profiled' strategy is not available"):
        fit_joint(
            [_oscillating_series(), _relaxing_series()],
            [_shared_background()],
            strategy="profiled",
        )


def test_an_unknown_strategy_names_the_two_that_exist() -> None:
    with pytest.raises(ValueError, match="least_squares"):
        fit_joint(
            [_oscillating_series(), _relaxing_series()],
            [_shared_background()],
            strategy="lsmr",
        )


def test_the_same_parameter_cannot_be_two_shared_quantities() -> None:
    shared = [
        _shared_background(),
        SharedParameter(
            name="A_bg_again",
            members={"ordered": "A_bg", "paramagnetic": "A_bg"},
            value=0.0,
        ),
    ]
    with pytest.raises(ValueError, match="can only be one"):
        fit_joint([_oscillating_series(), _relaxing_series()], shared)


# --------------------------------------------------------------------------- #
# The regression gate: global_fit goes through the joint builder now
# --------------------------------------------------------------------------- #

_EXPONENTIAL = MODELS["ExponentialRelaxation"].function


def _regression_series() -> tuple[list[MuonDataset], dict[int, ParameterSet]]:
    """The fixture the pre-change literals below were captured from."""
    rng = np.random.default_rng(20260918)
    time = np.linspace(0.05, 8.0, 120)
    datasets: list[MuonDataset] = []
    inits: dict[int, ParameterSet] = {}
    for index, lam in enumerate((0.35, 0.8, 1.4)):
        clean = _EXPONENTIAL(time, A0=21.0, Lambda=lam, baseline=1.5)
        datasets.append(_dataset(index, time, clean + rng.normal(0.0, 0.4, time.size), 0.4))
        params = ParameterSet()
        params.add(Parameter("A0", 18.0, min=0.0))
        params.add(Parameter("baseline", 0.0))
        params.add(Parameter("Lambda", 0.5, min=0.0))
        inits[index] = params
    return datasets, inits


#: Captured from ``main`` before ``_build_coupled_global_problem`` learned about
#: blocks. The joint generalisation rewrites the column layout of every coupled
#: fit in the app, so these literals are the gate that a global fit still lands
#: on exactly the same minimum, to solver noise.
_GLOBAL_FIT_BASELINE = {
    "joint": {
        "A0": 20.937755162596826,
        "baseline": 1.5662183353881352,
        "lambda": (0.3525652542082669, 0.8169283351511394, 1.4283803478801689),
        "chi2": (110.5319339213011, 121.30774170734502, 117.89284420157776),
        "sigma_A0": 0.09854718933903328,
    },
    "least_squares": {
        "A0": 20.93775676302667,
        "baseline": 1.5661684948802403,
        "lambda": (0.35256269405489954, 0.8169214852636951, 1.4283386892134866),
        "chi2": (110.53139988378435, 121.30624160897978, 117.89488011569034),
        "sigma_A0": 0.09859688623954416,
    },
}


#: How closely a fresh fit must match the captured baseline, per strategy. The
#: literals were captured on one machine; both solvers stop inside their own
#: convergence tolerance (Minuit's EDM, the trust-region ftol/xtol), and where
#: they stop drifts between BLAS builds — the Linux CI runner lands ~2e-9
#: (Minuit) and ~2e-5 (least squares) relative from the macOS capture on the
#: least-constrained rate. The byte-for-byte guarantee on ``global_fit`` is
#: carried by the pre-existing engine tests, which run unmodified; this
#: snapshot exists to catch a column-order or packing regression in the block
#: builder, which shows up as a gross difference, not a fifth-decimal one.
_BASELINE_TOLERANCE = {"joint": 1e-6, "least_squares": 1e-4}


@pytest.mark.parametrize("strategy", ["joint", "least_squares"])
def test_global_fit_still_lands_on_the_pre_joint_minimum(strategy: str) -> None:
    """``global_fit`` through the block builder reproduces its pre-change numbers."""
    datasets, inits = _regression_series()
    expected = _GLOBAL_FIT_BASELINE[strategy]
    tol = _BASELINE_TOLERANCE[strategy]

    results, fitted = FitEngine().global_fit(
        datasets,
        _EXPONENTIAL,
        ["A0", "baseline"],
        ["Lambda"],
        inits,
        strategy=strategy,
    )

    assert fitted["A0"].value == pytest.approx(expected["A0"], rel=tol)
    assert fitted["baseline"].value == pytest.approx(expected["baseline"], rel=tol)
    for run in (0, 1, 2):
        result = results[run]
        assert result.parameters["Lambda"].value == pytest.approx(expected["lambda"][run], rel=tol)
        assert result.chi_squared == pytest.approx(expected["chi2"][run], rel=tol)
        assert result.uncertainties["A0"] == pytest.approx(expected["sigma_A0"], rel=tol)
        assert result.dof == 117
