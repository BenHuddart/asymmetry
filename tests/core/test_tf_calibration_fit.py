"""Shared TF calibration constraints and independent Poisson information controls."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.fitting import count_domain, tf_calibration
from asymmetry.core.fitting.count_domain import fit_fb_alpha
from asymmetry.core.fitting.count_results import CountFitParameter
from asymmetry.core.fitting.engine import FitCancelledError
from asymmetry.core.fitting.likelihood import FitInputError, Objective
from asymmetry.core.fitting.models import oscillatory
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.fitting.tf_calibration import TFCalibrationRequest, fit_tf_calibration
from asymmetry.core.utils.constants import MUON_LIFETIME_US
from tests.core.test_beta_estimation import _fb_beta_run


def test_joint_fit_retains_both_backgrounds_and_unique_dof():
    ds = _fb_beta_run(background_per_bin=5, seed=71)
    params = ParameterSet(
        [
            Parameter("alpha", 1.1, min=0.1, max=5),
            Parameter("N0", float("nan"), min=0),
            Parameter("background", 5, min=0),
            Parameter("background_b", 5, min=0),
            Parameter("A0", 20, min=0, max=100),
            Parameter("frequency", 0.5, min=0),
            Parameter("phase", 0.3),
            Parameter("Lambda", 0.1, min=0),
            Parameter("baseline", 0, fixed=True),
        ]
    )
    result = fit_fb_alpha(ds, 1, 2, oscillatory, params, estimate_beta=True)
    assert result.success, result.message
    joint = result.joint_fit
    assert joint is not None
    names = joint.covariance_parameters
    assert "background" in names and "background_b" in names
    covariance = np.asarray(joint.covariance)
    assert covariance[names.index("background"), names.index("background_b")] != 0
    assert joint.dof == sum(r.residuals.size for r in result.group_results.values()) - len(names)
    assert joint.statistic == pytest.approx(
        sum(r.chi_squared for r in result.group_results.values())
    )


def _raw_tf(*, beta=0.8, seed=None, backgrounds=(7.0, 11.0), phase=0.17, n_bins=600):
    """Independently authored count equations on a native acquisition grid."""
    time = np.arange(n_bins) * 0.016
    decay = np.exp(-time / MUON_LIFETIME_US)
    wave = 0.2 * np.exp(-0.15 * time) * np.cos(2 * np.pi * 1.3 * time + phase)
    means = np.array(
        [
            20000 * decay * (1 + wave) + backgrounds[0],
            16000 * decay * (1 - beta * wave) + backgrounds[1],
        ]
    )
    counts = np.rint(means) if seed is None else np.random.default_rng(seed).poisson(means)
    run = Run(
        7101,
        [Histogram(y, 0.016) for y in counts],
        grouping={
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "first_good_bin": 4,
            "last_good_bin": 498,
            "deadtime_correction": False,
            "background_correction": False,
        },
    )
    return MuonDataset(np.array([]), np.array([]), np.array([]), run=run)


def _request(*, fit_beta=False, parameters=(), **kwargs):
    return TFCalibrationRequest(
        1, 2, fit_beta=fit_beta, frequency_mhz=1.3, parameters=parameters, **kwargs
    )


@pytest.mark.parametrize("fit_beta,beta", [(False, 1), (True, 0.8)])
def test_shared_fit_known_truth_and_full_nuisance_information(fit_beta, beta):
    result = fit_tf_calibration(_raw_tf(beta=beta), _request(fit_beta=fit_beta))
    assert result.ok, result.message
    assert result.fit.success
    assert result.method == "tf_calibration_fit"
    assert result.model == "exponential_common_phase"
    assert result.fit.objective is Objective.POISSON
    assert result.alpha == pytest.approx(1.25, abs=1e-4)
    assert result.beta == pytest.approx(beta, abs=1e-4)
    assert result.beta_fixed == (not fit_beta)
    assert result.alpha_error > 0
    assert result.fit.uncertainty_status == "available"
    assert result.fit.dof == 990 - (9 if fit_beta else 8)
    assert result.fit.statistic < 1
    if fit_beta:
        assert result.beta_error > 0
        assert abs(result.alpha_beta_correlation) < 1
    else:
        assert result.beta_error is None
        assert result.alpha_beta_correlation is None
        assert "beta" not in result.fit.covariance_parameters
    parameters = {p.name: p for p in result.fit.parameters}
    assert parameters["A0"].unit == "%"
    assert parameters["frequency"].unit == "MHz"
    assert parameters["phase"].unit == "rad"
    assert parameters["Lambda"].unit == "µs⁻¹"
    assert parameters["background_b"].unit == "counts/bin"
    # Independent Poisson information in (Nf, alpha_m, beta, a, f, phi, lambda, cF, cB).
    # Transform the full inverse into the solver's (N0, alpha_A, beta, A0, ...) units.
    t = np.arange(4, 499) * 0.016
    e = np.exp(-t / MUON_LIFETIME_US)
    env = np.exp(-0.15 * t)
    angle = 2 * np.pi * 1.3 * t + 0.17
    p = env * np.cos(angle)
    dp = -env * np.sin(angle)
    nf, am, a = 20000.0, 0.8, 0.2
    jf = np.column_stack(
        [
            e * (1 + a * p),
            np.zeros(t.size),
            np.zeros(t.size),
            nf * e * p,
            nf * e * a * dp * 2 * np.pi * t,
            nf * e * a * dp,
            -nf * e * a * t * p,
            np.ones(t.size),
            np.zeros(t.size),
        ]
    )
    jb = np.column_stack(
        [
            am * e * (1 - beta * a * p),
            nf * e * (1 - beta * a * p),
            -nf * am * e * a * p,
            -nf * am * e * beta * p,
            -nf * am * e * beta * a * dp * 2 * np.pi * t,
            -nf * am * e * beta * a * dp,
            nf * am * e * beta * a * t * p,
            np.zeros(t.size),
            np.ones(t.size),
        ]
    )
    mu = np.r_[nf * e * (1 + a * p) + 7, nf * am * e * (1 - beta * a * p) + 11]
    names = ["nf", "am", "beta", "a", "frequency", "phase", "Lambda", "background", "background_b"]
    cols = [i for i, name in enumerate(names) if fit_beta or name != "beta"]
    jac = np.vstack([jf, jb])[:, cols]
    scale = np.array([20000 if i == 0 else 1 for i in cols])
    scaled = jac * scale
    cov = np.linalg.inv(scaled.T @ (scaled / mu[:, None])) * scale[:, None] * scale[None, :]
    transform = np.zeros((len(result.fit.covariance_parameters), len(cols)))
    for i, name in enumerate(result.fit.covariance_parameters):
        if name == "alpha":
            transform[i, cols.index(1)] = -1 / am**2
        elif name == "N0":
            transform[i, cols.index(0)] = np.sqrt(am)
            transform[i, cols.index(1)] = nf / (2 * np.sqrt(am))
        elif name == "A0":
            transform[i, cols.index(3)] = 100
        else:
            transform[i, cols.index(names.index(name))] = 1
    expected = transform @ cov @ transform.T
    actual = np.asarray(result.fit.covariance)
    np.testing.assert_allclose(np.sqrt(np.diag(actual)), np.sqrt(np.diag(expected)), rtol=0.035)
    norm = np.sqrt(np.diag(expected))
    np.testing.assert_allclose(
        actual / np.outer(norm, norm), expected / np.outer(norm, norm), atol=0.04
    )


def test_one_free_normalisation_has_exact_poisson_standard_error():
    ds = _raw_tf(beta=1, backgrounds=(0, 0))
    truth = dict(
        alpha=1.25,
        beta=1,
        A0=20,
        frequency=1.3,
        phase=0.17,
        Lambda=0.15,
        background=0,
        background_b=0,
        baseline=0,
    )
    overrides = tuple(CountFitParameter(n, v, fixed=True) for n, v in truth.items())
    result = fit_tf_calibration(ds, _request(parameters=overrides))
    t = np.arange(4, 499) * 0.016
    e = np.exp(-t / MUON_LIFETIME_US)
    p = 0.2 * np.exp(-0.15 * t) * np.cos(2 * np.pi * 1.3 * t + 0.17)
    c = np.sum(np.sqrt(1.25) * e * (1 + p) + e * (1 - p) / np.sqrt(1.25))
    total = sum(h.counts[4:499].sum() for h in ds.run.histograms)
    mle = total / c
    n0 = next(p.value for p in result.fit.parameters if p.name == "N0")
    assert result.ok
    assert result.fit.dof == 989
    assert n0 == pytest.approx(mle, rel=2e-5)
    assert result.fit.standard_error("N0") == pytest.approx(np.sqrt(mle / c), rel=0.002)
    assert result.alpha_error is None


@pytest.mark.parametrize("seed", [90235, 43, 89])
@pytest.mark.parametrize("fit_beta,beta", [(False, 1.0), (True, 0.8)])
def test_bounded_seeded_noisy_recoveries(seed, fit_beta, beta):
    result = fit_tf_calibration(_raw_tf(beta=beta, seed=seed), _request(fit_beta=fit_beta))
    assert result.ok, result.message
    assert abs(result.alpha - 1.25) < 5 * result.alpha_error
    if fit_beta:
        assert abs(result.beta - beta) < 5 * result.beta_error
    assert 0.7 < result.fit.statistic_per_dof < 1.3


def test_fixed_beta_is_the_same_solver_constraint_as_explicit_beta_one():
    ds = _raw_tf(beta=1)
    implicit = fit_tf_calibration(ds, _request())
    explicit = fit_tf_calibration(
        ds, _request(parameters=(CountFitParameter("beta", 1, fixed=True),))
    )
    assert implicit.fit == explicit.fit
    with pytest.raises(FrozenInstanceError):
        implicit.fit.parameters[0].value = 2
    with pytest.raises(TypeError):
        implicit.fit.covariance[0][0] = 2


@pytest.mark.parametrize(
    "parameter",
    [
        CountFitParameter("beta", 0.8, fixed=True),
        CountFitParameter("beta", 1),
        CountFitParameter("baseline", 1, fixed=True),
        CountFitParameter("dpsep", 0.324),
    ],
)
def test_inconsistent_or_unsupported_tf_parameters_rejected(parameter):
    with pytest.raises(ValueError, match="TF"):
        fit_tf_calibration(_raw_tf(), _request(parameters=(parameter,)))


def test_selected_support_uses_existing_native_bins_and_exclusion():
    result = fit_tf_calibration(_raw_tf(beta=1), _request(t_min=0.4, t_max=4, exclude=(1, 2)))
    f, b = result.fit.domains
    assert f.detector_indices == (0,) and b.detector_indices == (1,)
    assert f.raw_windows[0].first == 25
    assert f.raw_windows[-1].last == 250
    assert all(not 1 <= t <= 2 for t in f.time_us)
    assert f.time_us == b.time_us
    assert result.n_bins_used == len(f.time_us)


@pytest.mark.parametrize("kind", ["variance", "overlap", "fractional", "pulse", "deadtime"])
def test_unsupported_observations_preserve_loading_and_requested_settings(kind):
    ds = _raw_tf(beta=1)
    request = _request()
    if kind == "variance":
        ds.run.histograms[0].count_variance = ds.run.histograms[0].counts.copy()
    elif kind == "overlap":
        ds.run.grouping["groups"][2] = [1, 2]
    elif kind == "fractional":
        ds.run.histograms[0].counts[10] += 0.5
    elif kind == "pulse":
        with pytest.raises(FitInputError, match="pulse"):
            _request(pulse_separation_us=0.324)
        return
    else:
        ds.run.grouping.update(
            deadtime_correction=True,
            deadtime_mode="manual",
            deadtime_manual_us=0.001,
            good_frames=1e8,
        )
    with pytest.raises(ValueError):
        fit_tf_calibration(ds, request)
    assert len(ds.run.histograms) == 2
    if kind == "deadtime":
        assert ds.run.grouping["deadtime_correction"] is True


@pytest.mark.parametrize("state", ["failed", "unavailable", "unreliable"])
def test_convergence_and_uncertainty_are_separate_without_step_size_errors(monkeypatch, state):
    solve = count_domain._solve

    def changed_solve(*args, **kwargs):
        m = solve(*args, **kwargs)

        class State:
            valid = state != "failed"
            accurate = state == "unavailable"
            covariance = None if state == "unavailable" else m.covariance
            values, errors, merrors, fval, nfcn, fmin = (
                m.values,
                m.errors,
                m.merrors,
                m.fval,
                m.nfcn,
                m.fmin,
            )

        return State()

    monkeypatch.setattr(count_domain, "_solve", changed_solve)
    result = fit_tf_calibration(_raw_tf(beta=1), _request())
    assert result.fit.success == (state != "failed")
    assert result.ok == (state != "failed")
    assert result.alpha_error is None
    assert result.beta_error is None
    assert result.fit.uncertainty_status == (
        "unreliable" if state == "unreliable" else "unavailable"
    )


def test_invalid_expected_counts_cannot_be_reported_as_a_success(monkeypatch):
    solve = count_domain._solve

    def invalid_optimum(free, *args, **kwargs):
        m = solve(free, *args, **kwargs)

        class Invalid:
            valid, accurate, covariance = True, m.accurate, m.covariance
            values = [900 if p.name == "A0" else m.values[p.name] for p in free]
            errors, merrors, fval, nfcn, fmin = m.errors, m.merrors, m.fval, m.nfcn, m.fmin

        return Invalid()

    monkeypatch.setattr(count_domain, "_solve", invalid_optimum)
    result = fit_tf_calibration(_raw_tf(beta=1), _request())
    assert not result.ok and not result.fit.success
    assert "invalid expected counts" in result.message
    assert result.alpha_error is None


@pytest.mark.parametrize("boundary", ["start", "objective", "final"])
def test_cooperative_cancellation_reaches_actual_work_and_final_boundary(monkeypatch, boundary):
    cancelled = boundary == "start"
    calls = 0

    def cancel():
        nonlocal calls
        calls += 1
        return cancelled or (boundary == "objective" and calls >= 5)

    if boundary == "final":
        original = tf_calibration.fit_fb_alpha

        def cancel_after_fit(*args, **kwargs):
            nonlocal cancelled
            result = original(*args, **kwargs)
            cancelled = True
            return result

        monkeypatch.setattr(tf_calibration, "fit_fb_alpha", cancel_after_fit)
    with pytest.raises(FitCancelledError):
        fit_tf_calibration(_raw_tf(beta=1), _request(), cancel_callback=cancel)
    assert calls <= 5 or boundary == "final"


def test_fixed_beta_does_not_acquire_free_beta_significance_gates(monkeypatch):
    def beta_gate_must_not_run(*args):
        raise AssertionError("free-beta signal gate used for alpha-only calibration")

    monkeypatch.setattr(tf_calibration, "_amplitude_is_significant", beta_gate_must_not_run)
    monkeypatch.setattr(tf_calibration, "_frequency_is_resolved", beta_gate_must_not_run)
    result = fit_tf_calibration(_raw_tf(beta=1), _request())
    assert result.ok and result.beta_fixed


def test_joint_error_and_dof_match_the_legacy_beta_adapter():
    from asymmetry.core.fitting.beta_calibration import estimate_beta_detailed
    from asymmetry.core.utils.constants import MUON_GYROMAGNETIC_RATIO_MHZ_PER_T

    ds = _raw_tf(beta=0.8, seed=90235)
    shared = fit_tf_calibration(ds, _request(fit_beta=True))
    legacy = estimate_beta_detailed(ds, 1, 2, field_tesla=1.3 / MUON_GYROMAGNETIC_RATIO_MHZ_PER_T)
    assert shared.ok and legacy.ok
    assert legacy.method == "count_fit"
    assert legacy.alpha == shared.alpha
    assert legacy.beta == shared.beta
    assert legacy.alpha_error == shared.alpha_error
    assert legacy.beta_error == shared.beta_error
    assert legacy.alpha_beta_correlation == shared.alpha_beta_correlation
    assert legacy.reduced_chi2 == shared.fit.statistic_per_dof


def test_full_hesse_runs_before_optional_minos_and_retains_intervals(monkeypatch):
    from iminuit import Minuit

    hesse, minos = Minuit.hesse, Minuit.minos
    events = []

    def record_hesse(self, *args, **kwargs):
        events.append("hesse")
        return hesse(self, *args, **kwargs)

    def record_minos(self, *args, **kwargs):
        events.append("minos")
        return minos(self, *args, **kwargs)

    monkeypatch.setattr(Minuit, "hesse", record_hesse)
    monkeypatch.setattr(Minuit, "minos", record_minos)
    result = fit_tf_calibration(_raw_tf(beta=1), _request(minos=True))
    assert result.ok
    assert events.index("hesse") < events.index("minos")
    assert "alpha" in [name for name, lo, hi in result.fit.minos_intervals if lo < 0 < hi]


@pytest.mark.parametrize(
    "beta,fit_beta,native_alpha,native_beta",
    [
        (0.8, True, 1 / 0.800001, 0.799999),
        (1.0, False, 1 / 0.800003, 1.0),
    ],
)
def test_matches_native_global_histogram_rounded_control(beta, fit_beta, native_alpha, native_beta):
    """musrfit1.12.1/708f777b, ROOT6.40.04 global histogram MAX_LIKELIHOOD.

    Owned raw16ns bins4..498; lifetime fixed, backgrounds0, phases0/180,
    free normalisation, alpha, fraction amplitude, MHz frequency, inverse-us rate.
    Native alpha=NB/NF is reciprocal; beta=AB/AF. Absolute point tolerance1e-4.
    Error equality is not asserted: installed native likelihood errors differ.
    """
    ds = _raw_tf(beta=beta, backgrounds=(0, 0), phase=0, n_bins=2000)
    parameters = (
        CountFitParameter("background", 0, fixed=True),
        CountFitParameter("background_b", 0, fixed=True),
        CountFitParameter("phase", 0, fixed=True),
    )
    result = fit_tf_calibration(ds, _request(fit_beta=fit_beta, parameters=parameters))
    assert result.ok, result.message
    assert result.alpha == pytest.approx(native_alpha, abs=1e-4)
    assert result.beta == pytest.approx(native_beta, abs=1e-4)


@pytest.mark.parametrize("alpha", [-1.25, 0.0, float("nan"), float("inf")])
def test_nonpositive_or_nonfinite_fixed_alpha_is_rejected(alpha):
    with pytest.raises(ValueError, match="fixed alpha.*positive"):
        fit_tf_calibration(
            _raw_tf(beta=1), _request(parameters=(CountFitParameter("alpha", alpha, fixed=True),))
        )


def test_explicitly_processed_detector_labels_are_not_raw_poisson_observations():
    ds = _raw_tf(beta=1)
    ds.run.metadata["histogram_labels"] = ["Forward PostPileUp rej.", "Backward"]
    with pytest.raises(FitInputError, match="PostPileUp"):
        fit_tf_calibration(ds, _request())
    assert len(ds.run.histograms) == 2
    assert np.all(ds.run.histograms[0].counts == np.floor(ds.run.histograms[0].counts))


def test_unselected_processed_channels_do_not_ban_the_acquisition():
    ds = _raw_tf(beta=1)
    ds.run.histograms.append(Histogram(ds.run.histograms[0].counts.copy(), 0.016))
    ds.run.metadata["histogram_labels"] = ["Forward", "Backward", "PostPileUp rej."]
    result = fit_tf_calibration(ds, _request())
    assert result.ok


def test_joint_fit_honours_an_independently_fixed_positive_frequency():
    result = fit_tf_calibration(
        _raw_tf(beta=0.8),
        _request(fit_beta=True, parameters=(CountFitParameter("frequency", 1.3, fixed=True),)),
    )
    assert result.fit.success
    assert result.ok, result.message
    assert result.beta == pytest.approx(0.8, abs=1e-4)
    assert result.beta_error > 0
    assert result.fit.standard_error("frequency") is None
    assert next(p for p in result.fit.parameters if p.name == "frequency").fixed


@pytest.mark.parametrize("frequency", [0.0, -1.3, float("nan"), float("inf")])
def test_joint_fit_rejects_nonpositive_or_nonfinite_fixed_frequency(frequency):
    with pytest.raises(ValueError, match="fixed frequency.*positive"):
        fit_tf_calibration(
            _raw_tf(beta=0.8),
            _request(
                fit_beta=True, parameters=(CountFitParameter("frequency", frequency, fixed=True),)
            ),
        )
