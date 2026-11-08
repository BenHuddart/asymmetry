"""Public TF selection against independent envelopes and calibration truth."""

from dataclasses import replace

import numpy as np
import pytest

from asymmetry.core.fitting import tf_calibration as tf
from asymmetry.core.fitting.count_results import CountFitParameter as P
from asymmetry.core.fitting.engine import FitCancelledError
from asymmetry.core.fitting.likelihood import Objective
from asymmetry.core.utils.constants import MUON_LIFETIME_US
from tests.core.test_tf_calibration_fit import _raw_tf


def _control(envelope, beta=1, *, power=1.5, corrected=False):
    ds = _raw_tf(beta=beta)
    t = ds.run.histograms[0].time_axis
    exponent = {
        "undamped": np.zeros_like(t),
        "exponential": 0.3 * t,
        "gaussian": (0.3 * t) ** 2,
        "stretched": (0.3 * t) ** power,
    }[envelope]
    q = 0.2 * np.exp(-exponent) * np.cos(2 * np.pi * 1.3 * t + 0.17)
    for h, mean in zip(
        ds.run.histograms,
        [
            20000 * np.exp(-t / MUON_LIFETIME_US) * (1 + q) + 7,
            16000 * np.exp(-t / MUON_LIFETIME_US) * (1 - beta * q) + 11,
        ],
        strict=True,
    ):
        h.counts = np.rint(mean if not corrected else mean / (1 + 0.02 * mean / 1600))
        if corrected:
            h.count_variance = 2 * h.counts
    if corrected:
        ds.run.grouping.update(
            deadtime_correction=True,
            deadtime_mode="manual",
            dead_time_us=[0.02, 0.02],
            good_frames=100000,
        )
    return ds


@pytest.mark.parametrize("fit_beta", [False, True])
@pytest.mark.parametrize("envelope", ["undamped", "exponential", "gaussian", "stretched"])
def test_public_selection_recovers_calibration_and_minimum_total_cost(envelope, fit_beta):
    beta = 0.8 if fit_beta else 1
    selection = tf.select_tf_calibration(
        _control(envelope, beta),
        tf.TFCalibrationRequest(
            1,
            2,
            fit_beta=fit_beta,
            frequency_mhz=1.3,
            parameters=(P("frequency", 1.3, fixed=True),),
        ),
    )
    assert [c.envelope for c in selection.candidates] == list(tf.TFEnvelope)
    best = selection.selected
    assert best is not None
    assert best is min((c for c in selection.candidates if c.ok), key=lambda c: c.fit.statistic)
    assert best.alpha == pytest.approx(1.25, abs=2e-4)
    assert best.beta == pytest.approx(beta, abs=2e-4)
    if best.fit.uncertainty_status == "available":
        assert best.alpha_error > 0
    else:
        assert not fit_beta
        assert best.alpha_error is None
    assert (best.beta_error is not None) == fit_beta
    assert best.fit.objective is Objective.POISSON
    assert best.model == f"{best.envelope.value}_common_phase"
    for candidate in selection.candidates:
        assert candidate.fit.domains == best.fit.domains
        names = {p.name for p in candidate.fit.parameters}
        assert ("Lambda" in names) == (
            candidate.envelope in {tf.TFEnvelope.EXPONENTIAL, tf.TFEnvelope.STRETCHED}
        )
        assert ("sigma" in names) == (candidate.envelope is tf.TFEnvelope.GAUSSIAN)
        assert ("stretch" in names) == (candidate.envelope is tf.TFEnvelope.STRETCHED)
        assert candidate.fit.dof == 990 - len(candidate.fit.covariance_parameters)
    if envelope in {"gaussian", "stretched"}:
        assert (
            best.envelope in {tf.TFEnvelope.GAUSSIAN, tf.TFEnvelope.STRETCHED}
            if envelope == "gaussian"
            else best.envelope is tf.TFEnvelope.STRETCHED
        )


@pytest.mark.parametrize("fit_beta", [False, True])
def test_selection_prepares_once_and_retains_corrected_variance(monkeypatch, fit_beta):
    original = tf.build_count_groups
    prepared = []

    def build(*args, **kwargs):
        prepared.append(original(*args, **kwargs))
        return prepared[-1]

    monkeypatch.setattr(tf, "build_count_groups", build)
    selection = tf.select_tf_calibration(
        _control("stretched", 0.8 if fit_beta else 1, corrected=True),
        tf.TFCalibrationRequest(
            1,
            2,
            fit_beta=fit_beta,
            frequency_mhz=1.3,
            parameters=(P("frequency", 1.3, fixed=True),),
        ),
    )
    assert len(prepared) == 1
    assert selection.selected.alpha == pytest.approx(1.25, abs=3e-4)
    assert all(c.fit.objective is Objective.GAUSSIAN for c in selection.candidates)
    assert all(c.fit.domains == selection.selected.fit.domains for c in selection.candidates)


def test_envelope_analytic_limits_and_units():
    t = np.array([-2.0, -0.1, 0.0, 0.1, 2.0])
    for power in [0.02, 1, 2, 3]:
        actual = tf._stretched_tf(t, 20, 1.3, 0.17, 0.3, power)
        expected = 20 * np.exp(-((0.3 * abs(t)) ** power)) * np.cos(2 * np.pi * 1.3 * t + 0.17)
        np.testing.assert_allclose(actual, expected, atol=1e-14)
    np.testing.assert_allclose(
        tf._stretched_tf(t, 20, 1.3, 0.17, 0.3, 1),
        tf.oscillatory(t, 20, 1.3, 0.17, 0.3),
        atol=1e-14,
    )
    np.testing.assert_allclose(
        tf._stretched_tf(t, 20, 1.3, 0.17, 0.3, 2),
        tf._gaussian_tf(t, 20, 1.3, 0.17, 0.3),
        atol=1e-14,
    )
    assert P("stretch", 1.5).unit == "1"


@pytest.mark.parametrize("corrected", [False, True])
def test_stretched_covariance_matches_independent_information(corrected):
    ds = _control("stretched", 0.8, corrected=corrected)
    request = tf.TFCalibrationRequest(
        1, 2, fit_beta=True, parameters=(P("frequency", 1.3, fixed=True),)
    )
    result = tf.select_tf_calibration(ds, request).selected
    assert result.envelope is tf.TFEnvelope.STRETCHED
    names = result.fit.covariance_parameters
    assert {"alpha", "beta", "stretch", "Lambda", "phase", "background", "background_b"} <= set(
        names
    )
    values = {p.name: p.value for p in result.fit.parameters}
    t = np.arange(4, 499) * 0.016

    def means(v):
        q = (
            0.01
            * v["A0"]
            * np.exp(-((v["Lambda"] * t) ** v["stretch"]))
            * np.cos(2 * np.pi * v["frequency"] * t + v["phase"])
        )
        e = np.exp(-t / MUON_LIFETIME_US)
        return np.r_[
            v["N0"] * np.sqrt(v["alpha"]) * e * (1 + q) + v["background"],
            v["N0"] / np.sqrt(v["alpha"]) * e * (1 - v["beta"] * q) + v["background_b"],
        ]

    jac = []
    for name in names:
        step = max(abs(values[name]), 1) * 1e-5
        jac.append(
            (
                means(values | {name: values[name] + step})
                - means(values | {name: values[name] - step})
            )
            / (2 * step)
        )
    jac = np.array(jac).T
    variance = (
        np.concatenate([h.count_variance[4:499] for h in ds.run.histograms])
        if corrected
        else means(values)
    )
    independent = np.linalg.inv(jac.T @ (jac / variance[:, None]))
    covariance = np.asarray(result.fit.covariance)
    np.testing.assert_allclose(covariance, independent, rtol=0.015, atol=2e-6)
    assert abs(result.fit.correlation("stretch", "beta")) > 0.001
    assert result.beta_error == pytest.approx(
        np.sqrt(independent[names.index("beta"), names.index("beta")]), rel=0.01
    )


def test_failed_stretched_candidate_keeps_usable_alternative():
    result = tf.select_tf_calibration(
        _control("stretched", 0.8), tf.TFCalibrationRequest(1, 2, fit_beta=True, frequency_mhz=1.3)
    )
    stretched = result.candidates[-1]
    # This deliberately difficult unconstrained control can hit the solver's EDM limit.
    # A failed candidate's lower cost must not outrank a usable fit.
    if not stretched.ok:
        assert stretched.message
        assert result.selected is not stretched
        assert result.selected.ok
    else:
        assert result.selected is stretched


@pytest.mark.parametrize(
    "mode", ["failure", "no-usable", "tie", "near-tie", "nonfinite", "covariance", "unreliable"]
)
def test_selection_policy_retains_candidate_states(monkeypatch, mode):
    original = tf._fit_fb_count_groups
    calls = []

    def solve(*args, **kwargs):
        grouped = original(*args, **kwargs)
        i = len(calls)
        fit = grouped.joint_fit
        success = mode != "no-usable" and not (mode == "failure" and i == 0)
        statistic = [1.0, 1.0, 1.0, 1.0][i]
        if mode == "near-tie":
            statistic -= i * 1e-9
        if mode == "nonfinite" and i == 0:
            statistic = float("nan")
        grouped.joint_fit = replace(
            fit,
            success=success,
            statistic=statistic,
            message="deliberate fit failure" if not success else fit.message,
            covariance=None if mode == "covariance" else fit.covariance,
            covariance_accurate=False if mode == "unreliable" else fit.covariance_accurate,
        )
        calls.append(grouped.joint_fit)
        return grouped

    monkeypatch.setattr(tf, "_fit_fb_count_groups", solve)
    result = tf.select_tf_calibration(
        _control("exponential"), tf.TFCalibrationRequest(1, 2, frequency_mhz=1.3)
    )
    assert len(result.candidates) == 4
    if mode == "no-usable":
        assert result.selected is None
        assert all(c.message == "deliberate fit failure" for c in result.candidates)
    else:
        index = 1 if mode in {"failure", "nonfinite"} else 3 if mode == "near-tie" else 0
        assert result.selected is result.candidates[index]
    if mode in {"covariance", "unreliable"}:
        assert result.selected.ok
        assert result.selected.alpha_error is None
        assert result.selected.beta_error is None
        assert result.selected.alpha_beta_correlation is None
        assert result.selected.fit.uncertainty_status == (
            "unavailable" if mode == "covariance" else "unreliable"
        )


def test_common_and_candidate_constraints_and_no_shape_cap():
    request = tf.TFCalibrationRequest(
        1,
        2,
        fit_beta=True,
        parameters=(P("frequency", 1.3, fixed=True), P("phase", 0.17, fixed=True)),
    )
    specifics = {
        tf.TFEnvelope.EXPONENTIAL: (P("Lambda", 0.3, min=0.3, max=0.3, fixed=True),),
        tf.TFEnvelope.GAUSSIAN: (P("sigma", 0.3, min=0.3, max=0.3, fixed=True),),
        tf.TFEnvelope.STRETCHED: (P("Lambda", 0.3, fixed=True), P("stretch", 3, fixed=True)),
    }
    result = tf.select_tf_calibration(
        _control("stretched", 0.8, power=3), request, envelope_parameters=specifics
    )
    assert result.selected.envelope is tf.TFEnvelope.STRETCHED
    assert result.selected.alpha == pytest.approx(1.25, abs=2e-4)
    for c in result.candidates:
        fitted = {p.name: p for p in c.fit.parameters}
        assert fitted["frequency"].fixed and fitted["frequency"].value == 1.3
        assert fitted["phase"].fixed
        for p in specifics.get(c.envelope, ()):
            assert fitted[p.name].fixed and fitted[p.name].value == p.value
            assert p in c.request.parameters
    # Positive exponents below the numerical default floor are explicit valid constraints.
    low = tf.select_tf_calibration(
        _control("exponential"),
        replace(request, fit_beta=False),
        envelope_parameters={tf.TFEnvelope.STRETCHED: (P("stretch", 0.02, min=0.01, max=0.04),)},
    )
    p = next(p for p in low.candidates[-1].fit.parameters if p.name == "stretch")
    assert p.min == 0.01 and p.max == 0.04


@pytest.mark.parametrize(
    "common,specific",
    [
        ((P("Lambda", 0.1),), {}),
        ((), {"unknown": ()}),
        ((), {tf.TFEnvelope.UNDAMPED: (P("Lambda", 0.1),)}),
        ((), {tf.TFEnvelope.GAUSSIAN: (P("Lambda", 0.1),)}),
        ((), {tf.TFEnvelope.STRETCHED: (P("stretch", 0, fixed=True),)}),
        ((), {tf.TFEnvelope.STRETCHED: (P("stretch", 1, min=0),)}),
        ((), {tf.TFEnvelope.STRETCHED: (P("stretch", 1), P("stretch", 2))}),
        ((P("frequency", 0, fixed=True),), {}),
        ((P("beta", 0.8, fixed=True),), {}),
    ],
)
def test_invalid_selection_constraints_fail_before_any_fit(monkeypatch, common, specific):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid request reached minimisation")

    monkeypatch.setattr(tf, "_fit_fb_count_groups", forbidden)
    with pytest.raises(ValueError):
        tf.select_tf_calibration(
            _control("exponential"),
            tf.TFCalibrationRequest(1, 2, fit_beta=True, parameters=common),
            envelope_parameters=specific,
        )


@pytest.mark.parametrize("stage", ["before", "during", "between"])
def test_selection_cooperatively_cancels(monkeypatch, stage):
    cancelled = stage == "before"
    original = tf._fit_fb_count_groups
    calls = []

    def solve(*args, **kwargs):
        nonlocal cancelled
        calls.append(1)
        if stage == "during":
            cancelled = True
        result = original(*args, **kwargs)
        if stage == "between":
            cancelled = True
        return result

    monkeypatch.setattr(tf, "_fit_fb_count_groups", solve)
    with pytest.raises(FitCancelledError):
        tf.select_tf_calibration(
            _control("exponential"),
            tf.TFCalibrationRequest(1, 2),
            cancel_callback=lambda: cancelled,
        )
    assert len(calls) == (0 if stage == "before" else 1)


def test_selection_minos_preserves_conditional_errors():
    request = tf.TFCalibrationRequest(
        1,
        2,
        minos=True,
        parameters=tuple(
            P(n, v, fixed=True)
            for n, v in dict(
                A0=20, frequency=1.3, phase=0.17, background=7, background_b=11
            ).items()
        ),
    )
    result = tf.select_tf_calibration(
        _control("stretched"),
        request,
        envelope_parameters={
            tf.TFEnvelope.EXPONENTIAL: (P("Lambda", 0.3, fixed=True),),
            tf.TFEnvelope.GAUSSIAN: (P("sigma", 0.3, fixed=True),),
            tf.TFEnvelope.STRETCHED: (P("Lambda", 0.3, fixed=True), P("stretch", 1.5, fixed=True)),
        },
    )
    best = result.selected
    assert best.envelope is tf.TFEnvelope.STRETCHED
    assert best.alpha_error > 0
    assert best.beta_error is None
    intervals = {name: (low, high) for name, low, high in best.fit.minos_intervals}
    assert intervals["alpha"][0] < 0 < intervals["alpha"][1]
    assert intervals["alpha"][1] == pytest.approx(best.alpha_error, rel=0.02)
    assert set(intervals) == {"alpha", "N0"}


@pytest.mark.parametrize(
    "common,specific",
    [
        ((P("phase", 0.5, min=-0.1, max=0.1, fixed=True),), {}),
        ((P("alpha", 1.25, min=1.3, max=2, fixed=True),), {}),
        ((), {tf.TFEnvelope.EXPONENTIAL: (P("Lambda", 0.3, min=0, max=0.2, fixed=True),)}),
        ((), {tf.TFEnvelope.GAUSSIAN: (P("sigma", 0.3, min=0.4, max=1, fixed=True),)}),
        ((), {tf.TFEnvelope.STRETCHED: (P("Lambda", 0.3, min=0, max=0.2, fixed=True),)}),
        ((), {tf.TFEnvelope.STRETCHED: (P("stretch", 3, min=0.1, max=2, fixed=True),)}),
        ((), {tf.TFEnvelope.STRETCHED: (P("stretch", 0.02, min=0.05, fixed=True),)}),
    ],
)
def test_fixed_selection_overrides_must_respect_effective_bounds(monkeypatch, common, specific):
    def forbidden(*args, **kwargs):
        pytest.fail("fixed value outside effective bounds reached minimisation")

    monkeypatch.setattr(tf, "_fit_fb_count_groups", forbidden)
    with pytest.raises(ValueError, match="TF fixed parameter"):
        tf.select_tf_calibration(
            _control("exponential"),
            tf.TFCalibrationRequest(1, 2, parameters=common),
            envelope_parameters=specific,
        )
