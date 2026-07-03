"""Fit-robustness guards from the GUI Round-4 magnetism evaluation.

#8a — the stretched exponential's ``beta`` carries a small *positive* floor so a
one-shot fit cannot wander into the degenerate ``beta -> 0`` limit (which, with
the ``|Lambda|`` sign-fold, is the documented spin-glass sign/exponent
degeneracy).

#8b — the fit wizard offers a Dynamic Gaussian Kubo-Toyabe candidate for KT-like
spectra and recovers a moderately-dynamic spectrum from its default seeds. (A
robust *multi-decade* hop-rate seed and the high-TF oscillatory envelope seed
are larger, regime-dependent problems deferred to a documented follow-up.)
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import COMPONENTS, CompositeModel
from asymmetry.core.fitting.engine import FitEngine
from asymmetry.core.fitting.fit_wizard import (
    CandidateTemplate,
    build_candidate_templates,
    build_fit_wizard_recommendation_for_templates,
    fingerprint_spectrum,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet, get_param_info

pytestmark = [pytest.mark.integration]


# --------------------------------------------------------------------------- #
# #8a — stretched-exponential beta floor
# --------------------------------------------------------------------------- #


def test_beta_has_positive_floor_while_rate_floors_stay_zero() -> None:
    beta_floor = get_param_info("beta").default_min
    assert beta_floor is not None
    assert beta_floor > 0.0
    assert beta_floor == pytest.approx(0.05)

    # The Lambda/sigma/Delta rate floors are unchanged (still 0 — already applied
    # before this change; we must not perturb them).
    assert get_param_info("Lambda").default_min == 0.0
    assert get_param_info("sigma").default_min == 0.0
    assert get_param_info("Delta").default_min == 0.0

    # The StretchedExponential component resolves its beta through the same global
    # ParamInfo the GUI parameter-table populate reads, so the floor is visible
    # via the component/param_info path too.
    stretched = COMPONENTS["StretchedExponential"]
    assert stretched.param_info["beta"].default_min == pytest.approx(0.05)


def test_stretched_exponential_fit_respects_beta_floor() -> None:
    # A fast early drop to a flat plateau (true beta = 0.03) is the degenerate
    # beta -> 0 regime. Without a floor the fit drifts below 0.05; with the
    # populate-path floor applied as the beta lower bound it stops at the floor.
    model = CompositeModel(["StretchedExponential", "Constant"], operators=["+"])
    t = np.linspace(0.05, 12.0, 400)
    y = 2.0 + 18.0 * np.exp(-((0.8 * t) ** 0.03))
    dataset = MuonDataset(time=t, asymmetry=y, error=np.full_like(t, 0.02))

    beta_floor = float(get_param_info("beta").default_min)

    def _fit_beta(beta_min: float) -> float:
        params = ParameterSet()
        params.add(Parameter("A_1", value=15.0, min=0.0, max=100.0))
        params.add(Parameter("Lambda", value=0.5, min=0.0, max=50.0))
        params.add(Parameter("beta", value=1.0, min=beta_min, max=3.0))
        params.add(Parameter("A_bg", value=2.0, min=-50.0, max=50.0))
        return FitEngine().fit(dataset, model.function, params).parameters["beta"].value

    # Unconstrained (the old default_min = 0) drifts into the degenerate regime...
    assert _fit_beta(0.0) < beta_floor
    # ...the positive floor keeps a one-shot fit out of it.
    assert _fit_beta(beta_floor) >= beta_floor - 1e-9


# --------------------------------------------------------------------------- #
# #8b — dynamic Gaussian KT wizard candidate
# --------------------------------------------------------------------------- #


def _dynamic_gkt_dataset(
    *,
    amplitude: float = 20.0,
    delta: float = 0.4,
    nu: float = 2.0,
    baseline: float = 2.0,
    noise: float = 0.05,
    seed: int = 3,
) -> tuple[CompositeModel, MuonDataset]:
    model = CompositeModel(["DynamicGaussianKT", "Constant"], operators=["+"])
    t = np.arange(0.0, 12.0, 0.02)
    y = model.function(t, A_1=amplitude, Delta=delta, nu=nu, B_L=0.0, A_bg=baseline)
    # Deterministic noise at the quoted error level — a noiseless curve leaves a
    # perfect zero-residual valley whose Hessian is numerically singular, which
    # is unrepresentative of real data.
    y = y + np.random.default_rng(seed).normal(0.0, noise, t.size)
    dataset = MuonDataset(time=t, asymmetry=y, error=np.full_like(t, noise))
    return model, dataset


def test_dynamic_gkt_candidate_offered_for_kt_like_data() -> None:
    # The wizard now offers the dynamic GKT candidate alongside the static one
    # when the spectrum is KT-like, so a user comparing models sees both frozen
    # and fluctuating local-field options. A near-static GKT spectrum has the
    # classic dip-and-recovery that triggers the KT-like hint.
    _model, dataset = _dynamic_gkt_dataset(
        amplitude=20.0, delta=0.8, nu=0.05, baseline=0.0, noise=0.01
    )
    fingerprint = fingerprint_spectrum(dataset)
    assert fingerprint.kt_like_hint
    keys = {template.key for template in build_candidate_templates(fingerprint)}
    assert "dynamic_gkt_constant" in keys
    assert "static_gkt_constant" in keys


def test_dynamic_gkt_wizard_recovers_moderate_dynamics() -> None:
    # A moderately-dynamic Gaussian KT spectrum (planted Delta = 0.4, nu = 2.0)
    # is recovered by the wizard's dynamic-GKT recommendation from its DEFAULT
    # seeds in one pass: Delta is seeded from the early-time curvature and nu
    # from the component default (1.0), and iminuit's gradient descent carries nu
    # to ~2.0 (the wizard's variant sweep scales A/Delta/phase but not nu). chi^2_r
    # ~ 1 and both shape parameters land near the planted values. (We assert
    # recovery rather than the iminuit ``valid`` flag: the dynamic-KT Hessian is
    # shallow along the nu direction, so the covariance-validity flag is
    # noise-sensitive even at a good minimum.)
    model, dataset = _dynamic_gkt_dataset(delta=0.4, nu=2.0)
    template = CandidateTemplate(
        key="dynamic_gkt_constant",
        title="Dynamic GKT + Constant",
        category="KT-like",
        rationale="test",
        model=model,
    )

    recommendation = build_fit_wizard_recommendation_for_templates(dataset, [template])
    assessment = recommendation.assessment_for_key("dynamic_gkt_constant")
    assert assessment is not None
    result = assessment.fit_result

    assert result.reduced_chi_squared < 1.5
    assert result.parameters["Delta"].value == pytest.approx(0.4, abs=0.06)
    assert result.parameters["nu"].value == pytest.approx(2.0, rel=0.3)
