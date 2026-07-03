"""MINOS asymmetric errors + χ² quality wiring (fit-workflow-diagnostics).

Covers the display-only MINOS overlay through the shared ``drive_minuit`` seam
(single, global, and count-domain sites) and the additive ``quality`` /
``uncertainties_asymmetric`` keys in ``fit_result_summary``.
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.count_domain import fit_fb_alpha
from asymmetry.core.fitting.engine import FitEngine, FitResult
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.fitting.result_summary import fit_result_summary
from asymmetry.core.simulate import build_builtin_template, simulate_run


def _expo(t, A0, lam):  # noqa: N803
    return A0 * np.exp(-lam * np.asarray(t, dtype=float))


def _flat_lowstat_dataset() -> MuonDataset:
    """A genuinely flat (λ_true = 0) low-statistics signal.

    With λ bounded at 0 the likelihood is non-parabolic — the canonical case where
    MINOS departs from HESSE: λ pins against the lower bound so its upward MINOS
    excursion far exceeds the (truncated) downward one.
    """
    rng = np.random.default_rng(20260612)
    t = np.linspace(0.0, 10.0, 80)
    err = np.full_like(t, 0.05)
    y = _expo(t, 0.20, 0.0) + rng.normal(0.0, err)
    return MuonDataset(time=t, asymmetry=y, error=err)


def _seed() -> ParameterSet:
    ps = ParameterSet()
    ps.add(Parameter(name="A0", value=0.2, min=0.0, max=1.0))
    ps.add(Parameter(name="lam", value=0.2, min=0.0))
    return ps


# --- single-fit MINOS -------------------------------------------------------


def test_minos_off_by_default_leaves_no_asymmetric_errors():
    result = FitEngine().fit(_flat_lowstat_dataset(), _expo, _seed())
    assert result.minos_errors is None
    assert result.dof == 80 - 2


def test_minos_asymmetric_against_lower_bound():
    ds = _flat_lowstat_dataset()
    result = FitEngine().fit(ds, _expo, _seed(), minos=True)
    assert result.success
    assert result.minos_errors is not None
    lo, hi = result.minos_errors["lam"]
    # λ is pinned at its lower bound: the upward excursion dominates the (truncated)
    # downward one — the documented asymmetric direction.
    assert hi > 0.0
    assert abs(hi) > abs(lo)
    assert abs(abs(hi) - abs(lo)) > 1e-4  # genuinely asymmetric, not numerical noise


def test_minos_does_not_change_symmetric_hesse_errors():
    ds = _flat_lowstat_dataset()
    sym = FitEngine().fit(ds, _expo, _seed(), minos=False)
    asym = FitEngine().fit(ds, _expo, _seed(), minos=True)
    assert sym.uncertainties.keys() == asym.uncertainties.keys()
    for name in sym.uncertainties:
        assert sym.uncertainties[name] == pytest.approx(asym.uncertainties[name], rel=1e-6)


def test_minos_well_determined_is_near_symmetric():
    rng = np.random.default_rng(7)
    t = np.linspace(0.0, 8.0, 400)
    err = np.full_like(t, 0.004)
    y = _expo(t, 0.22, 0.6) + rng.normal(0.0, err)
    ds = MuonDataset(time=t, asymmetry=y, error=err)
    ps = ParameterSet()
    ps.add(Parameter(name="A0", value=0.2, min=0.0, max=1.0))
    ps.add(Parameter(name="lam", value=0.5, min=0.0))
    result = FitEngine().fit(ds, _expo, ps, minos=True)
    for name, (lo, hi) in (result.minos_errors or {}).items():
        sigma = result.uncertainties[name]
        # A well-determined parameter has MINOS ≈ HESSE to a few percent.
        assert abs(hi) == pytest.approx(sigma, rel=0.05)
        assert abs(lo) == pytest.approx(sigma, rel=0.05)


# --- count-domain α MINOS ---------------------------------------------------


def _tf(t, A=20.0, f=1.5, phi=0.0):  # noqa: N803
    return A * np.cos(2.0 * np.pi * f * np.asarray(t, dtype=float) + phi)


def test_count_domain_alpha_minos_overlay():
    template = build_builtin_template("ideal_pulsed_fb")
    run = simulate_run(
        template, _tf, {"A": 20.0, "f": 1.5, "phi": 0.3}, total_events=40e6, alpha=1.25, seed=11
    )
    ds = MuonDataset(time=np.array([]), asymmetry=np.array([]), error=np.array([]), run=run)
    params = ParameterSet(
        [
            Parameter("alpha", 1.0, min=0.1, max=5.0),
            Parameter("N0", 1.5e5, min=0.0),
            Parameter("background", 0.0),
            Parameter("background_b", 0.0),
            Parameter("A", 18.0, min=0.0, max=50.0),
            Parameter("f", 1.5, min=0.0),
            Parameter("phi", 0.3),
        ]
    )
    result = fit_fb_alpha(ds, 1, 2, _tf, params, cost="gaussian", minos=True)
    forward = result.group_results[1]
    assert "alpha" in forward.minos_errors
    lo, hi = forward.minos_errors["alpha"]
    assert lo < 0.0 < hi
    # The symmetric HESSE σ (the value the promote path consumes) is still present
    # and untouched by MINOS.
    assert "alpha" in forward.uncertainties
    assert forward.uncertainties["alpha"] > 0.0


# --- quality verdict wiring -------------------------------------------------


def _fake_result(chi2: float, dof: int) -> FitResult:
    return FitResult(
        success=True,
        chi_squared=chi2,
        reduced_chi_squared=chi2 / dof,
        parameters=ParameterSet(),
        dof=dof,
    )


def test_result_summary_carries_quality_and_asymmetric_keys():
    ds = _flat_lowstat_dataset()
    result = FitEngine().fit(ds, _expo, _seed(), minos=True)
    summary = fit_result_summary(result)
    assert set(summary) >= {"quality", "uncertainties_asymmetric", "uncertainties"}
    # Asymmetric overlay present; symmetric HESSE preserved as the canonical errors.
    assert "lam" in summary["uncertainties_asymmetric"]
    assert summary["uncertainties"]["lam"] == pytest.approx(result.uncertainties["lam"])
    assert summary["quality"] is not None
    assert summary["quality"]["verdict"] in {"good", "poor", "overdone"}


def test_quality_verdict_regions_at_default():
    # ν = 100; at the muon-tuned 0.999 default the two-sided good band is the
    # wider χ²ᵣ ∈ [~0.54, ~1.47], so these values land clear of either edge.
    good = fit_result_summary(_fake_result(100.0, 100))["quality"]
    assert good["verdict"] == "good"
    poor = fit_result_summary(_fake_result(170.0, 100))["quality"]
    assert poor["verdict"] == "poor"
    overdone = fit_result_summary(_fake_result(50.0, 100))["quality"]
    assert overdone["verdict"] == "overdone"
    # The target band is reported and brackets χ²ᵣ = 1.
    assert good["band_low"] < 1.0 < good["band_high"]


def test_quality_none_when_dof_unusable():
    # No chi²/dof information and no MINOS overlay → verdict suppressed, not faked.
    summary = fit_result_summary(FitResult(success=True))
    assert summary["quality"] is None
    assert summary["uncertainties_asymmetric"] == {}
