"""Muonium oscillation components (WiMDA TFMuonium/LowTFMuonium/ZFmuonium ports).

Physics checks are against the WiMDA arithmetic; the fit round-trip uses a
self-consistent synthetic signal generated from the component itself (genuine
muonium, well-separated satellites), so the suite never depends on the WiMDA
corpus. Note: shallow-donor CdS is fit with three independent lines + link
groups, not these components (see docs/porting/muonium-triplet/).
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting import muonium as mu
from asymmetry.core.fitting.composite import COMPONENTS, CompositeModel
from asymmetry.core.fitting.engine import FitEngine
from asymmetry.core.fitting.parameters import Parameter, ParameterSet

_TWO_PI = 2.0 * np.pi


def _gmu() -> float:
    return mu.G_MU_MHZ_PER_G


# --- WiMDA arithmetic -------------------------------------------------------


def test_g_factors_match_wimda_literals() -> None:
    # WiMDA: gm = 0.01355342, ge = 2.8024 (MHz/G). Asymmetry's CODATA-based
    # constants agree to ~5e-7 (a negligible difference for the line positions).
    assert mu.G_MU_MHZ_PER_G == pytest.approx(0.01355342, abs=1e-6)
    assert mu.G_E_MHZ_PER_G == pytest.approx(2.8024, abs=1e-3)


def test_tf_satellites_straddle_larmor_with_splitting_a_hf() -> None:
    field, a_hf = 100.0, 0.242
    delta, e1, e2, e3, e4 = mu._tf_levels(field, a_hf)
    in_band = sorted([abs(e1 - e2), abs(e3 - e4)])
    out_band = sorted([abs(e1 - e4), abs(e2 - e3)])
    nu_d = _gmu() * field

    # In-band pair straddles the diamagnetic Larmor line, symmetric, split = A_hf.
    assert 0.5 * (in_band[0] + in_band[1]) == pytest.approx(nu_d, abs=1e-3)
    assert in_band[1] - in_band[0] == pytest.approx(a_hf, abs=2e-3)
    # The other two transitions are far out of band and carry weight (1-delta) ~ 0.
    assert min(out_band) > 100.0
    assert (1.0 - delta) < 1e-4


def test_tf_uses_positive_frequencies_same_phase() -> None:
    # Positive-frequency convention: every line shares +phase, so at t=0 the
    # normalised sum collapses to cos(phase) (weights sum to 1).
    for phase in (0.0, 0.5, -1.2):
        val = mu.tf_muonium(np.array([0.0]), 100.0, 0.242, phase)[0]
        assert val == pytest.approx(np.cos(phase), abs=1e-9)


def test_low_tf_has_two_lines_one_in_band() -> None:
    field, a_hf = 100.0, 0.5
    _delta, e1, e2, e3, _e4 = mu._tf_levels(field, a_hf)
    lines = sorted([abs(e1 - e2), abs(e2 - e3)])
    # One in-band satellite near the Larmor line, one far out of band.
    assert lines[0] < 5.0
    assert lines[1] > 100.0


def test_zf_line_frequencies_and_cutoff_weights() -> None:
    a_hf, d, f_cut = 3.0, 1.0, 0.0
    f1, f2, f3 = a_hf - d, a_hf + d / 2.0, 1.5 * d
    # With f_cut = 0 the weights are the bare 1, 2, 2 normalised by 6 -> 5/6 at t=0.
    assert mu.zf_muonium(np.array([0.0]), a_hf, d, f_cut, 0.0)[0] == pytest.approx(
        (1.0 + 2.0 + 2.0) / 6.0
    )
    # A finite cutoff rolls off the higher lines (Lorentzian).
    rolled = mu.zf_muonium(np.array([0.0]), a_hf, d, 1.0, 0.0)[0]
    a1 = 1.0 / (1.0 + (f1 / 1.0) ** 2)
    a2 = 2.0 / (1.0 + (f2 / 1.0) ** 2)
    a3 = 2.0 / (1.0 + (f3 / 1.0) ** 2)
    assert rolled == pytest.approx((a1 + a2 + a3) / 6.0)


def test_a_hf_moves_satellites_symmetrically() -> None:
    field = 100.0
    nu_d = _gmu() * field
    # The satellites are symmetric about the Larmor line in the small-A_hf limit;
    # the residual asymmetry grows slowly with A_hf, so use a modest tolerance.
    for a_hf in (0.2, 0.5, 1.0):
        _d, e1, e2, e3, e4 = mu._tf_levels(field, a_hf)
        lo, hi = sorted([abs(e1 - e2), abs(e3 - e4)])
        assert nu_d - lo == pytest.approx(hi - nu_d, abs=3e-3)  # symmetric
        assert hi - lo == pytest.approx(a_hf, abs=3e-3)  # separation = A_hf


# --- Engine self-consistency (genuine muonium, no corpus) -------------------


def test_muonium_tf_round_trip_recovers_hyperfine() -> None:
    model = CompositeModel.from_expression("MuoniumTF * Exponential + Constant")
    truth = {"A_1": 20.0, "field": 100.0, "A_hf": 2.0, "phase": 0.3, "Lambda": 0.2, "A_bg": 0.5}
    t = np.linspace(0.0, 12.0, 800)
    rng = np.random.default_rng(0)
    ds = MuonDataset(
        time=t,
        asymmetry=model.function(t, **truth) + rng.normal(0.0, 0.2, size=t.shape),
        error=np.full_like(t, 0.2),
        metadata={"run_number": 1},
    )
    positive = {"A_1", "A_hf", "Lambda", "A_bg"}
    seed = {**truth, "A_hf": 2.1}
    ps = ParameterSet(
        [
            Parameter(name=n, value=seed[n], min=(0.0 if n in positive else -np.inf))
            for n in model.param_names
        ]
    )
    result = FitEngine().fit(ds, model.function, ps, t_min=0.0, t_max=12.0)
    assert result.success
    assert result.reduced_chi_squared == pytest.approx(1.0, abs=0.2)
    fitted = {p.name: p.value for p in result.parameters}
    assert fitted["A_hf"] == pytest.approx(2.0, abs=0.02)
    assert "A_hf" in result.uncertainties


# --- Registry / composite / serialization integration -----------------------


def test_components_registered_with_expected_parameters() -> None:
    assert COMPONENTS["MuoniumTF"].param_names == ["A", "field", "A_hf", "phase"]
    assert COMPONENTS["MuoniumLowTF"].param_names == ["A", "field", "A_hf", "phase"]
    assert COMPONENTS["MuoniumZF"].param_names == ["A", "A_hf", "D_mu", "f_cut", "phase"]
    for name in ("MuoniumTF", "MuoniumLowTF", "MuoniumZF"):
        assert COMPONENTS[name].category == "Muonium"


def test_components_appear_in_builder_category_map() -> None:
    pytest.importorskip("PySide6")
    from asymmetry.gui.panels.fit_function_builder import _build_components_by_category

    grouped = _build_components_by_category()
    assert "Muonium" in grouped
    assert {"MuoniumTF", "MuoniumLowTF", "MuoniumZF"} <= set(grouped["Muonium"])


def test_cds_style_model_builds_with_hyperfine_param() -> None:
    model = CompositeModel.from_expression(
        "OscillatoryField*Exponential + MuoniumTF*Exponential + Constant"
    )
    assert "A_hf" in model.param_names


def test_model_with_muonium_component_round_trips() -> None:
    model = CompositeModel.from_expression("MuoniumTF * Exponential + Constant")
    restored = CompositeModel.from_dict(model.to_dict())
    assert restored.param_names == model.param_names
    assert restored.component_names == model.component_names


def test_muonium_functions_are_picklable() -> None:
    for fn in (mu.tf_muonium, mu.low_tf_muonium, mu.zf_muonium):
        assert pickle.loads(pickle.dumps(fn)) is fn


def test_tf_lineshapes_stay_finite_for_extreme_parameters() -> None:
    """x is clamped, so even a pathological A_hf can't overflow to a NaN curve."""
    t = np.linspace(0.0, 12.0, 200)
    for a_hf in (0.0, 1e-12, -0.01, 1e-200):
        for field in (0.0, 100.0, -100.0, 20000.0):
            assert np.all(np.isfinite(mu.tf_muonium(t, field, a_hf, 0.3)))
            assert np.all(np.isfinite(mu.low_tf_muonium(t, field, a_hf, 0.3)))
