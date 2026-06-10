"""Tests for the WiMDA fit-function-parity components.

Verification criteria from docs/porting/wimda-fit-function-parity/:
limit identities against established components, t = 0 normalisation,
registry/doc hygiene, serialization round-trips, and golden checks against
independent evaluations.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import erfc

from asymmetry.core.fitting.component_docs import FIT_COMPONENT_APPLICABILITY
from asymmetry.core.fitting.composite import COMPONENTS, CompositeModel
from asymmetry.core.fitting.models import (
    bessel_oscillation,
    gaussian_broadened_kt,
    longitudinal_field_kubo_toyabe,
    risch_kehr,
)
from asymmetry.core.fitting.muon_fluorine.polarization import (
    dynamic_fmuf_polarization,
    fmuf_triangle_polarization,
    general_fmuf_polarization,
    linear_fmuf_polarization,
)
from asymmetry.core.fitting.muonium import (
    VACUUM_MUONIUM_A_HF_MHZ,
    _tf_levels,
    high_tf_muonium,
    high_tf_muonium_aniso,
    muonium_lf_relaxation,
)
from asymmetry.core.fitting.nuclear_dipole import (
    dipolar_pair_field,
    dipolar_pair_kernel,
    dipolar_spin_j,
    electron_dipole,
    proton_dipole,
)

NEW_COMPONENTS = [
    "RischKehr",
    "Bessel",
    "GaussianBroadenedKT",
    "MuoniumHighTF",
    "MuoniumHighTFAniso",
    "MuoniumLFRelax",
    "DynamicFmuF",
    "FmuF_Triangle",
    "DipolarPairField",
    "ProtonDipole",
    "ElectronDipole",
    "DipolarSpinJ",
]

T = np.linspace(0.0, 16.0, 321)


# --- registry and documentation hygiene -------------------------------------


def test_new_components_registered_with_metadata() -> None:
    for name in NEW_COMPONENTS:
        definition = COMPONENTS[name]
        assert definition.domain == "time"
        assert definition.category in {
            "Relaxation",
            "Oscillation",
            "Kubo-Toyabe",
            "Muonium",
            "Nuclear dipolar",
        }
        assert definition.formula_template
        assert definition.latex_equation
        assert set(definition.param_defaults) == set(definition.param_names)
        for param in definition.param_names:
            assert param in definition.param_info


def test_new_components_have_applicability_docs() -> None:
    for name in NEW_COMPONENTS:
        text = FIT_COMPONENT_APPLICABILITY[name]
        assert len(text) > 100


def test_new_components_finite_and_normalised_at_defaults() -> None:
    for name in NEW_COMPONENTS:
        definition = COMPONENTS[name]
        y = definition.function(T, **definition.param_defaults)
        assert y.shape == T.shape, name
        assert np.all(np.isfinite(y)), name
        # All new components are cosine-like at phase 0 / pure relaxation:
        # the t = 0 value equals the amplitude default (A = 25).
        assert y[0] == pytest.approx(25.0, rel=1e-9), name


def test_new_components_serialization_round_trip() -> None:
    for name in NEW_COMPONENTS:
        model = CompositeModel([name, "Constant"], ["+"])
        restored = CompositeModel.from_dict(model.to_dict())
        assert restored.component_names == model.component_names


# --- RischKehr ---------------------------------------------------------------


def test_risch_kehr_matches_erfc_form_and_asymptote() -> None:
    gamma = 1.3
    g = gamma * T[1:]
    direct = np.exp(g) * erfc(np.sqrt(g))  # safe for these moderate g values
    assert np.allclose(risch_kehr(T[1:], gamma), direct, rtol=1e-12)

    t_large = np.array([1.0e4])
    tail = risch_kehr(t_large, gamma)[0]
    assert tail == pytest.approx(1.0 / np.sqrt(np.pi * gamma * 1.0e4), rel=1e-2)
    # Continuity across WiMDA's branch point Gamma*t = 20 is inherent to erfcx;
    # check smoothness there anyway.
    near = risch_kehr(np.array([19.99, 20.01]), 1.0)
    assert abs(near[1] - near[0]) < 1e-4


def test_risch_kehr_negative_rate_uses_magnitude() -> None:
    assert np.allclose(risch_kehr(T, -2.0), risch_kehr(T, 2.0))


# --- Bessel ------------------------------------------------------------------


def test_bessel_matches_overhauser_integral() -> None:
    freq = 0.7  # MHz
    omega = 2.0 * np.pi * freq
    t = np.linspace(0.0, 8.0, 50)
    # MS-Intro eqn 6.45: P(t) = (1/pi) int_-B1^B1 dB cos(gamma B t)/sqrt(B1^2-B^2)
    phi = np.linspace(0.0, np.pi, 20001)  # B = B1 cos(phi) substitution
    integral = np.trapezoid(np.cos(omega * np.outer(t, np.cos(phi))), phi, axis=1) / np.pi
    assert np.allclose(bessel_oscillation(t, freq), integral, atol=1e-10)
    assert bessel_oscillation(np.array([0.0]), freq)[0] == 1.0


# --- Gaussian-broadened KT ---------------------------------------------------


def test_gbkt_zero_width_reduces_to_lf_kt() -> None:
    for b_l in (0.0, 30.0):
        broadened = gaussian_broadened_kt(T, 0.5, b_l, 0.0)
        static = longitudinal_field_kubo_toyabe(T, 1.0, 0.5, b_l)
        assert np.allclose(broadened, static, atol=1e-12)


def test_gbkt_matches_brute_force_average() -> None:
    width = 0.3
    nodes = np.linspace(-5.0, 5.0, 2001)
    pdf = np.exp(-0.5 * nodes**2) / np.sqrt(2.0 * np.pi)
    brute = np.zeros_like(T)
    for x, p in zip(nodes, pdf, strict=True):
        brute += (
            p
            * (nodes[1] - nodes[0])
            * longitudinal_field_kubo_toyabe(T, 1.0, abs(0.5 * (1.0 + width * x)), 30.0)
        )
    assert np.allclose(gaussian_broadened_kt(T, 0.5, 30.0, width), brute, atol=1e-4)


def test_gbkt_broadening_softens_the_dip() -> None:
    sharp = gaussian_broadened_kt(T, 0.5, 0.0, 0.0)
    broad = gaussian_broadened_kt(T, 0.5, 0.0, 0.4)
    assert broad.min() > sharp.min()


# --- muonium -----------------------------------------------------------------


def test_high_tf_pair_frequencies_sum_to_hyperfine() -> None:
    a_hf = VACUUM_MUONIUM_A_HF_MHZ
    _d, e1, e2, e3, e4 = _tf_levels(3000.0, a_hf)
    assert abs(e1 - e2) + abs(e3 - e4) == pytest.approx(a_hf, rel=1e-12)


def test_high_tf_aniso_reduces_to_isotropic_pair() -> None:
    iso = high_tf_muonium(T, 3000.0, VACUUM_MUONIUM_A_HF_MHZ, 0.3)
    aniso = high_tf_muonium_aniso(T, 3000.0, VACUUM_MUONIUM_A_HF_MHZ, 0.0, 0.3)
    assert np.allclose(iso, aniso, atol=1e-10)


def test_high_tf_aniso_powder_average_damps_pair() -> None:
    iso = high_tf_muonium(T, 3000.0, VACUUM_MUONIUM_A_HF_MHZ, 0.0)
    aniso = high_tf_muonium_aniso(T, 3000.0, VACUUM_MUONIUM_A_HF_MHZ, 25.0, 0.0)
    # The distribution of anisotropy shifts dephases the pair envelope.
    assert np.max(np.abs(aniso[T > 4.0])) < np.max(np.abs(iso[T > 4.0]))


def test_muonium_lf_relaxation_quenches_with_field() -> None:
    low = muonium_lf_relaxation(T, 0.5, 0.01, 10.0, VACUUM_MUONIUM_A_HF_MHZ)
    high = muonium_lf_relaxation(T, 0.5, 0.01, 5000.0, VACUUM_MUONIUM_A_HF_MHZ)
    assert low[-1] < high[-1] <= 1.0
    assert low[0] == 1.0


# --- nuclear dipolar ---------------------------------------------------------


def test_spin_half_reduces_to_meier_pair() -> None:
    kernel = dipolar_pair_kernel(T, 2.0 * np.pi * 0.2, 0.0)
    spin_j = dipolar_spin_j(T, 0.2, 0.0, 0.5)
    assert np.allclose(kernel, spin_j, atol=1e-12)


def test_spin_j_quadrupole_inactive_for_spin_half() -> None:
    without = dipolar_spin_j(T, 0.2, 0.0, 0.5)
    with_quad = dipolar_spin_j(T, 0.2, 0.7, 0.5)
    # A spin-1/2 nucleus has no quadrupole moment; the closed form keeps the
    # 1<->2 splitting quadrupole-free (the spectator levels shift together).
    assert np.allclose(without, with_quad, atol=1e-12)


def test_dipolar_pair_transverse_damping_preserves_static_sixth() -> None:
    damped = dipolar_pair_field(np.array([60.0]), 10.0, 5.0)[0]
    assert damped == pytest.approx(1.0 / 6.0, abs=1e-12)


def test_dipole_pair_frequency_scales_with_gyromagnetic_ratio() -> None:
    # At equal distance the electron pair beats ~660x faster than the proton
    # pair; check via the first zero-crossing ordering instead of exact ratios.
    t_short = np.linspace(0.0, 0.02, 2000)
    e = electron_dipole(t_short, 2.0, 0.0)
    p = proton_dipole(t_short, 2.0, 0.0)
    assert e.min() < 0.9  # electron pair has already oscillated
    assert p.min() > 0.999  # proton pair has barely moved


# --- F-mu-F dynamics and triangle ---------------------------------------------


def test_dynamic_fmuf_static_limit() -> None:
    static = linear_fmuf_polarization(T, 1.17)
    dynamic = dynamic_fmuf_polarization(T, 1.17, 0.0)
    assert np.allclose(static, dynamic, atol=1e-12)


def test_dynamic_fmuf_slow_fluctuation_damps_tail() -> None:
    static = linear_fmuf_polarization(T, 1.17)
    dynamic = dynamic_fmuf_polarization(T, 1.17, 0.3)
    assert dynamic[0] == pytest.approx(1.0)
    assert dynamic[-1] < static[-1]
    assert np.all(dynamic <= 1.0 + 1e-9)


def test_dynamic_fmuf_fast_fluctuation_motional_narrowing() -> None:
    from asymmetry.core.fitting.muon_fluorine.dipolar import omega_d_mu_f_rad_per_us

    omega_d = omega_d_mu_f_rad_per_us(1.17)
    nu = 50.0
    expected = np.exp(-2.0 * omega_d**2 * T / nu)
    assert np.allclose(dynamic_fmuf_polarization(T, 1.17, nu), expected, atol=1e-12)


def test_triangle_distant_third_fluorine_matches_general_collinear() -> None:
    # r3 -> infinity decouples the third spin; the remaining physics is the
    # collinear three-spin problem with F-F coupling = FmuF_General(r, r, 180).
    tri = fmuf_triangle_polarization(T, 1.17, 60.0, 90.0)
    gen = general_fmuf_polarization(T, 1.17, 1.17, 180.0)
    assert np.allclose(tri, gen, atol=1e-6)


def test_triangle_component_handles_invalid_trial_point() -> None:
    y = COMPONENTS["FmuF_Triangle"].function(T, A=0.8, r_muF=1.2, r3=-1.0, phi3=90.0)
    assert np.all(np.isfinite(y))
    assert y[0] > 100.0


def test_triangle_third_fluorine_changes_lineshape() -> None:
    near = fmuf_triangle_polarization(T, 1.17, 2.0, 90.0)
    far = fmuf_triangle_polarization(T, 1.17, 30.0, 90.0)
    assert np.max(np.abs(near - far)) > 0.01


# --- fit recovery smoke test ---------------------------------------------------


def test_risch_kehr_fit_recovery() -> None:
    from asymmetry.core.data.dataset import MuonDataset
    from asymmetry.core.fitting.engine import FitEngine
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet

    rng = np.random.default_rng(42)
    t = np.linspace(0.0, 12.0, 240)
    truth = {"A": 22.0, "Gamma": 1.5}
    definition = COMPONENTS["RischKehr"]
    y = definition.function(t, **truth) + rng.normal(0.0, 0.2, t.size)
    dataset = MuonDataset(time=t, asymmetry=y, error=np.full_like(t, 0.2))

    model = CompositeModel(["RischKehr"])
    definition_composite = model.to_model_definition()
    start = {"A_1": 18.0, "Gamma": 0.8}
    params = ParameterSet(
        [
            Parameter(name, value=start.get(name, 1.0), min=0.0)
            for name in definition_composite.param_names
        ]
    )
    result = FitEngine().fit(dataset, model.function, params)
    fitted = {p.name: p.value for p in result.parameters}
    assert fitted["A_1"] == pytest.approx(truth["A"], abs=0.5)
    assert fitted["Gamma"] == pytest.approx(truth["Gamma"], abs=0.2)
