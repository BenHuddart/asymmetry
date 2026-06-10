"""Tests for muon-fluorine polarization models."""

from __future__ import annotations

import time

import numpy as np

from asymmetry.core.fitting.component_docs import get_component_applicability
from asymmetry.core.fitting.composite import COMPONENTS
from asymmetry.core.fitting.muon_fluorine import polarization
from asymmetry.core.fitting.muon_fluorine.dipolar import omega_d_mu_f_rad_per_us
from asymmetry.core.fitting.muon_fluorine.polarization import (
    general_fmuf_polarization,
    linear_fmuf_polarization,
    mu_f_polarization,
)
from asymmetry.core.fitting.parameters import get_param_info
from asymmetry.gui.utils.latex_renderer import render_latex_png_bytes
from asymmetry.gui.widgets.component_info_dialog import build_component_info_html


def test_mu_f_dipolar_frequency_scales_as_inverse_cubic_distance() -> None:
    omega_12 = omega_d_mu_f_rad_per_us(1.2)
    omega_15 = omega_d_mu_f_rad_per_us(1.5)

    assert omega_12 > 0.0
    assert np.isclose(omega_15 / omega_12, (1.2 / 1.5) ** 3, rtol=1.0e-12)

    nu_12_mhz = omega_12 / (2.0 * np.pi)
    assert 0.15 < nu_12_mhz < 0.30


def test_analytical_mu_f_and_linear_fmuf_are_normalized_at_t0() -> None:
    t = np.linspace(0.0, 6.0, 200)

    muf = mu_f_polarization(t, r_muF=1.2)
    linear = linear_fmuf_polarization(t, r_muF=1.2)

    assert np.isclose(muf[0], 1.0)
    assert np.isclose(linear[0], 1.0)
    assert np.all(np.isfinite(muf))
    assert np.all(np.isfinite(linear))


def test_general_fmuf_approaches_linear_case_for_collinear_equal_distances() -> None:
    t = np.linspace(0.0, 4.0, 160)

    linear = linear_fmuf_polarization(t, r_muF=1.2)
    general = general_fmuf_polarization(t, r1=1.2, r2=1.2, theta=180.0)

    rms = float(np.sqrt(np.mean((general - linear) ** 2)))
    assert rms < 0.06


def test_general_fmuf_cache_hits_on_repeated_geometry() -> None:
    polarization._general_spectral_terms_cached.cache_clear()

    t = np.linspace(0.0, 2.0, 80)
    _ = general_fmuf_polarization(t, r1=1.18, r2=1.24, theta=150.0)
    first_info = polarization._general_spectral_terms_cached.cache_info()

    _ = general_fmuf_polarization(t, r1=1.18, r2=1.24, theta=150.0)
    second_info = polarization._general_spectral_terms_cached.cache_info()

    assert second_info.hits >= first_info.hits + 1


def test_muon_fluorine_components_registered() -> None:
    expected = {"MuF", "FmuF_Linear", "FmuF_General", "FmuF_Triangle", "DynamicFmuF"}
    assert expected.issubset(COMPONENTS)

    for name in expected:
        assert COMPONENTS[name].category == "Nuclear dipolar"


def test_muon_fluorine_components_return_finite_arrays() -> None:
    t = np.linspace(0.0, 2.0, 80)

    y1 = COMPONENTS["MuF"].function(t, A=0.8, r_muF=1.2)
    y2 = COMPONENTS["FmuF_Linear"].function(t, A=0.8, r_muF=1.2)
    y3 = COMPONENTS["FmuF_General"].function(t, A=0.8, r1=1.2, r2=1.3, theta=145.0)

    assert np.all(np.isfinite(y1))
    assert np.all(np.isfinite(y2))
    assert np.all(np.isfinite(y3))


def test_general_component_handles_invalid_theta_trial_point() -> None:
    t = np.linspace(0.0, 2.0, 80)

    y = COMPONENTS["FmuF_General"].function(t, A=0.8, r1=1.2, r2=1.3, theta=-5.0)

    assert np.all(np.isfinite(y))
    assert np.all(y == y[0])
    assert y[0] > 100.0


def test_theta_has_nonnegative_default_min_bound() -> None:
    info = get_param_info("theta")
    assert info.default_min == 0.0


def test_general_fmuf_caching_reduces_evaluation_time() -> None:
    t = np.linspace(0.0, 4.0, 220)
    r1 = 1.18
    r2 = 1.24
    theta = 150.0

    uncached_times: list[float] = []
    for _ in range(3):
        polarization._general_spectral_terms_cached.cache_clear()
        start = time.perf_counter()
        _ = general_fmuf_polarization(t, r1=r1, r2=r2, theta=theta)
        uncached_times.append(time.perf_counter() - start)

    polarization._general_spectral_terms_cached.cache_clear()
    _ = general_fmuf_polarization(t, r1=r1, r2=r2, theta=theta)

    cached_times: list[float] = []
    for _ in range(5):
        start = time.perf_counter()
        _ = general_fmuf_polarization(t, r1=r1, r2=r2, theta=theta)
        cached_times.append(time.perf_counter() - start)

    uncached_median = float(np.median(uncached_times))
    cached_median = float(np.median(cached_times))
    assert cached_median < uncached_median


def test_muon_fluorine_applicability_text_mentions_physical_use_cases() -> None:
    muf_text = get_component_applicability("MuF")
    linear_text = get_component_applicability("FmuF_Linear")
    general_text = get_component_applicability("FmuF_General")

    import re

    # Tolerate either unicode or ASCII renderings of the isotope/formula
    # glyphs — these assertions pin the physics content, not the typography.
    assert re.search(r"one dominant (?:¹⁹|19)f nucleus", muf_text.lower())
    assert "ionic fluorides" in linear_text.lower()
    assert "three coupled spins" in general_text.lower()
    assert re.search(r"hf[₂2]", general_text.lower())


def test_component_info_html_includes_muon_fluorine_physics_sections() -> None:
    html = build_component_info_html(COMPONENTS["FmuF_General"], render_latex_images=False)

    assert "Dipolar Hamiltonian" in html
    assert "Dipolar Frequency" in html
    assert "Stopping-State Scenario" in html
    assert "Measured Asymmetry Context" in html
    assert "Model Limits" in html
    assert "\\sum_{i&gt;j}" in html
    assert "\\omega_{ij}" in html


def test_component_info_html_includes_aps_style_references() -> None:
    html = build_component_info_html(COMPONENTS["FmuF_General"], render_latex_images=False)

    assert "References" in html
    assert "T. Lancaster et al., Phys. Rev. Lett. 99, 267601 (2007)." in html
    assert "J. H. Brewer et al., Phys. Rev. B 33, 7813 (1986)." in html


def test_muon_fluorine_hamiltonian_blocks_render_to_png() -> None:
    hamiltonian = (
        r"H = \sum_{i>j} H_{ij}, \quad "
        r"H_{ij}=\omega_{ij}\left[\mathbf{S}_i\cdot\mathbf{S}_j - 3(\mathbf{S}_i\cdot\hat{r}_{ij})(\mathbf{S}_j\cdot\hat{r}_{ij})\right]"
    )
    frequency = r"\omega_{ij}=\frac{\mu_0}{4\pi}\gamma_i\gamma_j\hbar\,r_{ij}^{-3}"

    assert render_latex_png_bytes(hamiltonian) is not None
    assert render_latex_png_bytes(frequency) is not None
