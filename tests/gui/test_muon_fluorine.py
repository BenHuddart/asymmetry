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


def _general_spectral_terms_loop(
    r1: float, r2: float, theta_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    """Independent per-orientation-loop reference for the general F-mu-F spectrum.

    Mirrors the pre-vectorization implementation (one ``np.linalg.eigh`` per
    powder orientation via ``three_spin_hamiltonian_rad_per_us``) so the batched
    ``_general_spectral_terms_cached`` can be checked against it directly.
    """
    from asymmetry.core.fitting.muon_fluorine.dipolar import (
        MUON_SIGMA_Z_THREE_SPIN,
        omega_d_f_f_rad_per_us,
        three_spin_hamiltonian_rad_per_us,
    )

    theta = np.deg2rad(theta_deg)
    v_f1 = np.array([0.0, 0.0, 1.0])
    v_f2 = np.array([np.sin(theta), 0.0, np.cos(theta)])

    rotations, weights = polarization._powder_rotations(
        polarization._DEFAULT_NUM_BETA,
        polarization._DEFAULT_NUM_ALPHA,
        polarization._DEFAULT_NUM_GAMMA,
    )
    n_mu_f1 = rotations @ v_f1
    n_mu_f2 = rotations @ v_f2
    f1_to_f2 = r2 * n_mu_f2 - r1 * n_mu_f1
    d_f1_f2 = np.linalg.norm(f1_to_f2, axis=1)
    n_f1_f2 = f1_to_f2 / d_f1_f2[:, None]

    c1 = omega_d_mu_f_rad_per_us(r1)
    c2 = omega_d_mu_f_rad_per_us(r2)
    c3 = omega_d_f_f_rad_per_us(float(np.mean(d_f1_f2)))
    dim = MUON_SIGMA_Z_THREE_SPIN.shape[0]

    freqs_list, amps_list = [], []
    for idx, w in enumerate(weights):
        h = three_spin_hamiltonian_rad_per_us(c1, c2, c3, n_mu_f1[idx], n_mu_f2[idx], n_f1_f2[idx])
        evals, evecs = np.linalg.eigh(h)
        sigma_eig = evecs.conj().T @ MUON_SIGMA_Z_THREE_SPIN @ evecs
        tw = (np.abs(sigma_eig) ** 2) / float(dim)
        omega_mn = (evals[:, None] - evals[None, :]).real
        freqs_list.append(omega_mn.ravel())
        amps_list.append((float(w) * tw).ravel().real)

    frequencies = np.concatenate(freqs_list)
    amplitudes = np.concatenate(amps_list)
    binned = np.round(frequencies, decimals=polarization._SPECTRUM_BIN_DECIMALS)
    unique_freq, inverse = np.unique(binned, return_inverse=True)
    binned_amps = np.zeros_like(unique_freq, dtype=float)
    np.add.at(binned_amps, inverse, amplitudes)
    total = float(np.sum(binned_amps))
    if total > 0.0:
        binned_amps /= total
    return unique_freq, binned_amps


def test_general_spectral_terms_match_orientation_loop() -> None:
    """The batched spectral build must equal the per-orientation loop reference."""
    vectorized = polarization._general_spectral_terms_cached.__wrapped__
    for r1, r2, theta in [
        (1.2, 1.2, 180.0),
        (1.18, 1.24, 150.0),
        (1.10, 1.30, 90.0),
        (1.05, 1.05, 120.0),
        (1.17, 1.22, 60.0),
    ]:
        freqs_v, amps_v = vectorized(
            r1,
            r2,
            theta,
            polarization._DEFAULT_NUM_BETA,
            polarization._DEFAULT_NUM_ALPHA,
            polarization._DEFAULT_NUM_GAMMA,
        )
        freqs_l, amps_l = _general_spectral_terms_loop(r1, r2, theta)

        assert freqs_v.shape == freqs_l.shape
        np.testing.assert_allclose(freqs_v, freqs_l, rtol=0.0, atol=1e-9)
        np.testing.assert_allclose(amps_v, amps_l, rtol=0.0, atol=1e-12)


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
