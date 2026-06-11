"""Tests for the muoniated-radical correlation spectrum.

Covers the pure-core combiner and forward map (`corr_fn`, `breit_rabi_pair`,
`correlation_spectrum`), the exact reconciliation with WiMDA's approximate
`rmatch` inverse, and the end-to-end `compute_average_group_spectrum`
`correlation` display mode on a synthetic muoniated-radical TF run.

Worked example: the cyclohexadienyl radical (Mu + benzene), A_µ = 514.4 MHz
(Blundell et al., *Muon Spectroscopy*, OUP 2022, §19.4 Example 19.8;
McKenzie et al., *J. Phys. Chem. B* 117, 13614 (2013)).
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.fourier.correlation import (
    DEFAULT_CORR_ORDER,
    _pair_frequencies,
    breit_rabi_pair,
    corr_fn,
    correlation_spectrum,
)
from asymmetry.core.fourier.spectrum import (
    GroupSpectrumConfig,
    compute_average_group_spectrum,
)
from asymmetry.core.fourier.units import gauss_to_mhz

GAMMA_MU_MHZ_PER_G = float(gauss_to_mhz(1.0))
CYCLOHEXADIENYL_A_MHZ = 514.4


def _wimda_rmatch(freq: float, field: float) -> float:
    """Verbatim transcription of WiMDA ``rmatch`` (``Plot.pas:515-523``)."""
    wemp = (2.81555 * field) ** 2
    wemn = 1.394225 * field
    wplus = 4.0 * (wemn - freq)
    return -(wemn - wemp / wplus)


def _radical_tf_run(
    *, field_gauss: float, couplings_mhz: tuple[float, ...], n: int = 2048, bin_width: float = 0.001
) -> Run:
    """A synthetic muoniated-radical TF run.

    Each coupling adds a Breit–Rabi line pair (ν₁₂, ν₃₄) on top of the
    diamagnetic line at γ_µ·B; fine (~ns) bins give the high Nyquist a
    continuous-source radical measurement needs.
    """
    rng = np.random.default_rng(11)
    time = np.arange(n, dtype=float) * bin_width
    diamag_mhz = field_gauss * GAMMA_MU_MHZ_PER_G
    relax = np.exp(-time / 1.5)
    signal_ac = 0.04 * np.cos(2.0 * np.pi * diamag_mhz * time)
    for a_mhz in couplings_mhz:
        nu12, nu34 = breit_rabi_pair(field_gauss, a_mhz)
        signal_ac = signal_ac + 0.10 * relax * (
            np.cos(2.0 * np.pi * nu12 * time) + np.cos(2.0 * np.pi * nu34 * time)
        )
    histograms: list[Histogram] = []
    for sign in (+1.0, -1.0):
        counts = 5.0e5 * np.exp(-time / 2.1969811) * (1.0 + sign * signal_ac)
        counts = rng.poisson(np.clip(counts, 1.0, None)).astype(float)
        histograms.append(Histogram(counts=counts, bin_width=bin_width, t0_bin=0))
    return Run(
        run_number=314,
        histograms=histograms,
        metadata={"field": float(field_gauss), "temperature": 298.0},
        grouping={
            "groups": {1: [1], 2: [2]},
            "group_names": {1: "Fwd", 2: "Bwd"},
            "first_good_bin": 0,
            "last_good_bin": n - 1,
            "deadtime_correction": False,
        },
    )


def _peak_x(x: np.ndarray, y: np.ndarray) -> float:
    return float(x[int(np.argmax(y))])


# ── corr_fn (WiMDA CorrFn port) ─────────────────────────────────────────────


def test_corr_fn_equal_amplitudes_is_product() -> None:
    assert float(corr_fn(2.0, 2.0, 2)) == pytest.approx(4.0)


def test_corr_fn_order_penalty() -> None:
    # 2·|4·1| / (4² + (1/4)²) = 8 / 16.0625
    assert float(corr_fn(4.0, 1.0, 2)) == pytest.approx(8.0 / (16.0 + 1.0 / 16.0))


def test_corr_fn_order_zero_is_plain_product() -> None:
    assert float(corr_fn(4.0, 1.0, 0)) == pytest.approx(4.0)


def test_corr_fn_zero_factor_is_zero() -> None:
    assert float(corr_fn(0.0, 5.0, 2)) == 0.0


def test_corr_fn_higher_order_suppresses_more() -> None:
    low = float(corr_fn(5.0, 1.0, 1))
    high = float(corr_fn(5.0, 1.0, 4))
    assert high < low  # larger order penalises unequal amplitudes harder


def test_corr_fn_vectorised() -> None:
    out = corr_fn(np.array([2.0, 4.0, 0.0]), np.array([2.0, 1.0, 5.0]), 2)
    np.testing.assert_allclose(out, [4.0, 8.0 / (16.0 + 1.0 / 16.0), 0.0])


# ── Breit–Rabi pair (exact, reuses muonium._tf_levels) ──────────────────────


@pytest.mark.parametrize("field", [1000.0, 2000.0, 2900.0, 5000.0])
@pytest.mark.parametrize("a_mhz", [200.0, 330.0, 514.4, 1200.0])
def test_breit_rabi_pair_sum_is_coupling(field: float, a_mhz: float) -> None:
    nu12, nu34 = breit_rabi_pair(field, a_mhz)
    assert nu12 + nu34 == pytest.approx(a_mhz, rel=1e-9)
    assert nu12 < nu34  # ν₁₂ is the lower line


def test_vectorised_pair_matches_scalar_reference() -> None:
    """The fast array form _pair_frequencies must equal the scalar breit_rabi_pair."""
    for field in (1000.0, 2900.0, 15000.0):
        a_axis = np.linspace(0.5, 1500.0, 400)
        nu12, nu34 = _pair_frequencies(field, a_axis)
        ref = [breit_rabi_pair(field, float(a)) for a in a_axis]
        np.testing.assert_allclose(nu12, [r[0] for r in ref], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(nu34, [r[1] for r in ref], rtol=1e-12, atol=1e-12)


def test_breit_rabi_pair_matches_wimda_rmatch_within_approximation() -> None:
    """The exact forward map agrees with WiMDA's approximate inverse.

    Documents the divergence: rmatch's rounded constants (2.81555/1.394225 vs
    CODATA 2.81605/1.394471) and high-field expansion drift A by ~0.01–0.03 MHz,
    while the forward-map sum is A to machine precision.
    """
    for field, a_mhz in [(1000.0, 500.0), (5000.0, 1200.0)]:
        nu12, nu34 = breit_rabi_pair(field, a_mhz)
        # rmatch maps the upper line (−ν₃₄ in WiMDA's negative convention) back
        # to its lower partner; |f₁+f₂| then estimates A.
        f1 = -nu34
        f2 = _wimda_rmatch(f1, field)
        a_from_rmatch = abs(f1 + f2)
        assert nu12 + nu34 == pytest.approx(a_mhz, rel=1e-9)  # exact
        assert a_from_rmatch == pytest.approx(a_mhz, abs=0.05)  # WiMDA, approximate


# ── correlation_spectrum (pure core) ────────────────────────────────────────


def _line_spectrum(freqs: np.ndarray, centres: tuple[float, ...], width: float = 1.5) -> np.ndarray:
    power = np.zeros_like(freqs)
    for c in centres:
        power += np.exp(-0.5 * ((freqs - c) / width) ** 2)
    return power


def test_correlation_spectrum_peaks_at_coupling() -> None:
    field, a_mhz = 2900.0, CYCLOHEXADIENYL_A_MHZ
    nu12, nu34 = breit_rabi_pair(field, a_mhz)
    freqs = np.arange(0.0, 500.0, 0.5)
    power = _line_spectrum(freqs, (field * GAMMA_MU_MHZ_PER_G, nu12, nu34))
    a_axis, corr = correlation_spectrum(freqs, power, field_gauss=field)
    assert a_axis.size > 0
    df = a_axis[1] - a_axis[0]
    assert _peak_x(a_axis, corr) == pytest.approx(a_mhz, abs=3.0 * df)


def test_correlation_spectrum_two_radicals_two_peaks() -> None:
    field = 2900.0
    a_lo, a_hi = 330.0, 514.4
    freqs = np.arange(0.0, 500.0, 0.5)
    centres = []
    for a in (a_lo, a_hi):
        centres.extend(breit_rabi_pair(field, a))
    power = _line_spectrum(freqs, tuple(centres))
    a_axis, corr = correlation_spectrum(freqs, power, field_gauss=field)
    # Two dominant peaks near the two couplings.
    from scipy.signal import find_peaks

    peaks, _ = find_peaks(corr, height=0.5 * corr.max())
    peak_positions = sorted(float(a_axis[p]) for p in peaks)
    df = a_axis[1] - a_axis[0]
    assert any(abs(p - a_lo) < 4.0 * df for p in peak_positions)
    assert any(abs(p - a_hi) < 4.0 * df for p in peak_positions)


def test_correlation_suppresses_near_dc_lower_line() -> None:
    """A candidate whose lower line ν₁₂ dips to ~0 must not borrow DC/baseline power."""
    field = 5000.0
    freqs = np.arange(0.0, 300.0, 0.5)
    # Strong low-frequency/baseline content near DC, nothing else.
    power = np.exp(-0.5 * ((freqs - 0.5) / 1.0) ** 2)
    a_axis, corr = correlation_spectrum(freqs, power, field_gauss=field)
    nu12, _nu34 = _pair_frequencies(field, a_axis)
    dip = int(np.argmin(nu12))
    assert nu12[dip] < 1.0  # ν₁₂ genuinely dips into the near-DC region here
    assert corr[dip] == 0.0  # ...and the spurious contribution is suppressed
    # No spurious peak survives from pure low-frequency content.
    assert float(np.max(corr)) == pytest.approx(0.0, abs=1e-9)


def test_correlation_spectrum_zero_field_is_empty() -> None:
    freqs = np.arange(0.0, 500.0, 0.5)
    power = _line_spectrum(freqs, (200.0, 300.0))
    a_axis, corr = correlation_spectrum(freqs, power, field_gauss=0.0)
    assert a_axis.size == 0 and corr.size == 0


# ── end-to-end through compute_average_group_spectrum ───────────────────────


def test_correlation_mode_peaks_at_coupling() -> None:
    run = _radical_tf_run(field_gauss=2900.0, couplings_mhz=(CYCLOHEXADIENYL_A_MHZ,))
    ds = compute_average_group_spectrum(run, GroupSpectrumConfig(display="Correlation"))
    assert ds is not None
    assert ds.metadata.get("correlation_axis") is True
    assert ds.metadata["x_label"].startswith("Muon hyperfine coupling")
    assert ds.metadata.get("fourier_correlation_field_gauss") == pytest.approx(2900.0)
    df = float(ds.time[1] - ds.time[0])
    assert _peak_x(ds.time, ds.asymmetry) == pytest.approx(CYCLOHEXADIENYL_A_MHZ, abs=5.0 * df)


def test_correlation_peak_is_field_independent() -> None:
    peaks = []
    for field in (2900.0, 5000.0):
        run = _radical_tf_run(field_gauss=field, couplings_mhz=(CYCLOHEXADIENYL_A_MHZ,))
        ds = compute_average_group_spectrum(run, GroupSpectrumConfig(display="Correlation"))
        assert ds is not None
        peaks.append(_peak_x(ds.time, ds.asymmetry))
    assert peaks[0] == pytest.approx(peaks[1], abs=6.0)


def test_correlation_reference_field_override() -> None:
    run = _radical_tf_run(field_gauss=2900.0, couplings_mhz=(CYCLOHEXADIENYL_A_MHZ,))
    # Drop the field from metadata; supply it via the config override instead.
    run.metadata.pop("field", None)
    cfg = GroupSpectrumConfig(display="Correlation", correlation_reference_field_gauss=2900.0)
    ds = compute_average_group_spectrum(run, cfg)
    assert ds is not None and ds.time.size > 0
    df = float(ds.time[1] - ds.time[0])
    assert _peak_x(ds.time, ds.asymmetry) == pytest.approx(CYCLOHEXADIENYL_A_MHZ, abs=5.0 * df)


# ── config round-trip + regression ──────────────────────────────────────────


def test_config_roundtrip_preserves_correlation_keys() -> None:
    cfg = GroupSpectrumConfig(
        display="Correlation",
        correlation_reference_field_gauss=2900.0,
        correlation_order=3,
    )
    restored = GroupSpectrumConfig.from_dict(cfg.to_dict())
    assert restored.correlation_reference_field_gauss == pytest.approx(2900.0)
    assert restored.correlation_order == 3
    assert restored.display == "Correlation"


def test_default_correlation_order_matches_wimda() -> None:
    assert GroupSpectrumConfig().correlation_order == DEFAULT_CORR_ORDER == 2


def test_non_correlation_mode_unaffected() -> None:
    run = _radical_tf_run(field_gauss=2900.0, couplings_mhz=(CYCLOHEXADIENYL_A_MHZ,))
    ds = compute_average_group_spectrum(run, GroupSpectrumConfig(display="(Power)^1/2"))
    assert ds is not None
    assert ds.metadata.get("correlation_axis") is None
    assert ds.metadata["x_label"] == "Frequency (MHz)"
