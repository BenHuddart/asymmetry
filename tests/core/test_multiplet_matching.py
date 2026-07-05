"""Tests for multiplet pattern matching and its physics inversion helpers.

Signal generators are the actual component physics functions (``muonium``,
``muon_fluorine``), so the matcher is validated against the same forward maps
the fit wizard will seed.
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.muon_fluorine.dipolar import (
    omega_d_mu_f_rad_per_us,
    r_mu_f_from_omega_d,
)
from asymmetry.core.fitting.muon_fluorine.polarization import linear_fmuf_polarization
from asymmetry.core.fitting.muonium import (
    VACUUM_MUONIUM_A_HF_MHZ,
    a_hf_from_low_tf_pair,
    high_tf_pair_frequencies,
    low_tf_muonium,
    low_tf_pair_frequencies,
)
from asymmetry.core.fitting.peak_detection import (
    DetectedPeak,
    PeakAnalysis,
    analyze_dataset_peaks,
    deserialize_multiplet_match,
    match_multiplets,
    serialize_multiplet_match,
)
from asymmetry.core.fitting.spectral import field_gauss_to_frequency_mhz


def _dataset(time: np.ndarray, asym: np.ndarray) -> MuonDataset:
    return MuonDataset(
        time=np.asarray(time, dtype=float),
        asymmetry=np.asarray(asym, dtype=float),
        error=np.full_like(np.asarray(time, dtype=float), 0.05),
        metadata={"run_number": 1},
    )


def _peaks_at(frequencies: list[float], snr: float = 20.0) -> PeakAnalysis:
    peaks = tuple(
        DetectedPeak(
            frequency_mhz=f,
            amplitude=1.0,
            snr=snr,
            width_mhz=0.05,
            prominence=1.0,
            source="fft",
        )
        for f in frequencies
    )
    return PeakAnalysis(
        peaks=peaks,
        noise_floor=0.05,
        resolution_mhz=0.05,
        nyquist_mhz=1.2 * max(frequencies),
        detrended=True,
    )


# --------------------------------------------------------------------------- #
# Inversion helpers
# --------------------------------------------------------------------------- #


def test_a_hf_inversion_round_trip_exact() -> None:
    for field, a_hf in ((20.0, VACUUM_MUONIUM_A_HF_MHZ), (50.0, 500.0), (5.0, 4463.302)):
        lo, hi = low_tf_pair_frequencies(field, a_hf)
        recovered = a_hf_from_low_tf_pair(field, lo, hi)
        assert recovered is not None
        assert recovered == pytest.approx(a_hf, rel=1e-6)


def test_a_hf_inversion_rejects_impossible_pairs() -> None:
    assert a_hf_from_low_tf_pair(20.0, 1.4, 1.4) is None  # zero splitting
    assert a_hf_from_low_tf_pair(20.0, 1.0, 60000.0) is None  # beyond bracket
    assert a_hf_from_low_tf_pair(0.0, 1.0, 2.0) is None  # no field
    assert a_hf_from_low_tf_pair(20.0, 2.0, 1.0) is None  # inverted order


def test_r_mu_f_inversion_round_trip() -> None:
    for r in (1.0, 1.17, 2.4):
        assert r_mu_f_from_omega_d(omega_d_mu_f_rad_per_us(r)) == pytest.approx(r)
    with pytest.raises(ValueError):
        r_mu_f_from_omega_d(0.0)


# --------------------------------------------------------------------------- #
# FFT-level matches on physics-generated signals
# --------------------------------------------------------------------------- #


def test_fmuf_triplet_recovers_distance() -> None:
    rng = np.random.default_rng(11)
    r_true = 1.17
    t = np.linspace(0.0, 32.0, 4096)
    y = linear_fmuf_polarization(t, r_true) + rng.normal(0.0, 0.01, t.size)
    analysis = analyze_dataset_peaks(_dataset(t, y), burg_check="never")

    matches = match_multiplets(analysis, field_gauss=None, geometry="ZF")
    fmuf = [m for m in matches if m.kind == "fmuf_linear"]
    assert fmuf, f"no fmuf match; peaks={[p.frequency_mhz for p in analysis.peaks]}"
    best = fmuf[0]
    assert best.family_key == "fmuf"
    r_derived = best.derived("r_muF_angstrom")
    assert r_derived is not None
    assert r_derived == pytest.approx(r_true, rel=0.03)


def test_low_tf_muonium_pair_recovers_hyperfine() -> None:
    rng = np.random.default_rng(12)
    field = 20.0
    t = np.linspace(0.0, 16.0, 16384)  # nyquist ~512 MHz, resolution 1/16 MHz
    y = 0.2 * low_tf_muonium(t, field, VACUUM_MUONIUM_A_HF_MHZ, 0.0)
    y = y + rng.normal(0.0, 0.005, t.size)
    analysis = analyze_dataset_peaks(_dataset(t, y), burg_check="never")

    matches = match_multiplets(analysis, field_gauss=field, geometry="TF")
    muonium = [m for m in matches if m.kind == "muonium_low_tf"]
    assert muonium, f"no match; peaks={[p.frequency_mhz for p in analysis.peaks]}"
    a_hf = muonium[0].derived("a_hf_mhz")
    assert a_hf is not None
    # Peak-position error propagates ~1/splitting into A_hf; 10 % is realistic.
    assert a_hf == pytest.approx(VACUUM_MUONIUM_A_HF_MHZ, rel=0.10)


def test_larmor_line_recognised() -> None:
    rng = np.random.default_rng(13)
    field = 100.0
    nu_d = field_gauss_to_frequency_mhz(field)
    t = np.linspace(0.0, 16.0, 4096)
    y = 0.2 * np.cos(2.0 * np.pi * nu_d * t) * np.exp(-0.1 * t)
    y = y + rng.normal(0.0, 0.01, t.size)
    analysis = analyze_dataset_peaks(_dataset(t, y), burg_check="never")

    matches = match_multiplets(analysis, field_gauss=field, geometry="TF")
    larmor = [m for m in matches if m.kind == "larmor"]
    assert larmor
    assert larmor[0].family_key == "oscillatory"
    derived_field = larmor[0].derived("field_gauss")
    assert derived_field is not None
    assert derived_field == pytest.approx(field, rel=0.05)


# --------------------------------------------------------------------------- #
# Unit-level matches on constructed peak lists
# --------------------------------------------------------------------------- #


def test_high_tf_pair_sum_identity() -> None:
    field = 5000.0
    a_hf = VACUUM_MUONIUM_A_HF_MHZ
    nu12, nu34 = high_tf_pair_frequencies(field, a_hf)
    assert nu12 + nu34 == pytest.approx(a_hf)

    analysis = _peaks_at([nu12, nu34])
    matches = match_multiplets(analysis, field_gauss=field, geometry="TF")
    high = [m for m in matches if m.kind == "muonium_high_tf"]
    assert high
    assert high[0].derived("a_hf_mhz") == pytest.approx(a_hf, rel=0.01)


def test_zf_muonium_relation() -> None:
    a_hf, d_mhz = 4463.302, 300.0
    f1, f2, f3 = a_hf - d_mhz, a_hf + d_mhz / 2.0, 1.5 * d_mhz
    analysis = _peaks_at([f1, f2, f3])
    matches = match_multiplets(analysis, field_gauss=None, geometry="ZF")
    zf = [m for m in matches if m.kind == "muonium_zf"]
    assert zf
    assert zf[0].derived("a_hf_mhz") == pytest.approx(a_hf, rel=0.01)
    assert zf[0].derived("d_mhz") == pytest.approx(d_mhz, rel=0.01)


def test_muf_triplet_ratios() -> None:
    r_true = 1.5
    omega_tilde = omega_d_mu_f_rad_per_us(r_true) / (2.0 * np.pi)
    analysis = _peaks_at([0.5 * omega_tilde, omega_tilde, 1.5 * omega_tilde])
    matches = match_multiplets(analysis, field_gauss=None, geometry="ZF")
    muf = [m for m in matches if m.kind == "muf"]
    assert muf
    assert muf[0].derived("r_muF_angstrom") == pytest.approx(r_true, rel=0.02)


# --------------------------------------------------------------------------- #
# Geometry / field gating
# --------------------------------------------------------------------------- #


def test_geometry_gates_rules() -> None:
    # An F-mu-F-shaped triplet is not matched in TF geometry...
    omega_tilde = omega_d_mu_f_rad_per_us(1.17) / (2.0 * np.pi)
    factors = (0.5 * (3.0 - np.sqrt(3.0)), np.sqrt(3.0), 0.5 * (3.0 + np.sqrt(3.0)))
    triplet = _peaks_at([f * omega_tilde for f in factors])
    assert not [
        m
        for m in match_multiplets(triplet, field_gauss=20.0, geometry="TF")
        if m.kind == "fmuf_linear"
    ]
    # ...but is with unknown geometry (metadata-poor data keeps its hints).
    assert [
        m
        for m in match_multiplets(triplet, field_gauss=None, geometry=None)
        if m.kind == "fmuf_linear"
    ]

    # A Larmor line is not matched in ZF geometry or without a field value.
    nu_d = field_gauss_to_frequency_mhz(100.0)
    line = _peaks_at([nu_d])
    assert not match_multiplets(line, field_gauss=100.0, geometry="ZF")
    assert not [
        m for m in match_multiplets(line, field_gauss=None, geometry="TF") if m.kind == "larmor"
    ]


def test_empty_analysis_no_matches() -> None:
    empty = PeakAnalysis(
        peaks=(),
        noise_floor=0.0,
        resolution_mhz=0.1,
        nyquist_mhz=10.0,
        detrended=False,
    )
    assert match_multiplets(empty, field_gauss=20.0, geometry="TF") == ()


# --------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------- #


def test_multiplet_match_round_trip() -> None:
    analysis = _peaks_at([1.0, 2.7320508, 3.7320508])
    matches = match_multiplets(analysis, field_gauss=None, geometry="ZF")
    assert matches
    for match in matches:
        restored = deserialize_multiplet_match(serialize_multiplet_match(match))
        assert restored == match
    assert deserialize_multiplet_match(None) is None
    assert deserialize_multiplet_match({"kind": "larmor"}) is not None
