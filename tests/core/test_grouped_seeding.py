"""Grouped nuisance seeds retain the count and absolute-phase conventions."""

import numpy as np
import pytest

from asymmetry.core.fitting.grouped_seeding import (
    _bounded_phase_seed_padding,
    seed_group_absolute_phases,
    seed_group_background_and_n0,
)
from asymmetry.core.fitting.grouped_time_domain import GroupedTimeDomainGroup
from asymmetry.core.utils.constants import (
    GAUSS_TO_TESLA,
    MUON_GYROMAGNETIC_RATIO_MHZ_PER_T,
    MUON_LIFETIME_US,
)


def test_background_and_n0_recover_lifetime_corrected_constant_signal():
    time = np.linspace(0.0, 8.0, 801)
    counts = 120.0 + 7.0 * np.exp(time / MUON_LIFETIME_US)
    assert seed_group_background_and_n0(counts, time=time) == pytest.approx((7.0, 120.0, 0.01))


@pytest.mark.parametrize(
    "counts, expected",
    [([], (0.0, 100.0, 0.2)), ([np.nan], (0.0, 100.0, 0.2)), ([4.0], (4.0, 1.0, 1.0))],
)
def test_background_seed_fallbacks(counts, expected):
    assert seed_group_background_and_n0(np.array(counts)) == pytest.approx(expected)


@pytest.mark.parametrize("phase", [0.4, -1.1, 2.5])
def test_phase_seed_is_absolute_radians(phase):
    time = np.linspace(0.0, 8.0, 801)
    counts = 100.0 * (1.0 + 0.18 * np.cos(2 * np.pi * 3.2 * time + phase))
    counts += 8.0 * np.exp(time / MUON_LIFETIME_US)
    group = GroupedTimeDomainGroup(7, "Group", time, counts, np.full_like(time, 0.05))
    assert seed_group_absolute_phases([group])["7"] == pytest.approx(phase, abs=0.25)


def test_short_trace_and_bounded_padding():
    group = GroupedTimeDomainGroup(7, "Group", np.arange(3.0), np.ones(3), np.ones(3))
    assert seed_group_absolute_phases([group]) == {"7": 0.0}
    assert [_bounded_phase_seed_padding(n) for n in (0, 100, 32768, 131073)] == [1, 8, 4, 1]


def test_grouped_phase_seed_uses_absolute_fft_peak_estimate() -> None:
    # The grouped fit seeds each group's phase nuisance with the group's
    # *absolute* FFT phase (the shared model phase is held at zero), not an
    # offset relative to the first group.
    time = np.linspace(0.0, 8.0, 801)
    frequency = 3.2
    background = 8.0
    n0 = 100.0
    amplitude = 0.18
    phase_a = np.deg2rad(25.0)
    phase_b = np.deg2rad(-55.0)

    def _counts(phase: float) -> np.ndarray:
        return n0 * (1.0 + amplitude * np.cos(2.0 * np.pi * frequency * time + phase)) + (
            background * np.exp(time / float(MUON_LIFETIME_US))
        )

    grouped_groups = [
        GroupedTimeDomainGroup(
            group_id=1,
            group_name="Forward",
            time=time,
            counts=_counts(phase_a),
            error=np.full_like(time, 0.05),
            metadata={"field": frequency / (MUON_GYROMAGNETIC_RATIO_MHZ_PER_T * GAUSS_TO_TESLA)},
        ),
        GroupedTimeDomainGroup(
            group_id=2,
            group_name="Backward",
            time=time,
            counts=_counts(phase_b),
            error=np.full_like(time, 0.05),
            metadata={"field": frequency / (MUON_GYROMAGNETIC_RATIO_MHZ_PER_T * GAUSS_TO_TESLA)},
        ),
    ]

    phases = seed_group_absolute_phases(grouped_groups)

    assert phases["1"] == pytest.approx(float(np.angle(np.exp(1j * phase_a))), abs=0.25)
    assert phases["2"] == pytest.approx(float(np.angle(np.exp(1j * phase_b))), abs=0.25)


def test_grouped_seed_estimates_background_n0_and_amplitude_from_counts_definition() -> None:
    time = np.linspace(0.0, 8.0, 801)
    n0 = 100.0
    background = 12.0
    amplitude = 0.18
    polarization = amplitude * np.cos(2.0 * np.pi * 1.2 * time)
    fast_component = 80.0 * np.exp(-time / 0.05)
    counts = (
        n0 * (1.0 + polarization)
        + background * np.exp(time / float(MUON_LIFETIME_US))
        + fast_component
    )

    estimated_background, estimated_n0, estimated_amplitude = seed_group_background_and_n0(
        counts,
        time=time,
    )

    assert estimated_background == pytest.approx(background, rel=0.2)
    assert estimated_n0 == pytest.approx(n0, rel=0.2)
    assert estimated_amplitude == pytest.approx(amplitude, rel=0.25)


def test_bounded_phase_seed_padding_caps_large_signals() -> None:
    """Phase-seed FFT padding is capped so huge histograms stay cheap."""
    from asymmetry.core.fitting.grouped_seeding import (
        _MAX_PHASE_SEED_FFT_POINTS,
        _bounded_phase_seed_padding,
    )

    # Small signals keep the full desired padding.
    assert _bounded_phase_seed_padding(1024, desired=8) == 8
    # Very large signals are capped to padding 1 (zero-padding only
    # interpolates, so this does not alias the phase seed).
    huge = _MAX_PHASE_SEED_FFT_POINTS * 4
    assert _bounded_phase_seed_padding(huge, desired=8) == 1
    # For a signal within the bound, padding never pushes the padded length
    # past the cap.
    n = _MAX_PHASE_SEED_FFT_POINTS // 2
    assert _bounded_phase_seed_padding(n, desired=8) * n <= _MAX_PHASE_SEED_FFT_POINTS
    assert _bounded_phase_seed_padding(0) == 1
