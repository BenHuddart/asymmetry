"""Phase 2: ISIS pulse-shape response and the interior exclusion window."""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.maxent import (
    MaxEntConfig,
    build_maxent_input,
    opus,
    pulse_amplitude_phase,
    pulse_response,
    run_cycles,
    tropus,
)


def _fb_run(histograms: list[Histogram]) -> Run:
    n = histograms[0].counts.size
    return Run(
        run_number=1,
        histograms=histograms,
        metadata={"field": 0.0},
        grouping={
            "groups": {1: [1], 2: [2]},
            "group_names": {1: "F", 2: "B"},
            "first_good_bin": 0,
            "last_good_bin": n - 1,
            "deadtime_correction": False,
        },
    )


def _pulsed_two_line_run(freqs=(1.0, 7.0), *, half_width_us=0.08, amp=0.18) -> Run:
    """F/B run whose lines are pre-attenuated by the single-pulse response."""
    rng = np.random.default_rng(3)
    bin_width = 0.016
    n = 400
    time = np.arange(n, dtype=float) * bin_width
    response_amp, response_delta = pulse_amplitude_phase(
        np.asarray(freqs), half_width_us=half_width_us, n_pulses=1
    )
    histograms: list[Histogram] = []
    for phase0 in (0.0, 180.0):
        signal = np.ones_like(time)
        for amplitude_r, delta, freq in zip(response_amp, response_delta, freqs):
            # Each line enters as the instrument sees it: R·cos(2πft + φ − δ).
            signal = signal + amp * amplitude_r * np.cos(
                2.0 * np.pi * freq * time + np.deg2rad(phase0) - delta
            )
        counts = 4000.0 * np.exp(-time / 2.1969811) * signal
        histograms.append(
            Histogram(
                counts=rng.poisson(np.clip(counts, 1.0, None)).astype(float),
                bin_width=bin_width,
                t0_bin=0,
            )
        )
    return _fb_run(histograms)


# ── kernel mathematics ──────────────────────────────────────────────────────


def test_pulse_response_dc_and_ignore() -> None:
    freqs = np.array([0.0, 1.0, 5.0])
    # DC is exactly (1, 0).
    p_cos, p_sin = pulse_response(freqs, half_width_us=0.08, n_pulses=1)
    assert p_cos[0] == pytest.approx(1.0)
    assert p_sin[0] == pytest.approx(0.0)
    # Ignore (n_pulses=0) is the identity response.
    ic, is_ = pulse_response(freqs, half_width_us=0.08, n_pulses=0)
    np.testing.assert_allclose(ic, 1.0)
    np.testing.assert_allclose(is_, 0.0)


def test_single_pulse_is_double_pulse_zero_separation_limit() -> None:
    freqs = np.linspace(0.0, 10.0, 64)
    single = pulse_response(freqs, half_width_us=0.08, n_pulses=1)
    double0 = pulse_response(freqs, half_width_us=0.08, separation_us=0.0, n_pulses=2)
    np.testing.assert_allclose(single[0], double0[0])
    np.testing.assert_allclose(single[1], double0[1])


def test_pulse_response_rolls_off_with_frequency() -> None:
    """The finite pulse acts as a low-pass: the response amplitude R(ν) falls
    with frequency, suppressing amplitudes well before ~10 MHz (the documented
    pulsed-source limit).  ``pulse_amplitude_phase`` returns ``(R, δ)`` — R is
    already the kernel magnitude."""
    freqs = np.linspace(0.0, 10.0, 200)
    amplitude, _phase = pulse_amplitude_phase(freqs, half_width_us=0.08, n_pulses=1)
    assert amplitude[0] == pytest.approx(1.0)
    # The response never amplifies (R ≤ 1 everywhere).
    assert amplitude.max() <= 1.0 + 1.0e-9
    # Monotone roll-off through the main lobe and strong suppression by 7 MHz.
    main_lobe = freqs < 6.0
    assert np.all(np.diff(amplitude[main_lobe]) <= 1.0e-9)
    i7 = int(np.abs(freqs - 7.0).argmin())
    assert amplitude[i7] < 0.3


# ── the response folds into the forward model, preserving the adjoint ────────


def test_opus_tropus_are_adjoint_with_pulse() -> None:
    run = _pulsed_two_line_run()
    config = MaxEntConfig(
        n_spectrum_points=64,
        f_min_mhz=0.2,
        f_max_mhz=9.0,
        auto_window=False,
        pulse_mode="single",
        pulse_half_width_us=0.08,
    )
    prepared = build_maxent_input(run, config)
    assert prepared.pulse_amp is not None  # pulse is active
    rng = np.random.default_rng(5)
    spectrum = rng.random(prepared.n_spectrum_points)
    values = {g.group_id: rng.normal(size=g.time_us.size) for g in prepared.groups}
    forward = opus(spectrum, prepared)
    lhs = sum(float(np.dot(forward[g.group_id], values[g.group_id])) for g in prepared.groups)
    rhs = float(np.dot(spectrum, tropus(values, prepared)))
    assert lhs == pytest.approx(rhs, rel=1e-10, abs=1e-10)


def test_pulse_correction_recovers_flat_amplitude() -> None:
    """The headline Phase-2 target: on pulsed data with equal-amplitude lines at
    1 and 7 MHz, the recovered spectral weight rolls off with frequency when the
    pulse response is off, and is restored ~flat when it is on."""
    run = _pulsed_two_line_run(freqs=(1.0, 7.0), half_width_us=0.08)

    def recovered_ratio(pulse_mode: str) -> float:
        config = MaxEntConfig(
            n_spectrum_points=256,
            f_min_mhz=0.2,
            f_max_mhz=9.0,
            auto_window=False,
            outer_cycles=12,
            inner_iterations=8,
            fit_phases=False,
            group_phase_degrees={1: 0.0, 2: 180.0},
            pulse_mode=pulse_mode,
            pulse_half_width_us=0.08,
        )
        maxent_input = build_maxent_input(run, config)
        result = run_cycles(maxent_input, config)
        frequencies = result.frequencies_mhz

        def weight_near(f0: float) -> float:
            band = np.abs(frequencies - f0) < 0.6
            return float(np.trapezoid(result.spectrum[band], frequencies[band]))

        return weight_near(7.0) / weight_near(1.0)

    ratio_off = recovered_ratio("ignore")
    ratio_on = recovered_ratio("single")

    # Disabled: the high-frequency line is visibly suppressed (rolled off).
    assert ratio_off < 0.5
    # Enabled: recovery is roughly flat and markedly better than disabled.
    assert 0.6 < ratio_on < 1.8
    assert ratio_on > 2.0 * ratio_off


# ── interior exclusion window ───────────────────────────────────────────────


def _single_line_run(freq=2.0) -> Run:
    rng = np.random.default_rng(9)
    bin_width = 0.02
    n = 300
    time = np.arange(n, dtype=float) * bin_width
    histograms: list[Histogram] = []
    for phase0 in (0.0, 180.0):
        signal = 1.0 + 0.2 * np.cos(2.0 * np.pi * freq * time + np.deg2rad(phase0))
        counts = 4000.0 * np.exp(-time / 2.1969811) * signal
        histograms.append(
            Histogram(
                counts=rng.poisson(np.clip(counts, 1.0, None)).astype(float),
                bin_width=bin_width,
                t0_bin=0,
            )
        )
    return _fb_run(histograms)


def test_exclusion_window_inflates_sigma_and_preserves_grid() -> None:
    run = _single_line_run()
    base = MaxEntConfig(n_spectrum_points=64, f_min_mhz=0.5, f_max_mhz=4.0, auto_window=False)
    excluded = MaxEntConfig(
        n_spectrum_points=64,
        f_min_mhz=0.5,
        f_max_mhz=4.0,
        auto_window=False,
        exclude_t_min_us=2.0,
        exclude_t_max_us=3.0,
    )
    base_input = build_maxent_input(run, base)
    excl_input = build_maxent_input(run, excluded)

    for base_group, excl_group in zip(base_input.groups, excl_input.groups):
        # The grid length and mask are untouched — points are de-weighted, not
        # dropped (FFT/grid length preserved).
        assert excl_group.time_us.size == base_group.time_us.size
        assert int(np.count_nonzero(excl_group.mask)) == int(np.count_nonzero(base_group.mask))
        inside = (excl_group.time_us >= 2.0) & (excl_group.time_us <= 3.0)
        # σ is hugely inflated inside the window and unchanged outside.
        assert np.all(excl_group.sigma[inside] > 1.0e6 * base_group.sigma[inside])
        np.testing.assert_allclose(excl_group.sigma[~inside], base_group.sigma[~inside])


def test_pulse_and_exclusion_changes_force_state_restart() -> None:
    run = _single_line_run()
    config = MaxEntConfig(n_spectrum_points=64, f_min_mhz=0.5, f_max_mhz=4.0, auto_window=False)
    prepared = build_maxent_input(run, config)
    from asymmetry.core.maxent import initialize_state

    state = initialize_state(prepared, config)

    for changed in (
        MaxEntConfig(
            n_spectrum_points=64,
            f_min_mhz=0.5,
            f_max_mhz=4.0,
            auto_window=False,
            pulse_mode="single",
        ),
        MaxEntConfig(
            n_spectrum_points=64,
            f_min_mhz=0.5,
            f_max_mhz=4.0,
            auto_window=False,
            exclude_t_min_us=2.0,
            exclude_t_max_us=3.0,
        ),
    ):
        changed_input = build_maxent_input(run, changed)
        with pytest.raises(ValueError, match="restart"):
            run_cycles(changed_input, changed, state=state, cycles=1)


def test_maxent_config_pulse_and_units_round_trip() -> None:
    config = MaxEntConfig(
        pulse_mode="double",
        pulse_half_width_us=0.08,
        pulse_separation_us=0.324,
        exclude_t_min_us=1.0,
        exclude_t_max_us=2.0,
    )
    restored = MaxEntConfig.from_dict(config.to_dict())
    assert restored.pulse_mode == "double"
    assert restored.pulse_half_width_us == pytest.approx(0.08)
    assert restored.pulse_separation_us == pytest.approx(0.324)
    assert restored.exclude_t_min_us == pytest.approx(1.0)
    assert restored.exclude_t_max_us == pytest.approx(2.0)
    # Malformed values degrade to defaults, not exceptions.
    bad = MaxEntConfig.from_dict({"pulse_mode": "wat"})
    assert bad.pulse_mode == "ignore"
