"""Phase 3: ZF/LF mode, SpecBG, spectrum/log export."""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.maxent import (
    MaxEntConfig,
    build_maxent_input,
    initialize_state,
    run_cycles,
    run_log_text,
    spectrum_to_text,
    subtract_zero_frequency,
)
from asymmetry.core.transform.deadtime import calibrate_deadtime_from_histograms
from asymmetry.core.utils.constants import MUON_LIFETIME_US


def _kubo_toyabe_fb_run(*, delta: float = 0.3, alpha: float = 1.2) -> Run:
    """Two F/B groups carrying a static Gaussian Kubo–Toyabe relaxation."""
    rng = np.random.default_rng(4)
    bin_width = 0.04
    n = 300
    time = np.arange(n, dtype=float) * bin_width
    kubo_toyabe = 1.0 / 3.0 + (2.0 / 3.0) * (1.0 - delta**2 * time**2) * np.exp(
        -0.5 * delta**2 * time**2
    )
    histograms: list[Histogram] = []
    for phase, efficiency in ((0.0, 1.0), (180.0, alpha)):
        asym = 0.25 * kubo_toyabe * np.cos(np.deg2rad(phase))
        counts = efficiency * 4000.0 * np.exp(-time / 2.1969811) * (1.0 + asym)
        histograms.append(
            Histogram(
                counts=rng.poisson(np.clip(counts, 1.0, None)).astype(float),
                bin_width=bin_width,
                t0_bin=0,
            )
        )
    return Run(
        run_number=1,
        histograms=histograms,
        metadata={"field": 0.0},
        grouping={
            "groups": {1: [1], 2: [2]},
            "group_names": {1: "F", 2: "B"},
            "alpha": alpha,
            "first_good_bin": 0,
            "last_good_bin": n - 1,
            "deadtime_correction": False,
        },
    )


def test_zf_lf_mode_pins_phases_ties_amplitudes_and_peaks_near_zero() -> None:
    alpha = 1.2
    run = _kubo_toyabe_fb_run(alpha=alpha)
    config = MaxEntConfig(
        n_spectrum_points=128,
        f_min_mhz=0.0,
        f_max_mhz=3.0,
        auto_window=False,
        outer_cycles=8,
        inner_iterations=6,
        mode="zf_lf",
        selected_group_ids=[1, 2],
    )
    maxent_input = build_maxent_input(run, config)
    assert [g.phase_degrees for g in maxent_input.groups] == [0.0, 180.0]
    assert maxent_input.zf_lf_alpha == pytest.approx(alpha)

    result = run_cycles(maxent_input, config)
    # Phases stay pinned; amplitudes obey the α tie F = α·B.
    assert result.state.phases == {1: 0.0, 2: 180.0}
    assert result.state.amplitudes[1] / result.state.amplitudes[2] == pytest.approx(alpha, rel=1e-6)

    # A Kubo–Toyabe field distribution is broad and centred near zero — most of
    # the spectral weight sits below the line, not in a sharp high-frequency peak.
    frequencies = result.frequencies_mhz
    spectrum = result.spectrum
    near_zero = float(np.trapezoid(spectrum[frequencies < 0.5], frequencies[frequencies < 0.5]))
    away = float(np.trapezoid(spectrum[frequencies > 1.5], frequencies[frequencies > 1.5]))
    assert near_zero > away
    assert frequencies[int(np.argmax(spectrum))] < 0.5


def test_zf_lf_mode_requires_exactly_two_groups() -> None:
    run = _kubo_toyabe_fb_run()
    with pytest.raises(ValueError, match="two selected groups"):
        build_maxent_input(run, MaxEntConfig(mode="zf_lf", selected_group_ids=[1]))


def test_zf_lf_mode_raises_when_a_group_has_no_in_window_data() -> None:
    # An exclusion/time window that empties one group must fail loudly rather
    # than silently degrade to an untied single-group fit.
    run = _kubo_toyabe_fb_run()
    config = MaxEntConfig(
        mode="zf_lf",
        selected_group_ids=[1, 2],
        f_min_mhz=0.0,
        f_max_mhz=3.0,
        auto_window=False,
        t_min_us=1000.0,  # past the end of the data → empties both groups
        t_max_us=2000.0,
    )
    with pytest.raises(ValueError, match="usable data|no valid grouped"):
        build_maxent_input(run, config)


def test_zf_lf_orders_pair_by_forward_backward_designation() -> None:
    # Backward group has the LOWER id: ordering by sorted id would pin the wrong
    # group at 0°.  The forward/backward designation must win.
    run = _kubo_toyabe_fb_run()
    run.grouping["forward_group"] = 2
    run.grouping["backward_group"] = 1
    config = MaxEntConfig(
        mode="zf_lf",
        selected_group_ids=[1, 2],
        f_min_mhz=0.0,
        f_max_mhz=3.0,
        auto_window=False,
    )
    maxent_input = build_maxent_input(run, config)
    # groups[0] is forward (group 2, pinned 0°); groups[1] is backward (group 1, 180°).
    assert maxent_input.groups[0].group_id == 2
    assert maxent_input.groups[0].phase_degrees == 0.0
    assert maxent_input.groups[1].group_id == 1
    assert maxent_input.groups[1].phase_degrees == 180.0


def test_zf_lf_tie_respects_disabled_amplitude_fit() -> None:
    run = _kubo_toyabe_fb_run(alpha=1.5)
    config = MaxEntConfig(
        n_spectrum_points=64,
        f_min_mhz=0.0,
        f_max_mhz=3.0,
        auto_window=False,
        outer_cycles=3,
        inner_iterations=3,
        mode="zf_lf",
        selected_group_ids=[1, 2],
        fit_amplitudes=False,
        fit_backgrounds=False,
        fit_constant_background=False,
    )
    result = run_cycles(build_maxent_input(run, config), config)
    # With amplitude fitting off, the α tie must not rewrite the frozen defaults.
    assert result.state.amplitudes[1] == pytest.approx(1.0)
    assert result.state.amplitudes[2] == pytest.approx(1.0)


def test_mode_round_trips_and_change_forces_restart() -> None:
    assert MaxEntConfig().mode == "general"
    assert MaxEntConfig.from_dict({"mode": "zf_lf"}).mode == "zf_lf"
    assert MaxEntConfig.from_dict({"mode": "nonsense"}).mode == "general"

    run = _kubo_toyabe_fb_run()
    base = MaxEntConfig(
        n_spectrum_points=64,
        f_min_mhz=0.0,
        f_max_mhz=3.0,
        auto_window=False,
        selected_group_ids=[1, 2],
    )
    prepared = build_maxent_input(run, base)
    state = initialize_state(prepared, base)
    zf = MaxEntConfig(
        n_spectrum_points=64,
        f_min_mhz=0.0,
        f_max_mhz=3.0,
        auto_window=False,
        selected_group_ids=[1, 2],
        mode="zf_lf",
    )
    with pytest.raises(ValueError, match="restart"):
        run_cycles(build_maxent_input(run, zf), zf, state=state, cycles=1)


def test_specbg_subtracts_central_peak_on_a_copy() -> None:
    frequencies = np.linspace(0.0, 3.0, 256)
    # A strong central peak plus a weak satellite at 1.5 MHz.
    central = np.exp(-((frequencies / 0.2) ** 2))
    satellite = 0.1 * np.exp(-(((frequencies - 1.5) / 0.1) ** 2))
    spectrum = central + satellite

    result = subtract_zero_frequency(
        frequencies,
        spectrum,
        gaussian_width=0.2,
        lorentzian_width=0.2,
        lorentzian_fraction=0.0,
    )
    # The central feature is strongly suppressed; the satellite survives.
    assert abs(result[0]) < 0.2 * spectrum[0]
    satellite_idx = int(np.argmin(np.abs(frequencies - 1.5)))
    assert result[satellite_idx] == pytest.approx(spectrum[satellite_idx], abs=0.02)
    # Display-only: the input array is untouched.
    assert spectrum[0] == pytest.approx(central[0] + satellite[0])


def test_apply_maxent_specbg_skips_when_window_excludes_zero() -> None:
    # SpecBG subtracts a zero-centred peak, so it must be a no-op for an LF-style
    # window that does not reach zero frequency (peak is at the Larmor line).
    import numpy as np

    from asymmetry.core.data.dataset import MuonDataset
    from asymmetry.core.representation.frequency import apply_maxent_specbg

    freqs = np.linspace(20.0, 24.0, 128)  # window far from zero
    values = 1.0 + 0.1 * np.exp(-(((freqs - 22.0) / 0.2) ** 2))
    dataset = MuonDataset(
        time=freqs, asymmetry=values.copy(), error=np.zeros_like(freqs), metadata={}
    )
    config = MaxEntConfig(
        mode="zf_lf",
        specbg_enabled=True,
        specbg_gaussian_width_mhz=0.2,
        specbg_lorentzian_width_mhz=0.2,
    )
    result = apply_maxent_specbg(dataset, config)
    np.testing.assert_allclose(result.asymmetry, values)


def test_as_dataset_is_the_single_specbg_application_point() -> None:
    # SpecBG now flows through MaxEntResult.as_dataset(config) — one place both
    # the on-demand representation and the live worker reach.  Passing the config
    # applies the zero-frequency subtraction; omitting it leaves the spectrum raw.
    run = _kubo_toyabe_fb_run()
    config = MaxEntConfig(
        n_spectrum_points=128,
        f_min_mhz=0.0,
        f_max_mhz=3.0,
        auto_window=False,
        outer_cycles=4,
        inner_iterations=4,
        mode="zf_lf",
        selected_group_ids=[1, 2],
        specbg_enabled=True,
        specbg_gaussian_width_mhz=0.2,
        specbg_lorentzian_width_mhz=0.2,
    )
    result = run_cycles(build_maxent_input(run, config), config)

    raw = result.as_dataset(run)
    subtracted = result.as_dataset(run, config)
    # The zero-frequency central peak is suppressed only on the config path.
    assert abs(subtracted.asymmetry[0]) < abs(raw.asymmetry[0])
    # as_dataset(config) and the re-exported helper agree (one implementation).
    from asymmetry.core.representation.frequency import apply_maxent_specbg

    direct = apply_maxent_specbg(result.as_dataset(run), config)
    np.testing.assert_allclose(subtracted.asymmetry, direct.asymmetry)


def test_spectrum_and_log_export_are_well_formed() -> None:
    run = _kubo_toyabe_fb_run()
    config = MaxEntConfig(
        n_spectrum_points=64,
        f_min_mhz=0.0,
        f_max_mhz=3.0,
        auto_window=False,
        outer_cycles=3,
        inner_iterations=3,
        selected_group_ids=[1, 2],
    )
    result = run_cycles(build_maxent_input(run, config), config)

    spectrum_text = spectrum_to_text(result, config)
    assert spectrum_text.startswith("# MaxEnt spectrum")
    assert "frequency_MHz" in spectrum_text
    data_lines = [ln for ln in spectrum_text.splitlines() if not ln.startswith("#")]
    assert len(data_lines) == result.frequencies_mhz.size
    assert len(data_lines[0].split("\t")) == 3

    log_text = run_log_text(result, config)
    assert "per-cycle convergence" in log_text
    assert "final group phases" in log_text


# ── deadtime fit: recover a known injected non-paralysable deadtime ──────────


def _decay_histogram(
    *,
    amplitude: float,
    bin_width: float,
    n_bins: int,
    tau_inj_us: float,
    num_good_frames: float,
) -> Histogram:
    """A muon-lifetime decay thinned by a known non-paralysable deadtime.

    The fit (``calibrate_deadtime_from_histograms``) reads bin ``k`` at time
    ``(k+1)·bin_width`` from ``t0``, so the synthetic counts use that same
    convention.  Each bin's true count ``N`` is reduced to the observed count by
    the first-order non-paralysable loss ``N → N(1 − Nτ/T)`` with
    ``T = num_good_frames·bin_width`` — exactly the rate-squared distortion the
    ``countfit`` model inverts.  ``tau_inj_us = 0`` gives a clean (un-thinned)
    histogram.
    """
    k = np.arange(n_bins, dtype=np.float64)
    times_us = (k + 1.0) * bin_width
    true_counts = amplitude * np.exp(-times_us / MUON_LIFETIME_US)
    frame_window = float(num_good_frames) * float(bin_width)
    observed = true_counts * (1.0 - true_counts * float(tau_inj_us) / frame_window)
    return Histogram(counts=np.clip(observed, 1.0, None), bin_width=bin_width, t0_bin=0)


def test_fit_deadtime_recovers_known_injected_value() -> None:
    # 5% loss at t0 (amplitude·τ/(frames·bin_width) = 0.05) — a realistic ISIS
    # deadtime distortion.  The countfit model carries the full
    # τ_µ(1−e^{−τ/τ_µ}) loss factor, so the recovered τ maps to the injected
    # one as τ_fit = −τ_µ·ln(1 − τ_inj/τ_µ); both agree to ~1% here.
    bin_width = 0.016
    frames = 2.0e5
    amplitude = 3200.0
    tau_inj = 0.05  # µs (50 ns)
    histograms = [
        _decay_histogram(
            amplitude=amplitude,
            bin_width=bin_width,
            n_bins=400,
            tau_inj_us=tau_inj,
            num_good_frames=frames,
        )
        for _ in range(2)
    ]
    recovered = calibrate_deadtime_from_histograms(histograms, num_good_frames=frames)
    assert recovered is not None
    assert len(recovered) == 2
    for tau_fit in recovered:
        assert tau_fit == pytest.approx(tau_inj, rel=0.05)


def test_fit_deadtime_returns_near_zero_for_a_clean_run() -> None:
    bin_width = 0.016
    frames = 2.0e5
    histograms = [
        _decay_histogram(
            amplitude=3200.0,
            bin_width=bin_width,
            n_bins=400,
            tau_inj_us=0.0,
            num_good_frames=frames,
        )
    ]
    recovered = calibrate_deadtime_from_histograms(histograms, num_good_frames=frames)
    assert recovered is not None
    # No injected loss → the fitted deadtime collapses to ≈0 (a few ps at most).
    assert recovered[0] == pytest.approx(0.0, abs=1.0e-3)
