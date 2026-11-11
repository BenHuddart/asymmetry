"""Quantity algebra and retained support through the public Fourier operation."""

import json
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.fitting.likelihood import GaussianObservations
from asymmetry.core.fourier.conditioning import (
    apply_spectrum_conditioning,
    pulse_compensation_gain,
    sigma_clip_baseline,
    unit_area_normalise,
)
from asymmetry.core.fourier.spectrum import GroupSpectrumConfig, compute_average_group_spectrum
from asymmetry.core.representation.frequency import FrequencyFFT
from asymmetry.core.utils.constants import MUON_LIFETIME_US, PION_LIFETIME_US
from tests.core.test_fourier_spectrum import _run


@pytest.mark.parametrize(
    "display, exponent",
    [
        ("Magnitude", 1),
        ("(Power)^1/2", 1),
        ("Power", 2),
        ("Cos", 1),
        ("Sin", 1),
        ("Imaginary", 1),
        ("Real", 1),
        ("Phase", 1),
        ("Real+Imag", 1),
    ],
)
def test_public_quantity_and_existing_group_errors(display, exponent):
    run = _run()
    original = [h.counts.copy() for h in run.histograms]
    config = GroupSpectrumConfig(display=display, estimate_average_error=True)
    plain = compute_average_group_spectrum(run, config)
    corrected = compute_average_group_spectrum(
        run,
        replace(
            config,
            pulse_compensation=True,
            pulse_half_width_us=0.05,
        ),
    )
    gain = pulse_compensation_gain(plain.time, half_width_us=0.05)
    np.testing.assert_allclose(corrected.asymmetry, plain.asymmetry * gain**exponent)
    np.testing.assert_allclose(corrected.error, plain.error * gain**exponent)
    singles = [
        compute_average_group_spectrum(run, replace(config, selected_group_ids=[g])) for g in [1, 2]
    ]
    expected = np.mean([s.asymmetry for s in singles], axis=0) * gain**exponent
    np.testing.assert_allclose(corrected.asymmetry, expected)
    if display == "Real+Imag":
        np.testing.assert_allclose(
            corrected.metadata["fourier_imag"], np.array(plain.metadata["fourier_imag"]) * gain
        )
    for histogram, before in zip(run.histograms, original):
        np.testing.assert_array_equal(histogram.counts, before)
    assert corrected.metadata["fourier_pulse_response"]["gain_exponent"] == exponent


def test_multigroup_power_is_mean_of_powers():
    config = GroupSpectrumConfig(pulse_compensation=True, pulse_half_width_us=0.05)
    power = compute_average_group_spectrum(_run(), replace(config, display="Power"))
    magnitude = compute_average_group_spectrum(_run(), replace(config, display="Magnitude"))
    assert not np.allclose(power.asymmetry, magnitude.asymmetry**2)


@pytest.mark.parametrize("display", ["Phase Spectrum", "Burg", "Correlation"])
def test_non_amplitude_and_derived_modes_do_not_consume_response(display):
    config = GroupSpectrumConfig(display=display)
    plain = compute_average_group_spectrum(_run(), config)
    corrected = compute_average_group_spectrum(
        _run(),
        replace(
            config,
            pulse_compensation=True,
            pulse_half_width_us=float("nan"),
        ),
    )
    np.testing.assert_array_equal(corrected.asymmetry, plain.asymmetry)
    assert "fourier_pulse_response" not in corrected.metadata


@pytest.mark.parametrize("padding", [1, 3])
@pytest.mark.parametrize("baseline_mode", ["none", "sigma_clip", "wimda"])
def test_cutoff_is_removed_before_normalisation_and_consumers(padding, baseline_mode):
    config = GroupSpectrumConfig(
        display="Power",
        padding=padding,
        estimate_average_error=True,
        pulse_compensation=True,
        pulse_half_width_us=0.2,
        pulse_max_gain=3,
        baseline_mode=baseline_mode,
        exclude_enabled=True,
        exclusion_ranges=[(0.5, 0.01), (8, 2)],
    )
    plain = compute_average_group_spectrum(
        _run(),
        replace(
            config,
            pulse_compensation=False,
            baseline_mode="none",
            exclude_enabled=False,
        ),
    )
    gain = pulse_compensation_gain(plain.time, half_width_us=0.2, max_gain=3)
    keep = gain > 0
    expected = plain.asymmetry[keep] * gain[keep] ** 2
    if baseline_mode != "none":
        baseline, _ = sigma_clip_baseline(
            expected,
            max_iter=1 if baseline_mode == "wimda" else 10,
            location="mean" if baseline_mode == "wimda" else "median",
        )
        expected -= baseline
    corrected = compute_average_group_spectrum(_run(), config)
    np.testing.assert_array_equal(corrected.time, plain.time[keep])
    np.testing.assert_allclose(corrected.asymmetry, expected)
    assert corrected.metadata["fourier_compensation_cutoff_mhz"] == plain.time[~keep][0]
    observations = GaussianObservations(corrected.time, corrected.asymmetry, corrected.error)
    assert len(observations.time) == keep.sum()
    density = compute_average_group_spectrum(
        _run(), replace(config, display_normalisation="unit_area")
    )
    normalised = unit_area_normalise(corrected.time, corrected.asymmetry, corrected.error)
    np.testing.assert_allclose(density.asymmetry, normalised.display)
    np.testing.assert_allclose(density.error, normalised.error)


@pytest.mark.parametrize("step", [0.1, 0.37])
def test_sampled_amplitude_limit_and_removed_helper_bins(step):
    freqs = np.arange(0, 40, step)
    # Independent integral of the normalised parabolic proton distribution.
    response = np.array(
        [
            quad(
                lambda t: 3 / (4 * 0.07) * (1 - (t / 0.07) ** 2) * np.cos(2 * np.pi * f * t),
                -0.07,
                0.07,
            )[0]
            / abs(1 + 1j * 2 * np.pi * f * PION_LIFETIME_US)
            for f in freqs
        ]
    )
    cut = np.flatnonzero(np.abs(response) < 1 / 25)[0]
    gain = pulse_compensation_gain(freqs, half_width_us=0.07)
    np.testing.assert_allclose(gain[:cut], 1 / np.abs(response[:cut]), rtol=1e-12)
    assert np.all(gain[cut:] == 0)
    result = apply_spectrum_conditioning(
        freqs,
        np.ones_like(freqs),
        np.zeros_like(freqs),
        quantity="power",
        pulse_compensation=True,
        pulse_half_width_us=0.07,
        baseline_mode="sigma_clip",
        exclusion_ranges=[(30, 10)],
    )
    assert np.isnan(result.display[cut:]).all()
    assert np.isnan(result.error[cut:]).all()
    assert np.all(result.error[:cut] == 0)
    assert np.max(gain**2) <= 625


def test_config_replay_preserves_quantity_and_consumed_provenance():
    config = GroupSpectrumConfig(
        display="Power",
        pulse_compensation=True,
        pulse_half_width_us=0.05,
        pulse_n_pulses=2,
        pulse_separation_us=0.324,
    )
    restored = GroupSpectrumConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    direct = compute_average_group_spectrum(_run(), config)
    rep = FrequencyFFT(recipe={"fourier_config": restored.to_dict()})
    replay = rep.compute(_run())[0]
    np.testing.assert_array_equal(replay.asymmetry, direct.asymmetry)
    np.testing.assert_array_equal(replay.error, direct.error)
    provenance = replay.metadata["fourier_pulse_response"]
    assert provenance["quantity"] == "power"
    assert provenance["amplitude_max_gain"] == 25
    assert provenance["shape"] == "parabolic_plus_pion"
    assert provenance["complex_sign"] == "P_cos-iP_sin"
    assert provenance["width_source"] == "explicit"


@pytest.mark.parametrize("n_pulses,separation", [(1, 0.0), (2, 0.0), (2, 0.324)])
def test_bin_centred_sinusoid_transfer_from_independent_arrival_integrals(n_pulses, separation):
    frequency, width = 1.25, 0.05
    omega = 2 * np.pi * frequency
    proton = quad(
        lambda t: 3 / (4 * width) * (1 - (t / width) ** 2) * np.cos(omega * t), -width, width
    )[0]
    pion = complex(
        quad(
            lambda t: np.exp(-t / PION_LIFETIME_US) / PION_LIFETIME_US * np.cos(omega * t),
            0,
            np.inf,
        )[0],
        -quad(
            lambda t: np.exp(-t / PION_LIFETIME_US) / PION_LIFETIME_US * np.sin(omega * t),
            0,
            np.inf,
        )[0],
    )
    births = np.array([0.0]) if n_pulses == 1 else np.array([-separation / 2, separation / 2])
    weights = np.exp(births / MUON_LIFETIME_US)
    response = proton * pion * np.sum(weights * np.exp(-1j * omega * births)) / weights.sum()
    time = np.arange(128) * 0.05
    datasets = []
    for transfer in [1.0, response]:
        counts = (
            10000
            * np.exp(-time / MUON_LIFETIME_US)
            * (1 + 0.2 * np.real(transfer * np.exp(1j * omega * time)))
        )
        run = Run(
            histograms=[
                Histogram(
                    counts, 0.05, t0_bin=0, count_variance=np.exp(-2 * time / MUON_LIFETIME_US)
                )
            ],
            grouping={"groups": {1: [1]}, "first_good_bin": 0, "last_good_bin": 127},
        )
        datasets.append(run)
    config = GroupSpectrumConfig(display="Magnitude")
    original = compute_average_group_spectrum(datasets[0], config)
    corrected = compute_average_group_spectrum(
        datasets[1],
        replace(
            config,
            pulse_compensation=True,
            pulse_half_width_us=width,
            pulse_n_pulses=n_pulses,
            pulse_separation_us=separation,
        ),
    )
    peak = np.flatnonzero(np.isclose(original.time, frequency))[0]
    assert corrected.asymmetry[peak] == pytest.approx(original.asymmetry[peak], rel=1e-11)
    assert corrected.error[peak] == 0


def test_companion_cutoff_and_baseline_share_retained_support():
    config = GroupSpectrumConfig(
        display="Real+Imag",
        pulse_compensation=True,
        pulse_half_width_us=0.2,
        pulse_max_gain=3,
        baseline_mode="sigma_clip",
    )
    plain = compute_average_group_spectrum(
        _run(), replace(config, pulse_compensation=False, baseline_mode="none")
    )
    corrected = compute_average_group_spectrum(_run(), config)
    gain = pulse_compensation_gain(plain.time, half_width_us=0.2, max_gain=3)
    keep = gain > 0
    expected = np.array(plain.metadata["fourier_imag"])[keep] * gain[keep]
    expected -= corrected.metadata["fourier_baseline"]
    np.testing.assert_allclose(corrected.metadata["fourier_imag"], expected)
    assert len(expected) == len(corrected.time)


@pytest.mark.parametrize("limit", [0.25, float("nan")])
def test_provenance_records_effective_amplitude_limit(limit):
    config = GroupSpectrumConfig(
        display="Power", pulse_compensation=True, pulse_half_width_us=0.05, pulse_max_gain=limit
    )
    spectrum = compute_average_group_spectrum(_run(), config)
    assert spectrum.metadata["fourier_pulse_response"]["amplitude_max_gain"] == 1.0
    assert spectrum.time.tolist() == [0.0]


@pytest.mark.parametrize("quantity", ["Power", "unknown"])
def test_conditioning_rejects_unknown_authored_quantity(quantity):
    with pytest.raises(ValueError, match="Spectrum quantity"):
        apply_spectrum_conditioning(
            [0, 1], [1, 1], quantity=quantity, pulse_compensation=True, pulse_half_width_us=0.05
        )


def test_ignore_does_not_consume_dormant_gain_limit():
    config = GroupSpectrumConfig(display="Power")
    plain = compute_average_group_spectrum(_run(), config)
    ignored = compute_average_group_spectrum(
        _run(),
        replace(
            config,
            pulse_compensation=True,
            pulse_n_pulses=0,
            pulse_half_width_us=float("nan"),
            pulse_separation_us=-1,
            pulse_max_gain=None,
        ),
    )
    np.testing.assert_array_equal(ignored.asymmetry, plain.asymmetry)
    np.testing.assert_array_equal(ignored.error, plain.error)
    assert ignored.metadata["fourier_pulse_response"]["mode"] == "ignore"
    assert ignored.metadata["fourier_pulse_response"]["amplitude_max_gain"] is None
