"""Automatic phase suitability on external oscillatory and relaxing runs."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from asymmetry.core.fourier.grouped import build_group_signal_dataset
from asymmetry.core.fourier.phase_fit import PhaseFitRejectedError
from asymmetry.core.fourier.spectrum import GroupSpectrumConfig, compute_average_group_spectrum
from asymmetry.core.fourier.units import gauss_to_mhz
from asymmetry.core.io import load
from asymmetry.core.transform.grouping import group_forward_backward
from asymmetry.core.workflow.reduction import ReductionSettings, resolve_reduction_grouping
from tests.corpus import corpus_path, requires_corpus

_AFM = corpus_path("Magnetism/AFM transition in high TF/data")
_GLASS = corpus_path("Magnetism/Spin Glass YMnAl/data/MUSR00024591.nxs")
pytestmark = [pytest.mark.integration, pytest.mark.slow]


@requires_corpus(_AFM)
@pytest.mark.parametrize("run_number", [*range(686, 694), *range(730, 740)])
def test_high_field_afm_keeps_an_accepted_line_on_the_recorded_field(run_number):
    assert _AFM is not None
    run = load(str(_AFM / f"tdc_hifi_2020_{run_number:05d}.mdu")).run
    run = replace(
        run, grouping=resolve_reduction_grouping(run, ReductionSettings(background="range"))
    )
    spectra = []
    for padding in (1, 4):
        result = compute_average_group_spectrum(
            run,
            GroupSpectrumConfig(
                display="phaseOptReal",
                selected_group_ids=[1, 5],
                t_max_us=6.0,
                window="lorentzian",
                padding=padding,
            ),
        )
        assert result is not None
        peak = np.argmax(result.asymmetry)
        # The integer-tesla header is rounded; allow the local field distribution.
        assert abs(result.time[peak] - gauss_to_mhz(run.field)) < 2.0
        assert result.asymmetry[peak] > 5 * np.median(np.abs(result.asymmetry))
        report = result.metadata["fourier_phase_fit"]
        assert report["method"] == "sampled_common_phase_v1"
        assert set(report["groups"]) == {"1", "5"}
        assert all(group["status"] == "accepted" for group in report["groups"].values())
        spectra.append(report)
    for group_id in ("1", "5"):
        first, second = (report["groups"][group_id] for report in spectra)
        assert first["phase_radians"] == pytest.approx(second["phase_radians"], abs=1e-7)
        assert first["standard_error_radians"] == pytest.approx(
            second["standard_error_radians"], abs=1e-7
        )
        assert first["acquisition"] == second["acquisition"]


@requires_corpus(_AFM)
@pytest.mark.parametrize("run_number", [686, 693, 730, 739])
@pytest.mark.parametrize("pair", [("1", "5"), ("2", "6"), ("3", "7"), ("4", "8")])
@pytest.mark.parametrize("t0_offset", [0, 3])
def test_high_field_afm_grouped_fft_preserves_reduction_window(run_number, pair, t0_offset):
    assert _AFM is not None
    run = load(str(_AFM / f"tdc_hifi_2020_{run_number:05d}.mdu")).run
    run = replace(
        run,
        grouping=resolve_reduction_grouping(
            run, ReductionSettings(pair=pair, t0_offset_bins=t0_offset)
        ),
    )
    reduced = group_forward_backward(run.histograms, run.grouping)
    first, last = (run.grouping[key] for key in ("first_good_bin", "last_good_bin"))
    expected_time = (np.arange(first, last + 1) - reduced.common_t0) * run.histograms[0].bin_width
    for gid, counts in zip(pair, (reduced.forward, reduced.backward)):
        signal = build_group_signal_dataset(
            run, int(gid), center_signal=False, apply_lifetime_correction=False
        )
        np.testing.assert_array_equal(signal.asymmetry, counts[first : last + 1])
        np.testing.assert_allclose(signal.time, expected_time, atol=1e-14)


@requires_corpus(_GLASS)
def test_unresolved_relaxation_is_not_presented_as_an_accepted_phase():
    assert _GLASS is not None
    run = load(str(_GLASS)).run
    with pytest.raises(PhaseFitRejectedError) as error:
        compute_average_group_spectrum(
            run,
            GroupSpectrumConfig(
                display="phaseOptReal", selected_group_ids=[1], t_max_us=6.0, window="lorentzian"
            ),
        )
    assert error.value.report.to_dict()["status"] == "rejected"
    assert error.value.report.reason
