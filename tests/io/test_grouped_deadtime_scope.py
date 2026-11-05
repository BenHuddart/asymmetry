"""Acquired observation transport and external reference cache scope."""

import hashlib
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from asymmetry.core.fitting.grouped_time_domain import build_count_groups
from asymmetry.core.fourier.grouped import build_group_signal_dataset
from asymmetry.core.fourier.spectrum import (
    GroupSpectrumConfig,
    compute_average_group_spectrum,
    precompute_group_fourier_inputs,
)
from asymmetry.core.io import load
from asymmetry.core.io.nexus_writer import write_nexus_v1
from asymmetry.core.maxent import MaxEntConfig, build_maxent_input
from asymmetry.core.project.profiles import (
    ProfileFingerprint,
    profile_from_payload,
    resolve_effective_grouping,
)
from asymmetry.core.simulate import reduce_run_to_dataset
from tests.core.test_grouped_deadtime_scope import scoped_run
from tests.corpus import corpus_path
from tests.io.test_file_window_timing import CORPUS
from tests.io.test_psi_root_examples import EXAMPLES

pytestmark = pytest.mark.io


@pytest.mark.parametrize("source", [*range(len(CORPUS)), *EXAMPLES])
def test_acquired_grouped_counts_and_signal_use_original_observations(source):
    if isinstance(source, int):
        path = corpus_path(CORPUS[source])
    else:
        locator = os.environ.get("ASYMMETRY_MUSRFIT_DATA")
        path = None if locator is None else Path(locator) / EXAMPLES[source][0]
        if path is not None:
            assert hashlib.sha256(path.read_bytes()).hexdigest() == EXAMPLES[source][1]
    if path is None:
        pytest.skip("optional acquired inputs are not configured")
    dataset = load(path)
    if isinstance(dataset, list):
        dataset = dataset[0]
    run = dataset.run
    profile = profile_from_payload(
        run.grouping, "Selected groups", ProfileFingerprint("", len(run.histograms))
    )
    profile = replace(
        profile, groups={1: [1], 2: [2]}, forward_group=1, backward_group=2, projections=[]
    )
    grouping = resolve_effective_grouping(profile, run)
    width, frames = run.histograms[0].bin_width, grouping["good_frames"]
    # Synthetic coefficients check transport, not physical detector calibration.
    coefficient = 0.01 * width * frames / max(float(np.max(h.counts)) for h in run.histograms[:2])
    run.grouping = dict(
        grouping,
        deadtime_correction=True,
        deadtime_mode="file",
        deadtime_file_us=[coefficient, -coefficient],
        bunching_factor=1,
    )
    dataset = reduce_run_to_dataset(run)
    # Explicit-pair reduction replaces the unavailable default-group diagnostic.
    dataset.metadata.pop("correction_error", None)
    originals = [histogram.counts.copy() for histogram in run.histograms]
    groups = build_count_groups(dataset, [1, 2], lifetime_corrected=False)
    for index, group in enumerate(groups):
        assert group.detector_indices == (index,)
        assert group.deadtime.status == "applied"
        ranges = group.deadtime.effective["scope"]["raw_bin_ranges"][str(index)]
        bins = np.concatenate([np.arange(first, last + 1) for first, last in ranges])
        observed = run.histograms[index].counts[bins]
        tau = coefficient if index == 0 else -coefficient
        np.testing.assert_allclose(
            group.counts, observed / (1 - observed * tau / (width * frames)), rtol=1e-13
        )
        variance = run.histograms[index].count_variance
        np.testing.assert_allclose(
            group.error**2,
            np.maximum(observed if variance is None else variance[bins], 1),
            rtol=1e-13,
        )
        signal = build_group_signal_dataset(
            run, index + 1, center_signal=False, apply_lifetime_correction=False
        )
        np.testing.assert_allclose(signal.asymmetry, group.counts, rtol=1e-13)
        np.testing.assert_allclose(signal.time, group.time, atol=1e-13)
    spectrum = compute_average_group_spectrum(run, GroupSpectrumConfig(selected_group_ids=[1]))
    assert spectrum is not None
    assert spectrum.metadata["deadtime_outcome"]["status"] == "applied"
    maxent = build_maxent_input(
        run,
        MaxEntConfig(
            auto_steer=False,
            auto_phase_seed=False,
            auto_window=False,
            selected_group_ids=[1],
            n_spectrum_points=16,
            f_min_mhz=0.0,
            f_max_mhz=0.5 / width,
            t_min_us=0.0,
            t_max_us=5.0,
        ),
    )
    assert len(maxent.groups) == 1
    assert maxent.metadata["deadtime_outcome"]["status"] == "applied"
    assert np.all(np.isfinite(maxent.groups[0].signal))
    for histogram, original in zip(run.histograms, originals, strict=True):
        np.testing.assert_array_equal(histogram.counts, original)


def test_external_reference_cache_reuses_only_matching_source_and_scope(tmp_path):
    sample = scoped_run("missing")
    reference = scoped_run("missing")
    for histogram in reference.histograms:
        histogram.counts = np.full(64, 2.0)
    reference.grouping["deadtime_file_us"] = [0.1, 0.1, None]
    path = tmp_path / "reference.nxs"
    write_nexus_v1(reference, path, effective=True)
    sample.grouping.update(
        background_correction=True,
        background_mode="reference_run",
        background_run={"source_file": str(path), "run_number": 123, "scale": 1.0},
    )
    cache = {}
    first = build_group_signal_dataset(
        sample,
        1,
        center_signal=False,
        apply_lifetime_correction=False,
        background_reference_cache=cache,
    )
    assert first.metadata["deadtime_outcome"]["status"] == "applied"
    with pytest.warns(UserWarning, match="deadtime"):
        union = compute_average_group_spectrum(
            sample, GroupSpectrumConfig(selected_group_ids=[1, 3])
        )
    assert union.metadata["deadtime_outcome"]["status"] == "skipped"
    prepared, _ = precompute_group_fourier_inputs(sample, [1])
    reference.histograms[0].counts = np.full(64, 3.0)
    write_nexus_v1(reference, path, effective=True)
    with pytest.raises(ValueError, match="requested source"):
        build_group_signal_dataset(sample, 1, prepared=prepared)
    second = build_group_signal_dataset(
        sample,
        1,
        center_signal=False,
        apply_lifetime_correction=False,
        background_reference_cache=cache,
    )
    expected = build_group_signal_dataset(
        sample, 1, center_signal=False, apply_lifetime_correction=False
    )
    np.testing.assert_array_equal(second.asymmetry, expected.asymmetry)
    assert not np.array_equal(first.asymmetry, second.asymmetry)
