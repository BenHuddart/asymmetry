"""Optional acquired selections retain raw identities through common timing edits."""

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from asymmetry.core.io import load
from asymmetry.core.project.profiles import (
    AlphaPolicy,
    BackgroundPolicy,
    DeadtimePolicy,
    GoodWindowPolicy,
    T0Policy,
    profile_fingerprint_for_run,
    profile_from_payload,
    resolve_effective_grouping,
)
from asymmetry.core.simulate import reduce_run_to_dataset
from asymmetry.core.transform.grouping import group_forward_backward
from tests.corpus import corpus_path
from tests.io.test_psi_root_examples import EXAMPLES

pytestmark = [pytest.mark.io, pytest.mark.integration]

CORPUS = [
    "Magnetism/Ferromagnetic nickel/Data/emu00124218.nxs",
    "Semiconductors/Photo-muSR in silicon/Data/HIFI00103277.nxs",
    "Magnetism/Magnetic ordering in EuO/data/deltat_pta_gps_2923.bin",
    "Magnetism/AFM transition in high TF/data/tdc_hifi_2020_00739.mdu",
]


@pytest.mark.parametrize("source", [*range(len(CORPUS)), *EXAMPLES])
def test_acquired_raw_membership_with_file_and_manual_timing(source):
    if isinstance(source, int):
        path = corpus_path(CORPUS[source])
    else:
        locator = os.environ.get("ASYMMETRY_MUSRFIT_DATA")
        path = None if locator is None else Path(locator) / EXAMPLES[source][0]
    if path is None:
        pytest.skip("optional external inputs are not configured")
    dataset = load(path)
    if isinstance(dataset, list):
        dataset = dataset[0]
    run = dataset.run
    profile = profile_from_payload(run.grouping, "selected", profile_fingerprint_for_run(run))
    profile = replace(
        profile,
        groups={1: [1], 2: [2]},
        forward_group=1,
        backward_group=2,
        projections=[],
        alpha_policy=AlphaPolicy(),
        deadtime_policy=DeadtimePolicy(),
        background_policy=BackgroundPolicy(),
    )
    selected = []
    results = []
    originals = [h.counts.copy() for h in run.histograms]
    for offset in (0, 5):
        grouping = resolve_effective_grouping(
            replace(profile, t0_policy=T0Policy(mode="manual", offset_bins=offset)), run
        )
        counts = group_forward_backward(run.histograms, grouping).counts
        selected.append(
            tuple(group.counts[counts.window.slice] for group in (counts.forward, counts.backward))
        )
        results.append(reduce_run_to_dataset(replace(run, grouping=grouping)))
    for before, after in zip(*selected):
        np.testing.assert_array_equal(before, after)
    np.testing.assert_allclose(
        results[1].time, results[0].time - 5 * run.histograms[0].bin_width, rtol=1e-13, atol=1e-13
    )
    np.testing.assert_array_equal(results[1].asymmetry, results[0].asymmetry)
    np.testing.assert_array_equal(results[1].error, results[0].error)
    first = counts.alignment.first_bin + 5
    last = min(first + 20, counts.alignment.first_bin + counts.alignment.length - 1)
    t0 = grouping["t0_bin"]
    manual = replace(
        profile,
        t0_policy=T0Policy(mode="manual", offset_bins=5),
        good_window_policy=GoodWindowPolicy(
            mode="manual", first_offset_bins=first - t0, last_offset_bins=last - t0
        ),
    )
    resolved = resolve_effective_grouping(manual, run)
    observed = group_forward_backward(run.histograms, resolved).counts
    for group, gid in zip(
        (observed.forward, observed.backward),
        (resolved["forward_group"], resolved["backward_group"]),
    ):
        offsets = dict(observed.alignment.detector_offsets)
        raw = np.arange(first, last + 1) - observed.alignment.first_bin
        from asymmetry.core.transform.grouping import effective_group_indices

        oracle = sum(
            run.histograms[i].counts[raw - offsets[i]]
            for i in effective_group_indices(resolved, gid, n_histograms=len(run.histograms))
        )
        np.testing.assert_array_equal(group.counts[observed.window.slice], oracle)
    for histogram, original in zip(run.histograms, originals):
        np.testing.assert_array_equal(histogram.counts, original)
