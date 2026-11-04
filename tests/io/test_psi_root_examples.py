"""Optional acquired ROOT interval comparisons against musrfit 1.12.1 examples."""

import hashlib
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from asymmetry.core.io import load
from asymmetry.core.project.profiles import (
    ProfileFingerprint,
    profile_from_payload,
    resolve_effective_grouping,
)
from asymmetry.core.simulate import reduce_run_to_dataset
from tests.io import test_psi_good_intervals

pytestmark = [pytest.mark.io, pytest.mark.integration]
musrfit_export = test_psi_good_intervals.musrfit_export

# External inputs from musrfit revision 708f777b; no acquisition is vendored.
EXAMPLES = {
    "lem": (
        "lem12_his_2466.root",
        "58a799e3b0bce6e3e7aa29da73fc233e1dded186e3d9979c8aab9cd087388913",
    ),
    "flame": (
        "../../tutorials/data/deltat_tdc_flame_2022_0137.root",
        "e41725a36daad83b2d84de52551f0311e032ef19d408e59346885f5a91401379",
    ),
    "hifi": (
        "../../tutorials/data/deltat_tdc_hifi_2025_0002.root",
        "17a5da47172ea76fb8adc1768d7b44e712e956c558e8ff705497730733c56617",
    ),
}


@pytest.mark.parametrize(
    "example,pair,good,omitted",
    [
        pytest.param(
            "lem",
            (1, 3),
            ((2741, 66600), (2741, 66600)),
            ((2792, 66600), (2792, 66600)),
            id="lem-terminal",
        ),
        pytest.param(
            "flame",
            (2, 1),
            ((1615, 103310), (1625, 103320)),
            ((1697, 103413), (1707, 103423)),
            id="flame-interior",
        ),
        pytest.param(
            "hifi",
            (1, 2),
            ((20297, 409190), (20267, 409160)),
            ((20506, 409600), (20476, 409570)),
            id="hifi-default-pair",
        ),
        pytest.param(
            "hifi",
            (2, 3),
            ((20267, 409190), (20252, 409175)),
            ((20476, 409600), (20461, 409585)),
            id="hifi-populated-pair",
        ),
    ],
)
@pytest.mark.parametrize("mode", ["matched-counts", "matched-fb", "omitted-fb"])
def test_external_root_selected_observations(musrfit_export, example, pair, good, omitted, mode):
    """Compare actual raw identities, then counts/ratios and applicable errors."""
    locator = os.environ.get("ASYMMETRY_MUSRFIT_DATA")
    if locator is None:
        pytest.skip("ASYMMETRY_MUSRFIT_DATA not set; external musrfit examples unavailable")
    relative, digest = EXAMPLES[example]
    path = (Path(locator) / relative).resolve()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    dataset = load(path)
    run = dataset.run
    histograms = [run.histograms[i - 1] for i in pair]
    profile = profile_from_payload(
        run.grouping,
        "Compared pair",
        ProfileFingerprint("", len(run.histograms)),
    )
    profile = replace(
        profile, groups={1: [pair[0]], 2: [pair[1]]}, forward_group=1, backward_group=2
    )
    grouping = resolve_effective_grouping(profile, run)
    shift = max(h.t0_bin for h in histograms) - histograms[0].t0_bin
    assert (grouping["first_good_bin"], grouping["last_good_bin"]) == (
        good[0][0] + shift,
        good[0][1] + shift,
    )
    reduced = reduce_run_to_dataset(replace(run, grouping=grouping))
    bounds = omitted if mode == "omitted-fb" else good
    raw = [np.arange(first, last + 1) for first, last in bounds]
    f, b = [h.counts[indices] for h, indices in zip(histograms, raw, strict=True)]
    assert len(f) == len(b)
    numbers = [str(run.grouping["root_histo_numbers"][i - 1]) for i in pair]
    asymmetry = mode != "matched-counts"
    data = (
        ""
        if mode == "omitted-fb"
        else "data "
        + " ".join(
            str(value)
            for first, last in (good if asymmetry else good[:1])
            for value in (first, last + 1)
        )
    )
    exported = musrfit_export(
        path,
        "MUSR-ROOT",
        data,
        forward=numbers[0] if asymmetry else " ".join(numbers),
        backward=numbers[1],
        asymmetry=asymmetry,
    )
    assert exported.shape == (len(f), 4)
    width, anchor = histograms[0].bin_width, histograms[0].t0_bin
    np.testing.assert_array_equal(np.rint(exported[:, 0] / width + anchor), raw[0])
    np.testing.assert_array_equal(raw[1] - raw[0], histograms[1].t0_bin - anchor)
    np.testing.assert_allclose(exported[:, 0], (raw[0] - anchor) * width, rtol=5e-6, atol=1e-15)
    if not asymmetry:
        np.testing.assert_array_equal(exported[:, 1], f + b)
        np.testing.assert_allclose(exported[:, 2], np.sqrt(np.maximum(f + b, 1)), rtol=5e-6)
    else:
        nonempty = f + b > 0
        np.testing.assert_allclose(
            exported[nonempty, 1], ((f - b) / np.maximum(f + b, 1))[nonempty], rtol=5e-6, atol=5e-7
        )
        positive = (f > 0) & (b > 0)
        if np.any(positive):
            np.testing.assert_allclose(
                exported[positive, 2],
                2 * np.sqrt(f[positive] * b[positive] / (f[positive] + b[positive]) ** 3),
                rtol=5e-6,
            )
        if mode == "matched-fb":
            np.testing.assert_allclose(
                exported[nonempty, 1], reduced.asymmetry[nonempty] / 100, rtol=5e-6, atol=5e-7
            )
            if np.any(positive):
                np.testing.assert_allclose(
                    exported[positive, 2], reduced.error[positive] / 100, rtol=5e-6
                )
            np.testing.assert_allclose(exported[:, 0], reduced.time, rtol=5e-6, atol=1e-15)
