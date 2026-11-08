"""Count-fit observations stay raw even when a usable reference is available."""

from copy import deepcopy

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.fitting.grouped_time_domain import (
    build_count_group,
    build_count_groups,
    build_grouped_time_domain_groups,
)
from asymmetry.core.fitting.likelihood import CountSource
from asymmetry.core.io import resolve_background_reference
from asymmetry.core.transform.corrections import (
    CorrectionUnavailableError,
    reference_observation_metadata,
)
from asymmetry.core.transform.reduce import corrected_grouped_counts


@pytest.fixture
def reference_pair():
    grouping = {
        "groups": {1: [1], 2: [2]},
        "forward_group": 1,
        "backward_group": 2,
        "background_correction": True,
        "background_mode": "reference_run",
        "background_run": {
            "run_number": 2,
            "scale_request": {"mode": "manual", "coefficient": 0.5},
        },
    }
    sample = Run(
        run_number=1,
        histograms=[Histogram(np.full(32, n), 0.016) for n in (200.0, 150.0)],
        grouping=grouping,
    )
    reference_run = Run(
        run_number=2,
        histograms=[Histogram(np.full(32, n), 0.016) for n in (30.0, 20.0)],
    )
    reference_dataset = MuonDataset([], [], [], run=reference_run)
    reference = resolve_background_reference(
        grouping["background_run"], datasets=[reference_dataset], allow_file=False
    )
    return MuonDataset([], [], [], run=sample), reference


def _count_inputs(dataset, route, *, lifetime_corrected=False):
    if route == "single":
        return [build_count_group(dataset, 1, lifetime_corrected=lifetime_corrected)]
    if route == "fb":
        return build_count_groups(dataset, [1, 2], lifetime_corrected=lifetime_corrected)
    return build_grouped_time_domain_groups(dataset, lifetime_corrected=lifetime_corrected)


@pytest.mark.parametrize("route", ["single", "fb", "grouped"])
@pytest.mark.parametrize("lifetime_corrected", [False, True])
def test_resolved_reference_is_unsupported_for_count_fit_inputs(
    reference_pair, route, lifetime_corrected
):
    dataset, reference = reference_pair
    run = dataset.run
    reduced = corrected_grouped_counts(
        histograms=run.histograms,
        grouping=run.grouping,
        forward_idx=[0],
        backward_idx=[1],
        use_deadtime=False,
        deadtime_mode="off",
        use_background=True,
        metadata=reference_observation_metadata(run),
        reference_resolver=lambda _: reference,
    )
    for group, count, variance in (
        (reduced.forward, 185.0, 207.5),
        (reduced.backward, 140.0, 155.0),
    ):
        np.testing.assert_array_equal(group.counts, count)
        np.testing.assert_array_equal(group.variance, variance)
    before = deepcopy(run.grouping)
    with pytest.raises(
        CorrectionUnavailableError, match="reference-run subtraction is unsupported"
    ) as error:
        _count_inputs(dataset, route, lifetime_corrected=lifetime_corrected)
    assert "Disable background correction or choose a non-reference background mode" in str(
        error.value
    )
    assert "use an asymmetry fit for reference-subtracted data" in str(error.value)
    assert run.grouping == before


@pytest.mark.parametrize("route", ["single", "fb", "grouped"])
@pytest.mark.parametrize(
    "mode,enabled", [("reference_run", False), ("none", True), ("fixed", True), ("range", True)]
)
def test_count_controls_preserve_full_observations(reference_pair, route, mode, enabled):
    dataset, _ = reference_pair
    dataset.run.grouping.update(
        background_mode=mode,
        background_correction=enabled,
        background_fixed_values=[5.0, 7.0],
        background_range=[0, 3],
    )
    groups = _count_inputs(dataset, route)
    assert len(groups) == (1 if route == "single" else 2)
    for group, expected in zip(groups, (200.0, 150.0), strict=False):
        assert group.count_source is CountSource.RAW
        np.testing.assert_array_equal(group.counts, expected)
        np.testing.assert_allclose(group.error**2, expected, rtol=1e-14)
