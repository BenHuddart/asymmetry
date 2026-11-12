"""Selecting loader-cached periods retains their actual execution attribution."""

from dataclasses import replace

import pytest

from asymmetry.core.workflow.reduction import ReductionSettings, reduce_run


@pytest.mark.parametrize("revision", [None, "historical", "1"])
def test_selecting_cached_periods_preserves_their_captured_execution(revision):
    from asymmetry.core.io.periods import select_period
    from tests.core.test_pooled_periods import two_periods

    data = reduce_run(two_periods(), ReductionSettings())
    data.run.grouping["period_reduced"] = [(data.time, data.asymmetry, data.error)] * 2
    data.histogram_observation = replace(
        data.histogram_observation, cached_computation_revisions=(revision, revision)
    )
    selected = select_period(data, "red")
    assert selected.computation_revision == revision
    data.histogram_observation = replace(
        data.histogram_observation, cached_computation_revisions=()
    )
    assert select_period(data, "red").computation_revision is None
