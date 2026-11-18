"""Inconclusive recognition preserves authored project grouping on reopen."""

import copy

import pytest

from asymmetry.core.project.profiles import reconcile_instrument_for_payload
from asymmetry.core.project.schema import CURRENT_SCHEMA_VERSION, load_project, save_project
from tests.core.test_grouping_profiles import _run


@pytest.mark.parametrize("n_hist", [64, 96])
def test_reconcile_unknown_count_preserves_authored_musr_grouping(n_hist, tmp_path):
    run = _run(instrument="Unknown", n_hist=n_hist, metadata={"instrument": "Unknown"})
    run.source_file = "run000123.nxs"
    payload = {
        "instrument": "MuSR",
        "groups": {"1": [1, 4, 9], "2": [2, 7]},
        "group_names": {"1": "Sample forward", "2": "Sample backward"},
        "forward_group": 2,
        "backward_group": 1,
        "alpha": 1.23,
        "grouping_preset": "Custom",
    }
    snapshot = copy.deepcopy(payload)
    path = tmp_path / "unknown.asymp"
    save_project(
        {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "datasets": [
                {"run_number": 1, "source_file": run.source_file, "grouping_overrides": payload}
            ],
        },
        path,
    )
    payload = load_project(path)["datasets"][0]["grouping_overrides"]
    run_grouping = copy.deepcopy(run.grouping)
    reconciled, note = reconcile_instrument_for_payload(run, payload)
    assert note is None
    assert reconciled is payload
    assert payload == snapshot
    assert run.grouping == run_grouping
    assert run.metadata["instrument"] == "Unknown"
