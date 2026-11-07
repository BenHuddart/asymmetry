"""File-origin exposure is distinct from DT values and from scalar broadcasts."""

import numpy as np
import pytest

from asymmetry.core.io.frame_exposure import nexus_period_exposures, read_nexus_frames
from asymmetry.core.transform.reference_scale import ReferenceExposure


@pytest.mark.parametrize(
    "origin,values,periods,expected",
    [
        ("good_frames", [1], 1, [1]),
        ("goodfrm", [200, 100], 2, [200, 100]),
        ("periods/good_frames", [200, 100], 2, [200, 100]),
        ("instrument/beam/frames_period_daq", [14055, 14053], 2, [14055, 14053]),
        ("good_frames", [200], 2, [None, None]),
        ("instrument/beam/frames_good", [200], 2, [None, None]),
        ("instrument/beam/frames_period", [0, 100, 0, 200], 2, [None, None]),
        ("instrument/beam/frames", [200], 1, [None]),
        ("good_frames", [float("nan")], 1, [None]),
        ("good_frames", [-1], 1, [None]),
        ("good_frames", [], 1, [None]),
    ],
)
def test_only_applicable_measured_frames_are_subtraction_exposure(
    origin, values, periods, expected
):
    actual = nexus_period_exposures(np.asarray(values), origin, periods)
    assert [item["value"] for item in actual] == expected
    assert [item["period_number"] for item in actual] == list(range(1, periods + 1))


@pytest.mark.parametrize("legacy", [True, False])
def test_file_parser_retains_dt_precedence_but_unusable_exposure_does_not_block_loading(legacy):
    entry = {"good_frames": [200], "goodfrm": [300], "periods": {"good_frames": [400]}}

    def read(node, key, default=None):
        return node.get(key, default) if node is not None else default

    facts = read_nexus_frames(entry, read, legacy=legacy)
    assert facts.values.tolist() == [200]
    entry["good_frames"] = ["unusable"]
    bad = read_nexus_frames(entry, read, legacy=legacy)
    assert bad.values.size == 0
    assert bad.exposures(1)[0]["value"] is None


def test_untagged_dt_number_is_unknown_even_when_equal_to_one():
    assert ReferenceExposure.from_grouping({"good_frames": 1}).value is None


@pytest.mark.parametrize(
    "exposure",
    [
        ReferenceExposure(origin="whole_run_deadtime"),
        ReferenceExposure(1, "user"),
        ReferenceExposure(200, "user"),
    ],
)
@pytest.mark.io
def test_nexus_export_preserves_exposure_separately_from_dt_normalisation(tmp_path, exposure):
    from asymmetry.core.io import load
    from asymmetry.core.io.nexus_writer import write_nexus_v1
    from tests.io.test_nexus_writer import _template

    run = _template()
    run.grouping.update(good_frames=1, reference_exposure=exposure.to_dict())
    path = tmp_path / "reference.nxs"
    write_nexus_v1(run, path)
    restored = load(str(path)).run
    assert restored.grouping["good_frames"] == 1
    assert ReferenceExposure.from_grouping(restored.grouping) == exposure


def test_unrepresentable_exposure_declaration_is_unknown():
    exposure = ReferenceExposure.from_dict({"value": 10**310, "origin": "user"})
    assert exposure.value is None
    assert exposure.origin == "unknown:unusable declaration"


@pytest.mark.parametrize("value", [True, False])
@pytest.mark.io
def test_boolean_file_exposure_remains_loadable_with_unknown_exposure(tmp_path, value):
    import json

    import h5py

    from asymmetry.core.io import load
    from asymmetry.core.io.nexus_writer import write_nexus_v1
    from tests.io.test_nexus_writer import _template

    run = _template()
    path = tmp_path / "reference.nxs"
    write_nexus_v1(run, path)
    with h5py.File(path, "a") as handle:
        entry = handle["run"]
        del entry["asymmetry_reference_exposure"]
        entry.create_dataset(
            "asymmetry_reference_exposure", data=json.dumps({"value": value, "origin": "user"})
        )
    restored = load(str(path)).run
    assert restored.grouping["good_frames"] == run.grouping["good_frames"]
    assert ReferenceExposure.from_grouping(restored.grouping).value is None


@pytest.mark.parametrize("value", [True, False])
def test_boolean_nexus_frames_are_not_exposure_but_retain_dt_parsing(value):
    facts = read_nexus_frames(
        {"good_frames": [value]},
        lambda node, key, default=None: node.get(key, default) if node is not None else default,
        legacy=True,
    )
    assert facts.values.tolist() == [float(value)]
    assert facts.exposures(1)[0]["value"] is None
