"""Profile binning is typed when consumed, while saved invalid payloads still load."""

import numpy as np
import pytest

from asymmetry.core.project.profiles import ProfileFingerprint, profile_from_payload
from asymmetry.core.project.schema import CURRENT_SCHEMA_VERSION, load_project, save_project
from asymmetry.core.transform.rebin import BinningSpec


@pytest.mark.parametrize(
    "key,value",
    [
        ("bunching_factor", "abc"),
        ("bunching_factor", 2.7),
        ("bunching_factor", True),
        ("binning_mode", "Varaible"),
        ("bin0_us", -1),
        ("bin10_us", np.nan),
    ],
)
def test_profile_rejects_invalid_binning_at_its_boundary(key, value):
    with pytest.raises(ValueError, match="binning mode|bunching factor|bin0_us|bin10_us"):
        profile_from_payload({key: value}, "test", ProfileFingerprint("test", 2))


def test_profile_stores_one_binning_spec_and_keeps_the_wire_keys():
    profile = profile_from_payload(
        {"binning_mode": "variable", "bunching_factor": 3, "bin0_us": 0.1, "bin10_us": 0.4},
        "test",
        ProfileFingerprint("test", 2),
    )
    assert profile.binning == BinningSpec("variable", 3, 0.1, 0.4)
    raw = profile.to_dict()
    assert raw["binning_mode"] == "variable" and raw["bunching_factor"] == 3
    assert raw["bin0_us"] == 0.1 and raw["bin10_us"] == 0.4
    assert type(profile).from_dict(raw).binning == profile.binning


def test_invalid_saved_binning_loads_and_fails_when_profile_is_consumed(tmp_path):
    payload = {"bunching_factor": "abc"}
    state = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "datasets": [{"run_number": 1, "source_file": "sample.nxs", "grouping_overrides": payload}],
    }
    path = tmp_path / "binning.asymp"
    save_project(state, path)
    loaded = load_project(path)["datasets"][0]["grouping_overrides"]
    assert loaded == payload
    with pytest.raises(ValueError, match="bunching factor"):
        profile_from_payload(loaded, "test", ProfileFingerprint("test", 2))


def test_profile_base_binning_cannot_silently_discard_a_display_merge():
    from asymmetry.core.project.profiles import GroupingProfile

    with pytest.raises(ValueError, match="base binning; merge must be 1"):
        GroupingProfile("test", ProfileFingerprint("test", 2), binning=BinningSpec(merge=2))
