"""Reference manifests describe the actual complete owned native input."""

from pathlib import Path

import numpy as np
import pytest

from tests.reference_cases import load_reference_case

pytestmark = pytest.mark.io
h5py = pytest.importorskip("h5py")
FIXTURES = Path(__file__).parents[1] / "fixtures/reference_cases/manual-phase"


def test_manual_phase_owned_counts_and_requests_match_manifest():
    case = load_reference_case(FIXTURES / "reference.json")
    t = np.arange(1001) * 0.01
    expected = np.rint(200000 + 30000 * np.cos(2 * np.pi * 1.3 * t + 0.41))
    with h5py.File(FIXTURES / "synthetic.nxs") as handle:
        detector = handle["raw_data_1/instrument/detector_1"]
        np.testing.assert_array_equal(detector["counts"][0], [expected, expected])
        np.testing.assert_array_equal(detector["raw_time"], t)
        np.testing.assert_array_equal(detector["counts"].attrs["t0_bin"], [0, 0])
        assert detector["raw_time"].attrs["units"] == "microseconds"
    zero = (FIXTURES / "phase0.msr").read_text()
    phased = (FIXTURES / "phase37.msr").read_text()
    assert phased == zero.replace("phase 0", "phase 37")
    assert case["settings"]["phase_degrees"] == [0, 37]
    assert case["settings"]["packing"] == 1
    for setting in (
        "data 3 993",
        "t0 0",
        "packing 1",
        "apodization none",
        "backgr.fit 2",
        "2 bg 200000 0 none",
        "deadtime-cor no",
        "range 0.03 9.93",
    ):
        assert setting in zero
