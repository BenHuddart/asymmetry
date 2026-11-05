"""Matched raw observations against musrfit 1.12.1/708f777b (ROOT 6.40.04).

The native single-histogram ASCII export has six significant digits. Its
integer channel timing is compared separately from exact NeXus calibration.
"""

import shutil
import subprocess
from dataclasses import replace

import numpy as np
import pytest

from asymmetry.core.io.nexus import NexusLoader
from asymmetry.core.project.profiles import (
    GoodWindowPolicy,
    GroupingProfile,
    ProfileFingerprint,
    T0Policy,
    resolve_effective_grouping,
)
from asymmetry.core.simulate import reduce_run_to_dataset
from asymmetry.core.transform.grouping import group_forward_backward
from tests.io.test_deadtime_reference_programs import _synthetic_nexus

h5py = pytest.importorskip("h5py")
pytestmark = [pytest.mark.io, pytest.mark.integration]


@pytest.mark.parametrize("exact", [0.2074, 0.21])
@pytest.mark.parametrize("channel", [1, 2])
@pytest.mark.parametrize("shift,manual", [(0, False), (5, False), (5, True)])
def test_musrfit_matched_explicit_data_channels(tmp_path, channel, exact, shift, manual):
    program = shutil.which("musrfit")
    if program is None:
        pytest.skip("musrfit executable is not installed")
    version = subprocess.check_output([program, "--version"], text=True, timeout=15)
    if "1.12.1" not in version or "708f777b" not in version:
        pytest.skip("reference requires musrfit 1.12.1 revision 708f777b")
    path = tmp_path / "synthetic.nxs"
    _synthetic_nexus(path, 0)
    raw = np.stack([100 + np.arange(32), 300 + 2 * np.arange(32)])
    with h5py.File(path, "a") as f:
        d = f["raw_data_1/instrument/detector_1"]
        for name in ("counts", "raw_time", "resolution"):
            del d[name]
        d.create_dataset("counts", data=raw[None, :, :]).attrs["t0_bin"] = [11, 11]
        d.create_dataset("raw_time", data=np.arange(33) * 0.02).attrs["units"] = "us"
        d.create_dataset("resolution", data=[20000.0]).attrs["units"] = "picoseconds"
        for name, value in [
            ("time_zero", exact),
            ("first_good_time", 0.22),
            ("last_good_time", 0.30),
        ]:
            d.create_dataset(name, data=value).attrs["units"] = "us"
        del f["raw_data_1/detector_1/counts"]
        f["raw_data_1/detector_1/counts"] = d["counts"]
    (tmp_path / "musrfit_startup.xml").write_text(
        '<?xml version="1.0"?><musrfit><data_path>.</data_path></musrfit>'
    )
    first, end = (11 + shift, 15 + shift) if manual else (11, 15)
    (tmp_path / "reference.msr").write_text(f"""Synthetic raw observations
###############################################################
FITPARAMETER
1 norm 100 0 none
2 bg 0 0 none
3 asym 0 0 none
###############################################################
THEORY
asymmetry 3
###############################################################
RUN synthetic EMU ISIS NEXUS
fittype 0
norm 1
backgr.fit 2
forward {channel}
t0 {10 + shift}
data {first} {end}
fit -1 2
packing 1
###############################################################
COMMANDS
SCALE_N0_BKG FALSE
MIGRAD
SAVE
###############################################################
PLOT 0
runs 1
range -1 2
""")
    result = subprocess.run(
        [program, "reference.msr", "-c", "--dump", "ascii"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    exported = np.loadtxt(tmp_path / "reference_0.dat", comments="%", delimiter=",")
    np.testing.assert_allclose(exported[:, 1], raw[channel - 1, first:end], rtol=5e-6)
    np.testing.assert_allclose(exported[:, 2], np.sqrt(raw[channel - 1, first:end]), rtol=5e-6)
    loaded = NexusLoader().load(path)
    # The explicit native interval has an exclusive upper channel. Native t0
    # stamps channel 10 at zero; Asymmetry retains the exact raw-edge origin.
    profile = GroupingProfile(
        name="matched",
        fingerprint=ProfileFingerprint("EMU", 2),
        groups={1: [1], 2: [2]},
        t0_policy=T0Policy(mode="manual", offset_bins=shift),
        good_window_policy=GoodWindowPolicy(mode="manual", first_offset_bins=1, last_offset_bins=4)
        if manual
        else GoodWindowPolicy(),
    )
    grouping = resolve_effective_grouping(profile, loaded.run)
    counts = group_forward_backward(loaded.run.histograms, grouping).counts
    native_time = (np.arange(first, end) - 10 - shift) * 0.02
    np.testing.assert_allclose(exported[:, 0], native_time, atol=1e-15)
    np.testing.assert_allclose(
        reduce_run_to_dataset(replace(loaded.run, grouping=grouping)).time - exported[:, 0],
        0.21 - exact,
        atol=1e-15,
    )
    value = (counts.forward, counts.backward)[channel - 1]
    np.testing.assert_allclose(value.counts[counts.window.slice], exported[:, 1], rtol=5e-6)
    np.testing.assert_allclose(
        np.sqrt(value.independent_variance[counts.window.slice]), exported[:, 2], rtol=5e-6
    )
    assert [(h.good_bin_start, h.good_bin_end) for h in loaded.run.histograms] == [(11, 14)] * 2
