"""Matched raw observations against musrfit 1.12.1/708f777b (ROOT 6.40.04).

The native single-histogram ASCII export has six significant digits. Its
integer channel timing is compared separately from exact NeXus calibration.
"""

import shutil
import subprocess

import numpy as np
import pytest

from asymmetry.core.io.nexus import NexusLoader
from tests.io.test_deadtime_reference_programs import _synthetic_nexus

h5py = pytest.importorskip("h5py")
pytestmark = [pytest.mark.io, pytest.mark.integration]


@pytest.mark.parametrize("exact", [0.2074, 0.21])
@pytest.mark.parametrize("channel", [1, 2])
def test_musrfit_matched_explicit_data_channels(tmp_path, channel, exact):
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
t0 10
data 11 15
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
    np.testing.assert_allclose(exported[:, 1], raw[channel - 1, 11:15], rtol=5e-6)
    np.testing.assert_allclose(exported[:, 2], np.sqrt(raw[channel - 1, 11:15]), rtol=5e-6)
    loaded = NexusLoader().load(path)
    # The explicit native interval has an exclusive upper channel. Native t0
    # stamps channel 10 at zero; Asymmetry retains the exact raw-edge origin.
    np.testing.assert_allclose(exported[:, 0], [0.02, 0.04, 0.06, 0.08], atol=1e-15)
    np.testing.assert_allclose(loaded.time - exported[:, 0], 0.21 - exact, atol=1e-15)
    assert [(h.good_bin_start, h.good_bin_end) for h in loaded.run.histograms] == [(11, 14)] * 2
