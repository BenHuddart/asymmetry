"""Synthetic executable references for deadtime-corrected count uncertainties.

musrfit 1.12.1 (708f777b, ROOT 6.40.04) exports single-histogram errors
from corrected counts. The ASCII export has six significant digits, so
comparisons allow 5e-6 relative error. No reference implementation source is used.
"""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

pytestmark = [pytest.mark.io, pytest.mark.integration]
h5py = pytest.importorskip("h5py")


def _synthetic_nexus(path, coefficient):
    """Write an ISIS IDF V2 acquisition with two unequal, constant detectors."""
    with h5py.File(path, "w") as handle:
        handle.attrs.update(NeXus_version="4.4.3", HDF5_version=h5py.version.hdf5_version)
        entry = handle.create_group("raw_data_1")
        entry.attrs["NX_class"] = "NXentry"
        for name, value in {
            "definition": "muonTD",
            "title": "Synthetic deadtime uncertainty",
            "start_time": "2000-01-01T00:00:00",
            "end_time": "2000-01-01T01:00:00",
            "experiment_identifier": "synthetic",
            "beamline": "EMU",
            "instrument/name": "EMU",
            "instrument/source/name": "ISIS",
            "instrument/source/type": "pulsed muon source",
            "instrument/source/probe": "positive muons",
            "sample/name": "synthetic",
            "sample/shape": "synthetic",
        }.items():
            entry.create_dataset(name, data=np.bytes_(value))
        for name, value in {"IDF_version": 2, "run_number": 1, "good_frames": 100000}.items():
            entry.create_dataset(name, data=[value])
        entry.create_dataset("sample/temperature", data=[1.0]).attrs["units"] = "kelvin"
        entry.create_dataset("sample/magnetic_field", data=[0.0]).attrs["units"] = "gauss"
        detector = entry.create_group("instrument/detector_1")
        detector.attrs["NX_class"] = "NXdetector"
        counts = detector.create_dataset(
            "counts", data=np.broadcast_to(np.array([100000, 25000])[None, :, None], (1, 2, 100))
        )
        counts.attrs.update(
            signal=1, axes="period_index,spectrum_index,time_bin", t0_bin=[0, 0], units="counts"
        )
        detector.create_dataset("spectrum_index", data=[1, 2])
        detector.create_dataset("period_index", data=[1])
        detector.create_dataset("raw_time", data=np.arange(100) * 0.1).attrs["units"] = (
            "microseconds"
        )
        detector.create_dataset("resolution", data=[100000.0]).attrs["units"] = "picoseconds"
        detector.create_dataset("dead_time", data=[coefficient, coefficient]).attrs.update(
            available=1, units="microseconds"
        )
        data = entry.create_group("detector_1")
        data.attrs["NX_class"] = "NXdata"
        data["counts"] = counts


@pytest.mark.parametrize("coefficient", [-0.02, 0.0, 0.005, 0.02, 0.05])
@pytest.mark.parametrize("packing", [1, 4])
@pytest.mark.parametrize("detectors", ["1", "1 2"])
def test_musrfit_single_histogram_uncertainty(tmp_path, coefficient, packing, detectors):
    executable = shutil.which("musrfit")
    if executable is None:
        pytest.skip("musrfit executable is not installed")
    version = subprocess.run(
        [executable, "--version"], capture_output=True, text=True, check=True, timeout=15
    ).stdout
    if "1.12.1" not in version or "708f777b" not in version:
        pytest.skip("reference requires musrfit 1.12.1, revision 708f777b")
    _synthetic_nexus(tmp_path / "synthetic.nxs", coefficient)
    (tmp_path / "musrfit_startup.xml").write_text(
        '<?xml version="1.0"?><musrfit><data_path>.</data_path></musrfit>'
    )
    (tmp_path / "reference.msr").write_text(
        f"""Synthetic deadtime uncertainty
###############################################################
FITPARAMETER
1 norm 100000 0 none
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
forward {detectors}
t0 0
data 1 99
deadtime-cor file
fit 0 9
packing {packing}
###############################################################
COMMANDS
SCALE_N0_BKG FALSE
MIGRAD
SAVE
###############################################################
PLOT 0
runs 1
range 0 9
"""
    )
    result = subprocess.run(
        [executable, "reference.msr", "-c", "--dump", "ascii"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    exported = np.loadtxt(tmp_path / "reference_0.dat", comments="%", delimiter=",")
    raw = np.array([100000.0]) if detectors == "1" else np.array([100000.0, 25000.0])
    corrected = np.sum(raw / (1 - raw * coefficient / 10000.0))
    np.testing.assert_allclose(exported[:, 1], corrected * packing, rtol=5e-6)
    np.testing.assert_allclose(exported[:, 2], np.sqrt(corrected * packing), rtol=5e-6)
