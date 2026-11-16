"""Optional fresh musrfit checks of our serialized synthetic PSI fixtures."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tests.reference_programs import musrfit_executable

pytestmark = [pytest.mark.io, pytest.mark.integration]
FIXTURES = Path(__file__).parents[1] / "fixtures" / "psi_root"


@pytest.mark.parametrize(
    "filename,format_name",
    [
        ("synthetic.bin", "PSI-BIN"),
        ("legacy-lem.root", "ROOT-NPP"),
        ("musrroot-folder.root", "MUSR-ROOT"),
    ],
)
@pytest.mark.parametrize("file_t0", [False, True])
@pytest.mark.parametrize(
    "asymmetry,packing,native_groups",
    [(False, 1, 40), (True, 1, 40), (False, 3, 13), (False, 4, 9)],
)
def test_native_psi_selected_observations(
    tmp_path, filename, format_name, asymmetry, file_t0, packing, native_groups
):
    executable = musrfit_executable()
    shutil.copyfile(FIXTURES / filename, tmp_path / filename)
    (tmp_path / "musrfit_startup.xml").write_text(
        '<?xml version="1.0"?><musrfit><data_path>.</data_path></musrfit>'
    )
    settings = (
        "fittype 2\nalpha 1\nbackgr.fix 0 0\nforward 1\nbackward 2\nt0 4 7\ndata 20 60 23 63"
        if asymmetry
        else "fittype 0\nnorm 2\nbackgr.fit 3\nforward 1 2\nt0 4 7\ndata 20 60"
    )
    if file_t0:
        settings = settings.replace("t0 4 7\n", "")
    (tmp_path / "reference.msr").write_text(
        f"""Synthetic PSI selected observations
###############################################################
FITPARAMETER
1 alpha 1 0 none
2 norm 1000 0 none
3 bg 0 0 none
4 asym 0 0 none
###############################################################
THEORY
asymmetry 4
###############################################################
RUN {Path(filename).stem} GPS PSI {format_name}
{settings}
fit -1 10
packing {packing}
###############################################################
COMMANDS
SCALE_N0_BKG FALSE
MIGRAD
SAVE
###############################################################
PLOT {2 if asymmetry else 0}
runs 1
range -1 10
"""
    )
    result = subprocess.run(
        [executable, "reference.msr", "-c", "--dump", "ascii", "--use-no-of-threads", "1"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    exported = np.loadtxt(tmp_path / "reference_0.dat", comments="%", delimiter=",")
    assert exported.shape == (native_groups, 4)
    # The native packing-4 export omits the terminal complete group; retain
    # that difference rather than changing the requested data interval.
    raw = np.arange(20, 20 + native_groups * packing)
    forward, backward = 100 + raw, 300 + 3 * (raw + 3)
    time = ((raw - 4) * 0.01).reshape(-1, packing).mean(axis=1)
    forward = forward.reshape(-1, packing).sum(axis=1)
    backward = backward.reshape(-1, packing).sum(axis=1)
    # The theory column is a fit prediction, not a reduction reference.
    np.testing.assert_allclose(exported[:, 0], time, rtol=3e-8)
    if asymmetry:
        np.testing.assert_allclose(
            exported[:, 1], (forward - backward) / (forward + backward), rtol=5e-6
        )
        np.testing.assert_allclose(
            exported[:, 2], 2 * np.sqrt(forward * backward / (forward + backward) ** 3), rtol=5e-6
        )
    else:
        np.testing.assert_array_equal(exported[:, 1], forward + backward)
        np.testing.assert_allclose(exported[:, 2], np.sqrt(forward + backward), rtol=5e-6)
