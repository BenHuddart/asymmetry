"""Optional native feature comparisons, distinct from timing application checks."""

import re
import shutil
import subprocess

import numpy as np
import pytest

from asymmetry.core.io import load
from asymmetry.core.transform.t0 import find_t0_for_run
from tests.io.test_psi_loader import _write_psi_bin

pytestmark = [pytest.mark.io, pytest.mark.integration]


@pytest.mark.parametrize("pulsed", [False, True])
def test_native_musrt0_unique_features(tmp_path, pulsed):
    executable = shutil.which("musrt0")
    if executable is None:
        pytest.skip("musrt0 is not installed")
    version = subprocess.run(
        [executable, "--version"], capture_output=True, text=True, timeout=15, check=True
    ).stdout
    if "1.12.1" not in version or "708f777b" not in version:
        pytest.skip("comparison requires musrt0 1.12.1/708f777b")
    raw = np.full((2, 128), 10, dtype="<i4")
    if pulsed:
        for i, start in enumerate((20, 31)):
            raw[i, :start] = 0
            raw[i, start : start + 11] = np.arange(11) * 100
        maxima, edges = [30, 41], [25, 36]
    else:
        raw[0, 20], raw[1, 31] = 10000, 20000
        maxima = edges = [20, 31]
    path = tmp_path / "feature.bin"
    _write_psi_bin(path, counts=raw, t0_values=[4, 7], first_good_values=[50, 61])
    (tmp_path / "musrfit_startup.xml").write_text(
        '<?xml version="1.0"?><musrfit><data_path>.</data_path></musrfit>'
    )
    msr = tmp_path / "feature.msr"
    msr.write_text("""Synthetic timing feature
FITPARAMETER
1 alpha 1 0 none
2 asym 0 0 none
THEORY
asymmetry 2
RUN feature GPS PSI PSI-BIN
fittype 2
alpha 1
backgr.fix 0 0
forward 1
backward 2
t0 4 7
data 50 100 61 111
fit 0 1
packing 1
COMMANDS
MIGRAD
SAVE
PLOT 2
runs 1
range 0 1
""")
    result = subprocess.run(
        [executable, msr.name, "-g"], cwd=tmp_path, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr
    native = re.search(r"^t0\s+([\d.]+)\s+([\d.]+)", msr.read_text(), re.MULTILINE)
    assert native is not None
    assert [float(v) for v in native.groups()] == maxima
    run = load(path).run
    search = find_t0_for_run(run.histograms, pulsed=pulsed)
    # Exact integer comparisons: unique maxima agree; half-height is a different feature.
    assert [est.t0_bin for est in search.estimates] == edges
