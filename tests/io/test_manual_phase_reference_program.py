"""Explicitly configured fresh native musrview phase control."""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tests.reference_cases import load_reference_case
from tests.reference_programs import musrfit_executable

pytestmark = [pytest.mark.io, pytest.mark.integration]
FIXTURES = Path(__file__).parents[1] / "fixtures/reference_cases/manual-phase"


def test_fresh_native_musrview_manual_phase(tmp_path):
    if not os.environ.get("ASYMMETRY_MUSRVIEW"):
        pytest.skip("set ASYMMETRY_MUSRVIEW and ASYMMETRY_ROOT for fresh native phase execution")
    executable = musrfit_executable("musrview")
    root = os.environ.get("ASYMMETRY_ROOT")
    assert root, "configured musrview requires ASYMMETRY_ROOT for public canvas extraction"
    version = subprocess.run(
        [root, "--version"], capture_output=True, text=True, check=True, timeout=15
    )
    assert "ROOT Version: 6.40.04" in (version.stdout + version.stderr).splitlines(), (
        "ROOT identity mismatch or empty version output"
    )
    case = load_reference_case(FIXTURES / "reference.json")
    assert (
        hashlib.sha256(Path(executable).resolve().read_bytes()).hexdigest()
        == (case["identity"]["musrview"]["executable_sha256"])
    ), "musrview binary identity mismatch"
    for name in case["inputs"]:
        shutil.copyfile(FIXTURES / name, tmp_path / name)
    spectra = []
    for phase in case["settings"]["phase_degrees"]:
        result = subprocess.run(
            [executable, f"phase{phase}.msr", "-f", "--root", "--timeout", "15"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
            env=os.environ | {"MUSRFITPATH": str(tmp_path)},
        )
        canvas = tmp_path / f"phase{phase}_0_F.root"
        assert canvas.is_file(), result.stdout + result.stderr
        output = f"phase{phase}.csv"
        macro = f"export_canvas.C({json.dumps(canvas.name)},{json.dumps(output)})"
        subprocess.run(
            [root, "-n", "-l", "-b", "-q", macro],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        data = np.loadtxt(tmp_path / output, delimiter=",")
        assert data.shape == (case["native_output"]["bins"], 3), "empty/truncated native output"
        spectra.append(data)
    recorded = np.loadtxt(FIXTURES / case["native_output"]["file"], delimiter=",")
    for data, columns in zip(spectra, ([0, 1, 2], [0, 3, 4]), strict=True):
        np.testing.assert_allclose(data, recorded[:, columns], rtol=1e-7, atol=1e-7)
    for name, digest in case["inputs"].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest
