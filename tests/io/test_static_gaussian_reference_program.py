"""Optional musrfit 1.12.1 static LF Gaussian objective checks on owned input."""

import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tests.core.test_static_gaussian_accuracy import hayano_integral
from tests.reference_programs import musrfit_executable

pytestmark = [pytest.mark.io, pytest.mark.integration]


@pytest.mark.parametrize("ratio,total", [(0.06, 1.19978e-5), (1, 1.25459e-12), (10, 2.09439e-10)])
def test_native_static_gaussian_objective(tmp_path, ratio, total):
    executable = musrfit_executable()
    fixture = Path(__file__).parents[1] / "fixtures/reference_cases/constant"
    shutil.copy(fixture / "musrfit_startup.xml", tmp_path / "musrfit_startup.xml")
    time = np.linspace(0, 5, 41)
    expected = hayano_integral(time, ratio)
    # Unit supplied errors make the native chi-square the squared residual norm.
    (tmp_path / "owned.dat").write_text(
        "HEADER\nTITLE: Independent static Gaussian integral\nDATA\n"
        + "".join(f"{t:.17g}, {value:.17g}, 1\n" for t, value in zip(time, expected, strict=True))
    )
    request = (
        (fixture / "owned.msr")
        .read_text()
        .replace(
            "1 a 3 0.1 none",
            f"1 a 1 0 none\n2 freq {ratio / (2 * np.pi):.17g} 0 none\n3 sigma 1 0 none",
        )
    )
    request = request.replace("asymmetry 1", "asymmetry 1\nstatGssKTLF 2 3")
    request = request.replace("fit -0.5 2.5", "fit -0.1 5.1").replace("range 0 2", "range 0 5")
    (tmp_path / "owned.msr").write_text(request)
    result = subprocess.run(
        [executable, "owned.msr", "--chisq-only", "--use-no-of-threads", "1"],
        cwd=tmp_path,
        env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    match = re.search(r">> chisq = ([^,]+)", result.stdout)
    assert match is not None, result.stdout + result.stderr
    # The total retains the weak-field discrepancy, rather than certifying agreement.
    # Five significant printed digits and platform rounding set the replay tolerance.
    np.testing.assert_allclose(float(match[1]), total, rtol=1e-4, atol=1e-14)
