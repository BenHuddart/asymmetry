"""Fresh type8 constant fit against the recorded three-point native case."""

import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tests.reference_cases import load_reference_case
from tests.reference_programs import musrfit_executable

pytestmark = [pytest.mark.io, pytest.mark.integration]
FIXTURES = Path(__file__).parents[1] / "fixtures/reference_cases/constant"


def test_fresh_musrfit_supplied_error_constant(tmp_path):
    executable = musrfit_executable()
    case = load_reference_case(FIXTURES / "reference.json")
    for name in case["inputs"]:
        shutil.copyfile(FIXTURES / name, tmp_path / name)
    subprocess.run(
        [
            executable,
            "owned.msr",
            "--dump",
            "ascii",
            "--keep-mn2-output",
            "--use-no-of-threads",
            "1",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    output = (tmp_path / "MINUIT2.OUTPUT").read_text()
    parameter = re.search(r"^\s*1\s+a\s+(\S+)\s+(\S+)", output, re.MULTILINE)
    objective = re.search(r"chisq\s*=\s*(\S+), NDF = (\d+)", output)
    assert parameter is not None and objective is not None, "invalid/empty musrfit fit output"
    assert int(objective[2]) == 2, "native fit did not retain all three observations"
    reference = case["native_output"]["musrfit"]
    np.testing.assert_allclose(
        [float(parameter[1]), float(parameter[2]), float(objective[1])],
        [reference["value"], reference["standard_error"], reference["chi_squared"]],
        rtol=0,
        atol=case["tolerances"]["musrfit"]["atol"],
    )
