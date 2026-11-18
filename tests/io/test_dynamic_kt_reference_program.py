"""Optional musrfit 1.12.1 fixed-parameter dynamic KT objective replays."""

import json
from pathlib import Path

import numpy as np
import pytest

from tests.reference_programs import musrfit_kt_objective

pytestmark = [pytest.mark.io, pytest.mark.integration]
_REFERENCE = json.loads(
    (Path(__file__).parents[1] / "fixtures/kt_renewal_reference.json").read_text()
)


@pytest.mark.parametrize(
    "kind,rate,ratio,total",
    [
        ("gaussian", 2.4, 0, 9.86018e-6),
        ("gaussian", 100, 1, 9.97993e-10),
        ("lorentzian", 2.4, 1, 6.16668e-6),
        ("lorentzian", 100, 1, 3.42067e-8),
    ],
)
def test_native_dynamic_kt_objective(tmp_path, kind, rate, ratio, total):
    case = next(
        c
        for c in _REFERENCE["cases"]
        if c["kind"] == kind and c["rate_over_width"] == rate and c["omega_over_width"] == ratio
    )
    theory = "dynGssKTLF" if kind == "gaussian" else "dynExpKTLF"
    objective = musrfit_kt_objective(
        tmp_path,
        np.array(case["dimensionless_time"]),
        np.array(case["polarisation"], float),
        f"{theory} 2 3 4",
        [("a", 1.0), ("freq", ratio / (2 * np.pi)), ("width", 1.0), ("hop", rate)],
    )
    # Recorded totals retain the numerical discrepancies; they are not an agreement oracle.
    # Five significant printed digits and platform rounding set the replay tolerance.
    np.testing.assert_allclose(objective, total, rtol=1e-4, atol=1e-14)
