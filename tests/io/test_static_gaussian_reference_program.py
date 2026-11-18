"""Optional musrfit 1.12.1 static LF Gaussian objective checks on owned input."""

import numpy as np
import pytest

from tests.core.test_static_gaussian_accuracy import hayano_integral
from tests.reference_programs import musrfit_kt_objective

pytestmark = [pytest.mark.io, pytest.mark.integration]


@pytest.mark.parametrize("ratio,total", [(0.06, 1.19978e-5), (1, 1.25459e-12), (10, 2.09439e-10)])
def test_native_static_gaussian_objective(tmp_path, ratio, total):
    time = np.linspace(0, 5, 41)
    expected = hayano_integral(time, ratio)
    objective = musrfit_kt_objective(
        tmp_path,
        time,
        expected,
        "statGssKTLF 2 3",
        [("a", 1.0), ("freq", ratio / (2 * np.pi)), ("sigma", 1.0)],
    )
    # The total retains the weak-field discrepancy, rather than certifying agreement.
    # Five significant printed digits and platform rounding set the replay tolerance.
    np.testing.assert_allclose(objective, total, rtol=1e-4, atol=1e-14)
