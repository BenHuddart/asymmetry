"""Independent coordinate and signed-count uncertainty controls."""

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram
from asymmetry.core.transform.asymmetry import compute_asymmetry_with_count_errors
from asymmetry.core.transform.rebin import BinningSpec
from asymmetry.core.transform.reduce import reduce_grouped_asymmetry


def test_raw_coordinate_error_transforms_with_the_value():
    """A central-difference Jacobian converts unbalanced r noise to corrected A.

    This is first-order coordinate arithmetic, not native-program execution or
    equivalence of nonlinear least-squares objectives for finite residuals.
    """
    raw_ratio = 0.2
    raw_error = np.sqrt(0.96 / 100)
    alpha, beta = 1.25, 0.8
    step = 1e-5
    ratios = raw_ratio + np.array([-step, 0.0, step])
    forward_fraction = (1 + ratios) / 2
    backward_fraction = 1 - forward_fraction
    corrected = (forward_fraction - alpha * backward_fraction) / (
        beta * forward_fraction + alpha * backward_fraction
    )
    jacobian = (corrected[2] - corrected[0]) / (2 * step)
    value, error = compute_asymmetry_with_count_errors(
        np.array([60.0]), np.array([40.0]), np.sqrt([60.0]), np.sqrt([40.0]), alpha, beta
    )
    assert value[0] == pytest.approx(5 / 49, rel=1e-12)
    assert error[0] == pytest.approx(abs(jacobian) * raw_error, rel=1e-9)
    assert error[0] == pytest.approx(0.11477201002211895, rel=1e-12)


@pytest.mark.parametrize(
    "background, expected_value, expected_variance",
    [(5.0, -100.0, 2000.0), (10.0, -300.0, 48000.0), (14.0, -1900.0, 52400000.0)],
)
def test_signed_background_reduction_retains_raw_variance(
    background, expected_value, expected_variance
):
    """Fixed subtraction keeps var(F)=5, var(B)=10 even at corrected F=0.

    Independent derivatives at F=0,-5,-9 and B=10 give fractional variances
    0.2, 4.8, and 5240. The public reduction scales their errors by 100.
    """
    result = reduce_grouped_asymmetry(
        histograms=[Histogram(np.array([count]), 0.1) for count in (5.0, 10.0)],
        grouping={
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "background_mode": "fixed",
            "background_fixed_values": [background, 0.0],
        },
        binning=BinningSpec(),
        forward_idx=[0],
        backward_idx=[1],
        alpha=1.0,
        beta=1.0,
        use_deadtime=False,
        deadtime_mode="off",
        use_background=True,
    )
    assert result.asymmetry[0] == pytest.approx(expected_value)
    assert result.error[0] ** 2 == pytest.approx(expected_variance)
