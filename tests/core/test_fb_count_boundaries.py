"""Count-array boundaries reject mismatches instead of selecting a partial record."""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.transform.asymmetry import (
    compute_asymmetry,
    compute_asymmetry_with_count_errors,
    estimate_alpha,
    estimate_alpha_detailed,
)
from asymmetry.core.transform.background import subtract_scaled_counts
from asymmetry.core.transform.integral import integrate_asymmetry


@pytest.mark.parametrize(
    "operation",
    [
        lambda f, b: compute_asymmetry(f, b),
        lambda f, b: compute_asymmetry_with_count_errors(f, b, np.sqrt(f), np.sqrt(b)),
        lambda f, b: estimate_alpha(f, b),
        lambda f, b: estimate_alpha_detailed(f, b, method="ratio"),
        lambda f, b: integrate_asymmetry(f, b),
        lambda f, b: subtract_scaled_counts(f, b, 1.0),
    ],
    ids=["asymmetry", "count-errors", "alpha", "detailed-alpha", "integral", "reference"],
)
def test_mismatched_count_arrays_are_rejected(operation) -> None:
    with pytest.raises(ValueError, match="shape"):
        operation(np.array([10.0, 20.0]), np.array([5.0]))


def test_count_errors_must_have_the_count_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        compute_asymmetry_with_count_errors(
            np.array([10.0, 20.0]), np.array([5.0, 8.0]), np.array([3.0]), np.array([2.0, 3.0])
        )


def test_alpha_time_axis_must_have_the_count_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        estimate_alpha_detailed(
            np.array([10.0, 20.0]),
            np.array([5.0, 8.0]),
            time_us=np.array([0.0]),
            method="ratio",
        )
