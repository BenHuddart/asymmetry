"""Entire exponential functions for relaxation memory integrals.

The phi-function definition follows Hochbruck & Ostermann, Acta Numerica 19,
209 (2010), doi:10.1017/S0962492910000048.
"""

from math import factorial

import numpy as np
from numpy.typing import ArrayLike, NDArray

_PHI2_COEFFICIENTS = tuple(1.0 / factorial(k + 2) for k in range(17))


def phi2(z: ArrayLike) -> NDArray:
    """Evaluate (exp(z) - 1 - z)/z², continuously extended to phi2(0) = 1/2.

    Inside |z| <= 1, a 17-term Taylor polynomial avoids cancellation. Its
    omitted tail is bounded by 1/(19! (1 - 1/20)) < 8.7e-18. Outside that
    disc the closed form is well conditioned in the left half-plane used by
    decaying correlation functions. Real inputs retain a real result.
    """
    argument = np.asarray(z)
    argument = np.asarray(argument, dtype=np.result_type(argument.dtype, float))
    result = np.empty_like(argument)
    small = np.abs(argument) <= 1
    result[small] = np.polynomial.polynomial.polyval(argument[small], _PHI2_COEFFICIENTS)
    large = argument[~small]
    result[~small] = (np.expm1(large) - large) / large / large
    return result
