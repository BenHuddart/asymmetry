r"""Explicit asymmetry units — the fraction/percent divide, named.

Two conventions coexist across the API and both are internally consistent:

* the dimensionless **fraction** :math:`A \in [-1, 1]`, what the low-level
  primitives (:func:`~asymmetry.core.transform.compute_asymmetry`,
  :func:`~asymmetry.core.transform.binned_fb_asymmetry`,
  :func:`~asymmetry.core.transform.integrate_asymmetry`) return; and
* **percent** :math:`0`–:math:`100`, what the loaders, the reductions
  (:func:`~asymmetry.core.transform.reduce_grouped_asymmetry`),
  :attr:`~asymmetry.core.data.dataset.MuonDataset.asymmetry` and every built-in
  fit model's amplitude use (``A0`` defaults to ``25``).

Mixing them is a silent factor of 100 that reads as a plausible-looking result:
a 20 % amplitude fitted as 0.2 %, or a fit that converges to a degenerate
minimum. This module gives that distinction a name so a call site can *state*
the scale instead of a reader inferring it. Nothing here changes any existing
return value — see "Asymmetry units across the API" in the documentation for
the per-function table, and :class:`asymmetry.core.fitting.AsymmetryScaleWarning`
for the runtime guard that catches the mismatch during a fit.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "ASYMMETRY_FRACTION",
    "ASYMMETRY_PERCENT",
    "PERCENT_PER_FRACTION",
    "AsymmetryUnit",
    "convert_asymmetry",
    "to_fraction",
    "to_percent",
]

#: Percent per unit fraction — the whole content of the conversion, named once so
#: no call site has to spell a bare ``100.0``.
PERCENT_PER_FRACTION = 100.0


class AsymmetryUnit(Enum):
    r"""The scale an asymmetry value is expressed on.

    A two-member enum, deliberately: unlike
    :class:`~asymmetry.core.fitting.knight_shift.KnightShiftUnit` there is no
    ``AUTO`` member and no ppm, because an asymmetry's scale is a *property of
    the producing function*, not a display preference to be guessed — the point
    is that every producer can state which of the two it means.
    """

    #: Dimensionless fraction, :math:`A \in [-1, 1]`.
    FRACTION = "fraction"
    #: Percent, :math:`0`–:math:`100` (fraction × 100).
    PERCENT = "percent"

    @property
    def per_fraction(self) -> float:
        """Multiplier taking a fraction-scale value onto this unit."""
        return PERCENT_PER_FRACTION if self is AsymmetryUnit.PERCENT else 1.0

    @property
    def label(self) -> str:
        """Short axis-label suffix: ``"%"`` for percent, empty for a fraction."""
        return "%" if self is AsymmetryUnit.PERCENT else ""

    def scale_to(self, other: AsymmetryUnit) -> float:
        """Multiplicative factor converting a value in *this* unit to ``other``.

        ``ASYMMETRY_FRACTION.scale_to(ASYMMETRY_PERCENT)`` is ``100.0``; the
        reverse is ``0.01``; either unit to itself is exactly ``1.0``.
        """
        return other.per_fraction / self.per_fraction


#: The dimensionless fraction :math:`A \in [-1, 1]`. Module-level alias for
#: :attr:`AsymmetryUnit.FRACTION`, so a keyword argument or a docstring can name
#: the scale without importing the enum.
ASYMMETRY_FRACTION = AsymmetryUnit.FRACTION

#: Percent, :math:`0`–:math:`100`. Module-level alias for
#: :attr:`AsymmetryUnit.PERCENT`.
ASYMMETRY_PERCENT = AsymmetryUnit.PERCENT


def convert_asymmetry(
    values: ArrayLike,
    frm: AsymmetryUnit,
    to: AsymmetryUnit,
) -> NDArray[np.float64]:
    """Convert asymmetry (or asymmetry-error) values between the two scales.

    Parameters
    ----------
    values : array_like
        Asymmetry or asymmetry-error values on the ``frm`` scale. Errors convert
        by the same factor as the values they belong to.
    frm, to : AsymmetryUnit
        Source and target scale. Both must be stated — there is no inference,
        because guessing the input scale from its magnitude is exactly the
        failure mode this module exists to remove.

    Returns
    -------
    NDArray
        A new ``float64`` array on the ``to`` scale. A same-unit conversion
        still returns a copy, so the result is never an alias of the input.
    """
    out = np.array(values, dtype=np.float64, copy=True)
    factor = frm.scale_to(to)
    if factor != 1.0:
        out *= factor
    return out


def to_percent(
    values: ArrayLike, *, frm: AsymmetryUnit = ASYMMETRY_FRACTION
) -> NDArray[np.float64]:
    """Return ``values`` on the percent scale (fraction input by default).

    The one-liner for feeding a fraction-scale primitive's output to a
    percent-scale consumer::

        asym, err = compute_asymmetry(forward, backward, alpha)
        engine.fit_arrays(time, to_percent(asym), to_percent(err), model, params)
    """
    return convert_asymmetry(values, frm, ASYMMETRY_PERCENT)


def to_fraction(
    values: ArrayLike, *, frm: AsymmetryUnit = ASYMMETRY_PERCENT
) -> NDArray[np.float64]:
    """Return ``values`` on the fractional scale (percent input by default).

    The counterpart of :func:`to_percent`, and the free-function equivalent of
    :attr:`asymmetry.core.data.dataset.MuonDataset.asymmetry_fraction`.
    """
    return convert_asymmetry(values, frm, ASYMMETRY_FRACTION)
