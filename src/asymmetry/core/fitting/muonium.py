"""Muonium oscillation line-shapes, ported from WiMDA's muonium functions.

Faithful ports of WiMDA's ``TFMuonium`` / ``LowTFMuonium`` / ``ZFmuonium``
(``src/Extrafunctions/muoniumfunctions.dpr``), adapted to Asymmetry conventions:

* frequencies in MHz, time in µs, the ``2π`` factor explicit;
* **phase in radians** (WiMDA uses degrees);
* g-factors taken from :mod:`asymmetry.core.utils.constants` rather than the
  literals ``gm = 0.01355342`` / ``ge = 2.8024`` (they agree to 6 figures);
* these return the *normalised* oscillation; the leading amplitude ``A`` is
  applied by the composite-model wrapper (as for every other component).

**Positive-frequency (same-phase) convention.** WiMDA's ``TFMuonium`` uses the
*signed* transition frequency ``w12`` (which is negative), so its
``cos(2π w12 t + φ)`` puts the lower satellite at phase ``−φ``; yet WiMDA's own
``LowTFMuonium`` *negates* ``w12`` to make it positive — an internal
inconsistency. Physically the muonium precession lines share one initial phase,
so here every line uses ``|w|`` (positive frequency, ``+φ`` for all lines). This
is the deliberate, documented deviation from ``TFMuonium``'s literal signed form
(it matches ``LowTFMuonium``'s negation and same-phase data).

The transverse-field functions take the applied field ``B`` (Gauss) and the
hyperfine coupling ``A_hf`` (MHz); the central diamagnetic Mu⁺ line is **not**
included here (model it separately with ``OscillatoryField``), matching WiMDA.

In the shallow-donor regime (small ``A_hf``) ``TFMuonium`` reduces to two
satellites straddling ``ν_d = γ_µ·B`` with separation ``≈ A_hf``: the two extra
transitions carry weight ``(1−δ) → 0``. For that shallow-donor case the muonium
satellites are better fit as three independent oscillating lines (WiMDA's own
recommendation, = Asymmetry's link groups); these components target genuine
muonium where all transitions and their relative weights matter. See
``docs/porting/muonium-triplet/``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.utils.constants import (
    ELECTRON_GYROMAGNETIC_RATIO_RAD_PER_US_PER_G,
    GAUSS_TO_TESLA,
    MUON_GYROMAGNETIC_RATIO_MHZ_PER_T,
)

#: Muon and electron gyromagnetic ratios in MHz/G (WiMDA's ``gm`` and ``ge``).
G_MU_MHZ_PER_G = MUON_GYROMAGNETIC_RATIO_MHZ_PER_T * GAUSS_TO_TESLA
G_E_MHZ_PER_G = ELECTRON_GYROMAGNETIC_RATIO_RAD_PER_US_PER_G / (2.0 * np.pi)

_TWO_PI = 2.0 * np.pi


def _tf_levels(field: float, A_hf: float) -> tuple[float, float, float, float, float]:
    """Return ``(delta, E1, E2, E3, E4)`` for the TF muonium level structure.

    Mirrors WiMDA: ``x = (g_e+g_µ)·B/A_hf`` (with ``x → 1e20`` when ``A_hf ≤ 0``,
    as WiMDA guards), ``δ = x/√(1+x²)``, and the four energy levels ``E_i``.

    ``x`` is clamped to ``±1e20`` (WiMDA's own saturation sentinel) so that a
    pathologically small ``A_hf`` can never overflow ``x²`` to ``inf`` and yield
    a NaN curve — keeping the model finite for every trial value the minimiser
    might probe.
    """
    if A_hf > 0.0:
        x = (G_E_MHZ_PER_G + G_MU_MHZ_PER_G) * field / A_hf
    else:
        x = 1.0e20
    x = float(np.clip(x, -1.0e20, 1.0e20))
    d = (G_E_MHZ_PER_G - G_MU_MHZ_PER_G) / (G_E_MHZ_PER_G + G_MU_MHZ_PER_G)
    root = np.sqrt(1.0 + x * x)
    delta = x / root
    e1 = A_hf / 4.0 * (1.0 + 2.0 * d * x)
    e2 = A_hf / 4.0 * (-1.0 + 2.0 * root)
    e3 = A_hf / 4.0 * (1.0 - 2.0 * d * x)
    e4 = A_hf / 4.0 * (-1.0 - 2.0 * root)
    return delta, e1, e2, e3, e4


def tf_muonium(
    t: NDArray[np.float64], field: float, A_hf: float, phase: float
) -> NDArray[np.float64]:
    """Normalised transverse-field muonium oscillation (WiMDA ``TFMuonium``).

    Four Mu⁰ transitions ``w12, w14, w34, w23`` with weights ``(1±δ)``; returns
    ``¼·Σ (1±δ)·cos(2π |w| t + φ)`` (positive-frequency convention, see module
    docstring). ``field`` in Gauss, ``A_hf`` in MHz.
    """
    t = np.asarray(t, dtype=float)
    delta, e1, e2, e3, e4 = _tf_levels(field, A_hf)
    w12, w14, w34, w23 = abs(e1 - e2), abs(e1 - e4), abs(e3 - e4), abs(e2 - e3)
    return 0.25 * (
        (1.0 + delta) * np.cos(_TWO_PI * w12 * t + phase)
        + (1.0 - delta) * np.cos(_TWO_PI * w14 * t + phase)
        + (1.0 + delta) * np.cos(_TWO_PI * w34 * t + phase)
        + (1.0 - delta) * np.cos(_TWO_PI * w23 * t + phase)
    )


def low_tf_muonium(
    t: NDArray[np.float64], field: float, A_hf: float, phase: float
) -> NDArray[np.float64]:
    """Normalised low-field TF muonium oscillation (WiMDA ``LowTFMuonium``).

    Two transitions ``w12, w23`` (positive-frequency convention; WiMDA's
    ``LowTFMuonium`` likewise negates ``w12`` so its lower line is positive).
    """
    t = np.asarray(t, dtype=float)
    delta, e1, e2, e3, _e4 = _tf_levels(field, A_hf)
    w12, w23 = abs(e1 - e2), abs(e2 - e3)
    return 0.25 * (
        (1.0 + delta) * np.cos(_TWO_PI * w12 * t + phase)
        + (1.0 - delta) * np.cos(_TWO_PI * w23 * t + phase)
    )


def zf_muonium(
    t: NDArray[np.float64], A_hf: float, D: float, f_cut: float, phase: float
) -> NDArray[np.float64]:
    """Normalised zero-field axial muonium oscillation (WiMDA ``ZFmuonium``).

    Three lines ``f1=A_hf−D``, ``f2=A_hf+D/2``, ``f3=3D/2`` with weights
    ``a1,a2,a3`` (a Lorentzian roll-off above ``f_cut`` when ``f_cut > 0``),
    normalised by ``1/6``.
    """
    t = np.asarray(t, dtype=float)
    f1 = A_hf - D
    f2 = A_hf + D / 2.0
    f3 = 1.5 * D
    if f_cut > 0.0:
        a1 = 1.0 / (1.0 + (f1 / f_cut) ** 2)
        a2 = 2.0 / (1.0 + (f2 / f_cut) ** 2)
        a3 = 2.0 / (1.0 + (f3 / f_cut) ** 2)
    else:
        a1, a2, a3 = 1.0, 2.0, 2.0
    return (
        a1 * np.cos(_TWO_PI * f1 * t + phase)
        + a2 * np.cos(_TWO_PI * f2 * t + phase)
        + a3 * np.cos(_TWO_PI * f3 * t + phase)
    ) / 6.0
