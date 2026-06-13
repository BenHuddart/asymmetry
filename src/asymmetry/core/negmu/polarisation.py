"""EXPERIMENTAL — WORK IN PROGRESS. Negative-muon (μ⁻) capture-lifetime analysis.

This API is UNVALIDATED against real μ⁻ elemental-analysis data. No μ⁻ corpus
exists in this project; every result here has been exercised only against
synthetic histograms. The element lifetime values are literature-anchored
(Suzuki, Measday & Roalsvig, Phys. Rev. C 35, 2212 (1987), via Blundell et al.,
Muon Spectroscopy: An Introduction, OUP 2022, Table C.1), but the fitting,
capture-ratio, and background machinery have NOT been checked against an
established tool (WiMDA, Mantid) on measured data. The API, parameter names, and
return shapes MAY CHANGE without notice. Do not rely on results for publication
without independent verification. This feature is deliberately NOT exposed in the
GUI fit builders. Promotion trigger for a GUI: real ISIS μ⁻ data AND a user.

μ⁻SR polarisation functions and a polarisation-aware capture count model.

In μ⁻SR a surviving spin polarisation (≥5/6 is lost in the muonic cascade;
Blundell et al. 2022, §22.1) gives rise to an oscillatory modulation of the
muonic-atom count decay.  The modulation multiplies the raw exponential sum by
``(1 + P_pol(t))``, leaving the flat background unchanged.

Two polarisation functions are provided:

* ``lorentzian_gaussian_polarisation`` (WiMDA "LorGau") — damped precession
* ``diamagnetic_polarisation`` (WiMDA "Diamagnetic") — undamped precession

Both return **P_pol(t)** (the modulation depth × envelope × oscillation), not
the full multiplier ``(1 + P_pol(t))``.  The composite model builder
``build_capture_count_model_with_polarisation`` wraps the multiplier.

Parameterisation conventions confirmed against WiMDA ``Analyse.pas``
``Polfunc()`` (RadioGroup1.itemindex = 2, lines 1322–1323):

    LorGau forward: ``(1 + a0·exp(−λ·t)·cos(2π·freq·t + phase))``

where ``freq`` is in MHz, ``t`` in μs, ``λ`` in μs⁻¹, and ``phase`` in
radians.  Diamagnetic is the undamped (λ = 0) limit.  WiMDA's Diamagnetic
RadioGroup mode (itemindex = 3) has no implementation in ``Polfunc`` (falls
through with undefined result — a WiMDA bug), so the Asymmetry formula is
derived from the textbook physics rather than the WiMDA source.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.negmu.model import CaptureComponent

#: Recognised polarisation mode names.
POLARISATION_MODES: tuple[str, ...] = ("lorgau", "diamagnetic")


def lorentzian_gaussian_polarisation(
    t: NDArray[np.float64],
    a0: float,
    lam: float,
    freq: float,
    phase: float,
) -> NDArray[np.float64]:
    """Lorentzian-damped oscillation — WiMDA "LorGau" polarisation function.

    Returns ``a0·exp(−λ·t)·cos(2π·freq·t + phase)`` (the modulation part
    P_pol(t) only; the full multiplier is ``1 + P_pol(t)``).

    Parameters
    ----------
    t:
        Time array in μs.
    a0:
        Precession amplitude (dimensionless, typically ≤ 1/3 from muonic
        cascade geometry).
    lam:
        Lorentzian damping rate λ in μs⁻¹.
    freq:
        Precession frequency in MHz.
    phase:
        Initial phase in radians.
    """
    t_arr = np.asarray(t, dtype=np.float64)
    return a0 * np.exp(-lam * t_arr) * np.cos(2.0 * np.pi * freq * t_arr + phase)


def diamagnetic_polarisation(
    t: NDArray[np.float64],
    a0: float,
    freq: float,
    phase: float,
) -> NDArray[np.float64]:
    """Undamped diamagnetic precession — WiMDA "Diamagnetic" polarisation.

    Returns ``a0·cos(2π·freq·t + phase)`` (P_pol(t) only; full multiplier is
    ``1 + P_pol(t)``).  This is the λ → 0 limit of
    :func:`lorentzian_gaussian_polarisation`.

    Parameters
    ----------
    t:
        Time array in μs.
    a0:
        Precession amplitude (dimensionless).
    freq:
        Precession frequency in MHz.
    phase:
        Initial phase in radians.
    """
    t_arr = np.asarray(t, dtype=np.float64)
    return a0 * np.cos(2.0 * np.pi * freq * t_arr + phase)


def build_capture_count_model_with_polarisation(
    components: Sequence[CaptureComponent],
    polarisation: str | None = None,
) -> Callable[..., NDArray[np.float64]]:
    """Return ``f(t, **params)`` → raw counts with optional polarisation multiplier.

    When ``polarisation`` is ``None`` the result is **bit-identical** to
    :func:`~asymmetry.core.negmu.model.build_capture_count_model` on the same
    inputs — verified by ``np.array_equal``.

    When ``polarisation`` is ``"lorgau"`` or ``"diamagnetic"`` the model is:

        N(t) = [Σ_i amp_i·exp(−t/τ_i)] · (1 + P_pol(t)) + background

    where P_pol(t) is the polarisation function evaluated from the following
    named parameters in ``**params``:

    * ``"lorgau"``: ``pol_a0``, ``pol_lam``, ``pol_freq``, ``pol_phase``
    * ``"diamagnetic"``: ``pol_a0``, ``pol_freq``, ``pol_phase``

    Missing pol params default to 0.  Unknown params are ignored.

    Parameters
    ----------
    components:
        Capture components (same as :func:`build_capture_count_model`).
    polarisation:
        ``None`` (no polarisation), ``"lorgau"``, or ``"diamagnetic"``.
    """
    if polarisation is not None and polarisation not in POLARISATION_MODES:
        raise ValueError(
            f"Unknown polarisation {polarisation!r}; expected one of {POLARISATION_MODES} or None"
        )

    comp_list = list(components)

    # Bind the polarisation kernel once at build time; avoids per-call string dispatch.
    if polarisation == "lorgau":

        def _apply_pol(t_arr: NDArray[np.float64], kw: dict) -> NDArray[np.float64]:
            return lorentzian_gaussian_polarisation(
                t_arr,
                float(kw.get("pol_a0", 0.0)),
                float(kw.get("pol_lam", 0.0)),
                float(kw.get("pol_freq", 0.0)),
                float(kw.get("pol_phase", 0.0)),
            )

    elif polarisation == "diamagnetic":

        def _apply_pol(t_arr: NDArray[np.float64], kw: dict) -> NDArray[np.float64]:
            return diamagnetic_polarisation(
                t_arr,
                float(kw.get("pol_a0", 0.0)),
                float(kw.get("pol_freq", 0.0)),
                float(kw.get("pol_phase", 0.0)),
            )

    else:
        _apply_pol = None

    def model(t: NDArray[np.float64], **params) -> NDArray[np.float64]:
        t_arr = np.asarray(t, dtype=np.float64)
        exp_sum = np.zeros_like(t_arr)
        for comp in comp_list:
            amp = float(params.get(f"amp_{comp.label}", 0.0))
            tau = float(params.get(f"tau_{comp.label}", comp.tau_us))
            exp_sum += amp * np.exp(-t_arr / tau)
        bg = float(params.get("background", 0.0))
        if _apply_pol is None:
            return exp_sum + bg
        return exp_sum * (1.0 + _apply_pol(t_arr, params)) + bg

    return model
