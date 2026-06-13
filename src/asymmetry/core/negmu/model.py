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

Multi-exponential μ⁻ capture count model:  N(t) = Σ_i amp_i·exp(−t/τ_i) + bg.
A raw-count model, NOT an asymmetry model — see the study comparison.md §3 for
why the single-envelope count model used for μ⁺ cannot express this.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CaptureComponent:
    """One exponential component in the capture count model."""

    label: str
    tau_us: float


def build_capture_count_model(
    components: Sequence[CaptureComponent],
) -> Callable[..., NDArray[np.float64]]:
    """Return ``f(t, **params)`` → raw counts for the fixed component order.

    Recognised params: ``amp_<label>`` (per component), optional ``tau_<label>``
    (overrides the component seed when the lifetime is freed), and ``background``
    (flat). Unknown params are ignored. Vectorised over t (μs).
    """
    comp_list = list(components)

    def model(t: NDArray[np.float64], **params) -> NDArray[np.float64]:
        t_arr = np.asarray(t, dtype=np.float64)
        result = np.zeros_like(t_arr)
        for comp in comp_list:
            amp = float(params.get(f"amp_{comp.label}", 0.0))
            tau = float(params.get(f"tau_{comp.label}", comp.tau_us))
            result += amp * np.exp(-t_arr / tau)
        result += float(params.get("background", 0.0))
        return result

    return model


def evaluate_capture_model(
    components: Sequence[CaptureComponent],
    params: dict[str, float],
    t: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Convenience: build + call in one step (used by background.py and tests)."""
    return build_capture_count_model(components)(t, **params)
