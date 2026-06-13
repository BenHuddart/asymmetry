"""Fit raw data in the rotating reference frame via a frequency offset.

The statistically exact way to fit in a rotating frame is **not** to fit the
demodulated display curve (its low-pass correlates neighbouring bins and
distorts lineshapes — see :mod:`asymmetry.core.transform.rrf`): it is to fit
the *raw* lab-frame data with a model whose precession frequencies are the
rotating-frame offsets δν plus the frame frequency ν₀.  The fitted
parameters then read directly in the rotating frame — envelope-scale numbers
for a high-TF line — while χ² and the uncertainties keep their exact
per-bin Poisson statistics.

WiMDA precedent: ``MusrFun`` shifts exactly its three rotation component
types by ``rotfreq`` (``$WIMDA_SRC/src/Analyse.pas``), but fits raw data and
guards the combination only by disabling the fit form while the RRF display
is on; a fit run in that state would bias the fitted frequency by a full ν₀
(study comparison ledger, item 4).  Here the offset owns the parameter
semantics instead: fitted frequency ≡ δν, lab value = δν + ν₀ via
:func:`apply_rrf_offsets`.  WiMDA additionally shifts every phase by −φ₀;
that is a display-frame convention deliberately not replicated — fitted
phases remain lab-frame (the wrapper never touches them).

Only components that are pure frame rotations may be shifted —
``Oscillatory`` (frequency in MHz) and ``OscillatoryField`` (field in Gauss,
offset by the Gauss equivalent of ν₀, exact because ν = γ_μB/2π is linear).
Any other oscillating component in the composite (muonium — nonlinear
Breit–Rabi frequencies; ``Bessel`` — a J₀ argument shift is not a frame
rotation; the F-μ-F and dipolar families — multi-frequency) raises
:class:`UnsupportedRRFComponentError`: a silently un-shifted line would
reintroduce the WiMDA trap.  Envelope components pass through untouched.

Migrated (study, implementation-options §B, 2026-06-13): the engine-level
``frequency_offsets`` argument is now live —
:meth:`asymmetry.core.fitting.engine.FitEngine.fit` accepts the resolved
``{param: ν₀}`` dict and applies the same :func:`offset_model_function`
shift, so the GUI fit path resolves :func:`rrf_frequency_offsets` once and
passes it straight to the engine rather than building a callable wrapper.
The thin :func:`rrf_offset_model` wrapper is retained for standalone
callers and pins the engine path's equivalence.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.fourier.units import mhz_to_gauss

if TYPE_CHECKING:
    from asymmetry.core.fitting.composite import CompositeModel

__all__ = [
    "RRF_ROTATION_COMPONENTS",
    "UnsupportedRRFComponentError",
    "apply_rrf_offsets",
    "offset_model_function",
    "rrf_frequency_offsets",
    "rrf_offset_model",
]

#: Components that are pure frame rotations, mapped to the local name of the
#: parameter that sets the precession frequency and that parameter's unit.
#: This is WiMDA's three-rotation-types statement made explicit (its scaled
#: variant has no Asymmetry counterpart).  python-user-functions note: a
#: plugin registering a new pure-rotation component extends this dict.
RRF_ROTATION_COMPONENTS: dict[str, tuple[str, str]] = {
    "Oscillatory": ("frequency", "MHz"),
    "OscillatoryField": ("field", "Gauss"),
}

#: Component categories whose members are pure (non-precessing) envelopes and
#: may pass through the offset untouched.  The check is default-closed: any
#: component outside the rotation registry AND outside this set raises, so a
#: plugin component in a new category fails loudly instead of being silently
#: left in the lab frame (the WiMDA ledger-item-4 trap).
_ENVELOPE_CATEGORIES = frozenset({"Relaxation", "Kubo-Toyabe", "Background"})


class UnsupportedRRFComponentError(ValueError):
    """A composite contains an oscillating component the offset cannot shift.

    Carries the offending component ``name`` so callers can produce targeted
    guidance without parsing the message.
    """

    def __init__(self, name: str) -> None:
        super().__init__(
            f"Component '{name}' oscillates but is not a pure frame rotation; "
            "fitting it with an RRF frequency offset would silently leave its "
            "frequencies in the lab frame. Fit the raw data directly instead, "
            f"or restrict the composite to {sorted(RRF_ROTATION_COMPONENTS)}."
        )
        self.name = name


def rrf_frequency_offsets(model: CompositeModel, frequency_mhz: float) -> dict[str, float]:
    """Map each rotation parameter's unique name to its additive frame offset.

    The offsets are what must be *added* to the fitted (rotating-frame)
    parameter values to evaluate the model against raw lab-frame data — and,
    identically, what converts fitted δν back to lab-frame values for
    reporting (:func:`apply_rrf_offsets`).
    """
    freq = float(frequency_mhz)
    if not np.isfinite(freq) or freq <= 0.0:
        raise ValueError(f"frequency_mhz must be positive and finite, got {frequency_mhz!r}.")

    offset_by_unit = {"MHz": freq, "Gauss": float(mhz_to_gauss(freq))}
    offsets: dict[str, float] = {}
    for component, mapping in zip(model.components, model.parameter_mapping(), strict=True):
        spec = RRF_ROTATION_COMPONENTS.get(component.name)
        if spec is None:
            if component.category not in _ENVELOPE_CATEGORIES:
                raise UnsupportedRRFComponentError(component.name)
            continue
        local_name, unit = spec
        offsets[mapping[local_name]] = offset_by_unit[unit]
    if not offsets:
        raise ValueError(
            f"Composite '{model.component_expression_string()}' contains no rotation "
            "component to offset; add Oscillatory/OscillatoryField or fit without RRF."
        )
    return offsets


def offset_model_function(
    base_fn: Callable[..., NDArray],
    offsets: dict[str, float],
) -> Callable[..., NDArray[np.float64]]:
    """Wrap ``f(t, **params)`` so the named parameters are shifted additively.

    The returned callable adds ``offsets[name]`` to each named parameter before
    delegating to ``base_fn`` — the single shift seam shared by
    :func:`rrf_offset_model` and the engine-level ``frequency_offsets``
    argument (:meth:`asymmetry.core.fitting.engine.FitEngine.fit`), so fitting
    raw data through either path is bit-for-bit identical.  A parameter named in
    ``offsets`` but absent from the call raises :class:`ValueError`.
    """
    offset_items = tuple(offsets.items())

    def wrapped(t: NDArray, **params: float) -> NDArray[np.float64]:
        shifted = dict(params)
        for name, off in offset_items:
            if name not in shifted:
                raise ValueError(
                    f"RRF-offset model expected parameter '{name}'; got {sorted(shifted)}."
                )
            shifted[name] = float(shifted[name]) + off
        return base_fn(t, **shifted)

    return wrapped


def rrf_offset_model(
    model: CompositeModel,
    frequency_mhz: float,
) -> Callable[..., NDArray[np.float64]]:
    """Wrap ``model.function`` so its rotation frequencies read as δν.

    The returned callable has the same ``f(t, **params)`` contract the fit
    engine expects; rotation parameters are shifted by ν₀ (or its Gauss
    equivalent) before delegation, so the data fitted are raw and the fitted
    values are rotating-frame offsets.  The offsets used are attached as
    ``rrf_offsets`` (with ``rrf_frequency_mhz``) for reporting.

    The shift itself is :func:`offset_model_function`; the GUI fit path instead
    resolves :func:`rrf_frequency_offsets` once and passes them straight to the
    engine's ``frequency_offsets`` argument, which applies the same seam.
    """
    offsets = rrf_frequency_offsets(model, frequency_mhz)
    wrapped = offset_model_function(model.function, offsets)
    wrapped.rrf_frequency_mhz = float(frequency_mhz)
    wrapped.rrf_offsets = dict(offsets)
    return wrapped


def apply_rrf_offsets(
    values: dict[str, float],
    offsets: dict[str, float],
) -> dict[str, float]:
    """Convert fitted rotating-frame values to lab-frame ones for reporting.

    ``lab = δν + ν₀`` for each offset parameter; everything else (amplitudes,
    relaxation rates, phases) is frame-invariant and passes through.
    """
    return {name: float(value) + offsets.get(name, 0.0) for name, value in values.items()}
