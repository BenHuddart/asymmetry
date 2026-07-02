"""Per-member fit-quality contract shared across series engines.

A batch/series fit produces one :class:`~asymmetry.core.fitting.engine.FitResult`
per member.  Both the block-separable F-B path
(:func:`~asymmetry.core.fitting.series.fit_asymmetry_series`) and the grouped
time-domain path
(:func:`~asymmetry.core.fitting.grouped_time_domain.fit_grouped_series`) turn
those results into the same :class:`MemberQuality` record so trend tables and
plots read one shape regardless of representation — and a future engine merge has
a single seam.

The record carries the reduced χ², the per-parameter HESSE σ, and a set of
advisory *quality flags*.  Flags are diagnostic only: they never mutate a
member's trend-inclusion gate (``FitSlot.include_in_trend``) — per design
decision D3, exclusion is always the user's call, never automatic.

Flag vocabulary (:data:`MEMBER_QUALITY_FLAGS`):

* ``"failed"`` — the minimiser did not converge (``FitResult.success`` is False).
* ``"large_rel_err"`` — a free parameter's ``|σ / value|`` exceeds
  :data:`DEFAULT_REL_ERR_THRESHOLD` (or its value collapsed to ~0 with a finite
  σ): the data barely constrained it.
* ``"bound_pinned"`` — a free parameter sits on a finite ``min``/``max`` bound
  (reuses :func:`parameters_at_bound`).
* ``"spurious_reseeded"`` — the member landed on the spurious near-transition
  branch (amplitude collapse / frequency discontinuity), whether or not a
  reseed recovered it.  Only the F-B series path, which knows the scan trend,
  sets this.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

#: The full advisory flag vocabulary (see module docstring).
MEMBER_QUALITY_FLAGS: tuple[str, ...] = (
    "failed",
    "large_rel_err",
    "bound_pinned",
    "spurious_reseeded",
)

#: A free parameter whose relative HESSE error ``|σ / value|`` exceeds this is
#: essentially undetermined by the data (σ larger than the value itself). A value
#: that collapsed to ~0 with a finite σ is flagged regardless of the ratio.
DEFAULT_REL_ERR_THRESHOLD = 1.0

#: Relative tolerance for "a free parameter sits on a finite bound". Scale-aware
#: (see :func:`parameters_at_bound`) so it fires on both an exact rail and a
#: near-rail without flagging interior fits.
_BOUND_REL_TOL = 1.0e-3

#: Below this magnitude a fitted value is treated as "collapsed to ~0" for the
#: relative-error test, so a finite σ on it always flags rather than dividing by
#: a near-zero denominator.
_ZERO_VALUE_ATOL = 1.0e-9


def parameters_at_bound(parameters: Any, *, rel_tol: float = _BOUND_REL_TOL) -> list[str]:
    """Names of *free* parameters whose fitted value sits on a finite bound.

    A free (non-fixed, non-tied, non-linked-follower) parameter pinned at its
    ``min`` or ``max`` is a sign the fit is poorly constrained: the data did not
    determine it and the optimiser parked it on the boundary (maleic A_Mu→0,
    FµF 17319 r→2.5 Å, ZF-KT Δ→1.0, weak-signal rails). Such a fit can still
    report "converged" with finite errors, so this drives an advisory badge
    alongside the χ² verdict.

    Parameters the user *fixed* or tied are excluded by ranging over
    :attr:`ParameterSet.free_parameters` only — they are *meant* to hold a value.
    Only *finite* bounds are considered. The tolerance is scale-relative (a value
    within ``rel_tol`` of the bound, floored at ``rel_tol`` in absolute terms via
    the ``max(…, 1.0)`` scale) so an exact rail and a near-rail both fire while
    interior fits do not. Qt-free and headless-testable.
    """
    free = getattr(parameters, "free_parameters", None)
    if free is None:
        return []
    names: list[str] = []
    for p in free:
        try:
            value = float(p.value)
        except (TypeError, ValueError, AttributeError):
            continue
        if not math.isfinite(value):
            continue
        for bound in (getattr(p, "min", -math.inf), getattr(p, "max", math.inf)):
            if not math.isfinite(bound):
                continue
            if abs(value - bound) <= rel_tol * max(abs(bound), abs(value), 1.0):
                names.append(str(p.name))
                break
    return names


def large_relative_error_params(
    fit_result: Any, *, threshold: float = DEFAULT_REL_ERR_THRESHOLD
) -> list[str]:
    """Names of *free* parameters whose HESSE σ swamps their fitted value.

    ``|σ / value| > threshold`` — or a value that collapsed to ~0 while carrying
    a finite σ — means the data barely pinned the parameter (the run-2949 EuO
    A₁ = 967 ± 316 % / run-2947 A₁ ≈ 1.4e-5 pathologies). Fixed and tied
    parameters carry no free error and are skipped.
    """
    parameter_set = getattr(fit_result, "parameters", None)
    free = getattr(parameter_set, "free_parameters", None)
    if free is None:
        return []
    uncertainties = getattr(fit_result, "uncertainties", {}) or {}
    names: list[str] = []
    for p in free:
        name = str(p.name)
        sigma = uncertainties.get(name)
        if sigma is None:
            continue
        try:
            sigma = float(sigma)
            value = float(p.value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(sigma) or sigma <= 0.0:
            continue
        if not math.isfinite(value) or abs(value) <= _ZERO_VALUE_ATOL:
            # Value collapsed to ~0 (or non-finite) with a real σ → undetermined.
            names.append(name)
            continue
        if sigma / abs(value) > threshold:
            names.append(name)
    return names


@dataclass
class MemberQuality:
    """Per-member fit-quality summary — the shared trend-gating contract.

    ``chi2_reduced`` is ``None`` when no meaningful reduced χ² is available
    (non-finite or a computed/model-less member). ``param_errors`` is the HESSE
    σ per parameter. ``quality_flags`` is a subset of :data:`MEMBER_QUALITY_FLAGS`
    and is advisory only — it never changes trend inclusion.
    """

    chi2_reduced: float | None = None
    param_errors: dict[str, float] = field(default_factory=dict)
    quality_flags: set[str] = field(default_factory=set)

    def to_payload(self) -> dict:
        """Return a JSON-serialisable projection (flags as a sorted list)."""
        return {
            "chi2_reduced": self.chi2_reduced,
            "param_errors": dict(self.param_errors),
            "quality_flags": sorted(self.quality_flags),
        }

    @classmethod
    def from_payload(cls, data: dict | None) -> MemberQuality:
        """Reconstruct a :class:`MemberQuality` from :meth:`to_payload` output."""
        if not isinstance(data, dict):
            return cls()
        raw_chi2 = data.get("chi2_reduced")
        try:
            chi2 = None if raw_chi2 is None else float(raw_chi2)
        except (TypeError, ValueError):
            chi2 = None
        raw_errors = data.get("param_errors")
        errors: dict[str, float] = {}
        if isinstance(raw_errors, dict):
            for name, value in raw_errors.items():
                try:
                    errors[str(name)] = float(value)
                except (TypeError, ValueError):
                    continue
        raw_flags = data.get("quality_flags")
        flags = {str(f) for f in raw_flags} if isinstance(raw_flags, (list, set, tuple)) else set()
        return cls(chi2_reduced=chi2, param_errors=errors, quality_flags=flags)


def _reduced_chi2(fit_result: Any) -> float | None:
    """The member's reduced χ², or ``None`` when non-finite / unavailable."""
    try:
        value = float(getattr(fit_result, "reduced_chi_squared", 0.0) or 0.0)
    except (TypeError, ValueError):
        return None
    if value <= 0.0 or not math.isfinite(value):
        return None
    return value


def member_quality_flags(
    fit_result: Any,
    *,
    extra_flags: Any = (),
    rel_err_threshold: float = DEFAULT_REL_ERR_THRESHOLD,
) -> set[str]:
    """Derive the advisory quality flags for a single member's fit result.

    ``extra_flags`` lets a caller that has trend context (the F-B series engine)
    inject ``"spurious_reseeded"``; unknown flag names are ignored so the set
    stays within :data:`MEMBER_QUALITY_FLAGS`.
    """
    flags: set[str] = set()
    if not bool(getattr(fit_result, "success", False)):
        flags.add("failed")
    if large_relative_error_params(fit_result, threshold=rel_err_threshold):
        flags.add("large_rel_err")
    if parameters_at_bound(getattr(fit_result, "parameters", None)):
        flags.add("bound_pinned")
    for flag in extra_flags:
        if flag in MEMBER_QUALITY_FLAGS:
            flags.add(str(flag))
    return flags


def assess_member_quality(
    fit_result: Any,
    *,
    extra_flags: Any = (),
    rel_err_threshold: float = DEFAULT_REL_ERR_THRESHOLD,
) -> MemberQuality:
    """Build the :class:`MemberQuality` record for one member's fit result."""
    return MemberQuality(
        chi2_reduced=_reduced_chi2(fit_result),
        param_errors={
            str(k): float(v)
            for k, v in (getattr(fit_result, "uncertainties", {}) or {}).items()
            if _is_finite_number(v)
        },
        quality_flags=member_quality_flags(
            fit_result, extra_flags=extra_flags, rel_err_threshold=rel_err_threshold
        ),
    )


def _is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
