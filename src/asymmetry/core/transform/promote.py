"""Promote fitted count-domain calibrations into the run grouping.

These mirror the deadtime *Send-to-Group* pattern in
:func:`asymmetry.core.transform.deadtime.promote_deadtime_to_grouping`: a value
measured from the sample's own data (α, time-zero, flat background) is written
into the persisted grouping so the next reduction applies it. Each is
**suggest-only** — the GUI invokes it on an explicit button press, never
automatically — and each returns a ``{"before": …, "after": …}`` dict so the
panel can disclose the change before the user commits.

The textbook grounding is the same for all three (and for deadtime): α "is
dependent on sample position and detector efficiencies" and "needs to be
determined for each sample", and the analysis time-zero "is the beginning of
the spin dynamics", not necessarily the prompt-peak feature. These are
per-sample, per-setup calibrations, so a value fitted from the run's counts is
a legitimate calibration to persist.
"""

from __future__ import annotations


def promote_alpha_to_grouping(
    grouping: dict,
    alpha: float,
    *,
    alpha_error: float | None = None,
    reference_run: int | None = None,
) -> dict[str, dict[str, float]]:
    """Write a count-fit detector balance ``alpha`` into the grouping.

    The α-free forward/backward count fit (``count_domain.fit_fb_alpha``)
    estimates the detector balance with the full Poisson likelihood and a
    proper (α, amplitude) covariance. This promotes that value to
    ``grouping["alpha"]`` with ``alpha_method="count_fit"`` provenance,
    mirroring the three grouping-dialog estimators that already write the same
    keys. Suggest-only; returns the before/after α for the GUI to display.
    """
    before = {"alpha": _as_float(grouping.get("alpha"), 1.0)}
    new_alpha = float(alpha)
    grouping["alpha"] = new_alpha
    grouping["alpha_method"] = "count_fit"
    if alpha_error is not None:
        grouping["alpha_error"] = float(alpha_error)
    if reference_run is not None:
        grouping["alpha_reference_run"] = int(reference_run)
    return {"before": before, "after": {"alpha": new_alpha}}


def promote_t0_to_grouping(
    grouping: dict,
    t0_us: float,
    *,
    bin_width_us: float,
    reference_run: int | None = None,
    group_id: int | None = None,
) -> dict:
    """Write a fitted count-fit time-zero offset (µs) into the grouping ``t0_bin``.

    The count-fit ``t0`` nuisance is a continuous µs offset relative to the
    grouping's current time-zero, fitted per group. The persisted authority
    (``grouping["t0_bin"]``) is an integer bin index applied run-wide, so the
    promotion:

    - converts the offset to bins via ``bin_width_us`` and rounds to the
      nearest bin, **disclosing the sub-bin residual** (``residual_us`` in the
      return) the integer t0_bin cannot represent;
    - applies the *fitted group's* value run-wide (``t0_bin`` is run-level, not
      per-group) — the caller's suggest dialog says so, and ``group_id`` is
      echoed back for that message.

    Suggest-only. Provenance keys mirror α's (``t0_method`` /
    ``t0_reference_run``). Raises ``ValueError`` for a non-positive bin width.
    """
    bw = float(bin_width_us)
    if bw <= 0.0:
        raise ValueError("bin width must be positive to convert a t0 offset to bins")

    current = _as_int(grouping.get("t0_bin"), 0)
    delta_bins = int(round(float(t0_us) / bw))
    new_bin = max(0, current + delta_bins)
    # The disclosed residual is the part of the offset the integer t0_bin cannot
    # represent. Compute it from the delta actually applied (after the ≥0 clamp),
    # not the rounded delta — otherwise a clamp would understate the lost shift.
    applied_bins = new_bin - current
    residual_us = float(t0_us) - applied_bins * bw

    grouping["t0_bin"] = new_bin
    grouping["t0_method"] = "count_fit"
    if reference_run is not None:
        grouping["t0_reference_run"] = int(reference_run)
    return {
        "before": {"t0_bin": current},
        "after": {"t0_bin": new_bin},
        "residual_us": residual_us,
        "group_id": group_id,
    }


def promote_background_to_grouping(
    grouping: dict,
    *,
    forward: float | None = None,
    backward: float | None = None,
    reference_run: int | None = None,
) -> dict[str, dict[str, float]]:
    """Write fitted flat count backgrounds into the grouping ``fixed`` mode.

    The count-fit ``background`` (and ``background_b`` for the backward side of
    a forward/backward fit) is a flat per-bin count measured on the raw
    histogram — the same quantity the grouping's ``fixed`` background mode
    stores. This promotes the fitted value(s) into
    ``grouping["background_fixed_values"]`` (a ``[forward, backward]`` pair) and
    sets ``background_mode="fixed"`` with ``background_method="count_fit"``
    provenance. A side left as ``None`` keeps its existing fixed value (a
    single-histogram fit promotes only the side it fitted). Suggest-only;
    returns the before/after pair for the GUI to display.
    """
    existing_f, existing_b = _existing_fixed_pair(grouping)
    new_f = float(forward) if forward is not None else existing_f
    new_b = float(backward) if backward is not None else existing_b

    before = {"forward": existing_f, "backward": existing_b}
    grouping["background_fixed_values"] = [new_f, new_b]
    grouping["background_mode"] = "fixed"
    # Self-enable the correction (the reduction gate keys off
    # ``background_correction``), mirroring the deadtime promote's
    # ``deadtime_correction = True`` — otherwise the promoted background is inert
    # on the next reduction despite the "Re-reduce to apply" message.
    grouping["background_correction"] = True
    grouping["background_method"] = "count_fit"
    if reference_run is not None:
        grouping["background_reference_run"] = int(reference_run)
    return {"before": before, "after": {"forward": new_f, "backward": new_b}}


def _existing_fixed_pair(grouping: dict) -> tuple[float, float]:
    """Return the current fixed background pair, or ``(0.0, 0.0)`` if unset."""
    for key in ("background_fixed_values", "background_fix", "bkg_fix"):
        value = grouping.get(key)
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return _as_float(value[0], 0.0), _as_float(value[1], 0.0)
    return 0.0, 0.0


def _as_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
