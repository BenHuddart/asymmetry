"""Heuristics for spotting a transverse-field alpha-calibration run.

Alpha (the detector-balance parameter) is measured once, on a dedicated
*calibration run* — a small transverse field applied to a paramagnetic or
non-relaxing sample, where the muon precesses and the asymmetry oscillates
symmetrically about zero exactly when alpha balances the forward/backward
detector efficiencies (WiMDA's diamagnetic estimate; see
:func:`asymmetry.core.transform.asymmetry.estimate_alpha_detailed`). The
grouping UI wants to *point the user at* the most likely calibration run out
of everything loaded, without picking one for them irreversibly.

This module is the pure-core classifier the GUI dropdown highlights from. It is
deliberately conservative and metadata-only: it never touches histograms and it
never *infers* geometry from the field magnitude alone (a genuine TF run can sit
near zero field; a high longitudinal field is not a calibration run). A run is
flagged as a likely weak-TF calibration run when **either**

* the loader classified its applied-field geometry as ``"Transverse"`` (the
  structured NeXus ``magnetic_field_state`` or an explicit ``TF``/``tra`` token
  in the PSI free text — see
  :func:`asymmetry.core.io.base.field_direction_from_text`), **or**
* the run title / comment carries an explicit transverse-field token
  (``TF``/``wTF``/``transverse``) — this catches runs whose structured geometry
  the loader could not populate,

**and**, when an applied field magnitude is recorded, it sits in the
weak-to-moderate transverse window :data:`WEAK_TF_FIELD_RANGE_GAUSS`. A run with
an explicit transverse token but no recorded field is still flagged (the token
is the stronger signal); a run whose only evidence is a field magnitude in the
window is *not* flagged, because the magnitude alone is ambiguous (that is the
field-geometry policy the loaders already follow).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

#: Applied-field window (Gauss, on ``abs(field)``) a *weak* transverse-field
#: calibration run conventionally sits in. The lower bound rejects a nominal
#: zero-field run that merely carries a stray transverse token; the upper bound
#: rejects a high-field measurement run. These are advisory bounds for
#: *highlighting* a likely calibration run, not a hard classification — the user
#: can always pick any run from the dropdown.
WEAK_TF_FIELD_RANGE_GAUSS: tuple[float, float] = (5.0, 500.0)

#: Explicit transverse-field tokens in free text (title / comment). Mirrors the
#: transverse pattern in :data:`asymmetry.core.io.base._FIELD_DIRECTION_TAGS`
#: (``TF``/``wTF``/``transverse``) and additionally admits the ``tra`` stem the
#: PSI setup free text sometimes uses. The leading ``\b`` keeps it off substrings
#: of sample names.
_TF_TEXT_PATTERN = re.compile(
    r"\b(?:transverse|tra(?:ns)?|w?tf(?=\d|\b))",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TFCalibrationVerdict:
    """Whether a run looks like a weak-TF alpha-calibration run, and why.

    ``is_candidate`` is the flag the GUI highlights from; ``reason`` is a short
    human-readable justification for a tooltip / log line; ``field_gauss`` is the
    applied field magnitude the verdict considered (``None`` when unrecorded).
    """

    is_candidate: bool
    reason: str
    field_gauss: float | None = None


def _as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None  # drop NaN


def _has_tf_text(*texts: object) -> bool:
    """Return whether any free-text field carries an explicit transverse token."""
    blob = " ".join(str(t) for t in texts if t)
    return bool(_TF_TEXT_PATTERN.search(blob))


def classify_tf_calibration_run(metadata: dict[str, Any] | None) -> TFCalibrationVerdict:
    """Classify a run's metadata as a likely weak-TF calibration run.

    Parameters
    ----------
    metadata
        A run's metadata mapping. The keys consulted are ``field_direction``
        (the loader's structured geometry classification: ``"Transverse"`` /
        ``"Longitudinal"`` / ``"Zero field"`` / ``""``), ``field`` (applied field
        magnitude in Gauss), and the ``title`` / ``comment`` free text.

    Returns
    -------
    TFCalibrationVerdict
        ``is_candidate`` is ``True`` when the run shows explicit transverse-field
        evidence (structured geometry or a text token) and any recorded field
        magnitude sits in :data:`WEAK_TF_FIELD_RANGE_GAUSS`.
    """
    metadata = metadata if isinstance(metadata, dict) else {}

    direction = str(metadata.get("field_direction", "")).strip().lower()
    title = metadata.get("title")
    comment = metadata.get("comment")
    field = _as_float(metadata.get("field"))

    explicit_transverse = direction == "transverse" or _has_tf_text(title, comment)
    explicit_other = direction in ("longitudinal", "zero field")

    # An explicit non-transverse geometry vetoes the run outright, even if a
    # stray "tf" appears in the sample text.
    if explicit_other and not (direction == "transverse"):
        return TFCalibrationVerdict(
            is_candidate=False,
            reason=f"{direction} geometry — not a transverse-field calibration run",
            field_gauss=field,
        )

    if not explicit_transverse:
        return TFCalibrationVerdict(
            is_candidate=False,
            reason="no transverse-field evidence in metadata",
            field_gauss=field,
        )

    lo, hi = WEAK_TF_FIELD_RANGE_GAUSS
    if field is not None and not (lo <= abs(field) <= hi):
        return TFCalibrationVerdict(
            is_candidate=False,
            reason=(
                f"transverse field {abs(field):.0f} G outside the weak-TF "
                f"window [{lo:.0f}, {hi:.0f}] G"
            ),
            field_gauss=field,
        )

    if field is not None:
        reason = f"transverse field {abs(field):.0f} G (weak-TF calibration window)"
    else:
        reason = "transverse-field geometry (weak-TF calibration run)"
    return TFCalibrationVerdict(is_candidate=True, reason=reason, field_gauss=field)


def best_calibration_run_index(metadatas: list[dict[str, Any] | None]) -> int | None:
    """Return the index of the best weak-TF calibration candidate, or ``None``.

    Scans *metadatas* in order and returns the index of the first run classified
    as a weak-TF calibration candidate. When several qualify, the one whose
    recorded field sits closest to the centre of
    :data:`WEAK_TF_FIELD_RANGE_GAUSS` (in log-field space, so 10 G and 1000 G are
    treated symmetrically about the geometric-mean centre) is preferred; a
    candidate with a recorded field always beats one with none. ``None`` when no
    run qualifies.
    """
    lo, hi = WEAK_TF_FIELD_RANGE_GAUSS
    centre = (lo * hi) ** 0.5  # geometric mean of the window

    best_index: int | None = None
    best_key: tuple[int, float] | None = None
    for index, metadata in enumerate(metadatas):
        verdict = classify_tf_calibration_run(metadata)
        if not verdict.is_candidate:
            continue
        if verdict.field_gauss is None:
            # No field recorded: rank behind any candidate with a field.
            key = (1, 0.0)
        else:
            distance = abs(_log(abs(verdict.field_gauss)) - _log(centre))
            key = (0, distance)
        if best_key is None or key < best_key:
            best_key = key
            best_index = index
    return best_index


def _log(value: float) -> float:
    """``log`` guarded for non-positive inputs (returns a large sentinel)."""
    import math

    return math.log(value) if value > 0.0 else -1.0e30


__all__ = [
    "WEAK_TF_FIELD_RANGE_GAUSS",
    "TFCalibrationVerdict",
    "classify_tf_calibration_run",
    "best_calibration_run_index",
]
