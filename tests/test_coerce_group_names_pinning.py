"""Behaviour-pinning for the unified ``optional_float`` / ``group_names`` helpers
(reconciliation N2).

Two small helpers were duplicated across ``core/fourier/spectrum.py`` and
``core/maxent/engine.py``:

- ``_group_names`` was byte-identical in both — pinned against a verbatim copy.
- ``_optional_float`` was **not** identical: the maxent copy added an
  ``np.isfinite`` guard the spectrum copy lacked, so the two diverged on
  non-finite numeric input (``inf``/``nan``/``"inf"``/``"nan"``). The
  reconciliation converges on the finite-checking (maxent) behaviour — the
  strictly safer choice, since the only valid inputs for the fields it parses
  (times, fields, frequencies) are finite. This module pins that decision:
  exact equality with the maxent copy everywhere, and equality with the
  spectrum copy for every input *except* the documented non-finite divergence.

The GUI's ``_parse_optional_float`` (``maxent_panel.py``) is a text-parsing
variant — a different concern — and is intentionally left in place.
"""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from asymmetry.core.data.dataset import Run
from asymmetry.core.transform.grouping import group_names
from asymmetry.core.utils.coerce import optional_float


def _legacy_optional_float_spectrum(value: object) -> float | None:
    """Verbatim copy of the removed ``spectrum._optional_float`` (no finite check)."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _legacy_optional_float_maxent(value: object) -> float | None:
    """Verbatim copy of the removed ``maxent._optional_float`` (finite check)."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


_FINITE_INPUTS = [
    None,
    0,
    1,
    -3,
    1.5,
    -2.25,
    "2.5",
    "  4.0 ",
    "abc",
    "",
    [],
    (1, 2),
    {},
    object(),
    True,
    False,
    Decimal("3.14"),
    np.float64(7.0),
    np.int64(9),
]

_NONFINITE_INPUTS = [
    float("inf"),
    float("-inf"),
    float("nan"),
    "inf",
    "-inf",
    "nan",
    np.float64("inf"),
    np.float64("nan"),
]


@pytest.mark.parametrize("value", _FINITE_INPUTS)
def test_optional_float_matches_both_copies_on_finite(value: object) -> None:
    """For finite/None/invalid inputs the old copies agreed — pin both."""
    result = optional_float(value)
    assert result == _legacy_optional_float_spectrum(value) or (
        result is None and _legacy_optional_float_spectrum(value) is None
    )
    assert result == _legacy_optional_float_maxent(value) or (
        result is None and _legacy_optional_float_maxent(value) is None
    )


@pytest.mark.parametrize("value", _FINITE_INPUTS + _NONFINITE_INPUTS)
def test_optional_float_matches_maxent_everywhere(value: object) -> None:
    """The unified helper follows the finite-checking (maxent) copy exactly."""
    result = optional_float(value)
    expected = _legacy_optional_float_maxent(value)
    assert (result is None and expected is None) or result == expected


@pytest.mark.parametrize("value", _NONFINITE_INPUTS)
def test_optional_float_diverges_from_spectrum_on_nonfinite(value: object) -> None:
    """Documented divergence: non-finite numbers now map to None (was passed through)."""
    assert optional_float(value) is None
    # The removed spectrum copy would have returned a non-finite float here.
    legacy = _legacy_optional_float_spectrum(value)
    assert legacy is not None and not np.isfinite(legacy)


# --- group_names -----------------------------------------------------------


def _legacy_group_names_spectrum(run: Run) -> dict[int, str]:
    """Verbatim copy of the removed ``spectrum._group_names`` body.

    This is the behaviour the unified ``group_names`` adopts: an explicitly
    ``None`` name falls back to ``"Group <id>"``.
    """
    grouping = run.grouping if isinstance(run.grouping, dict) else {}
    groups = grouping.get("groups") if isinstance(grouping, dict) else None
    if not isinstance(groups, dict):
        return {}
    raw_names = grouping.get("group_names")
    names = raw_names if isinstance(raw_names, dict) else {}
    resolved: dict[int, str] = {}
    for raw_id in groups:
        try:
            gid = int(raw_id)
        except (TypeError, ValueError):
            continue
        name = names.get(gid, names.get(str(gid)))
        resolved[gid] = str(name) if name is not None else f"Group {gid}"
    return resolved


def _legacy_group_names_maxent(run: Run) -> dict[int, str]:
    """Verbatim copy of the removed ``maxent._group_names`` body.

    Differs from the spectrum copy on exactly one input: an explicitly
    ``None`` name stringifies to ``"None"`` here (``str(names.get(...))``),
    where the spectrum copy falls back to ``"Group <id>"``. Group names always
    arrive as strings through the grouping dialog, so this never surfaces for
    real data; the unification keeps the spectrum (sensible-fallback) behaviour.
    """
    grouping = run.grouping if isinstance(run.grouping, dict) else {}
    groups = grouping.get("groups") if isinstance(grouping, dict) else None
    if not isinstance(groups, dict):
        return {}
    raw_names = grouping.get("group_names")
    names = raw_names if isinstance(raw_names, dict) else {}
    resolved: dict[int, str] = {}
    for raw_id in groups:
        try:
            gid = int(raw_id)
        except (TypeError, ValueError):
            continue
        resolved[gid] = str(names.get(gid, names.get(str(gid), f"Group {gid}")))
    return resolved


_GROUPINGS = [
    None,
    {},
    {"groups": None},
    {"groups": {}},
    {"groups": {1: [1, 2], 2: [3, 4]}},
    {"groups": {"1": [1], "2": [2]}},
    {"groups": {1: [1], 2: [2]}, "group_names": {1: "Forward", 2: "Backward"}},
    {"groups": {1: [1], 2: [2]}, "group_names": {"1": "Fwd"}},
    {"groups": {1: [1], "bad": [2], 3: [3]}, "group_names": {3: None}},
    {"groups": {1: [1]}, "group_names": "not-a-dict"},
]


def _run_with_grouping(grouping: object) -> Run:
    run = Run(run_number=1, grouping=grouping if grouping is not None else {})
    if grouping is None:
        run.grouping = None  # type: ignore[assignment]
    return run


@pytest.mark.parametrize("grouping", _GROUPINGS)
def test_group_names_matches_spectrum_copy(grouping: object) -> None:
    """The unified helper reproduces the spectrum copy across the input matrix."""
    run = _run_with_grouping(grouping)
    assert group_names(run) == _legacy_group_names_spectrum(run)


@pytest.mark.parametrize("grouping", _GROUPINGS)
def test_group_names_matches_maxent_copy_except_explicit_none(grouping: object) -> None:
    """Agrees with the maxent copy on every input that has no explicit-None name."""
    run = _run_with_grouping(grouping)
    raw = grouping.get("group_names") if isinstance(grouping, dict) else None
    has_explicit_none = isinstance(raw, dict) and any(v is None for v in raw.values())
    if not has_explicit_none:
        assert group_names(run) == _legacy_group_names_maxent(run)


def test_group_names_explicit_none_follows_spectrum_fallback() -> None:
    """Documented divergence: explicit-None name → 'Group <id>', not 'None'."""
    run = _run_with_grouping({"groups": {3: [3]}, "group_names": {3: None}})
    assert group_names(run) == {3: "Group 3"}
    assert _legacy_group_names_maxent(run) == {3: "None"}
