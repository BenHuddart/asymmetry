"""Unit tests for the shared fit-parameter display formatters."""

from __future__ import annotations

import math

import pytest

from asymmetry.gui.utils.formatting import (
    format_param_label,
    format_value_error,
    format_value_uncertainty,
)


@pytest.mark.parametrize(
    ("value", "error", "expected"),
    [
        # One significant digit while the error's leading digit is 3–9 …
        (35.8, 0.5, "35.8(5)"),
        (1.24, 0.03, "1.24(3)"),
        (2.85, 0.043, "2.85(4)"),
        (0.048, 0.0032, "0.048(3)"),
        (0.048, 0.032, "0.05(3)"),
        # … two while it is 1 or 2, where a single digit would round away a
        # large share of the uncertainty.
        (2.85, 0.13, "2.85(13)"),
        (1.2345, 0.0021, "1.2345(21)"),
        # An error bigger than the value still sets the precision.
        (0.048, 0.32, "0.0(3)"),
        (-35.8, 0.5, "-35.8(5)"),
        # No usable error: four significant digits of the value alone.
        (35.8123, None, "35.81"),
        (35.8123, 0.0, "35.81"),
        (35.8123, float("nan"), "35.81"),
    ],
)
def test_format_value_uncertainty(value: float, error: float | None, expected: str) -> None:
    assert format_value_uncertainty(value, error) == expected


def test_format_value_uncertainty_rounds_the_value_to_the_errors_last_place() -> None:
    # 0.043 keeps one digit, so both value and error round to two decimals.
    assert format_value_uncertainty(2.8549, 0.0432) == "2.85(4)"
    # 0.13 keeps two, and the value follows to the same place.
    assert format_value_uncertainty(2.8549, 0.1324) == "2.85(13)"


def test_format_value_error_matches_the_value_precision_to_the_error() -> None:
    assert format_value_error(0.4667, 0.0016) == "0.4667 ± 0.0016"
    assert format_value_error(0.4667, 0.0) == "0.4667"
    assert format_value_error(math.nan, 0.1) == "—"


def test_format_param_label_uses_the_registry_symbol_and_unit() -> None:
    assert format_param_label("Lambda") == "λ (µs⁻¹)"
    assert format_param_label("not_a_parameter") == "not_a_parameter"
