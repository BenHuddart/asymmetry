"""Tests for the weak-TF alpha-calibration-run heuristic (pure core)."""

from __future__ import annotations

import pytest

from asymmetry.core.data.calibration import (
    WEAK_TF_FIELD_RANGE_GAUSS,
    best_calibration_run_index,
    classify_tf_calibration_run,
)


def test_structured_transverse_geometry_in_window_is_candidate() -> None:
    verdict = classify_tf_calibration_run({"field_direction": "Transverse", "field": 100.0})
    assert verdict.is_candidate
    assert verdict.field_gauss == pytest.approx(100.0)
    assert "100" in verdict.reason


def test_transverse_token_in_title_is_candidate() -> None:
    verdict = classify_tf_calibration_run({"title": "FeSe 9p4 TF100", "field": 100.0})
    assert verdict.is_candidate


def test_tra_token_in_comment_is_candidate() -> None:
    verdict = classify_tf_calibration_run({"comment": "Transverse calibration", "field": 50.0})
    assert verdict.is_candidate


def test_transverse_token_without_field_is_still_candidate() -> None:
    """The explicit token is the stronger signal; a missing field does not veto."""
    verdict = classify_tf_calibration_run({"field_direction": "Transverse"})
    assert verdict.is_candidate
    assert verdict.field_gauss is None


def test_field_magnitude_alone_is_not_a_candidate() -> None:
    """A field in the window with no transverse evidence must not be flagged —
    the magnitude alone is ambiguous (the loaders' field-geometry policy)."""
    verdict = classify_tf_calibration_run({"field": 100.0})
    assert not verdict.is_candidate
    assert "no transverse-field evidence" in verdict.reason


def test_high_transverse_field_outside_window_is_rejected() -> None:
    verdict = classify_tf_calibration_run({"field_direction": "Transverse", "field": 5000.0})
    assert not verdict.is_candidate
    assert "outside" in verdict.reason


def test_near_zero_transverse_field_below_window_is_rejected() -> None:
    verdict = classify_tf_calibration_run({"field_direction": "Transverse", "field": 1.0})
    assert not verdict.is_candidate


def test_window_bounds_are_inclusive() -> None:
    lo, hi = WEAK_TF_FIELD_RANGE_GAUSS
    assert classify_tf_calibration_run({"field_direction": "Transverse", "field": lo}).is_candidate
    assert classify_tf_calibration_run({"field_direction": "Transverse", "field": hi}).is_candidate


def test_negative_field_uses_magnitude() -> None:
    verdict = classify_tf_calibration_run({"field_direction": "Transverse", "field": -100.0})
    assert verdict.is_candidate


def test_longitudinal_geometry_vetoes_stray_tf_text() -> None:
    """An explicit longitudinal geometry is not a calibration run even if a
    'tf'-looking token appears elsewhere in the metadata."""
    verdict = classify_tf_calibration_run(
        {"field_direction": "Longitudinal", "title": "tfoo sample", "field": 100.0}
    )
    assert not verdict.is_candidate
    assert "longitudinal" in verdict.reason


def test_zero_field_geometry_is_not_a_candidate() -> None:
    verdict = classify_tf_calibration_run({"field_direction": "Zero field", "field": 0.0})
    assert not verdict.is_candidate


def test_non_dict_metadata_is_not_a_candidate() -> None:
    assert not classify_tf_calibration_run(None).is_candidate
    assert not classify_tf_calibration_run("nope").is_candidate  # type: ignore[arg-type]


def test_tf_token_does_not_fire_on_substrings() -> None:
    """'half' must not read as LF, 'tffactor' must not read as TF via the stem."""
    verdict = classify_tf_calibration_run({"title": "tffactor study", "field": 100.0})
    assert not verdict.is_candidate


def test_best_calibration_run_index_prefers_window_centre() -> None:
    """Among candidates, the field closest to the window centre (log space) wins."""
    lo, hi = WEAK_TF_FIELD_RANGE_GAUSS
    centre = (lo * hi) ** 0.5
    metadatas = [
        {"field_direction": "Longitudinal", "field": 3000.0},  # not a candidate
        {"field_direction": "Transverse", "field": lo},  # candidate, edge
        {"field_direction": "Transverse", "field": centre},  # candidate, centre
        {"field": 200.0},  # not a candidate (no evidence)
    ]
    assert best_calibration_run_index(metadatas) == 2


def test_best_calibration_run_index_field_beats_no_field() -> None:
    metadatas = [
        {"field_direction": "Transverse"},  # candidate, no field
        {"field_direction": "Transverse", "field": 100.0},  # candidate, with field
    ]
    assert best_calibration_run_index(metadatas) == 1


def test_best_calibration_run_index_returns_none_when_no_candidate() -> None:
    metadatas = [
        {"field_direction": "Longitudinal", "field": 3000.0},
        {"field": 100.0},
        None,
    ]
    assert best_calibration_run_index(metadatas) is None


def test_best_calibration_run_index_picks_first_when_no_fields() -> None:
    metadatas = [
        {"field_direction": "Transverse"},
        {"title": "wTF sample"},
    ]
    # Both are candidates with no field; the first wins (stable order).
    assert best_calibration_run_index(metadatas) == 0
