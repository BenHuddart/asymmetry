"""Wizard criteria migrations recover sample counts without retaining redundant scores."""

from copy import deepcopy

import pytest

from asymmetry.core.fitting.information_criteria import InformationCriteria, SelectionMetric
from asymmetry.core.project.schema import migrate_to_current


@pytest.mark.parametrize(
    "row,expected",
    [
        (
            {
                "fit_result": {"chi_squared": 1.0},
                "aic": 7.0,
                "aicc": 31.0,
                "bic": 5.828313737302301,
            },
            5,
        ),
        (
            {
                "fit_result": {"chi_squared": 1.0},
                "aic": 7.0,
                "aicc": None,
                "bic": 5.1588830833596715,
                "selected_score": 7.0,
            },
            4,
        ),
        (
            {
                "fit_results_by_run": {"1": {"chi_squared": 10.0}, "2": {"chi_squared": 20.0}},
                "aic": 42.0,
                "bic": 50.40718428997293,
            },
            30,
        ),
        ({"fit_result": {"chi_squared": 1.0}, "aic": 1.0, "bic": 1.0}, None),
        ({"prescreen_only": True, "fit_results_by_run": {}, "aic": 48.0, "bic": 50.7}, None),
        (
            {"fit_result": {"chi_squared": float("inf")}, "aic": float("inf"), "bic": float("inf")},
            None,
        ),
    ],
)
def test_migration_recovers_samples_and_drops_criteria(row, expected):
    original = {"schema_version": 26, "nested": [{"assessment": row}]}
    before = deepcopy(original)
    migrated = migrate_to_current(original)
    restored = migrated["nested"][0]["assessment"]
    assert migrated["schema_version"] == 27
    assert restored["sample_count"] == expected
    assert not set(restored) & {"aic", "aicc", "bic", "selected_score"}
    if expected is not None:
        statistic = (
            row["fit_result"]["chi_squared"]
            if "fit_result" in row
            else sum(item["chi_squared"] for item in row["fit_results_by_run"].values())
        )
        k = round((row["aic"] - statistic) / 2)
        criteria = InformationCriteria(statistic, k, restored["sample_count"])
        assert criteria.value(SelectionMetric.BIC) == pytest.approx(row["bic"], abs=1e-9)
        if row.get("aicc") is not None:
            assert criteria.value(SelectionMetric.AICC) == pytest.approx(row["aicc"], abs=1e-9)
        elif "aicc" in row:
            assert criteria.value(SelectionMetric.AICC) is None
    assert original == before
    assert migrate_to_current(migrated) == migrated


def test_recorded_analysed_points_take_precedence():
    payload = {
        "schema_version": 26,
        "cache": {
            "fingerprint": {"analysed_points": 123},
            "assessments": [
                {"fit_result": {"chi_squared": 1.0}, "aic": 7.0, "bic": 5.1588830833596715}
            ],
        },
    }
    migrated = migrate_to_current(payload)
    assert migrated["cache"]["assessments"][0]["sample_count"] == 123
