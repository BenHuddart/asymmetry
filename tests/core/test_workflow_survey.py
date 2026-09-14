"""Tests for :mod:`asymmetry.core.workflow.survey`."""

from __future__ import annotations

from pathlib import Path

import pytest

from asymmetry.core.workflow.survey import survey_folder
from tests.core.conftest import (
    ALL_RUNS,
    CALIBRATION_FIELD_G,
    CALIBRATION_RUN,
    DEADTIME_RUN,
    ZF_RUNS,
    ZF_TEMPERATURES,
)


@pytest.fixture(scope="module")
def survey(workflow_folder: Path):
    return survey_folder(workflow_folder)


def test_survey_lists_every_run_file(survey) -> None:
    assert [row.run_number for row in survey.runs] == list(ALL_RUNS)
    assert not survey.truncated


def test_survey_reads_instrument_and_geometry_metadata(survey) -> None:
    calibration = survey.row(CALIBRATION_RUN)
    assert calibration.instrument == "SIM"
    assert calibration.facility == "ISIS"
    assert calibration.prefix == "SIM"
    assert calibration.field == pytest.approx(CALIBRATION_FIELD_G)
    assert calibration.geometry == "TF"
    assert calibration.n_histograms == 8
    assert calibration.n_points > 0
    assert calibration.bin_width_us == pytest.approx(0.016)


def test_zero_field_runs_report_zf_despite_the_files_tf_stamp(survey) -> None:
    # Every run in the fixture carries field_state "TF"; the zero-field ones
    # must still survey as ZF, or a whole ISIS scan would be mislabelled.
    for run_number in ZF_RUNS:
        row = survey.row(run_number)
        assert row.field_direction == "Transverse"
        assert row.geometry == "ZF"


def test_survey_reports_detector_orientation_and_the_runs_note(survey) -> None:
    # Both are what an agent has to reason from when a file records no field
    # geometry; the orientation is a bank position, never a field direction.
    row = survey.row(CALIBRATION_RUN)
    assert row.detector_orientation == "Longitudinal"
    assert row.notes == f"simulated {row.title}"
    assert row.geometry == "TF"

    data = survey.to_dict()["runs"][0]
    assert data["detector_orientation"] == "Longitudinal"
    assert data["notes"] == row.notes


def test_survey_records_start_time_and_duration(survey) -> None:
    calibration = survey.row(CALIBRATION_RUN)
    assert calibration.start_time == "2024-03-01T09:00:00"
    assert calibration.duration_s == pytest.approx(1800.0)


def test_has_file_deadtime_is_true_only_for_the_run_that_carries_values(survey) -> None:
    flagged = {row.run_number for row in survey.runs if row.has_file_deadtime}
    assert flagged == {DEADTIME_RUN}


def test_survey_finds_the_weak_tf_calibration_candidate(survey) -> None:
    assert [c.run_number for c in survey.calibration_candidates] == [CALIBRATION_RUN]
    assert survey.calibration_candidates[0].best
    assert survey.best_calibration_run == CALIBRATION_RUN


def test_survey_groups_the_zero_field_scan_with_temperature_as_axis(survey) -> None:
    scans = [scan for scan in survey.scans if scan.axis == "temperature"]
    assert len(scans) == 1
    scan = scans[0]
    assert scan.geometry == "ZF"
    assert scan.field == pytest.approx(0.0)
    assert scan.runs == list(ZF_RUNS)
    assert scan.values == pytest.approx(list(ZF_TEMPERATURES))


def test_survey_reports_no_field_scan_when_no_two_runs_share_a_temperature(survey) -> None:
    assert [scan for scan in survey.scans if scan.axis == "field"] == []


def test_survey_round_trips_through_its_dict(survey) -> None:
    data = survey.to_dict()
    assert data["best_calibration_run"] == CALIBRATION_RUN
    assert [row["run_number"] for row in data["runs"]] == list(ALL_RUNS)
    assert data["scans"][0]["axis"] == "temperature"


def test_row_raises_key_error_for_an_absent_run(survey) -> None:
    with pytest.raises(KeyError):
        survey.row(999999)


def test_survey_rejects_a_path_that_is_not_a_directory(tmp_path: Path) -> None:
    missing = tmp_path / "nowhere"
    with pytest.raises(ValueError):
        survey_folder(missing)
