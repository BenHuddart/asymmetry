"""Tests for :mod:`asymmetry.core.workflow.survey`."""

from __future__ import annotations

from pathlib import Path

import pytest

from asymmetry.core.workflow.survey import (
    PRECESSION_SNR_FLOOR,
    PrecessionEvidence,
    calibration_verdict,
    resolve_row_geometry,
    survey_folder,
)
from tests.core.conftest import (
    ALL_RUNS,
    CALIBRATION_FIELD_G,
    CALIBRATION_RUN,
    DEADTIME_RUN,
    DECOUPLING_FIELD_G,
    DECOUPLING_RUN,
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
    candidate = survey.calibration_candidates[0]
    assert candidate.best
    assert survey.best_calibration_run == CALIBRATION_RUN
    # It is the measurement, not the file's TF stamp, that makes it a candidate:
    # every run in this folder carries that stamp.
    assert candidate.source == "measured"
    assert candidate.snr == pytest.approx(survey.row(CALIBRATION_RUN).precession.snr)
    assert "Larmor frequency" in candidate.reason


# -- measured precession ----------------------------------------------------


def test_the_calibration_run_precesses_at_the_larmor_frequency(survey) -> None:
    evidence = survey.row(CALIBRATION_RUN).precession
    assert evidence.state == "larmor"
    assert evidence.larmor_mhz == pytest.approx(0.01355 * CALIBRATION_FIELD_G, rel=1e-3)
    assert evidence.frequency_mhz == pytest.approx(evidence.larmor_mhz, rel=0.25)
    assert evidence.snr > PRECESSION_SNR_FLOOR
    assert evidence.note == ""


def test_the_calibration_run_is_the_strongest_measured_candidate(survey) -> None:
    measured = [c for c in survey.calibration_candidates if c.source == "measured"]
    assert max(measured, key=lambda c: c.snr).run_number == CALIBRATION_RUN


def test_zero_field_runs_report_no_measurement_and_say_why(survey) -> None:
    for run_number in ZF_RUNS:
        evidence = survey.row(run_number).precession
        assert evidence.state is None
        assert evidence.frequency_mhz is None
        assert evidence.snr is None
        assert "the applied field is zero" in evidence.note


def test_a_field_run_whose_spectrum_is_a_plain_decay_reports_no_precession(survey) -> None:
    # The decoupling run records 110 G and carries the same "TF" stamp as
    # everything else, but nothing in it precesses — the case that makes the
    # file's stamp worthless.
    row = survey.row(DECOUPLING_RUN)
    assert row.field == pytest.approx(DECOUPLING_FIELD_G)
    assert row.field_direction == "Transverse"
    assert row.precession.state == "none"
    assert row.precession.frequency_mhz is None
    assert row.precession.larmor_mhz == pytest.approx(0.01355 * DECOUPLING_FIELD_G, rel=1e-3)


def test_a_run_without_measured_precession_is_not_a_calibration_candidate(survey) -> None:
    assert DECOUPLING_RUN not in [c.run_number for c in survey.calibration_candidates]


def test_the_classifier_decides_alone_only_where_nothing_could_be_measured() -> None:
    tf_stamp = {"field": 110.0, "field_state": "TF"}
    silent = PrecessionEvidence(
        state="none", frequency_mhz=None, snr=3.0, larmor_mhz=1.491, note=""
    )
    # The file claims TF and the classifier would agree — but the spectrum was
    # read and there is no precession in it, so the claim loses.
    assert calibration_verdict(tf_stamp, 110.0, silent)[0] is None

    above_nyquist = PrecessionEvidence(
        state=None, frequency_mhz=None, snr=None, larmor_mhz=54.2, note="above Nyquist"
    )
    assert calibration_verdict(tf_stamp, 110.0, above_nyquist)[0] == "metadata"


def test_larmor_precession_outside_the_weak_tf_window_is_not_a_calibration() -> None:
    strong = PrecessionEvidence(
        state="larmor", frequency_mhz=13.6, snr=200.0, larmor_mhz=13.55, note=""
    )
    source, reason = calibration_verdict({"field": 1000.0}, 1000.0, strong)
    assert source is None
    assert "weak-TF window" in reason


def test_the_survey_carries_the_precession_evidence_into_its_dict(survey) -> None:
    data = survey.to_dict()
    calibration = next(row for row in data["runs"] if row["run_number"] == CALIBRATION_RUN)
    assert calibration["precession"] == "larmor"
    assert calibration["precession_snr"] > PRECESSION_SNR_FLOOR
    assert calibration["geometry_source"] == "measured"

    zero_field = next(row for row in data["runs"] if row["run_number"] == ZF_RUNS[0])
    assert zero_field["precession"] is None
    assert "the applied field is zero" in zero_field["precession_note"]

    candidate = data["calibration_candidates"][0]
    assert candidate["source"] == "measured"
    assert candidate["snr"] is not None


# -- geometry ---------------------------------------------------------------


def test_the_calibration_runs_geometry_is_measured_not_read(survey) -> None:
    row = survey.row(CALIBRATION_RUN)
    assert (row.geometry, row.geometry_source) == ("TF", "measured")


def test_measured_precession_gives_a_geometry_a_blank_file_cannot() -> None:
    # An ISIS EMU file from 2024 records no field state at all; without the
    # measurement this run would have no geometry to report.
    blank = {"field": 100.0}
    larmor = PrecessionEvidence(
        state="larmor", frequency_mhz=1.40, snr=418.0, larmor_mhz=1.355, note=""
    )
    silent = PrecessionEvidence(
        state="none", frequency_mhz=None, snr=3.0, larmor_mhz=1.355, note=""
    )
    assert resolve_row_geometry(blank, larmor) == ("TF", "measured")
    assert resolve_row_geometry(blank, silent) == (None, "none")


def test_a_zero_field_run_is_zf_before_any_measurement_is_consulted() -> None:
    unmeasured = PrecessionEvidence(
        state=None, frequency_mhz=None, snr=None, larmor_mhz=0.0, note="the applied field is zero"
    )
    assert resolve_row_geometry({"field": 0.0, "field_state": "TF"}, unmeasured) == (
        "ZF",
        "field",
    )


def test_the_files_own_token_stands_when_nothing_was_measured() -> None:
    unmeasured = PrecessionEvidence(
        state=None, frequency_mhz=None, snr=None, larmor_mhz=None, note="no field recorded"
    )
    assert resolve_row_geometry({"field_state": "LF"}, unmeasured) == ("LF", "file")


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
