"""Tests for :mod:`asymmetry.core.workflow.survey`."""

from __future__ import annotations

from pathlib import Path

import pytest

from asymmetry.core.workflow.survey import (
    PRECESSION_SNR_FLOOR,
    CalibrationCandidate,
    PrecessionEvidence,
    RunRow,
    _group_geometry,
    _scan_groups,
    alpha_steps,
    calibration_verdict,
    coil_geometry,
    notes_scans,
    repeat_sets,
    resolve_row_geometry,
    survey_folder,
    temperature_departures,
)
from tests.core.conftest import (
    ALL_RUNS,
    CALIBRATION_ALPHA,
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


_NOT_MEASURED = PrecessionEvidence(
    state=None, frequency_mhz=None, snr=None, larmor_mhz=None, note="not measured"
)


def _row(
    *,
    run_number: int,
    temperature: float,
    field: float = 0.0,
    instrument: str = "SIM",
    geometry: str | None = "ZF",
    geometry_source: str = "field",
    notes: str = "",
    title: str = "",
    precession: PrecessionEvidence = _NOT_MEASURED,
    bin_width_us: float = 0.016,
    n_periods: int = 1,
    total_events: int = 1000,
    sample_temperature_logged: float | None = None,
) -> RunRow:
    """A :class:`RunRow` for the grouping tests, which read only a few of its fields."""
    return RunRow(
        run_number=run_number,
        file=f"{instrument}{run_number:08d}.nxs",
        prefix=instrument,
        instrument=instrument,
        facility="ISIS",
        title=title,
        sample=None,
        temperature=temperature,
        sample_temperature_logged=sample_temperature_logged,
        sample_temperature_log_source=None,
        field=field,
        field_direction="",
        geometry=geometry,
        geometry_source=geometry_source,
        precession=precession,
        detector_orientation="",
        notes=notes,
        n_histograms=2,
        n_points=100,
        total_events=total_events,
        bin_width_us=bin_width_us,
        start_time=None,
        duration_s=None,
        has_file_deadtime=False,
        n_periods=n_periods,
    )


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


def test_survey_reports_the_gross_event_total(workflow_folder: Path, survey) -> None:
    from asymmetry.core.io import load

    calibration = survey.row(CALIBRATION_RUN)
    dataset = load(workflow_folder / calibration.file)
    expected = round(sum(float(histogram.counts.sum()) for histogram in dataset.run.histograms))

    assert calibration.total_events == expected
    assert survey.to_dict()["runs"][0]["total_events"] == expected


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
    # Each candidate carries its own measured balance; one candidate, no step.
    assert candidate.alpha == pytest.approx(CALIBRATION_ALPHA, rel=0.02)
    assert survey.alpha_steps == []


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


def test_zero_field_runs_are_searched_for_a_spontaneous_line(survey) -> None:
    # The simulated zero-field runs only relax: no static order, no line.
    for run_number in ZF_RUNS:
        evidence = survey.row(run_number).precession
        assert evidence.state == "none"
        assert evidence.frequency_mhz is None
        assert evidence.larmor_mhz == 0.0
        assert "no spontaneous line" in evidence.note


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
    assert zero_field["precession"] == "none"
    assert "zero field" in zero_field["precession_note"]

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
    assert resolve_row_geometry(blank, silent) == (None, "refuted")


def test_a_file_tf_stamp_is_refuted_by_a_spectrum_with_no_line_in_it() -> None:
    # A field the muon does not precess in is not a transverse field, whatever
    # the file claims; reporting the claim would mislabel a decoupling run.
    stamped = {"field": 110.0, "field_state": "TF"}
    silent = PrecessionEvidence(
        state="none", frequency_mhz=None, snr=3.0, larmor_mhz=1.491, note=""
    )
    assert resolve_row_geometry(stamped, silent) == (None, "refuted")


def test_an_internal_field_line_refutes_nothing_and_leaves_the_file_standing() -> None:
    # An ordered magnet precessing in its own internal field is perfectly
    # consistent with a transverse applied field, so "other" is not evidence
    # against the file's stamp the way "none" is.
    stamped = {"field": 100.0, "field_state": "TF"}
    internal = PrecessionEvidence(
        state="other", frequency_mhz=6.14, snr=25.0, larmor_mhz=1.355, note=""
    )
    assert resolve_row_geometry(stamped, internal) == ("TF", "file")
    assert resolve_row_geometry({"field": 100.0}, internal) == (None, "none")


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


def _coils(main: float, x: float, y: float, z: float) -> dict:
    """Run metadata logging HiFi's four field coils, each steady over the run."""
    return {
        "field": main + z,
        "field_state": "TF",
        "nexus_time_series": {
            name: {"time": [0.0, 60.0], "values": [value, value]}
            for name, value in (
                ("Field_Main", main),
                ("Field_X", x),
                ("Field_Y", y),
                ("Field_Z", z),
            )
        },
    }


@pytest.mark.parametrize(
    ("main", "x", "y", "z", "expected"),
    [
        (19810.0, 0.0, 0.0, 0.0, "LF"),  # the main solenoid alone
        (0.0, 0.01, 19.96, 11.99, "TF"),  # TF20 on the Y coil, Z compensating stray field
        (20900.0, 0.0, 0.07, 160.0, "LF"),  # a Z sweep on a persistent main
        (0.0, 0.0, 50.0, 100.0, None),  # neither dominates
        (0.0, 0.0, 0.0, 0.0, None),  # nothing applied
    ],
)
def test_logged_coils_name_the_geometry_when_one_component_dominates(
    main: float, x: float, y: float, z: float, expected: str | None
) -> None:
    assert coil_geometry(_coils(main, x, y, z)) == expected


def test_a_file_without_coil_logs_has_no_coil_geometry() -> None:
    assert coil_geometry({"field": 100.0, "field_state": "TF"}) is None
    partial = _coils(19810.0, 0.0, 0.0, 0.0)
    del partial["nexus_time_series"]["Field_Y"]
    assert coil_geometry(partial) is None


def test_logged_coils_rank_below_measured_precession_and_above_the_rest() -> None:
    # HiFi stamps TF on its longitudinal runs; the coils it logged say LF.
    longitudinal = _coils(19810.0, 0.0, 0.0, 0.0)
    unmeasured = PrecessionEvidence(
        state=None, frequency_mhz=None, snr=None, larmor_mhz=268.5, note="above Nyquist"
    )
    silent = PrecessionEvidence(state="none", frequency_mhz=None, snr=3.0, larmor_mhz=1.0, note="")
    larmor = PrecessionEvidence(
        state="larmor", frequency_mhz=0.27, snr=90.0, larmor_mhz=0.27, note=""
    )
    assert resolve_row_geometry(longitudinal, unmeasured) == ("LF", "coils")
    assert resolve_row_geometry(longitudinal, silent) == ("LF", "coils")
    assert resolve_row_geometry(_coils(0.0, 0.01, 19.96, 11.99), larmor) == ("TF", "measured")


def test_survey_groups_the_zero_field_scan_with_temperature_as_axis(survey) -> None:
    scans = [scan for scan in survey.scans if scan.axis == "temperature"]
    assert len(scans) == 1
    scan = scans[0]
    assert scan.instrument == "SIM"
    assert scan.geometry == "ZF"
    assert scan.geometry_note == ""
    assert scan.field == pytest.approx(0.0)
    assert scan.runs == list(ZF_RUNS)
    assert scan.values == pytest.approx(list(ZF_TEMPERATURES))


def test_a_scan_whose_geometry_resolves_only_in_part_stays_one_scan() -> None:
    # A transverse-field scan through a magnetic transition resolves above it
    # and not below; splitting it on geometry would report one experiment as
    # two. The group holds together and says how its members broke down.
    members = [
        _row(run_number=1, temperature=380.0, geometry="TF", geometry_source="measured"),
        _row(run_number=2, temperature=370.0, geometry="TF", geometry_source="measured"),
        _row(run_number=3, temperature=350.0, geometry=None, geometry_source="none"),
    ]
    geometry, note = _group_geometry(members)
    assert geometry is None
    assert note == "TF measured on 2 of 3 runs; 1 unresolved"


def test_a_scan_whose_members_all_agree_reports_that_geometry_and_no_note() -> None:
    members = [
        _row(run_number=1, temperature=380.0, geometry="TF", geometry_source="measured"),
        _row(run_number=2, temperature=370.0, geometry="TF", geometry_source="file"),
    ]
    assert _group_geometry(members) == ("TF", "")


def test_a_zero_field_run_beside_one_field_run_is_not_a_field_scan() -> None:
    # With geometry out of the scan key, a magnet's fine ZF re-scan and its TF
    # scan share temperatures run for run; each pair would otherwise be
    # reported as a spurious "0 to 100 G field scan".
    pair = [
        _row(run_number=1, temperature=360.0, field=0.0),
        _row(run_number=2, temperature=360.0, field=100.0, geometry="TF"),
    ]
    assert [scan for scan in _scan_groups(pair)[0] if scan.axis == "field"] == []

    # Add a second non-zero field and it is a field scan, zero-field point and
    # all — that is what a decoupling curve looks like.
    decoupling = [*pair, _row(run_number=3, temperature=360.0, field=2000.0, geometry=None)]
    scans = [scan for scan in _scan_groups(decoupling)[0] if scan.axis == "field"]
    assert len(scans) == 1
    assert scans[0].values == pytest.approx([0.0, 100.0, 2000.0])


def test_field_scans_at_one_temperature_split_by_note_and_merge_interleaved_passes() -> None:
    # An ALC campaign at 300 K: two interleaved passes of one scan (offset by
    # 50 G), then a second scan of another region under another note.
    first = [
        _row(run_number=run, temperature=300.0, field=field, notes="CHMu(0) scan")
        for run, field in zip(range(1, 9), [*range(19000, 19400, 100), *range(19050, 19450, 100)])
    ]
    second = [
        _row(run_number=run, temperature=300.0, field=field, notes="o-p scan")
        for run, field in zip(range(9, 13), range(28500, 28900, 100))
    ]
    scans = [scan for scan in _scan_groups(first + second)[0] if scan.axis == "field"]
    assert [scan.runs[:2] for scan in scans] == [[1, 5], [9, 10]]
    assert [len(scan.runs) for scan in scans] == [8, 4]
    assert [scan.notes for scan in scans] == ["CHMu(0) scan", "o-p scan"]


def test_a_decoupling_sweep_run_on_into_an_even_scan_is_two_field_scans() -> None:
    # Widening steps up to 3500 G, then even 100 G steps from 5000 G: a
    # decoupling curve and a level-crossing scan taken back to back.
    fields = [0.0, 1.0, 1.8, 3.0, 5.4, 10.0, 18.0, 30.0, 54.0, 100.0, 180.0, 300.0, 540.0]
    fields += [1000.0, 1800.0, 2500.0, 3500.0, *range(5000, 6000, 100)]
    rows = [
        _row(run_number=run, temperature=50.0, field=float(field), geometry=None)
        for run, field in enumerate(fields, start=1)
    ]
    scans = [scan for scan in _scan_groups(rows)[0] if scan.axis == "field"]
    assert [(scan.values[0], scan.values[-1]) for scan in scans] == [
        (0.0, 3500.0),
        (5000.0, 5900.0),
    ]


def test_a_transverse_calibration_run_stays_out_of_a_longitudinal_field_scan() -> None:
    larmor = PrecessionEvidence(
        state="larmor", frequency_mhz=0.27, snr=40.0, larmor_mhz=0.271, note=""
    )
    rows = [
        _row(run_number=1, temperature=50.0, field=20.0, geometry="TF", precession=larmor),
        *[
            _row(run_number=run, temperature=50.0, field=field, geometry=None)
            for run, field in zip(range(2, 6), (0.0, 100.0, 1000.0, 5000.0))
        ],
    ]
    (scan,) = [scan for scan in _scan_groups(rows)[0] if scan.axis == "field"]
    assert scan.runs == [2, 3, 4, 5]


def test_a_temperature_returned_to_starts_a_new_field_scan() -> None:
    def block(first_run: int, temperature: float) -> list[RunRow]:
        return [
            _row(run_number=first_run + index, temperature=temperature, field=field)
            for index, field in enumerate((100.0, 1000.0, 5000.0))
        ]

    rows = [*block(1, 420.0), *block(4, 400.0), *block(7, 420.0)]
    scans = [scan for scan in _scan_groups(rows)[0] if scan.axis == "field"]
    assert [scan.runs for scan in scans] == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    # A temperature scan is not cut: fields alternate within a temperature point.
    (temperature_scan, *_) = [scan for scan in _scan_groups(rows)[0] if scan.axis == "temperature"]
    assert temperature_scan.runs == [4, 1, 7]


def test_a_field_scan_measured_alternately_at_two_temperatures_stays_whole() -> None:
    # Each field taken at 20 K and then at 2 K: no field repeats within a
    # temperature, so each temperature is one field scan however they interleave.
    rows = [
        _row(run_number=2 * index + offset, temperature=temperature, field=field)
        for index, field in enumerate((200.0, 800.0, 1600.0, 3200.0))
        for offset, temperature in ((1, 20.0), (2, 2.0))
    ]
    scans, _ = _scan_groups(rows)
    field_scans = [scan for scan in scans if scan.axis == "field"]
    assert sorted(len(scan.runs) for scan in field_scans) == [4, 4]


def test_a_return_sweep_in_one_visit_stays_in_its_scan() -> None:
    fields = (10.0, 100.0, 1000.0, 4000.0, 1000.0, 100.0)
    rows = [
        _row(run_number=run, temperature=300.0, field=field)
        for run, field in enumerate(fields, start=1)
    ]
    (scan,) = [scan for scan in _scan_groups(rows)[0] if scan.axis == "field"]
    assert len(scan.runs) == 6


def test_a_transverse_scan_keeps_the_runs_too_slow_to_show_their_line() -> None:
    larmor = PrecessionEvidence(
        state="larmor", frequency_mhz=1.4, snr=40.0, larmor_mhz=1.36, note=""
    )
    rows = [
        _row(run_number=run, temperature=350.0, field=field, precession=larmor)
        for run, field in enumerate((20.0, 40.0, 60.0, 80.0), start=1)
    ] + [
        _row(run_number=run, temperature=350.0, field=field) for run, field in ((5, 5.0), (6, 10.0))
    ]
    (scan,) = [scan for scan in _scan_groups(rows)[0] if scan.axis == "field"]
    assert len(scan.runs) == 6


def test_temperature_cross_sections_of_a_field_scan_grid_are_counted_not_listed() -> None:
    rows = [
        _row(run_number=10 * index + step, temperature=temperature, field=1000.0 + 100.0 * step)
        for index, temperature in enumerate((300.0, 325.0, 350.0))
        for step in range(8)
    ]
    scans, cross_sections = _scan_groups(rows)
    assert [scan.axis for scan in scans] == ["field"] * 3
    assert cross_sections == 8


def test_a_line_free_point_of_a_longitudinal_field_scan_is_named_in_a_tf_scan() -> None:
    tf = [
        _row(run_number=run, temperature=t, field=100.0, geometry="TF", geometry_source="measured")
        for run, t in ((1, 100.0), (2, 250.0), (3, 280.0))
    ]
    # A decoupling scan at 40 K whose 100 G point lands in the 100 G TF scan.
    lf = [
        _row(run_number=run, temperature=40.0, field=b, geometry=None, geometry_source="refuted")
        for run, b in ((10, 50.0), (11, 80.0), (12, 100.0))
    ]
    scan = next(s for s in _scan_groups(tf + lf)[0] if s.axis == "temperature")
    assert scan.runs == [12, 1, 2, 3]
    assert "run 12 also belongs to a field scan with no transverse line" in scan.geometry_note

    # In a grid of fields by temperatures every run is in both kinds of scan,
    # and most runs resolve no line: nothing is singled out.
    grid = [
        _row(run_number=10 * i + j, temperature=t, field=b, geometry=None)
        for i, t in enumerate((2.0, 3.0, 4.0))
        for j, b in enumerate((20.0, 40.0))
    ]
    assert all("also belong" not in s.geometry_note for s in _scan_groups(grid)[0])


def test_two_instruments_in_one_folder_never_share_a_scan() -> None:
    rows = [
        _row(run_number=1, temperature=10.0, instrument="EMU"),
        _row(run_number=2, temperature=20.0, instrument="EMU"),
        _row(run_number=3, temperature=10.0, instrument="MUSR"),
        _row(run_number=4, temperature=20.0, instrument="MUSR"),
    ]
    scans = _scan_groups(rows)[0]
    assert sorted(scan.instrument for scan in scans) == ["EMU", "MUSR"]
    assert all(len(scan.runs) == 2 for scan in scans)


def test_a_return_sweep_stays_whole_while_another_instruments_runs_interleave() -> None:
    """Two instruments sharing run numbers interleave in run order; neither splits the other."""
    rows = [
        _row(run_number=run, temperature=temperature, field=field, instrument=instrument)
        for run, field in enumerate((100.0, 200.0, 100.0, 300.0), start=1)
        for instrument, temperature in (("EMU", 10.0), ("MUSR", 20.0))
    ]
    field_scans = [scan for scan in _scan_groups(rows)[0] if scan.axis == "field"]
    assert sorted((scan.instrument, len(scan.runs)) for scan in field_scans) == [
        ("EMU", 4),
        ("MUSR", 4),
    ]


def test_runs_of_another_bin_width_never_share_a_scan() -> None:
    # A field scan at 294 K on one TDC binning, then repeats at its top field
    # at 300 K on another: a changed set-up, not a 294 -> 300 K temperature scan.
    scan = [
        _row(run_number=run, temperature=294.0, field=field, geometry="TF", bin_width_us=0.00039)
        for run, field in zip(range(1, 5), (200.0, 1000.0, 2000.0, 3000.0))
    ]
    repeats = [
        _row(run_number=run, temperature=300.0, field=3000.0, geometry="TF", bin_width_us=0.000586)
        for run in range(11, 16)
    ]
    scans, _ = _scan_groups(scan + repeats)
    assert [(s.axis, s.runs) for s in scans] == [("field", [1, 2, 3, 4])]

    # One nominal width read back through float32 is still one set-up.
    nominal = [
        _row(run_number=1, temperature=10.0, bin_width_us=0.01600000262260437),
        _row(run_number=2, temperature=20.0, bin_width_us=0.016000010073184967),
    ]
    assert [s.runs for s in _scan_groups(nominal)[0]] == [[1, 2]]


def test_two_period_runs_form_their_own_scan_which_says_so() -> None:
    rows = [
        _row(run_number=run, temperature=300.0, field=field, n_periods=periods)
        for run, field, periods in (
            (1, 28500.0, 1),
            (2, 28600.0, 1),
            (3, 28500.0, 2),
            (4, 28600.0, 2),
        )
    ]
    scans = [scan for scan in _scan_groups(rows)[0] if scan.axis == "field"]
    assert [(scan.runs, scan.n_periods) for scan in scans] == [([1, 2], 1), ([3, 4], 2)]
    assert scans[1].to_dict()["n_periods"] == 2


def test_consecutive_runs_at_one_condition_are_a_repeat_set() -> None:
    field_scan = [
        _row(run_number=run, temperature=294.0, field=field, geometry="TF")
        for run, field in zip(range(1, 4), (200.0, 1000.0, 3000.0))
    ]
    repeats = [
        _row(run_number=run, temperature=300.0, field=3000.0, geometry="TF", notes="3 kG")
        for run in range(11, 16)
    ]
    (repeat,) = repeat_sets(field_scan + repeats)
    assert (repeat.runs, repeat.temperature, repeat.field) == ([11, 12, 13, 14, 15], 300.0, 3000.0)
    assert repeat.co_add
    assert repeat.to_dict()["runs"] == [11, 12, 13, 14, 15]


def test_another_instruments_runs_between_repeats_do_not_break_them() -> None:
    rows = [
        _row(run_number=run, temperature=temperature, instrument=instrument)
        for run in (1, 2)
        for instrument, temperature in (("EMU", 10.0), ("MUSR", 20.0))
    ]
    assert sorted((r.instrument, r.runs) for r in repeat_sets(rows)) == [
        ("EMU", [1, 2]),
        ("MUSR", [1, 2]),
    ]


@pytest.mark.parametrize(
    "second",
    [
        {"notes": "4 foils"},
        {"field": 100.0},
        {"n_periods": 2},
        {"bin_width_us": 0.008},
        # A cryostat still cooling: the setpoint repeats, the sample does not.
        {"sample_temperature_logged": 250.0},
    ],
)
def test_a_changed_condition_ends_a_repeat_set(second: dict) -> None:
    first = {"temperature": 290.0, "sample_temperature_logged": 285.0}
    rows = [_row(run_number=1, **first), _row(run_number=2, **(first | second))]
    assert repeat_sets(rows) == []


def test_a_run_aborted_beside_its_restart_is_not_a_repeat_set() -> None:
    # Co-adding 2 % more events gains nothing worth a finding; half as many does.
    aborted = [
        _row(run_number=1, temperature=8.0, total_events=30_000_000),
        _row(run_number=2, temperature=8.0, total_events=550_000),
    ]
    assert repeat_sets(aborted) == []
    halves = [
        _row(run_number=1, temperature=8.0, total_events=30_000_000),
        _row(run_number=2, temperature=8.0, total_events=15_000_000),
    ]
    assert [r.runs for r in repeat_sets(halves)] == [[1, 2]]


def test_runs_whose_note_names_a_scan_are_not_repeats_to_co_add() -> None:
    # The files record one condition, but the note says something was stepped
    # between runs (a laser delay) that they do not record.
    rows = [
        _row(run_number=run, temperature=291.0, field=-100.0, notes="Laser on, P scan")
        for run in (1, 2, 3)
    ]
    (repeat,) = repeat_sets(rows)
    assert not repeat.co_add


def _stepped(notes: list[str], *, first: int = 1, **fields) -> list[RunRow]:
    """Consecutive runs at 295 K and 100 G whose notes are *notes*."""
    condition = {"temperature": 295.0, "field": 100.0, "geometry": "TF"} | fields
    return [
        _row(run_number=first + offset, notes=note, **condition)
        for offset, note in enumerate(notes)
    ]


def test_a_number_stepped_in_the_notes_at_one_condition_is_a_notes_scan() -> None:
    currents = ["0.0", "0.5", "1.0", "-0.5", "-1.0"]
    rows = _stepped([f"Steering {current} A" for current in currents])
    rows += _stepped(["Steering 0.0 A"], first=20, temperature=10.0)
    (scan,) = notes_scans(rows)
    assert (scan.source, scan.template, scan.quantity) == ("notes", "Steering <x> A", "steering")
    assert (scan.instrument, scan.temperature, scan.field) == ("SIM", 295.0, 100.0)
    assert scan.runs == [5, 4, 1, 2, 3]
    assert scan.values == [-1.0, -0.5, 0.0, 0.5, 1.0]
    assert scan.to_dict()["quantity"] == "steering"


def test_a_note_saying_less_of_the_template_joins_its_scan() -> None:
    # The other numbers of the template stay in it; a run whose note leaves
    # words out, its one number where the step is, is a point of the scan —
    # and one whose number stands elsewhere is not.
    notes = ["2 plates", "3 (50um) plates", "5 (50um) plates", "7 (50um) plates", "50um plates"]
    (scan,) = notes_scans(_stepped(notes))
    assert scan.template == "<x> (50um) plates"
    assert (scan.runs, scan.values) == ([1, 2, 3, 4], [2.0, 3.0, 5.0, 7.0])


def test_a_number_stepped_in_the_title_is_a_notes_scan_of_the_title() -> None:
    rows = [
        _row(run_number=run, temperature=5.0, title=f"Mix x={x} T=5 F=0")
        for run, x in ((1, "0.1"), (2, "0.2"), (3, "0.4"))
    ]
    (scan,) = notes_scans(rows)
    assert (scan.source, scan.template, scan.quantity) == ("title", "Mix x=<x> T=5 F=0", "x")


def test_a_scan_in_the_notes_is_not_found_again_in_the_title() -> None:
    rows = [
        _row(run_number=run, temperature=5.0, notes=f"slit {run}5", title=f"Run {run}5 here")
        for run in (1, 2, 4)
    ]
    assert [scan.source for scan in notes_scans(rows)] == ["notes"]


def test_a_stepped_quantity_named_like_a_recorded_one_is_renamed() -> None:
    (scan,) = notes_scans(_stepped([f"field coil {amps} A" for amps in (1, 3, 4)]))
    assert scan.quantity == "coil"
    (scan,) = notes_scans(_stepped([f"offset field {amps} G" for amps in (1, 3, 4)]))
    assert scan.quantity == "notes_field"


@pytest.mark.parametrize(
    "notes",
    [
        # A repeat counter, not a quantity.
        ["run 1 of 3", "run 2 of 3", "run 3 of 3"],
        ["Day 4", "Day 5", "Day 6", "Day 7"],
        # Dates and times are text.
        ["started 12/05/2024", "started 13/05/2024", "started 15/05/2024"],
        ["2024-05-12 cooled", "2024-05-13 cooled", "2024-05-15 cooled"],
        ["begun 06:30", "begun 09:45", "begun 13:10"],
        # Two points are a comparison; two numbers stepping are not one scan.
        ["Steering 0 A", "Steering 1 A", "Steering 1 A"],
        ["x 1 y 1", "x 2 y 3", "x 4 y 5"],
        # Too few runs.
        ["Steering 0 A", "Steering 2 A"],
    ],
)
def test_notes_that_step_no_single_quantity_are_no_notes_scan(notes: list[str]) -> None:
    assert notes_scans(_stepped(notes)) == []


@pytest.mark.parametrize(
    "fields",
    [{"temperature": 10.0}, {"field": 20.0}, {"instrument": "MUSR"}, {"bin_width_us": 0.008}],
)
def test_a_notes_scan_holds_one_condition(fields: dict) -> None:
    rows = _stepped(["Steering 0 A", "Steering 1 A"]) + _stepped(
        ["Steering 3 A"], first=3, **fields
    )
    assert notes_scans(rows) == []


def test_two_instruments_sharing_a_run_number_each_get_their_own_alpha(tmp_path: Path) -> None:
    pytest.importorskip("h5py")
    from asymmetry.core.io.nexus_writer import write_nexus_v1
    from asymmetry.core.simulate import simulate_run
    from tests.core.conftest import _calibration_signal, _template_for

    for prefix, alpha in (("EMU", 1.25), ("MUSR", 0.8)):
        title = f"Calibrant {prefix}"
        run = simulate_run(
            _template_for(temperature=5.0, field=CALIBRATION_FIELD_G, title=title),
            _calibration_signal,
            total_events=2.0e6,
            seed=1,
            alpha=alpha,
            run_number=42,
            title=title,
        )
        write_nexus_v1(run, tmp_path / f"{prefix}00000042.nxs")

    result = survey_folder(tmp_path)

    assert sorted(row.file for row in result.runs) == ["EMU00000042.nxs", "MUSR00000042.nxs"]
    alphas = {c.prefix: c.alpha for c in result.calibration_candidates}
    assert alphas["EMU"] == pytest.approx(1.25, rel=0.05)
    assert alphas["MUSR"] == pytest.approx(0.8, rel=0.05)
    assert sum(c.best for c in result.calibration_candidates) == 1
    assert [row.file for row in survey_folder(tmp_path, instrument="MUSR").runs] == [
        "MUSR00000042.nxs"
    ]


def test_survey_reports_no_field_scan_when_no_two_runs_share_a_temperature(survey) -> None:
    assert [scan for scan in survey.scans if scan.axis == "field"] == []


def test_survey_round_trips_through_its_dict(survey) -> None:
    data = survey.to_dict()
    assert data["best_calibration_run"] == CALIBRATION_RUN
    assert [row["run_number"] for row in data["runs"]] == list(ALL_RUNS)
    assert data["scans"][0]["axis"] == "temperature"
    assert data["repeats"] == []
    assert data["notes_scans"] == []


def test_row_raises_key_error_for_an_absent_run(survey) -> None:
    with pytest.raises(KeyError):
        survey.row(999999)


def test_survey_rejects_a_path_that_is_not_a_directory(tmp_path: Path) -> None:
    missing = tmp_path / "nowhere"
    with pytest.raises(ValueError):
        survey_folder(missing)


def test_survey_of_a_folder_with_only_subfolders_of_runs_names_them(tmp_path: Path) -> None:
    """No run files directly in *folder*: point at the sub-folder that holds them.

    ``scan_run_files`` never opens a file, so bare touched names are enough
    to exercise this without a real (h5py-backed) NeXus fixture.
    """
    rg1 = tmp_path / "RG1"
    rg1.mkdir()
    for run in range(700, 703):
        (rg1 / f"SIM{run:08d}.nxs").touch()
    rg2 = tmp_path / "RG2"
    rg2.mkdir()
    (rg2 / "SIM00000710.nxs").touch()
    # Two levels down: scan_run_files (and survey_folder) never recurse this far.
    nested = rg1 / "nested"
    nested.mkdir()
    (nested / "SIM00000720.nxs").touch()

    with pytest.raises(ValueError, match=r"RG1 \(3 runs\), RG2 \(1 run\)"):
        survey_folder(tmp_path)


def test_survey_of_a_genuinely_empty_folder_is_still_an_empty_survey(tmp_path: Path) -> None:
    """A folder with nothing in it at all is not the "point at a sub-folder" case."""
    empty = survey_folder(tmp_path)
    assert empty.runs == []


def _candidate(run_number: int, alpha: float, prefix: str = "SIM") -> CalibrationCandidate:
    return CalibrationCandidate(
        run_number=run_number,
        prefix=prefix,
        field_gauss=100.0,
        reason="",
        source="measured",
        snr=100.0,
        alpha=alpha,
        best=False,
    )


def test_an_alpha_step_is_reported_between_consecutive_candidates_in_run_order() -> None:
    # Listed out of order, with scatter inside each block and one step.
    candidates = [
        _candidate(281, 1.401),
        _candidate(252, 1.027),
        _candidate(280, 1.068),
        _candidate(293, 1.406),
    ]
    steps = alpha_steps(candidates)
    assert [step.to_dict() for step in steps] == [
        {"before_run": 280, "after_run": 281, "alpha_before": 1.068, "alpha_after": 1.401}
    ]
    assert alpha_steps(candidates[1:3]) == []


def test_alpha_steps_take_run_order_one_instrument_at_a_time() -> None:
    """Interleaving two instruments' shared run numbers would invent a step at every run."""
    candidates = [
        _candidate(1, 1.0, "EMU"),
        _candidate(1, 1.4, "MUSR"),
        _candidate(2, 1.01, "emu"),
        _candidate(2, 1.41, "MUSR"),
    ]
    assert [(step.before_run, step.after_run) for step in alpha_steps(candidates)] == [(2, 1)]


def test_a_logged_temperature_far_from_its_setpoint_is_a_departure() -> None:
    from dataclasses import replace

    rows = [
        replace(_row(run_number=1, temperature=15.0), sample_temperature_logged=285.2),
        # Close in kelvin but not in proportion, and close in proportion but
        # not in kelvin: thermometer scatter either way, not a departure.
        replace(_row(run_number=2, temperature=1.6), sample_temperature_logged=1.8),
        replace(_row(run_number=3, temperature=300.0), sample_temperature_logged=302.0),
        replace(_row(run_number=4, temperature=2.0), sample_temperature_logged=8.2),
        _row(run_number=5, temperature=10.0),
    ]
    assert temperature_departures(rows) == [1, 4]


def _fingerprint(**fields):
    from types import SimpleNamespace

    base = {
        "oscillatory_hint": True,
        "dominant_fft_frequency_mhz": 0.096,
        "dominant_fft_snr": 100.0,
        "dominant_fft_cycles_in_window": 0.75,
        "damped_line_frequency_mhz": 0.0,
        "damped_line_snr": 0.0,
    }
    namespace = SimpleNamespace(**(base | fields))
    namespace.has_damped_line_candidate = namespace.damped_line_frequency_mhz > 0.0
    return namespace


@pytest.mark.parametrize(
    ("fingerprint", "expected"),
    [
        # A sub-cycle "line" away from Larmor is leakage; the damped scan's
        # line (a muonium line in a 2 G field, here) is the precession.
        (
            {"damped_line_frequency_mhz": 2.8, "damped_line_snr": 68.0},
            ("other", 2.8),
        ),
        # ... and with no damped line there is no precession at all.
        ({}, ("none", None)),
        # A slow line at the Larmor frequency is a weak-TF calibration, kept.
        ({"dominant_fft_frequency_mhz": 0.0271}, ("larmor", 0.0271)),
        # A resolved line away from Larmor is an internal field, as before.
        (
            {"dominant_fft_frequency_mhz": 30.0, "dominant_fft_cycles_in_window": 200.0},
            ("other", 30.0),
        ),
    ],
)
def test_a_sub_cycle_line_away_from_larmor_is_not_precession(
    monkeypatch, fingerprint, expected
) -> None:
    import numpy as np

    from asymmetry.core.data.dataset import MuonDataset
    from asymmetry.core.workflow import survey as survey_module

    monkeypatch.setattr(
        survey_module, "fingerprint_spectrum", lambda dataset: _fingerprint(**fingerprint)
    )
    time = np.arange(0.0, 8.0, 0.016)
    dataset = MuonDataset(time, np.zeros_like(time), np.ones_like(time), {"run_number": 1})
    evidence = survey_module.precession_evidence(dataset, 2.0)
    assert (evidence.state, evidence.frequency_mhz) == expected


@pytest.mark.parametrize(
    ("fingerprint", "expected"),
    [
        # An ordered magnet's line, found by the damped-line scan behind a
        # sub-cycle leakage "line" (EuO at 10 K: 29.9 MHz).
        ({"damped_line_frequency_mhz": 29.9, "damped_line_snr": 21.6}, ("other", 29.9)),
        # A resolved dominant line stands on its own.
        (
            {"dominant_fft_frequency_mhz": 1.56, "dominant_fft_cycles_in_window": 12.0},
            ("other", 1.56),
        ),
        # A Kubo-Toyabe or paramagnetic relaxation has no line.
        ({}, ("none", None)),
    ],
)
def test_a_zero_field_line_is_spontaneous_precession(monkeypatch, fingerprint, expected) -> None:
    import numpy as np

    from asymmetry.core.data.dataset import MuonDataset
    from asymmetry.core.workflow import survey as survey_module

    monkeypatch.setattr(
        survey_module, "fingerprint_spectrum", lambda dataset: _fingerprint(**fingerprint)
    )
    time = np.arange(0.0, 8.0, 0.016)
    dataset = MuonDataset(time, np.zeros_like(time), np.ones_like(time), {"run_number": 1})
    evidence = survey_module.precession_evidence(dataset, 0.0)
    assert (evidence.state, evidence.frequency_mhz) == expected


def test_departs_needs_both_an_absolute_and_a_relative_offset() -> None:
    from asymmetry.core.workflow.survey import departs

    assert departs(2.0, 8.2)
    assert not departs(1.6, 1.8)
    assert not departs(300.0, 302.0)
    assert not departs(None, 5.0)
    assert not departs(5.0, None)


def test_a_transverse_field_across_another_pair_names_that_pair() -> None:
    import numpy as np

    from asymmetry.core.simulate import InstrumentTemplate, simulate_run_from_group_signals
    from asymmetry.core.workflow.survey import precessing_pair

    # A GPS-like run: the field precesses the spin in the plane of Up and Down,
    # so the file's own Forw/Back pair sees no line at all.
    template = InstrumentTemplate(
        key="gps_like",
        label="GPS-like",
        description="",
        n_detectors=4,
        n_bins=8000,
        bin_width_us=0.001,
        t0_bin=500,
        forward_detectors=(1,),
        backward_detectors=(2,),
        groups={1: (1,), 2: (2,), 3: (3,), 4: (4,)},
        group_names={1: "Forw", 2: "Back", 3: "Up", 4: "Down"},
        default_forward_group=1,
        default_backward_group=2,
    ).build()

    def line(sign: float):
        return lambda t: sign * 0.2 * np.cos(2 * np.pi * 1.355 * np.asarray(t))

    run = simulate_run_from_group_signals(
        template, {3: line(1.0), 4: line(-1.0)}, total_events=40e6, seed=2
    )

    found = precessing_pair(run, 7, 100.0)

    assert (found.forward, found.backward) == ("Up", "Down")
    assert found.precession.state == "larmor"


@pytest.mark.parametrize(
    ("sample", "title", "expected"),
    [
        (None, "Benzene(aq) T=350.0 F=160.0", "Benzene(aq)"),
        (None, "nickel_T=100_F=0", "nickel"),
        ("LiFeAs powder", "LFA T=2 B=400", "LiFeAs powder"),
        (None, "", ""),
    ],
)
def test_a_runs_sample_is_its_own_name_or_its_title_before_the_conditions(
    sample, title, expected
) -> None:
    from types import SimpleNamespace

    from asymmetry.core.workflow.survey import sample_name

    assert sample_name(SimpleNamespace(sample=sample, title=title)) == expected


def test_a_scan_crossing_numbered_samples_at_one_setpoint_is_a_composition_series() -> None:
    from asymmetry.core.workflow.survey import ScanGroup, composition_sets

    titles = {
        1: ("Solution T=290 F=2", "pure water"),
        2: ("0.25 Acid/Water T=290 F=2", ""),
        3: ("0.5 acid/water T=290 F=2", ""),
        4: ("0.25 Acid/Water T=300 F=2", ""),
        5: ("0.5 Acid/Water T=300 F=2", ""),
    }
    rows = {
        run: _row(
            run_number=run, temperature=290.0 if run <= 3 else 300.0, title=title, notes=notes
        )
        for run, (title, notes) in titles.items()
    }
    scan = ScanGroup(
        axis="temperature",
        instrument="SIM",
        geometry=None,
        geometry_note="",
        temperature=None,
        field=2.0,
        runs=list(titles),
        values=[290.0, 290.0, 290.0, 300.0, 300.0],
    )
    # 300 K holds two samples only: too few for a slope.
    (found,) = composition_sets(scan, rows)
    assert (found.setpoint, found.runs, found.values) == (290.0, [1, 2, 3], [None, 0.25, 0.5])
