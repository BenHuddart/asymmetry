"""Unit tests for the shared NXlog-series → summary-field scorer.

Pure-function coverage (no Qt) of the sensor-selection rules used by both the
Data Browser and the Run Info dialog. The HIGH bug this guards: a HiFi run logs
``DetectorTemp1`` (detector electronics, ~298 K) alongside ``Temp_Cryostat``;
both used to score 10 and the alphabetical tie-break picked ``DetectorTemp1`` so
"Use temperature from log" showed room temperature for every run.
"""

from __future__ import annotations

import pytest

from asymmetry.gui.utils.series_scoring import score_series_path

pytestmark = [pytest.mark.gui]


def test_detector_temp_is_excluded() -> None:
    assert score_series_path("temperature", "DetectorTemp1") == 0
    assert score_series_path("temperature", "DetectorTemp2") == 0
    assert score_series_path("temperature", "selog/DetectorTemp1/value_log") == 0


def test_cryostat_beats_detector_on_hifi() -> None:
    # The Sn HiFi situation: both sensors logged; the cryostat must win.
    cryostat = score_series_path("temperature", "Temp_Cryostat")
    detector = score_series_path("temperature", "DetectorTemp1")
    assert cryostat > detector
    assert detector == 0
    assert cryostat > 10  # preferred above a bare *Temp* log


def test_sample_thermometer_outranks_cryostat() -> None:
    assert score_series_path("temperature", "Temp_Sample") > score_series_path(
        "temperature", "Temp_Cryostat"
    )


@pytest.mark.parametrize(
    "path",
    ["Temp_Cryostat", "Temp_He", "Temp_VTI", "Temp_Dilution", "Temp_He3"],
)
def test_cryostat_family_preferred_over_bare_temp(path: str) -> None:
    bare = score_series_path("temperature", "Temp")  # unspecified thermometer
    assert score_series_path("temperature", path) > bare > 0


def test_plain_cryostat_only_case_still_selected() -> None:
    # A normal run that logs only a cryostat sensor still resolves to it.
    assert score_series_path("temperature", "Temp_Cryostat") > 0


def test_non_temperature_paths_score_zero() -> None:
    assert score_series_path("temperature", "Field_Magnet") == 0
    assert score_series_path("temperature", "beamlog_current") == 0


def test_detector_temp_recognised_via_name_child() -> None:
    # Native HDF4: generic Vgroup path, real sensor name in the ``name`` child.
    assert score_series_path("temperature", "selog/log_3", {"name": "DetectorTemp1"}) == 0


def test_cryostat_recognised_via_name_child() -> None:
    # The (b) rescue: a generically-named log whose ``name`` is the sensor.
    generic = score_series_path("temperature", "selog/log_1", {"name": "Temp_Cryostat"})
    assert generic > 10
    assert generic == score_series_path("temperature", "Temp_Cryostat")


def test_ancestor_detector_path_not_excluded() -> None:
    # "detector" in an ancestor segment must NOT veto a real sample thermometer.
    assert score_series_path("temperature", "instrument/detector/Temp_Sample") > 0


def test_field_scoring_unchanged() -> None:
    assert score_series_path("field", "Field_Danfysik") > 0
    assert score_series_path("field", "Sample_Magnetic_Field") > score_series_path(
        "field", "Field_Danfysik"
    )
    assert score_series_path("field", "Temp_Cryostat") == 0


def test_role_primary_shortcut() -> None:
    assert (
        score_series_path("temperature", "x", {"role": "sample_temperature", "primary": True})
        == 100
    )
    assert score_series_path("temperature", "x", {"role": "sample_temperature"}) == 70
    assert score_series_path("field", "x", {"role": "sample_field", "primary": True}) == 80
