"""Canonical identity migration retains the old automatic-window RF convention."""

import pytest

from asymmetry.core.data.dataset import Run
from asymmetry.core.instruments.resolution import (
    InstrumentSelection,
    RecordedInstrumentChoice,
    UnresolvedInstrumentIdentity,
)
from asymmetry.core.transform.background import grouped_background_ranges, resolve_facility
from asymmetry.core.transform.corrections import reference_observation_metadata


@pytest.mark.parametrize("label,last", [("PSI", 57), ("TRIUMF", 53)])
def test_opaque_legacy_hint_retains_automatic_rf_window_without_raw_metadata(label, last):
    choice = RecordedInstrumentChoice(UnresolvedInstrumentIdentity(label), ())
    run = Run(instrument_choice=choice)
    facility = resolve_facility(metadata=run.metadata, grouping={}, instrument_choice=choice)
    transported = resolve_facility(metadata=reference_observation_metadata(run))
    assert facility == transported == label
    for value in (facility, transported):
        windows = grouped_background_ranges(
            {},
            t0_bin=100,
            acquisition_start_bin=0,
            length=200,
            bin_width_us=0.005,
            facility=value,
        )
        assert [(window.first, window.last) for window in windows] == [(10, last), (10, last)]
    assert run.metadata == {}


def test_canonical_gps_identity_does_not_introduce_a_psi_rf_default():
    choice = RecordedInstrumentChoice(
        InstrumentSelection("psi.gps", "bin_combined", "1", "legacy"), ()
    )
    run = Run(instrument_choice=choice)
    assert resolve_facility(instrument_choice=choice) == ""
    assert resolve_facility(metadata=reference_observation_metadata(run)) == ""
    windows = grouped_background_ranges(
        {}, t0_bin=100, acquisition_start_bin=0, length=200, bin_width_us=0.005, facility=""
    )
    assert [(window.first, window.last) for window in windows] == [(10, 60), (10, 60)]


def test_explicit_facility_and_window_opt_out_remain_authoritative():
    choice = RecordedInstrumentChoice(UnresolvedInstrumentIdentity("TRIUMF"), ())
    assert resolve_facility(metadata={"facility": "PSI"}, instrument_choice=choice) == "PSI"
    windows = grouped_background_ranges(
        {}, t0_bin=100, acquisition_start_bin=0, length=200, bin_width_us=0.005, facility=""
    )
    assert [(window.first, window.last) for window in windows] == [(10, 60), (10, 60)]
