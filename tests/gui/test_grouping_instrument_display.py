"""Tests for the instrument display helper (grouping editor terminology).

The grouping editor keys everything off a
:class:`~asymmetry.core.project.profiles.ProfileFingerprint`, but the interface
only ever names the *instrument*. :func:`instrument_display_for_fingerprint`
turns a fingerprint into that user-facing label, appending the detector count
only when two fingerprints share an instrument name.
"""

from __future__ import annotations

import pytest

from asymmetry.core.project.profiles import ProfileFingerprint
from asymmetry.gui.windows.grouping.profile_bridge import (
    instrument_display_for_fingerprint,
)

pytestmark = [pytest.mark.gui]


def test_single_instrument_shows_bare_name() -> None:
    """A lone fingerprint reads as just the instrument display name."""
    fp = ProfileFingerprint(instrument="GPS", histogram_count=6)
    assert instrument_display_for_fingerprint(fp, [fp]) == "GPS"


def test_variant_key_resolves_to_display_name() -> None:
    """A variant registry key (GPS-RD) still reads as its display name."""
    fp = ProfileFingerprint(instrument="GPS-RD", histogram_count=11)
    assert instrument_display_for_fingerprint(fp, [fp]) == "GPS"


def test_shared_instrument_name_appends_detector_count() -> None:
    """Two fingerprints of the same instrument disambiguate by detector count."""
    gps = ProfileFingerprint(instrument="GPS", histogram_count=6)
    gps_rd = ProfileFingerprint(instrument="GPS-RD", histogram_count=11)
    fingerprints = [gps, gps_rd]
    assert instrument_display_for_fingerprint(gps, fingerprints) == "GPS (6 detectors)"
    assert instrument_display_for_fingerprint(gps_rd, fingerprints) == "GPS (11 detectors)"


def test_distinct_instruments_stay_bare() -> None:
    """Different instruments never trigger the detector-count disambiguator."""
    gps = ProfileFingerprint(instrument="GPS", histogram_count=6)
    musr = ProfileFingerprint(instrument="MuSR", histogram_count=64)
    fingerprints = [gps, musr]
    assert instrument_display_for_fingerprint(gps, fingerprints) == "GPS"
    assert instrument_display_for_fingerprint(musr, fingerprints) == "MuSR"


def test_empty_instrument_falls_back_to_detector_count() -> None:
    """An unresolved instrument shows the detector count, not a blank label."""
    fp = ProfileFingerprint(instrument="", histogram_count=32)
    assert instrument_display_for_fingerprint(fp, [fp]) == "32 detectors"


def test_empty_instrument_single_detector_is_singular() -> None:
    fp = ProfileFingerprint(instrument="", histogram_count=1)
    assert instrument_display_for_fingerprint(fp, [fp]) == "1 detector"


def test_empty_instrument_and_no_count_is_unknown() -> None:
    """A wholly unidentified fingerprint reads as 'Unknown instrument'."""
    fp = ProfileFingerprint(instrument="", histogram_count=0)
    assert instrument_display_for_fingerprint(fp, [fp]) == "Unknown instrument"


def test_single_detector_uses_singular_noun() -> None:
    """The detector-count disambiguator pluralises correctly."""
    a = ProfileFingerprint(instrument="GPS", histogram_count=1)
    b = ProfileFingerprint(instrument="GPS-RD", histogram_count=6)
    assert instrument_display_for_fingerprint(a, [a, b]) == "GPS (1 detector)"


def test_duplicate_fingerprint_twin_does_not_disambiguate() -> None:
    """An identical fingerprint appearing twice is not treated as a clash."""
    fp = ProfileFingerprint(instrument="GPS", histogram_count=6)
    twin = ProfileFingerprint(instrument="GPS", histogram_count=6)
    assert instrument_display_for_fingerprint(fp, [fp, twin]) == "GPS"
