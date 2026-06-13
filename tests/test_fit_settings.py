"""Tests for the QSettings-backed fit-display settings (Item 4)."""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings

from asymmetry.gui.fit_settings import (
    FIT_QUALITY_CONFIDENCE_SETTINGS_KEY,
    fit_quality_confidence,
    set_fit_quality_confidence,
)


@pytest.fixture
def settings():
    s = QSettings("AsymmetryTest", "fit_settings_test")
    s.clear()
    yield s
    s.clear()


def test_default_is_wimda_rgoodfit(settings):
    assert fit_quality_confidence(settings) == pytest.approx(0.95)


def test_round_trip(settings):
    assert set_fit_quality_confidence(0.80, settings) == pytest.approx(0.80)
    assert fit_quality_confidence(settings) == pytest.approx(0.80)


def test_clamps_to_valid_range(settings):
    assert set_fit_quality_confidence(0.2, settings) == pytest.approx(0.5)
    assert set_fit_quality_confidence(0.9999, settings) == pytest.approx(0.999)


def test_unparseable_value_falls_back_to_default(settings):
    settings.setValue(FIT_QUALITY_CONFIDENCE_SETTINGS_KEY, "not-a-number")
    assert fit_quality_confidence(settings) == pytest.approx(0.95)
