"""Tests for the process-wide fit-time store both wizards share."""

from __future__ import annotations

import json
import logging

import pytest

pytest.importorskip("PySide6")

from asymmetry.core.fitting.fit_time_store import FitTimeStore  # noqa: E402
from asymmetry.gui.utils import fit_times  # noqa: E402
from tests.core.test_fit_time_store import _timed  # noqa: E402

pytestmark = [pytest.mark.gui]


def test_the_store_is_loaded_once_and_shared() -> None:
    assert fit_times.shared_fit_time_store() is fit_times.shared_fit_time_store()


def test_recording_a_run_saves_the_store() -> None:
    fit_times.record_fit_times([_timed(["Keren", "Constant"], 2.0, points=4000)])
    saved = json.loads(fit_times.fit_times_path().read_text(encoding="utf-8"))
    assert saved == {"version": 1, "seconds_per_kpoint": {"Keren": [0.5]}}
    assert fit_times.shared_fit_time_store().samples == {"Keren": (0.5,)}


def test_a_saved_store_is_read_at_first_use() -> None:
    FitTimeStore({"Keren": [0.5]}).save(fit_times.fit_times_path())
    assert fit_times.shared_fit_time_store().samples == {"Keren": (0.5,)}


def test_a_corrupt_store_is_discarded_with_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    path = fit_times.fit_times_path()
    path.write_text("{not json", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger=fit_times.__name__):
        assert fit_times.shared_fit_time_store().samples == {}
    assert str(path) in caplog.text
    fit_times.record_fit_times([_timed(["Keren", "Constant"], 1.0)])
    assert json.loads(path.read_text(encoding="utf-8"))["seconds_per_kpoint"] == {"Keren": [1.0]}
