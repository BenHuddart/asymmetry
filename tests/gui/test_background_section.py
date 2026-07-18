"""Tests for the inline background-configuration section (Corrections panel).

Covers the embeddable body that replaced the retired ``BackgroundDialog``: mode
selection, the ``changed`` signal, and the reference-run picker (which delegates
to ``QInputDialog``/``QFileDialog``, monkeypatched here).
"""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QInputDialog

from asymmetry.core.project.profiles import BackgroundPolicy
from asymmetry.gui.windows.grouping.background_section import (
    BackgroundReferenceRunCandidate,
    BackgroundSectionWidget,
    background_status_text,
)


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


def _candidates() -> list[BackgroundReferenceRunCandidate]:
    return [BackgroundReferenceRunCandidate(9001, "Run 9001 (run 9001)", "/tmp/x.nxs", 1000.0)]


def test_configure_seeds_mode_without_emitting(qapp: QApplication) -> None:
    section = BackgroundSectionWidget(_candidates)
    fired: list[int] = []
    section.changed.connect(lambda: fired.append(1))
    section.configure(
        available_modes=("range", "tail_fit"),
        has_fixed_values=False,
        mode="range",
        background_run_payload=None,
    )
    assert section.mode() == "range"
    assert not fired  # seeding never emits


def test_mode_change_emits_changed(qapp: QApplication) -> None:
    section = BackgroundSectionWidget(_candidates)
    section.configure(
        available_modes=("range", "tail_fit"),
        has_fixed_values=False,
        mode="none",
        background_run_payload=None,
    )
    fired: list[int] = []
    section.changed.connect(lambda: fired.append(1))
    idx = section._mode_combo.findData("range")
    section._mode_combo.setCurrentIndex(idx)
    assert section.mode() == "range"
    assert fired  # a user edit notifies the owner


def test_reference_pick_selects_run_and_emits(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    section = BackgroundSectionWidget(_candidates)
    section.configure(
        available_modes=("reference_run",),
        has_fixed_values=False,
        mode="none",
        background_run_payload=None,
    )
    # The picker returns the candidate run.
    monkeypatch.setattr(QInputDialog, "getItem", lambda *a, **k: ("Run 9001 (run 9001)", True))
    fired: list[int] = []
    section.changed.connect(lambda: fired.append(1))
    idx = section._mode_combo.findData("reference_run")
    section._mode_combo.setCurrentIndex(idx)  # first entry prompts the picker
    assert section.mode() == "reference_run"
    payload = section.background_run_payload()
    assert payload is not None and payload["run_number"] == 9001
    assert fired


def test_cancelled_pick_falls_back_to_none(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    section = BackgroundSectionWidget(_candidates)
    section.configure(
        available_modes=("reference_run",),
        has_fixed_values=False,
        mode="none",
        background_run_payload=None,
    )
    monkeypatch.setattr(QInputDialog, "getItem", lambda *a, **k: ("", False))
    idx = section._mode_combo.findData("reference_run")
    section._mode_combo.setCurrentIndex(idx)
    assert section.mode() == "none"  # a cancelled pick resets the mode


def test_status_text_helper() -> None:
    assert background_status_text(BackgroundPolicy(mode="none")) == "Background: none"
    assert "tail fit" in background_status_text(BackgroundPolicy(mode="tail_fit")).lower()
    ref = BackgroundPolicy(mode="reference_run", details={"background_run": {"run_number": 42}})
    assert "42" in background_status_text(ref)
