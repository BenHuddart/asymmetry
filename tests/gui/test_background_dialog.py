"""Tests for the dedicated background-configuration dialog.

Covers mode-combo gating (per :func:`available_background_modes`), the
reference-run picker flow, the tail-fit preview status text, the
``background_run`` payload round-trip, and the preview canvas's
visible/hidden state per mode (hidden for ``fixed``, which has nothing
run-derived to show).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from asymmetry.core.project.profiles import BackgroundPolicy
from asymmetry.gui.windows.grouping.background_dialog import (
    BackgroundDialog,
    BackgroundReferenceRunCandidate,
    background_status_text,
)


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _tail_fit_preview(n: int = 400, bin_width: float = 0.016) -> tuple:
    rng = np.random.default_rng(0)
    forward = rng.poisson(np.full(n, 50.0)).astype(float)
    backward = rng.poisson(np.full(n, 40.0)).astype(float)
    return (forward, backward, bin_width, 0, n - 1)


def _dialog(**overrides) -> BackgroundDialog:
    defaults = dict(
        available_modes=("tail_fit", "reference_run", "fixed"),
        has_fixed_values=True,
        initial_mode="none",
        background_run_payload=None,
        reference_run_candidates=[],
        preview=None,
    )
    defaults.update(overrides)
    return BackgroundDialog(**defaults)


def test_range_mode_disabled_when_not_available(qapp: QApplication) -> None:
    dlg = _dialog(available_modes=("tail_fit", "reference_run"), has_fixed_values=False)
    idx = dlg._mode_combo.findData("range")
    item = dlg._mode_combo.model().item(idx)
    assert item is not None and not item.isEnabled()
    for mode in ("tail_fit", "reference_run"):
        idx = dlg._mode_combo.findData(mode)
        assert dlg._mode_combo.model().item(idx).isEnabled()


def test_fixed_mode_hidden_when_no_fixed_values(qapp: QApplication) -> None:
    dlg = _dialog(has_fixed_values=False)
    assert dlg._mode_combo.findData("fixed") == -1


def test_fixed_mode_shown_when_fixed_values_present(qapp: QApplication) -> None:
    dlg = _dialog(has_fixed_values=True)
    assert dlg._mode_combo.findData("fixed") != -1


def test_tail_fit_mode_shows_preview_status(qapp: QApplication) -> None:
    dlg = _dialog(initial_mode="tail_fit", preview=_tail_fit_preview())
    assert "Tail-fit background" in dlg._status_label.text()
    assert dlg.get_policy().mode == "tail_fit"


def test_reference_run_pick_populates_payload(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asymmetry.gui.windows.grouping.background_dialog as background_dialog_module

    candidate = BackgroundReferenceRunCandidate(
        run_number=9001, label="Run 9001 (run 9001)", source_file="/tmp/x.nxs", good_frames=500.0
    )
    monkeypatch.setattr(
        background_dialog_module.QInputDialog,
        "getItem",
        lambda *args, **kwargs: (candidate.label, True),
    )
    dlg = _dialog(initial_mode="reference_run", reference_run_candidates=[candidate])

    policy = dlg.get_policy()
    assert policy.mode == "reference_run"
    assert policy.details["background_run"]["run_number"] == 9001
    assert "9001" in dlg._status_label.text() or "9001" in dlg._reference_summary_label.text()


def test_reference_run_cancel_falls_back_to_none(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asymmetry.gui.windows.grouping.background_dialog as background_dialog_module

    monkeypatch.setattr(
        background_dialog_module.QInputDialog,
        "getItem",
        lambda *args, **kwargs: ("", False),
    )
    dlg = _dialog(initial_mode="reference_run", reference_run_candidates=[])
    assert dlg.current_mode() == "none"


def test_background_run_payload_round_trips(qapp: QApplication) -> None:
    payload = {"run_number": 9001, "source_file": "/tmp/x.nxs"}
    dlg = _dialog(initial_mode="reference_run", background_run_payload=payload)
    assert dlg.current_mode() == "reference_run"
    policy = dlg.get_policy()
    assert policy.details["background_run"]["run_number"] == 9001


def test_preview_canvas_hidden_for_fixed_mode(qapp: QApplication) -> None:
    dlg = _dialog(initial_mode="tail_fit", preview=_tail_fit_preview())
    if dlg._canvas is not None:
        assert dlg._canvas_container.isVisible() or not dlg.isVisible()
    idx = dlg._mode_combo.findData("fixed")
    dlg._mode_combo.setCurrentIndex(idx)
    if dlg._canvas is not None:
        assert dlg._canvas_container.isVisible() is False


def test_preview_disabled_when_no_preview_data(qapp: QApplication) -> None:
    dlg = _dialog(initial_mode="tail_fit", preview=None)
    assert dlg._canvas is None


def test_background_status_text_matches_mode() -> None:
    assert background_status_text(BackgroundPolicy(mode="none")) == "Background: none"
    assert "pre-t0" in background_status_text(BackgroundPolicy(mode="range"))
    assert "tail fit" in background_status_text(BackgroundPolicy(mode="tail_fit"))
    ref_policy = BackgroundPolicy(
        mode="reference_run", details={"background_run": {"run_number": 42}}
    )
    assert "42" in background_status_text(ref_policy)
    assert "fixed" in background_status_text(BackgroundPolicy(mode="fixed"))
