"""Standalone unit tests for the window-agnostic DecisionTrail widget."""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QLabel

from asymmetry.core.fitting.wizard_narrative import TrailStep
from asymmetry.gui.widgets.decision_trail import DecisionTrail


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _steps() -> tuple[TrailStep, ...]:
    return (
        TrailStep("conditions", "Run conditions read.", "conditions", ("cond detail",)),
        TrailStep("families", "Families considered.", "families", ("fam detail",)),
        TrailStep("spectrum", "Spectral search done.", "spectrum", ("spec detail",)),
    )


def test_set_steps_renders_expandable_rows(qapp: QApplication) -> None:
    trail = DecisionTrail()
    trail.set_steps(_steps())
    assert trail.step_keys() == ("conditions", "families", "spectrum")
    # Rows start collapsed.
    assert trail.is_step_expanded("conditions") is False
    trail.set_step_expanded("conditions", True)
    assert trail.is_step_expanded("conditions") is True


def test_streaming_placeholders_and_activation(qapp: QApplication) -> None:
    trail = DecisionTrail()
    trail.stream_placeholders(_steps())
    assert trail.step_keys() == ("conditions", "families", "spectrum")
    # Placeholder rows are not expandable.
    assert trail.is_step_expanded("conditions") is False
    trail.set_step_expanded("conditions", True)
    assert trail.is_step_expanded("conditions") is False
    # Activating an unknown key is a no-op (no raise).
    trail.activate_step("nonexistent")
    trail.activate_step("families")


def test_status_line_shows_and_hides(qapp: QApplication) -> None:
    trail = DecisionTrail()
    trail.set_status("Working...")
    assert trail._status_label.isVisibleTo(trail) is True
    assert trail._status_label.text() == "Working..."
    trail.set_status("")
    assert trail._status_label.isVisibleTo(trail) is False


def test_injected_detail_widget_revealed_on_expand(qapp: QApplication) -> None:
    trail = DecisionTrail()
    trail.set_steps(_steps())
    panel = QLabel("deep panel")
    trail.set_step_detail_widget("spectrum", panel)
    # Injected panel is hidden until the step expands.
    assert panel.isVisibleTo(trail) is False
    trail.set_step_expanded("spectrum", True)
    assert panel.isVisibleTo(trail) is True
    # Panel is re-parented under the trail.
    assert panel.parent() is not None


def test_registered_panel_survives_set_steps_rebuild(qapp: QApplication) -> None:
    """A host may register a panel before the final trail is built."""
    trail = DecisionTrail()
    panel = QLabel("candidates panel")
    trail.set_step_detail_widget("candidates", panel)
    trail.set_steps((TrailStep("candidates", "Candidates fitted.", "candidates", ("detail",)),))
    trail.set_step_expanded("candidates", True)
    assert panel.isVisibleTo(trail) is True


def test_clear_does_not_delete_injected_panel(qapp: QApplication) -> None:
    """Rebuilding the trail must not destroy a host-owned panel it merely hosted."""
    trail = DecisionTrail()
    panel = QLabel("shared panel")
    trail.set_steps((TrailStep("verdict", "Verdict.", "verdict", ()),))
    trail.set_step_detail_widget("verdict", panel)
    # Rebuild with a fresh trail — the shared panel must remain alive.
    trail.set_steps((TrailStep("verdict", "New verdict.", "verdict", ()),))
    # Accessing the panel must not raise (it was detached, not deleted).
    assert panel.text() == "shared panel"
