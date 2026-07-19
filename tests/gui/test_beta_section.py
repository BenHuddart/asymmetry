"""Tests for the β (asymmetry balance) card in the grouping Corrections editor.

Pins the GUI items of ``docs/porting/beta-correction/verification-plan.md``:
card wiring + payload round-trip (emit-only-when-≠1), dirty marking, the
β compare stage (availability, payload invariance, pager stop), and the
scalar-only vector-mode behaviour (card hidden, payload omits the key).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.gui.windows.grouping.beta_section import BetaSectionWidget, beta_status_text
from asymmetry.gui.windows.grouping_dialog import GroupingDialog


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _dataset(*, beta: float | None = None, run_number: int = 4101) -> MuonDataset:
    h1 = Histogram(counts=np.array([100.0, 100.0, 100.0, 100.0]), bin_width=0.01)
    h2 = Histogram(counts=np.array([50.0, 50.0, 50.0, 50.0]), bin_width=0.01)
    grouping: dict = {
        "groups": {1: [1], 2: [2]},
        "forward_group": 1,
        "backward_group": 2,
        "alpha": 1.0,
        "first_good_bin": 0,
        "last_good_bin": 3,
    }
    if beta is not None:
        grouping["beta"] = beta
    run = Run(
        run_number=run_number,
        histograms=[h1, h2],
        metadata={"run_number": run_number, "title": "Beta Test"},
        grouping=grouping,
    )
    t = np.array([0.0, 0.01, 0.02, 0.03])
    return MuonDataset(
        time=t,
        asymmetry=np.zeros_like(t),
        error=np.full_like(t, 0.01),
        metadata={"run_number": run_number},
        run=run,
    )


def _vector_dataset(run_number: int = 4110) -> MuonDataset:
    h1 = Histogram(counts=np.array([100.0, 100.0, 100.0, 100.0]), bin_width=0.01)
    h2 = Histogram(counts=np.array([50.0, 50.0, 50.0, 50.0]), bin_width=0.01)
    run = Run(
        run_number=run_number,
        histograms=[h1, h2],
        metadata={"run_number": run_number, "title": "Vector Beta Test"},
        grouping={
            "groups": {1: [1], 2: [2], 3: [1], 4: [2], 5: [1], 6: [2]},
            "group_names": {
                1: "Pz Forward",
                2: "Pz Backward",
                3: "Py Top",
                4: "Py Bottom",
                5: "Px Left",
                6: "Px Right",
            },
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": 3,
        },
    )
    t = np.array([0.0, 0.01, 0.02, 0.03])
    return MuonDataset(
        time=t,
        asymmetry=np.zeros_like(t),
        error=np.full_like(t, 0.01),
        metadata={"run_number": run_number},
        run=run,
    )


# --------------------------------------------------------------------------- #
# Widget unit behaviour
# --------------------------------------------------------------------------- #


def test_beta_widget_value_roundtrip_and_leniency(qapp: QApplication) -> None:
    widget = BetaSectionWidget()
    assert widget.value() == 1.0
    assert not widget.is_active()
    widget.set_value(0.9)
    assert widget.value() == pytest.approx(0.9)
    assert widget.is_active()
    for degenerate in (float("nan"), float("inf"), 0.0, -3.0, "x", None):
        widget.set_value(degenerate)
        assert widget.value() == 1.0


def test_beta_status_text_format() -> None:
    assert beta_status_text(1.0) == "β = 1.0000"
    assert beta_status_text(0.87654321) == "β = 0.8765"


# --------------------------------------------------------------------------- #
# Dialog wiring
# --------------------------------------------------------------------------- #


def test_beta_card_registered_and_default_payload_omits_beta(qapp: QApplication) -> None:
    dialog = GroupingDialog([_dataset()])
    assert "beta" in dialog._correction_cards
    assert dialog._correction_cards["beta"] is dialog._beta_card
    # Do-nothing default: the payload stays byte-identical to a pre-β one.
    assert "beta" not in dialog._current_grouping_payload()
    assert not dialog._correction_stage_active("beta")


def test_beta_edit_reaches_payload_and_marks_dirty(qapp: QApplication) -> None:
    dialog = GroupingDialog([_dataset()])
    dialog._clear_dirty()
    dialog._beta_section.set_value(0.9)
    payload = dialog._current_grouping_payload()
    assert payload["beta"] == pytest.approx(0.9)
    assert dialog._draft_dirty
    assert dialog._correction_stage_active("beta")
    # Back to the default removes the key again.
    dialog._beta_section.set_value(1.0)
    assert "beta" not in dialog._current_grouping_payload()


def test_beta_seeded_from_run_grouping(qapp: QApplication) -> None:
    dialog = GroupingDialog([_dataset(beta=1.2)])
    assert dialog._beta_section.value() == pytest.approx(1.2)
    assert dialog._current_grouping_payload()["beta"] == pytest.approx(1.2)


# --------------------------------------------------------------------------- #
# Compare stage
# --------------------------------------------------------------------------- #


def test_beta_compare_available_only_when_active(qapp: QApplication) -> None:
    dialog = GroupingDialog([_dataset()])
    assert not dialog._compare_stage_available("beta")
    dialog._beta_section.set_value(0.9)
    assert dialog._compare_stage_available("beta")
    # β ≠ 1 alone also makes the compound "vs raw" compare meaningful.
    assert dialog._compare_stage_available("raw")


def test_beta_compare_never_touches_the_persisted_payload(qapp: QApplication) -> None:
    dialog = GroupingDialog([_dataset(beta=0.9)])
    before = dialog._current_grouping_payload()
    dialog._set_compare_stage("beta")
    assert dialog._compare_stage == "beta"
    assert dialog._current_grouping_payload() == before
    dialog._set_compare_stage(None)
    assert dialog._current_grouping_payload() == before


def test_beta_is_a_pager_stop_in_pipeline_order(qapp: QApplication) -> None:
    dialog = GroupingDialog([_dataset(beta=0.9)])
    dialog._set_compare_stage(None)
    seen: list[str | None] = []
    for _ in range(6):
        dialog._step_compare(1)
        seen.append(dialog._compare_stage)
    assert "beta" in seen
    # β follows α (which is unavailable here at α = 1), i.e. β precedes "raw".
    assert seen.index("beta") < seen.index("raw")


# --------------------------------------------------------------------------- #
# Vector mode (scalar-only)
# --------------------------------------------------------------------------- #


def test_beta_hidden_and_omitted_in_vector_mode(qapp: QApplication) -> None:
    dialog = GroupingDialog([_vector_dataset()])
    assert bool(dialog._vector_axis_pairs)
    dialog._beta_section.set_value(0.9)
    assert dialog._beta_card.isHidden()
    assert not dialog._compare_stage_available("beta")
    assert not dialog._correction_stage_active("beta")
    assert "beta" not in dialog._current_grouping_payload()
