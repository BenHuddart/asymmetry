"""Tests for the Corrections-tab compare pager (M3 of the corrections-tab UX plan).

The pager (`◀`/`▶` + a muted label, built by
:meth:`GroupingDialog._build_compare_pager`) steps ``_compare_stage`` through the
cycle ``[None, "deadtime", "background", "alpha", "raw"]``, skipping stages
:meth:`GroupingDialog._compare_stage_available` rejects. It rides the same
``_compare_stage`` the pipeline chips and the pager-row "raw" checkbox drive,
and is refreshed from the single :meth:`GroupingDialog._sync_compare_toggles`
sync seam — see ``docs/porting/correction-order-alpha-estimation/
corrections-tab-ux-plan.md`` (M3; per-section checkboxes retired by the
correction-cards milestone).
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
from asymmetry.gui.windows.grouping_dialog import GroupingDialog


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _dataset_with_histograms() -> MuonDataset:
    h1 = Histogram(counts=np.array([100.0, 100.0, 100.0, 100.0]), bin_width=0.01)
    h2 = Histogram(counts=np.array([50.0, 50.0, 50.0, 50.0]), bin_width=0.01)
    run = Run(
        run_number=4001,
        histograms=[h1, h2],
        metadata={"run_number": 4001, "title": "Compare Pager Test"},
        grouping={
            "groups": {1: [1], 2: [2]},
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
        metadata={"run_number": 4001},
        run=run,
    )


def _vector_dataset_with_histograms(run_number: int = 4010) -> MuonDataset:
    h1 = Histogram(counts=np.array([100.0, 100.0, 100.0, 100.0]), bin_width=0.01)
    h2 = Histogram(counts=np.array([50.0, 50.0, 50.0, 50.0]), bin_width=0.01)
    run = Run(
        run_number=run_number,
        histograms=[h1, h2],
        metadata={"run_number": run_number, "title": "Vector Compare Pager Test"},
        grouping={
            "groups": {
                1: [1],
                2: [2],
                3: [1],
                4: [2],
                5: [1],
                6: [2],
            },
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
            "alpha_x": 1.1,
            "alpha_y": 1.2,
            "alpha_z": 1.3,
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


def test_pager_disabled_and_off_on_a_fresh_dialog(qapp: QApplication) -> None:
    """Nothing configured (deadtime off, background none, α = 1): both arrows
    disabled and the label reads "off"; stepping is a no-op."""
    dialog = GroupingDialog([_dataset_with_histograms()])

    assert dialog._compare_stage is None
    assert dialog._compare_pager_label.text() == "Comparing: off"
    assert not dialog._compare_prev_btn.isEnabled()
    assert not dialog._compare_next_btn.isEnabled()

    dialog._step_compare(1)
    assert dialog._compare_stage is None
    dialog._step_compare(-1)
    assert dialog._compare_stage is None


def test_pager_cycles_forward_skipping_deadtime(qapp: QApplication) -> None:
    """With deadtime off, background configured, and α ≠ 1: ▶ from None walks
    None -> background -> alpha -> raw -> None, and the label matches each stop.
    """
    dialog = GroupingDialog([_dataset_with_histograms()])
    dialog._background_mode = "range"
    dialog._alpha_spin.setValue(1.2)
    dialog._set_compare_stage(None)  # reset any auto-focus from the α edit

    assert dialog._compare_stage_available("background")
    assert dialog._compare_stage_available("alpha")
    assert dialog._compare_stage_available("raw")
    assert not dialog._compare_stage_available("deadtime")

    dialog._step_compare(1)
    assert dialog._compare_stage == "background"
    assert dialog._compare_pager_label.text() == "Comparing: without background (1/3)"

    dialog._step_compare(1)
    assert dialog._compare_stage == "alpha"
    assert dialog._compare_pager_label.text() == "Comparing: α = 1 (2/3)"

    dialog._step_compare(1)
    assert dialog._compare_stage == "raw"
    assert dialog._compare_pager_label.text() == "Comparing: vs raw (3/3)"

    # Wraps back to off.
    dialog._step_compare(1)
    assert dialog._compare_stage is None
    assert dialog._compare_pager_label.text() == "Comparing: off"


def test_pager_cycles_backward(qapp: QApplication) -> None:
    """◀ from off walks the cycle in reverse: None -> raw -> alpha -> background."""
    dialog = GroupingDialog([_dataset_with_histograms()])
    dialog._background_mode = "range"
    dialog._alpha_spin.setValue(1.2)
    dialog._set_compare_stage(None)

    dialog._step_compare(-1)
    assert dialog._compare_stage == "raw"

    dialog._step_compare(-1)
    assert dialog._compare_stage == "alpha"

    dialog._step_compare(-1)
    assert dialog._compare_stage == "background"

    dialog._step_compare(-1)
    assert dialog._compare_stage is None


def test_pager_arrows_enabled_but_deadtime_unreachable_when_off(qapp: QApplication) -> None:
    """Arrows are enabled whenever anything is available, even though the
    unavailable "deadtime" stage itself is never a landing stage."""
    dialog = GroupingDialog([_dataset_with_histograms()])
    dialog._background_mode = "range"
    dialog._set_compare_stage(None)

    assert dialog._compare_prev_btn.isEnabled()
    assert dialog._compare_next_btn.isEnabled()

    for _ in range(4):
        dialog._step_compare(1)
        assert dialog._compare_stage != "deadtime"


def test_pager_skips_alpha_in_vector_mode(qapp: QApplication) -> None:
    """The α compare is unavailable in vector mode (its per-projection table
    lives on the Grouping tab), so the pager cycle skips it even though α is
    otherwise available (calibrated / off-unity)."""
    dialog = GroupingDialog([_vector_dataset_with_histograms()])
    dialog._background_mode = "range"
    dialog._set_compare_stage(None)

    assert dialog._vector_axis_pairs
    assert not dialog._compare_stage_available("alpha")

    seen = []
    for _ in range(4):
        dialog._step_compare(1)
        seen.append(dialog._compare_stage)
        if dialog._compare_stage is None:
            break

    assert "alpha" not in seen
    assert "background" in seen
    assert "raw" in seen


def test_pager_label_syncs_from_a_pipeline_chip(qapp: QApplication) -> None:
    """Clicking a stage's pipeline chip drives the same shared ``_compare_stage``,
    and the pager label reflects it via the shared sync seam. (The per-section
    checkboxes are retired; chips + pager are the compare controls.)
    """
    dialog = GroupingDialog([_dataset_with_histograms()])
    dialog._background_mode = "range"
    dialog._set_compare_stage(None)

    dialog._on_pipeline_chip_clicked("background")

    assert dialog._compare_stage == "background"
    assert dialog._compare_pager_label.text().startswith("Comparing: without background")


def test_pager_step_reaches_the_preview_request(qapp: QApplication) -> None:
    """Stepping to a stage forwards into the preview request's ``compare_stage``
    (mirrors ``test_compare_toggle_reaches_the_preview_request``)."""
    dialog = GroupingDialog([_dataset_with_histograms()])
    dialog._background_mode = "range"
    dialog._alpha_spin.setValue(1.2)
    dialog._set_compare_stage(None)

    dialog._step_compare(1)
    assert dialog._compare_stage == "background"
    dialog._refresh_preview()
    assert dialog._preview_pane._pending is not None
    assert dialog._preview_pane._pending.compare_stage == "background"

    dialog._step_compare(1)
    assert dialog._compare_stage == "alpha"
    dialog._refresh_preview()
    assert dialog._preview_pane._pending.compare_stage == "alpha"
