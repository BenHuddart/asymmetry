"""Generic layouts and manual grouping for unidentified instruments."""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

import asymmetry.gui.windows.grouping.dialog as grouping_dialog_dialog_module
from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.gui.styles import tokens
from asymmetry.gui.windows.grouping_dialog import GroupingDialog

pytestmark = [pytest.mark.gui]


def _unknown_instrument_dataset(
    grouping_extra: dict | None = None, *, n_histograms: int = 16
) -> MuonDataset:
    """A run from an instrument Asymmetry has no layout for."""
    histograms = [Histogram(counts=np.full(4, 100.0), bin_width=0.01) for _ in range(n_histograms)]
    metadata = {"run_number": 7001, "facility": "TRIUMF", "instrument": "M20"}
    run = Run(
        run_number=7001,
        histograms=histograms,
        metadata=dict(metadata),
        grouping={
            "groups": {1: list(range(1, 9)), 2: list(range(9, 17))},
            "forward_group": 1,
            "backward_group": 2,
            "instrument": "M20",
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": 3,
        }
        | (grouping_extra or {}),
    )
    t = np.array([0.0, 0.01, 0.02, 0.03])
    return MuonDataset(
        time=t,
        asymmetry=np.zeros_like(t),
        error=np.full_like(t, 0.01),
        metadata=dict(metadata),
        run=run,
    )


@pytest.mark.parametrize("n_histograms", [16, 64, 96])
def test_unknown_instrument_gets_a_generic_layout_without_presets(
    qapp: QApplication, n_histograms: int
) -> None:
    """Unknown identity keeps the run's detector count and offers no presets."""
    dialog = GroupingDialog([_unknown_instrument_dataset(n_histograms=n_histograms)])
    layout = dialog._current_instrument_layout()
    assert layout.name == "Generic"
    assert layout.n_detectors == n_histograms
    assert dialog._preset_combo.count() == 0
    assert dialog._preset_combo.isEnabled() is False
    assert dialog._preset_combo.placeholderText() == "No presets for this instrument"


def test_unknown_instrument_keeps_a_stored_hifi_fallback_grouping(qapp: QApplication) -> None:
    """A project saved under the old HiFi fallback opens with its groups intact."""
    dataset = _unknown_instrument_dataset({"grouping_preset": "Longitudinal"})
    dialog = GroupingDialog([dataset])
    assert dialog._groups == {1: list(range(0, 8)), 2: list(range(8, 16))}
    # "Longitudinal" is no preset of this run's layout, so it reads as Custom.
    assert dialog._grouping_preset_name is None
    assert dialog._preset_combo.currentIndex() == -1


def test_typed_detector_list_updates_the_draft(qapp: QApplication) -> None:
    dialog = GroupingDialog([_unknown_instrument_dataset()])
    dialog._group_table.item(0, 3).setText("1-4, 16")

    assert dialog._groups[1] == [0, 1, 2, 3, 15]
    assert dialog._current_grouping_payload()["groups"][1] == [1, 2, 3, 4, 16]
    assert dialog._group_table.item(0, 3).text() == "1, 2, 3, 4, 16"
    assert dialog._draft_dirty is True


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        (
            "1-20",
            "17-20 do not exist",
        ),
        ("", "needs at least one detector"),
        ("1, x", "Cannot read 'x' as a detector id or range"),
    ],
)
def test_invalid_detector_list_leaves_the_draft_and_marks_the_cell(
    qapp: QApplication, text: str, reason: str
) -> None:
    dialog = GroupingDialog([_unknown_instrument_dataset()])
    item = dialog._group_table.item(0, 3)
    item.setText(text)

    assert dialog._groups[1] == list(range(0, 8))
    assert reason in item.toolTip()
    assert item.background().color().name() == tokens.ERROR_SOFT

    # Correcting the cell clears the mark and takes the edit.
    item.setText("1-3")
    assert dialog._groups[1] == [0, 1, 2]
    assert item.toolTip() == grouping_dialog_dialog_module._DETECTORS_CELL_TIP
    assert item.data(Qt.ItemDataRole.BackgroundRole) is None


def test_typed_group_name_updates_the_draft_and_the_pair_pickers(qapp: QApplication) -> None:
    dialog = GroupingDialog([_unknown_instrument_dataset()])
    dialog._group_table.item(0, 2).setText("  Upstream ")

    assert dialog._group_names[1] == "Upstream"
    assert dialog._forward_combo.itemText(dialog._forward_combo.findData(1)) == "1: Upstream"

    dialog._group_table.item(0, 2).setText("")
    assert 1 not in dialog._group_names


def test_group_id_cell_is_not_editable(qapp: QApplication) -> None:
    dialog = GroupingDialog([_unknown_instrument_dataset()])
    flags = dialog._group_table.item(0, 0).flags()
    assert not flags & Qt.ItemFlag.ItemIsEditable
