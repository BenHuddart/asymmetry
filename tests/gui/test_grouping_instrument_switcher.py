"""Tests for the Instrument switcher in the grouping window (M2).

The grouping editor opened on the first/selected run's instrument. This adds an
Instrument combo in the profile row that lists every instrument present in the
loaded datasets and swaps the whole editor — draft, preview run, scope panel,
and preset list — when the user picks another one.
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run

pytestmark = [pytest.mark.gui]

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QMessageBox

from asymmetry.gui.windows.grouping_dialog import GroupingDialog


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _dataset(run_number: int, *, instrument: str, n_hist: int = 2) -> MuonDataset:
    histograms = [
        Histogram(counts=np.full(4, 100.0 - 10.0 * i), bin_width=0.01) for i in range(n_hist)
    ]
    run = Run(
        run_number=run_number,
        histograms=histograms,
        metadata={"run_number": run_number, "title": f"{instrument} {run_number}"},
        grouping={
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": 3,
            "instrument": instrument,
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


def test_single_instrument_hides_switcher(qapp: QApplication) -> None:
    """One instrument: the switcher (and its label) are hidden."""
    dialog = GroupingDialog([_dataset(1, instrument="MuSR"), _dataset(2, instrument="MuSR")])
    # The dialog is never shown offscreen, so use isHidden() (explicit hide state)
    # rather than isVisible() (which is False until a top-level window is shown).
    assert dialog._instrument_combo.isHidden() is True
    assert dialog._instrument_label.isHidden() is True


def test_multi_instrument_lists_both(qapp: QApplication) -> None:
    """Two instruments: the switcher lists both with their run counts."""
    dialog = GroupingDialog(
        [
            _dataset(1, instrument="MuSR"),
            _dataset(2, instrument="MuSR"),
            _dataset(3, instrument="GPS"),
        ]
    )
    combo = dialog._instrument_combo
    assert combo.isHidden() is False
    labels = [combo.itemText(i) for i in range(combo.count())]
    assert labels == ["MuSR — 2 runs", "GPS — 1 run"]


def test_switch_swaps_fingerprint_scope_and_preview(qapp: QApplication) -> None:
    """Switching instrument swaps the fingerprint, scope population and preview run."""
    dialog = GroupingDialog(
        [
            _dataset(1, instrument="MuSR"),
            _dataset(2, instrument="MuSR"),
            _dataset(3, instrument="GPS"),
        ]
    )
    assert dialog._fingerprint.instrument == "MuSR"
    assert dialog._scope_panel.inheriting_run_numbers() == {1, 2}

    # Activate the GPS entry (index 1).
    dialog._instrument_combo.setCurrentIndex(1)
    dialog._on_instrument_combo_activated(1)

    assert dialog._fingerprint.instrument == "GPS"
    assert int(dialog._reference_dataset.run_number) == 3
    assert dialog._scope_panel.inheriting_run_numbers() == {3}
    # The preview-run combo now lists only the GPS run.
    combo = dialog._reference_combo
    assert [combo.itemData(i) for i in range(combo.count())] == [3]


def test_switch_swaps_draft_and_presets(qapp: QApplication) -> None:
    """Switching instrument swaps the draft (default name) and rebuilds presets."""
    dialog = GroupingDialog(
        [
            _dataset(1, instrument="MuSR", n_hist=2),
            _dataset(3, instrument="GPS", n_hist=6),
        ]
    )
    assert "MuSR" in dialog._draft_name

    # Give the GPS instrument a saved active profile with a distinctive name, so
    # the switch is proven to adopt that instrument's own draft, not the MuSR one.
    from asymmetry.core.project.profiles import GroupingProfile, ProfileFingerprint

    gps_fp = ProfileFingerprint(instrument="GPS", histogram_count=6)
    dialog._project_profiles.append(
        GroupingProfile(name="GPS silver", fingerprint=gps_fp, active=True)
    )

    dialog._instrument_combo.setCurrentIndex(1)
    dialog._on_instrument_combo_activated(1)

    # The draft is now the GPS instrument's active profile.
    assert dialog._draft_name == "GPS silver"
    # The preset dropdown was rebuilt for the new instrument (non-empty).
    assert dialog._preset_combo.count() > 0


def test_dirty_draft_prompts_before_switch(qapp: QApplication, monkeypatch) -> None:
    """A dirty draft prompts to discard before an instrument switch; cancel aborts."""
    dialog = GroupingDialog(
        [
            _dataset(1, instrument="MuSR"),
            _dataset(3, instrument="GPS"),
        ]
    )
    dialog._draft_dirty = True

    # User cancels the discard prompt: the switch is aborted.
    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Cancel),
    )
    dialog._instrument_combo.setCurrentIndex(1)
    dialog._on_instrument_combo_activated(1)
    assert dialog._fingerprint.instrument == "MuSR"
    # The combo selection is restored to the current instrument.
    assert dialog._instrument_combo.currentIndex() == 0

    # User accepts the discard: the switch proceeds.
    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Discard),
    )
    dialog._draft_dirty = True
    dialog._on_instrument_combo_activated(1)
    assert dialog._fingerprint.instrument == "GPS"
