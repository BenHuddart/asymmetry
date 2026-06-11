"""Reconciliation Phase 3 — frequency-panel UX (F4, F3-hint).

F4 replaces the two independent diamagnetic checkboxes (``remove_diamag`` in
Conditioning, ``diamag_exclusion`` in Exclusions) with a single mutually
exclusive three-way control (*Leave / Fit & subtract / Exclude band*).  The
schema keys stay readable, so the spectrum a given path produces must not move:
the hashes below pin the three single-path spectra captured *before* the control
refactor.  The GUI tests then assert the new control maps onto those same keys,
including the legacy both-keys-set project.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from asymmetry.core.fourier.spectrum import (
    GroupSpectrumConfig,
    compute_average_group_spectrum,
)

from .test_fourier_finishers import _tf_run

# Golden md5 of the averaged spectrum (200 G TF run) for each single diamag
# path, captured before the F4 control refactor.  The shared frequency axis is
# identical across paths.
_TIME_HASH = "fe6072bc95cda39c7ad82c0c4b8db0d5"
_PATH_HASHES = {
    "leave": "0cd63232b5d7475c71804784a05d5e75",
    "subtract": "2f6c9b8f378d8fd50896ffeb4e99f8d4",
    "band": "f790fef3aabb631012af560123ed1df4",
}

_PATH_CONFIGS = {
    "leave": GroupSpectrumConfig(display="(Power)^1/2"),
    "subtract": GroupSpectrumConfig(display="(Power)^1/2", remove_diamag=True),
    "band": GroupSpectrumConfig(
        display="(Power)^1/2", diamag_exclusion=True, diamag_half_width_mhz=0.3
    ),
}


def _md5(values: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(values, dtype=float).tobytes()).hexdigest()


def test_single_path_spectra_pinned() -> None:
    """Each diamag path's spectrum is unchanged by the control refactor."""
    run = _tf_run(field_gauss=200.0)
    for name, config in _PATH_CONFIGS.items():
        ds = compute_average_group_spectrum(run, config)
        assert ds is not None
        assert _md5(ds.time) == _TIME_HASH, name
        assert _md5(ds.asymmetry) == _PATH_HASHES[name], name


def test_single_paths_are_distinct() -> None:
    """Leave, subtract, and exclude-band genuinely differ — the pins are real."""
    assert len({*_PATH_HASHES.values()}) == 3


# ── F4: the three-way control maps onto the two readable schema keys ──────


def test_diamag_combo_maps_to_schema_keys(qapp) -> None:
    from asymmetry.gui.panels.fourier_panel import FourierPanel

    panel = FourierPanel()

    panel._set_diamag_mode("leave")
    state = panel.get_state()
    assert state["remove_diamag"] is False
    assert state["diamag_exclusion"] is False
    assert panel._diamag_width_edit.isEnabled() is False

    panel._set_diamag_mode("subtract")
    state = panel.get_state()
    assert state["remove_diamag"] is True
    assert state["diamag_exclusion"] is False
    assert panel._diamag_width_edit.isEnabled() is False

    panel._set_diamag_mode("band")
    state = panel.get_state()
    assert state["remove_diamag"] is False
    assert state["diamag_exclusion"] is True
    # The band half-width is editable only when the band is selected.
    assert panel._diamag_width_edit.isEnabled() is True


def test_diamag_modes_are_mutually_exclusive(qapp) -> None:
    """No combo selection ever yields both keys true at once."""
    from asymmetry.gui.panels.fourier_panel import FourierPanel

    panel = FourierPanel()
    for mode in ("leave", "subtract", "band"):
        panel._set_diamag_mode(mode)
        state = panel.get_state()
        assert not (state["remove_diamag"] and state["diamag_exclusion"])


def test_diamag_state_roundtrip(qapp) -> None:
    from asymmetry.gui.panels.fourier_panel import FourierPanel

    for mode in ("leave", "subtract", "band"):
        source = FourierPanel()
        source._set_diamag_mode(mode)
        source._diamag_width_edit.setText("0.45")
        restored = FourierPanel()
        restored.restore_state(source.get_state())
        assert restored._diamag_mode() == mode
        assert restored.get_state() == source.get_state()


def test_diamag_legacy_both_keys_load_as_subtract(qapp) -> None:
    """A legacy project with both keys set loads as Fit & subtract, band kept."""
    from asymmetry.gui.panels.fourier_panel import FourierPanel

    panel = FourierPanel()
    panel.restore_state(
        {
            "remove_diamag": True,
            "diamag_exclusion": True,
            "diamag_half_width_mhz": 0.7,
        }
    )
    assert panel._diamag_mode() == "subtract"
    # The band half-width is preserved (noted, not discarded).
    assert float(panel._diamag_width_edit.text()) == pytest.approx(0.7)
    # Re-saving collapses to the single active path (mutual exclusivity).
    state = panel.get_state()
    assert state["remove_diamag"] is True
    assert state["diamag_exclusion"] is False
    assert state["diamag_half_width_mhz"] == pytest.approx(0.7)


# ── F3: inherited grouping-background hint ────────────────────────────────


def test_background_hint_label(qapp) -> None:
    from asymmetry.gui.panels.fourier_panel import FourierPanel

    panel = FourierPanel()
    panel.set_background_hint(None)
    assert panel._background_hint_label.text() == "Background: off"
    panel.set_background_hint("tail-fit")
    assert panel._background_hint_label.text() == "Background: tail-fit, inherited from grouping"


def test_background_hint_helper_reflects_grouping(qapp) -> None:
    from asymmetry.core.data.dataset import MuonDataset
    from asymmetry.gui import mainwindow as mw_module

    run = _tf_run(field_gauss=200.0)
    dataset = MuonDataset(
        time=np.zeros(2),
        asymmetry=np.zeros(2),
        error=np.ones(2),
        metadata={"run_number": run.run_number},
        run=run,
    )
    window = mw_module.MainWindow()
    try:
        # Correction off → no hint.
        assert window._fourier_background_hint_for_dataset(dataset) is None
        # tail-fit is available everywhere; the gate + label map resolve it.
        run.grouping["background_correction"] = True
        run.grouping["background_mode"] = "tail_fit"
        assert window._fourier_background_hint_for_dataset(dataset) == "tail-fit"
        # reference_run "available" everywhere but applies nothing without a run.
        run.grouping["background_mode"] = "reference_run"
        run.grouping.pop("background_run", None)
        assert window._fourier_background_hint_for_dataset(dataset) is None
        run.grouping["background_run"] = 1234
        assert window._fourier_background_hint_for_dataset(dataset) == "reference run"
    finally:
        window.close()


# ── F4: the <5 G fit-and-subtract fallback is disclosed in metadata ───────


def test_diamag_skip_reason_recorded_below_threshold() -> None:
    run = _tf_run(field_gauss=200.0)
    run.metadata["field"] = 0.0
    ds = compute_average_group_spectrum(
        run, GroupSpectrumConfig(display="(Power)^1/2", remove_diamag=True)
    )
    assert "fourier_diamag_field_gauss" not in ds.metadata
    assert "below" in ds.metadata.get("fourier_diamag_skipped", "")


def test_diamag_skip_reason_absent_when_subtracted() -> None:
    run = _tf_run(field_gauss=200.0)
    ds = compute_average_group_spectrum(
        run, GroupSpectrumConfig(display="(Power)^1/2", remove_diamag=True)
    )
    assert "fourier_diamag_field_gauss" in ds.metadata
    assert "fourier_diamag_skipped" not in ds.metadata
