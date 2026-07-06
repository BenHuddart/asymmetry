"""Reconciliation Phase 3 — frequency-panel UX (F4, F3-hint).

F4 replaces the two independent diamagnetic checkboxes (``remove_diamag`` in
Conditioning, ``diamag_exclusion`` in Exclusions) with a single mutually
exclusive three-way control (*Leave / Fit & subtract / Exclude band*).  The
schema keys stay readable, so the spectrum a given path produces must not move:
the tests below pin the three single-path spectra captured *before* the control
refactor — their pairwise distinctness and per-path invariants, computed
in-process so the pin is platform-stable (an exact float byte-hash drifts across
numpy/FFT backends).  The GUI tests then assert the new control maps onto those
same keys, including the legacy both-keys-set project.
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.fourier.spectrum import (
    GroupSpectrumConfig,
    compute_average_group_spectrum,
)
from asymmetry.core.fourier.units import gauss_to_mhz

from .test_fourier_finishers import _tf_run

_FIELD_GAUSS = 200.0
_DIAMAG_HALF_WIDTH_MHZ = 0.3

_PATH_CONFIGS = {
    "leave": GroupSpectrumConfig(display="(Power)^1/2"),
    "subtract": GroupSpectrumConfig(display="(Power)^1/2", remove_diamag=True),
    "band": GroupSpectrumConfig(
        display="(Power)^1/2",
        diamag_exclusion=True,
        diamag_half_width_mhz=_DIAMAG_HALF_WIDTH_MHZ,
    ),
}


def _spectra() -> dict[str, object]:
    run = _tf_run(field_gauss=_FIELD_GAUSS)
    return {name: compute_average_group_spectrum(run, cfg) for name, cfg in _PATH_CONFIGS.items()}


def test_single_path_invariants_pinned() -> None:
    """Each diamag path keeps its defining behaviour through the control refactor."""
    spectra = _spectra()
    leave, subtract, band = spectra["leave"], spectra["subtract"], spectra["band"]
    assert leave is not None and subtract is not None and band is not None

    # Leave: no diamag handling at all.
    assert "fourier_diamag_field_gauss" not in leave.metadata
    assert "fourier_diamag_skipped" not in leave.metadata

    # Fit & subtract: the line is fitted (field recovered) and removed pre-FFT.
    assert subtract.metadata["fourier_diamag_field_gauss"] == pytest.approx(_FIELD_GAUSS, abs=5.0)
    assert "fourier_diamag_skipped" not in subtract.metadata

    # Exclude band: the displayed spectrum is hard-zeroed in the Larmor window.
    ref_mhz = float(gauss_to_mhz(_FIELD_GAUSS))
    inside = np.abs(band.time - ref_mhz) <= _DIAMAG_HALF_WIDTH_MHZ
    assert np.any(inside)
    assert np.allclose(band.asymmetry[inside], 0.0)
    # The band is post-FFT only: it does not touch the time domain, so no fit.
    assert "fourier_diamag_field_gauss" not in band.metadata


def test_single_paths_are_distinct() -> None:
    """Leave, subtract, and exclude-band genuinely produce different spectra."""
    spectra = _spectra()
    arrays = {name: np.asarray(ds.asymmetry, dtype=float) for name, ds in spectra.items()}
    # Same frequency axis, pairwise-different amplitudes.
    assert not np.array_equal(arrays["leave"], arrays["subtract"])
    assert not np.array_equal(arrays["leave"], arrays["band"])
    assert not np.array_equal(arrays["subtract"], arrays["band"])


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
    # The inherited-background hint now renders as the ActionFooter hint line.
    hint = panel._action_footer._hint_label
    panel.set_background_hint(None)
    assert hint.text() == "Background: off"
    panel.set_background_hint("tail-fit")
    assert hint.text() == "Background: tail-fit, inherited from grouping"


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
