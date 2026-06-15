"""GUI + persistence tests for the spectral-moments feature.

Covers eligibility gating, the on-plot window overlay, send-to-trend recording a
computed ``FitSeries`` (replace-on-resend), the ``.asymp`` round-trip of the
series + recipe, and live-setting persistence without a schema bump. See
``docs/porting/spectral-moments/verification-plan.md`` (C9–C14).
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.project.schema import CURRENT_SCHEMA_VERSION
from asymmetry.core.representation.base import RepresentationType
from asymmetry.core.representation.project_model import ProjectModel
from asymmetry.gui.mainwindow import MainWindow

pytestmark = pytest.mark.gui


def _spectrum(run: int, *, display: str = "phase_corrected") -> MuonDataset:
    f = np.linspace(20.0, 60.0, 1001)
    amp = np.exp(-0.5 * ((f - 40.0) / 4.0) ** 2) + 0.3 * np.exp(-0.5 * ((f - 48.0) / 5.0) ** 2)
    return MuonDataset(
        time=f,
        asymmetry=amp,
        error=0.01 * np.ones_like(f),
        metadata={
            "fourier_display": display,
            "field": 3000.0,
            "temperature": 5.0,
            "run_number": run,
            "run_label": f"R{run}",
        },
    )


class _FakeBrowser:
    """Minimal data-browser stand-in exposing the selected datasets."""

    def __init__(self, datasets):
        self._datasets = list(datasets)

    def get_selected_datasets(self):
        return list(self._datasets)


def _activate_fft_spectrum(window: MainWindow, runs, *, display="phase_corrected"):
    cache = window._frequency_cache(RepresentationType.FREQ_FFT)
    datasets = [_spectrum(r, display=display) for r in runs]
    for ds in datasets:
        cache[int(ds.run_number)] = [ds]
    window._frequency_plot_panel.plot_dataset(datasets[0])
    window._data_browser = _FakeBrowser(datasets)
    window._refresh_spectral_moments()
    return datasets


@pytest.fixture
def window():
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    win = MainWindow()
    yield win
    win.close()


# ── C9: eligibility guard ───────────────────────────────────────────────────


def test_eligible_for_phase_corrected_fft(window):
    _activate_fft_spectrum(window, [1], display="phase_corrected")
    assert window._fourier_panel.moments_widget.is_eligible() is True


@pytest.mark.parametrize("mode", ["power", "magnitude", "phase_spectrum", "burg", "correlation"])
def test_ineligible_modes_grey_out_with_reason(window, mode):
    _activate_fft_spectrum(window, [1], display=mode)
    widget = window._fourier_panel.moments_widget
    assert widget.is_eligible() is False
    assert widget.toolTip()  # explanatory tooltip present
    assert not window._frequency_plot_panel._moments_overlay_visible


def test_inactive_representation_widget_greyed(window):
    _activate_fft_spectrum(window, [1])
    # FFT is active, so the MaxEnt widget must be greyed.
    assert window._maxent_panel.moments_widget.is_eligible() is False


# ── C14: window visible on the plot ─────────────────────────────────────────


def test_eligible_spectrum_draws_window_overlay(window):
    _activate_fft_spectrum(window, [1])
    panel = window._frequency_plot_panel
    assert panel._moments_overlay_visible is True
    assert len(panel._moments_span_artists) == 3  # span + two handles


def test_ineligible_clears_overlay(window):
    _activate_fft_spectrum(window, [1])
    assert window._frequency_plot_panel._moments_overlay_visible is True
    _activate_fft_spectrum(window, [1], display="power")
    assert window._frequency_plot_panel._moments_overlay_visible is False


# ── C10/C11: send to trend records a computed series; resend replaces ────────


def _moments_batches(window):
    return [b for bid, b in window._project_model.batches.items() if bid.startswith("moments-")]


def test_send_to_trend_records_computed_series(window):
    _activate_fft_spectrum(window, [1, 2])
    widget = window._fourier_panel.moments_widget
    window._on_moments_send_to_trend(widget)
    batches = _moments_batches(window)
    assert len(batches) == 1
    series = batches[0]
    assert series.canonical_model is None  # computed series
    assert series.is_computed is True
    assert sorted(series.member_run_numbers) == [1, 2]
    assert series.rep_type == RepresentationType.FREQ_FFT
    row = series.results_by_run[1]
    assert row["success"] is True
    assert "B_rms_mean" in row["parameters"] and "beta" in row["parameters"]
    assert series.extra.get("moments_recipe", {}).get("unit") == "gauss"


def test_resending_same_selection_replaces(window):
    _activate_fft_spectrum(window, [1, 2])
    widget = window._fourier_panel.moments_widget
    window._on_moments_send_to_trend(widget)
    first_ids = [b.batch_id for b in _moments_batches(window)]
    window._on_moments_send_to_trend(widget)
    second_ids = [b.batch_id for b in _moments_batches(window)]
    assert first_ids == second_ids
    assert len(second_ids) == 1


def test_different_selection_is_a_new_series(window):
    datasets = _activate_fft_spectrum(window, [1, 2])
    widget = window._fourier_panel.moments_widget
    window._on_moments_send_to_trend(widget)
    # Narrow the selection → a different member set → a different series.
    window._data_browser = _FakeBrowser([datasets[0]])
    window._on_moments_send_to_trend(widget)
    assert len(_moments_batches(window)) == 2


# ── C12: .asymp round-trip of the series + recipe ───────────────────────────


def test_moments_series_round_trips_through_project_model(window):
    _activate_fft_spectrum(window, [1, 2])
    widget = window._fourier_panel.moments_widget
    widget.set_cutoff_fraction(0.1)
    window._on_moments_send_to_trend(widget)
    original = _moments_batches(window)[0]

    restored_model = ProjectModel.from_dict(window._project_model.to_dict())
    restored = restored_model.batches[original.batch_id]
    assert sorted(restored.member_run_numbers) == [1, 2]
    assert restored.canonical_model is None
    assert restored.results_by_run[1]["parameters"]["B_ave"] == pytest.approx(
        original.results_by_run[1]["parameters"]["B_ave"]
    )
    recipe = restored.extra.get("moments_recipe", {})
    assert recipe.get("cutoff_fraction") == pytest.approx(0.1)
    assert recipe.get("unit") == "gauss"


# ── C13: live settings persist without a schema bump ────────────────────────


def test_live_settings_persist_in_fourier_state(window):
    widget = window._fourier_panel.moments_widget
    widget.set_cutoff_fraction(0.2)
    widget.set_range_mhz(25.0, 55.0)
    state = window._fourier_panel.get_state()
    assert "moments" in state
    assert state["moments"]["cutoff_fraction"] == pytest.approx(0.2)
    # Restore into a fresh panel tolerates and applies it.
    from asymmetry.gui.panels.fourier_panel import FourierPanel

    fresh = FourierPanel()
    fresh.restore_state(state)
    assert fresh.moments_widget.cutoff_fraction() == pytest.approx(0.2)
    assert fresh.moments_widget.range_mhz() == (25.0, 55.0)


def test_resending_in_a_different_unit_replaces_not_duplicates(window):
    from asymmetry.core.fourier.units import FieldUnit

    _activate_fft_spectrum(window, [1, 2])
    widget = window._fourier_panel.moments_widget
    window._on_moments_send_to_trend(widget)
    first = [b.batch_id for b in _moments_batches(window)]
    # Same selection + window, different display unit → still one series.
    widget._unit = FieldUnit.MHZ
    window._on_moments_send_to_trend(widget)
    second = [b.batch_id for b in _moments_batches(window)]
    assert first == second
    assert len(second) == 1


def test_non_overlapping_spectrum_resets_window_to_full():
    from asymmetry.gui.panels.spectral_moments_widget import SpectralMomentsWidget

    w = SpectralMomentsWidget()
    w.set_spectrum_bounds(20.0, 60.0)
    w.set_range_mhz(40.0, 55.0)  # a window inside run A
    # Switch to a run whose spectrum does not overlap [40,55].
    w.set_spectrum_bounds(100.0, 200.0)
    lo, hi = w.range_mhz()
    assert lo < hi  # not inverted
    assert (lo, hi) == (100.0, 200.0)  # fell back to the new full extent


def test_no_schema_version_bump():
    # Baseline guard: bumped to 10 for data-browser custom/renamable columns
    # (browser_state.extra_columns generalised from a key list to column-def
    # dicts). A future accidental bump must consciously update this literal.
    assert CURRENT_SCHEMA_VERSION == 10


def test_restore_state_tolerates_absent_moments():
    from asymmetry.gui.panels.fourier_panel import FourierPanel

    panel = FourierPanel()
    panel.restore_state({})  # no "moments" key — must not raise
    assert panel.moments_widget.cutoff_fraction() == 0.0
