"""GUI tests for the RRF controls widget and its plot-panel integration.

Verification-plan items 7-8: view gating, unit conversion, auto-seed,
display transform application, fit-overlay demodulation, frame badge, and
plot_state["rrf"] persistence (additive, tolerant restore).
"""

from __future__ import annotations

import numpy as np
import pytest

pyside6 = pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from asymmetry.core.data.dataset import MuonDataset, Run
from asymmetry.core.fourier.units import FieldUnit, convert
from asymmetry.gui.panels.plot_panel import PlotPanel
from asymmetry.gui.widgets.rrf_controls import (
    rrf_display_dataset,
    rrf_display_fit_curve,
)

pytestmark = [pytest.mark.gui]

NU = 30.0  # MHz
LAM = 0.3


def _dataset(field_gauss: float = 2213.0, dt: float = 0.004) -> MuonDataset:
    t = np.arange(0.0, 8.0, dt)
    envelope = 25.0 * np.exp(-LAM * t)
    run = Run(
        run_number=4242,
        metadata={"field": field_gauss, "temperature": 1.5, "title": "rrf test"},
    )
    return MuonDataset(
        time=t,
        asymmetry=envelope * np.cos(2.0 * np.pi * NU * t),
        error=np.full_like(t, 0.5),
        metadata={},
        run=run,
    )


@pytest.fixture
def qapp() -> QApplication:
    app = QApplication.instance() or QApplication([])
    return app


@pytest.fixture
def panel(qapp: QApplication):
    widget = PlotPanel()
    if not widget._has_mpl:
        widget.close()
        widget.deleteLater()
        pytest.skip("matplotlib required")
    yield widget
    widget.close()
    widget.deleteLater()


def _enable_rrf(panel, frequency_mhz=NU, component="real"):
    controls = panel._rrf_controls
    # RRF is gated behind the Options → Advanced toggle (default off); the tests
    # that exercise the transform enable the feature explicitly.
    controls.set_feature_enabled(True)
    controls._freq_spin.setValue(frequency_mhz)
    idx = controls._component_combo.findData(component)
    controls._component_combo.setCurrentIndex(idx)
    controls._enable_check.setChecked(True)
    return controls


class TestViewGating:
    def test_visible_only_on_fb_asymmetry(self, panel):
        controls = panel._rrf_controls
        controls.set_feature_enabled(True)  # gate is exercised in TestFeatureGate
        panel.set_time_view_modes(["fb_asymmetry", "groups", "raw_counts"])
        controls.set_active_view_token("fb_asymmetry")
        assert controls.applies_to_current_view()
        assert controls.isVisibleTo(panel)

        controls.set_active_view_token("integral_scan")
        assert not controls.applies_to_current_view()
        assert not controls.isVisibleTo(panel)

        controls.set_active_view_token("fb_asymmetry")
        panel.set_current_time_view_mode("groups")
        panel.time_view_changed.emit("groups")
        assert not controls.applies_to_current_view()


class TestFeatureGate:
    """The Options → Advanced toggle gates the entire RRF surface (default off)."""

    def test_controls_absent_until_feature_enabled(self, panel):
        controls = panel._rrf_controls
        panel.set_time_view_modes(["fb_asymmetry", "groups", "raw_counts"])
        controls.set_active_view_token("fb_asymmetry")
        # Default off: even on the FB-asymmetry view the controls are hidden
        # (zero layout footprint), and the display transform is inert.
        assert not controls.feature_enabled()
        assert not controls.isVisibleTo(panel)
        controls._freq_spin.setValue(NU)
        controls._enable_check.setChecked(True)
        assert not controls.is_active()
        dataset = _dataset()
        assert rrf_display_dataset(panel, dataset) is dataset  # transform inert
        # Enabling the feature reveals them under the existing view condition.
        controls.set_feature_enabled(True)
        assert controls.isVisibleTo(panel)
        assert controls.is_active()

    def test_panel_setter_toggles_feature_and_transform(self, panel):
        dataset = _dataset()
        _enable_rrf(panel)
        assert panel._rrf_controls.is_active()
        assert rrf_display_dataset(panel, dataset) is not dataset
        # Turning the feature back off makes the transform inert immediately.
        panel.set_rrf_feature_enabled(False)
        assert not panel._rrf_controls.is_active()
        assert rrf_display_dataset(panel, dataset) is dataset
        assert not panel._rrf_controls.isVisibleTo(panel)

    def test_rrf_fit_frequency_only_when_active(self, panel):
        panel.set_time_view_modes(["fb_asymmetry", "groups", "raw_counts"])
        panel._rrf_controls.set_active_view_token("fb_asymmetry")
        assert panel.rrf_fit_frequency_mhz() is None  # feature off
        _enable_rrf(panel, frequency_mhz=NU)
        assert panel.rrf_fit_frequency_mhz() == pytest.approx(NU)
        panel.set_rrf_feature_enabled(False)
        assert panel.rrf_fit_frequency_mhz() is None

    def test_transform_inactive_when_disabled(self, panel):
        dataset = _dataset()
        assert rrf_display_dataset(panel, dataset) is dataset

    def test_transform_skips_frequency_and_derived_datasets(self, panel):
        _enable_rrf(panel)
        freq_ds = MuonDataset(
            time=np.linspace(0, 10, 50),
            asymmetry=np.zeros(50),
            error=np.ones(50),
            metadata={"plot_domain": "frequency"},
        )
        assert rrf_display_dataset(panel, freq_ds) is freq_ds
        derived = MuonDataset(
            time=np.linspace(0, 10, 50),
            asymmetry=np.zeros(50),
            error=np.ones(50),
            metadata={"x_label": "Field (G)", "y_label": "Integral"},
        )
        assert rrf_display_dataset(panel, derived) is derived


class TestTransform:
    def test_display_dataset_is_demodulated(self, panel):
        dataset = _dataset()
        _enable_rrf(panel)
        shown = rrf_display_dataset(panel, dataset)
        assert shown is not dataset
        envelope = 25.0 * np.exp(-LAM * shown.time)
        ok = np.isfinite(shown.asymmetry)
        assert ok.sum() > dataset.time.size // 2
        np.testing.assert_allclose(shown.asymmetry[ok], envelope[ok], rtol=5e-3, atol=0.05)
        assert "frame" in shown.metadata.get("rrf_frame", "")
        # Source dataset untouched (provenance invariant).
        assert np.max(np.abs(dataset.asymmetry)) > 20.0

    def test_magnitude_component(self, panel):
        dataset = _dataset()
        _enable_rrf(panel, component="magnitude")
        shown = rrf_display_dataset(panel, dataset)
        envelope = 25.0 * np.exp(-LAM * shown.time)
        ok = np.isfinite(shown.asymmetry)
        np.testing.assert_allclose(shown.asymmetry[ok], envelope[ok], rtol=5e-3, atol=0.05)

    def test_fit_curve_overlay_demodulated_in_step(self, panel):
        dataset = _dataset()
        _enable_rrf(panel)
        shown = rrf_display_dataset(panel, dataset)
        t_fit = dataset.time
        y_fit = 25.0 * np.exp(-LAM * t_fit) * np.cos(2.0 * np.pi * NU * t_fit)
        out = rrf_display_fit_curve(panel, (t_fit, y_fit, "fit"), shown)
        assert out is not None
        _, y_out, label = out
        assert label == "fit"
        ok = np.isfinite(y_out)
        envelope = 25.0 * np.exp(-LAM * t_fit)
        np.testing.assert_allclose(y_out[ok], envelope[ok], rtol=5e-3, atol=0.05)

    def test_fit_curve_untouched_when_data_not_transformed(self, panel):
        # Pairing rule: the overlay transforms iff the displayed dataset did.
        _enable_rrf(panel)
        t_fit = np.linspace(0.0, 8.0, 100)
        y_fit = np.cos(2.0 * np.pi * NU * t_fit)
        raw = _dataset()  # NOT passed through rrf_display_dataset
        out = rrf_display_fit_curve(panel, (t_fit, y_fit, "fit"), raw)
        assert out == (t_fit, y_fit, "fit")

    def test_fit_curve_uses_data_resolved_bandwidth(self, panel):
        # An Auto bandwidth resolved on a coarse data grid must be reused on
        # the (finer) overlay grid, not re-resolved — otherwise the two
        # curves are filtered differently (aliased-image regime).
        dataset = _dataset(dt=0.016)  # fs = 62.5 MHz: image folds to 2.5 MHz
        _enable_rrf(panel)
        shown = rrf_display_dataset(panel, dataset)
        assert shown.metadata["rrf_bandwidth_mhz"] == pytest.approx(1.75)
        t_fit = np.arange(0.0, 8.0, 0.001)  # much finer grid
        y_fit = 25.0 * np.exp(-LAM * t_fit) * np.cos(2.0 * np.pi * NU * t_fit)
        out = rrf_display_fit_curve(panel, (t_fit, y_fit, "fit"), shown)
        ok = np.isfinite(out[1])
        envelope = 25.0 * np.exp(-LAM * t_fit)
        np.testing.assert_allclose(out[1][ok], envelope[ok], rtol=2e-2, atol=0.3)

    def test_plot_dataset_draws_envelope_and_badge(self, panel):
        dataset = _dataset()
        _enable_rrf(panel)
        panel.plot_dataset(dataset)
        # The cached plotted arrays are the demodulated ones.
        plotted = panel._last_plot_asymmetry
        ok = np.isfinite(plotted)
        envelope = 25.0 * np.exp(-LAM * panel._last_plot_time)
        np.testing.assert_allclose(plotted[ok], envelope[ok], rtol=5e-3, atol=0.05)
        badge = [
            artist
            for artist in panel._ax.texts
            if "frame: ν₀" in str(getattr(artist, "get_text", lambda: "")())
        ]
        assert badge, "frame badge missing from axes"

    def test_one_frame_for_all_overlaid_runs(self, panel):
        a = _dataset()
        b = _dataset()
        _enable_rrf(panel)
        panel.plot_datasets([a, b])
        plotted = panel._last_plot_asymmetry
        ok = np.isfinite(plotted)
        # Both runs demodulate to the same envelope; nothing oscillates at NU.
        assert np.nanmax(np.abs(plotted[ok])) < 26.0
        assert np.nanmin(plotted[ok]) > -2.0


class TestControls:
    def test_unit_toggle_round_trips_through_convert(self, panel):
        controls = _enable_rrf(panel, frequency_mhz=NU)
        assert controls.frequency_mhz() == pytest.approx(NU)
        idx = controls._unit_combo.findData(FieldUnit.GAUSS.value)
        controls._unit_combo.setCurrentIndex(idx)
        expected_gauss = float(convert(NU, FieldUnit.MHZ, FieldUnit.GAUSS))
        assert controls._freq_spin.value() == pytest.approx(expected_gauss, rel=1e-6)
        assert controls.frequency_mhz() == pytest.approx(NU, rel=1e-6)

    def test_auto_seed_from_field_metadata(self, panel):
        dataset = _dataset(field_gauss=2000.0)
        panel._current_dataset = dataset
        controls = panel._rrf_controls
        assert controls._freq_spin.value() == 0.0
        controls._enable_check.setChecked(True)
        expected = float(convert(2000.0, FieldUnit.GAUSS, FieldUnit.MHZ))
        assert controls.frequency_mhz() == pytest.approx(expected, abs=5e-4)

    def test_bandwidth_auto_is_none(self, panel):
        controls = _enable_rrf(panel)
        assert controls.bandwidth_mhz() is None
        controls._bandwidth_spin.setValue(2.5)
        assert controls.bandwidth_mhz() == pytest.approx(2.5)


class TestPersistence:
    def test_state_round_trip(self, panel, qapp):
        controls = _enable_rrf(panel, frequency_mhz=NU, component="magnitude")
        controls._phase_spin.setValue(15.0)
        controls._bandwidth_spin.setValue(3.0)
        state = panel.get_state()
        assert state["rrf"]["enabled"] is True
        assert state["rrf"]["frequency_mhz"] == pytest.approx(NU)
        assert state["rrf"]["component"] == "magnitude"

        restored = PlotPanel()
        try:
            if not restored._has_mpl:
                pytest.skip("matplotlib required")
            restored.restore_state(state)
            rrf = restored._rrf_controls
            # The frame parameters round-trip via plot_state["rrf"]; the feature
            # flag is app-level chrome (QSettings), NOT in the project, so a
            # fresh panel restores with the toggle off (the host auto-enables it
            # on open — see rrf_has_active_parameters). The restored frame is
            # therefore configured-but-inactive until the feature is enabled.
            assert restored.rrf_has_active_parameters()  # configured…
            assert not rrf.is_active()  # …but inactive until the feature is on
            assert rrf._enable_check.isChecked()
            assert rrf.frequency_mhz() == pytest.approx(NU)
            assert rrf.phase_deg() == pytest.approx(15.0)
            assert rrf.bandwidth_mhz() == pytest.approx(3.0)
            assert rrf.component() == "magnitude"
            restored.set_rrf_feature_enabled(True)
            assert rrf.is_active()
        finally:
            restored.close()
            restored.deleteLater()

    def test_restore_tolerates_absence_and_junk(self, panel):
        # Enable first: restoring a state saved WITHOUT the key must reset
        # RRF rather than keep the stale frame (pre-feature projects).
        _enable_rrf(panel)
        assert panel._rrf_controls.is_active()
        state = panel.get_state()
        state.pop("rrf", None)
        panel.restore_state(state)
        assert not panel._rrf_controls.is_active()
        assert panel._rrf_controls._freq_spin.value() == 0.0

        state["rrf"] = {"enabled": "maybe", "frequency_mhz": "soup", "display_unit": 7}
        panel.restore_state(state)
        # Junk coerces to safe defaults; the panel must not raise or activate
        # with a nonsensical frequency.
        assert not panel._rrf_controls.is_active()

    def test_state_is_json_serialisable(self, panel):
        import json

        _enable_rrf(panel)
        json.dumps(panel.get_state()["rrf"])
