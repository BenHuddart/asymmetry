"""GUI wiring for the calibrated FFT scale and unit-area p(ν) display.

Covers the Fourier panel's unit-area control state, the calibrated-percent and
density y-labels reaching the plot, the export column naming, and the y-unit
suffix the limit controls show.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

pytest.importorskip("PySide6")

from asymmetry.core.data.dataset import MuonDataset  # noqa: E402
from asymmetry.core.fourier.spectrum import UNIT_AREA_YLABEL  # noqa: E402


@pytest.mark.gui
def test_unit_area_checkbox_round_trips_state(qapp) -> None:
    from asymmetry.gui.panels.fourier_panel import FourierPanel

    panel = FourierPanel()
    try:
        # Default is the calibrated (amplitude) scale.
        assert panel.get_state()["display_normalisation"] == "calibrated"

        panel._unit_area_check.setChecked(True)
        assert panel.get_state()["display_normalisation"] == "unit_area"

        restored = FourierPanel()
        try:
            restored.restore_state(panel.get_state())
            assert restored._unit_area_check.isChecked()
            assert restored.get_state()["display_normalisation"] == "unit_area"
        finally:
            restored.deleteLater()
    finally:
        panel.deleteLater()


def _panel(qapp):
    from asymmetry.gui.panels.plot_panel import PlotPanel

    panel = PlotPanel(domain="frequency")
    if not getattr(panel, "_has_mpl", False):
        panel.deleteLater()
        pytest.skip("matplotlib not available")
    return panel


def _spectrum(*, y_label: str) -> MuonDataset:
    freq = np.linspace(0.0, 100.0, 64)
    amp = np.exp(-((freq - 40.0) ** 2) / (2 * 5.0**2))
    err = np.full_like(freq, 0.02)
    return MuonDataset(
        time=freq,
        asymmetry=amp,
        error=err,
        metadata={
            "run_number": 21,
            "plot_domain": "frequency",
            "y_label": y_label,
            "fourier_display": "Magnitude",
        },
    )


@pytest.mark.gui
def test_percent_y_label_reaches_the_axis(qapp) -> None:
    panel = _panel(qapp)
    try:
        panel.plot_dataset(_spectrum(y_label="FFT Magnitude (%)"))
        assert panel._ax.get_ylabel() == "FFT Magnitude (%)"
        assert panel._display_y_unit_suffix("FFT Magnitude (%)") == "%"
    finally:
        panel.deleteLater()


@pytest.mark.gui
def test_unit_area_density_label_and_suffix(qapp) -> None:
    panel = _panel(qapp)
    try:
        panel.plot_dataset(_spectrum(y_label=UNIT_AREA_YLABEL))
        assert panel._ax.get_ylabel() == UNIT_AREA_YLABEL
        assert panel._display_y_unit_suffix(UNIT_AREA_YLABEL) == "1/MHz"
    finally:
        panel.deleteLater()


@pytest.mark.gui
def test_export_column_named_density_for_unit_area(qapp, tmp_path: Path) -> None:
    panel = _panel(qapp)
    try:
        ds = _spectrum(y_label=UNIT_AREA_YLABEL)
        panel.plot_dataset(ds)
        payloads = panel.get_current_plot_export_data([ds])
        dat_path = tmp_path / "spectrum.dat"
        panel._write_data_file(dat_path, payloads[0])
        text = dat_path.read_text(encoding="utf-8")
        assert "density_per_MHz" in text
        assert f"Ylabel: {UNIT_AREA_YLABEL}" in text
    finally:
        panel.deleteLater()


@pytest.mark.gui
def test_export_column_named_amplitude_for_calibrated(qapp, tmp_path: Path) -> None:
    panel = _panel(qapp)
    try:
        ds = _spectrum(y_label="FFT Magnitude (%)")
        panel.plot_dataset(ds)
        payloads = panel.get_current_plot_export_data([ds])
        dat_path = tmp_path / "spectrum.dat"
        panel._write_data_file(dat_path, payloads[0])
        text = dat_path.read_text(encoding="utf-8")
        assert "  amplitude  " in text or " amplitude " in text
        assert "density_per_MHz" not in text
    finally:
        panel.deleteLater()
