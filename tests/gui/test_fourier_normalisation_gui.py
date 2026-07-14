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


def _spectrum(*, y_label: str, unit_area: bool | None = None) -> MuonDataset:
    freq = np.linspace(0.0, 100.0, 64)
    amp = np.exp(-((freq - 40.0) ** 2) / (2 * 5.0**2))
    err = np.full_like(freq, 0.02)
    metadata: dict = {
        "run_number": 21,
        "plot_domain": "frequency",
        "y_label": y_label,
        "fourier_display": "Magnitude",
    }
    # Default the stamp from the label so density-labelled fixtures carry the
    # real unit-area marker the display Jacobian keys off.
    if unit_area is None:
        unit_area = y_label == UNIT_AREA_YLABEL
    if unit_area:
        metadata["fourier_display_normalisation"] = "unit_area"
    return MuonDataset(time=freq, asymmetry=amp, error=err, metadata=metadata)


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


# ── density Jacobian: unit-area y follows the x-axis unit ────────────────────


def _gamma_mhz_per_gauss() -> float:
    from asymmetry.core.fourier.units import gauss_to_mhz

    return float(gauss_to_mhz(1.0))


@pytest.mark.gui
def test_unit_area_gauss_view_scales_density_by_jacobian(qapp) -> None:
    panel = _panel(qapp)
    try:
        ds = _spectrum(y_label=UNIT_AREA_YLABEL)
        panel.plot_dataset(ds)
        panel._frequency_x_unit_combo.setCurrentText("Field (G)")
        panel.plot_dataset(ds)

        gamma = _gamma_mhz_per_gauss()
        np.testing.assert_allclose(panel._last_plot_asymmetry, ds.asymmetry * gamma, rtol=1e-12)
        # The ±1σ band uses the error channel: same factor as the line.
        np.testing.assert_allclose(panel._last_plot_error, ds.error * gamma, rtol=1e-12)
        # The displayed curve integrates to 1 over the displayed (Gauss) axis.
        integral = float(np.trapezoid(panel._last_plot_asymmetry, panel._last_plot_time))
        canonical = float(np.trapezoid(ds.asymmetry, ds.time))
        assert integral == pytest.approx(canonical, rel=1e-9)
        assert panel._ax.get_ylabel() == "Field distribution p(B) (1/G)"
        assert panel._display_y_unit_suffix("Field distribution p(B) (1/G)") == "1/G"
    finally:
        panel.deleteLater()


@pytest.mark.gui
def test_unit_area_tesla_view_scales_density_by_jacobian(qapp) -> None:
    panel = _panel(qapp)
    try:
        ds = _spectrum(y_label=UNIT_AREA_YLABEL)
        panel.plot_dataset(ds)
        panel._frequency_x_unit_combo.setCurrentText("Field (T)")
        panel.plot_dataset(ds)

        gamma_per_tesla = _gamma_mhz_per_gauss() * 1.0e4  # MHz per T
        np.testing.assert_allclose(
            panel._last_plot_asymmetry, ds.asymmetry * gamma_per_tesla, rtol=1e-9
        )
        assert panel._ax.get_ylabel() == "Field distribution p(B) (1/T)"
        assert panel._display_y_unit_suffix("Field distribution p(B) (1/T)") == "1/T"
    finally:
        panel.deleteLater()


@pytest.mark.gui
def test_unit_area_unit_switch_round_trips_values_and_y_limits(qapp) -> None:
    panel = _panel(qapp)
    try:
        ds = _spectrum(y_label=UNIT_AREA_YLABEL)
        panel.plot_dataset(ds)
        y_min_mhz = float(panel._y_min.value())
        y_max_mhz = float(panel._y_max.value())
        values_mhz = np.array(panel._last_plot_asymmetry)

        gamma = _gamma_mhz_per_gauss()
        panel._frequency_x_unit_combo.setCurrentText("Field (G)")
        # The y-limit fields convert by the factor ratio on the switch itself
        # (to within the limit fields' 3-decimal display quantization).
        assert float(panel._y_min.value()) == pytest.approx(y_min_mhz * gamma, abs=5e-4)
        assert float(panel._y_max.value()) == pytest.approx(y_max_mhz * gamma, abs=5e-4)

        # Returning to MHz restores the exact y window from the per-mode stash
        # (no re-quantization drift), and the plotted values round-trip exactly.
        panel._frequency_x_unit_combo.setCurrentText("Frequency (MHz)")
        panel.plot_dataset(ds)
        np.testing.assert_allclose(panel._last_plot_asymmetry, values_mhz, rtol=1e-12)
        assert float(panel._y_min.value()) == pytest.approx(y_min_mhz, rel=1e-9)
        assert float(panel._y_max.value()) == pytest.approx(y_max_mhz, rel=1e-9)
    finally:
        panel.deleteLater()


@pytest.mark.gui
def test_calibrated_spectrum_y_untouched_by_field_units(qapp) -> None:
    panel = _panel(qapp)
    try:
        ds = _spectrum(y_label="FFT Magnitude (%)")
        panel.plot_dataset(ds)
        y_min = float(panel._y_min.value())
        panel._frequency_x_unit_combo.setCurrentText("Field (G)")
        panel.plot_dataset(ds)
        np.testing.assert_allclose(panel._last_plot_asymmetry, ds.asymmetry, rtol=0)
        assert float(panel._y_min.value()) == pytest.approx(y_min, rel=1e-12)
        assert panel._ax.get_ylabel() == "FFT Magnitude (%)"
    finally:
        panel.deleteLater()


@pytest.mark.gui
def test_unit_area_export_in_gauss_names_density_per_gauss(qapp, tmp_path: Path) -> None:
    panel = _panel(qapp)
    try:
        ds = _spectrum(y_label=UNIT_AREA_YLABEL)
        panel.plot_dataset(ds)
        panel._frequency_x_unit_combo.setCurrentText("Field (G)")
        panel.plot_dataset(ds)
        payloads = panel.get_current_plot_export_data([ds])
        dat_path = tmp_path / "spectrum.dat"
        panel._write_data_file(dat_path, payloads[0])
        text = dat_path.read_text(encoding="utf-8")
        assert "density_per_G" in text
        assert "!  Density unit: density_per_G" in text
        assert "Ylabel: Field distribution p(B) (1/G)" in text
        # Exported values match the plotted (Jacobian-scaled) ones.
        gamma = _gamma_mhz_per_gauss()
        rows = [
            [float(tok) for tok in ln.split()]
            for ln in text.splitlines()
            if ln and not ln.startswith("!")
        ]
        exported_y = np.array([row[1] for row in rows])
        np.testing.assert_allclose(exported_y, ds.asymmetry * gamma, rtol=1e-9)
    finally:
        panel.deleteLater()


@pytest.mark.gui
def test_unit_area_unknown_unit_key_falls_back_to_canonical(qapp) -> None:
    panel = _panel(qapp)
    try:
        ds = _spectrum(y_label=UNIT_AREA_YLABEL)
        panel.plot_dataset(ds)
        # Defensive guard: an unrecognised x-unit token (e.g. a mode merged in
        # later, like relative-ppm) keeps factor 1 and the canonical label.
        panel._current_frequency_x_unit = "relative_ppm"
        assert panel._frequency_density_display_factor(ds) == 1.0
        panel.plot_dataset(ds)
        np.testing.assert_allclose(panel._last_plot_asymmetry, ds.asymmetry, rtol=0)
        assert panel._ax.get_ylabel() == UNIT_AREA_YLABEL
    finally:
        panel.deleteLater()
