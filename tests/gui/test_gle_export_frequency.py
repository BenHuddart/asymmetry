"""Frequency-view (FFT/MaxEnt) GLE and text export coverage.

The GLE/text export path is shared with the time domain but must mirror the
on-screen *spectrum* render when the panel is in frequency mode: display-unit x
data and window, real axis labels, a piecewise-linear line + light shaded ±1σ
band idiom (not the time-domain errorbar dots or the GLE spline),
self-describing sidecar columns with Fourier
provenance, and digit-safe (``run_``-prefixed) sidecar filenames. These tests
drive the real gleplot API, matching ``test_gle_export.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

pytest.importorskip("PySide6")
pytest.importorskip("gleplot", reason="gle extra not installed")

import gleplot as glp  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from asymmetry.core.data.dataset import MuonDataset  # noqa: E402
from asymmetry.gui.panels.plot_panel import PlotPanel  # noqa: E402


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _spectrum(
    *,
    run_number: int = 20,
    field: float | None = 3000.0,
    error: float = 0.02,
    correlation: bool = False,
) -> MuonDataset:
    freq = np.linspace(0.0, 100.0, 64)
    amp = np.exp(-((freq - 40.0) ** 2) / (2 * 5.0**2))
    err = np.full_like(freq, error)
    metadata: dict = {
        "run_number": run_number,
        "plot_domain": "frequency",
        "y_label": "FFT Magnitude (a.u.)",
        "fourier_display": "Magnitude",
        "fourier_window": "hann",
        "fourier_padding": 2,
        # Array-valued Fourier keys must be excluded from the sidecar header.
        "fourier_imag": [1.0, 2.0, 3.0],
    }
    if field is not None:
        metadata["field"] = field
    if correlation:
        metadata["correlation_axis"] = True
        metadata["x_label"] = "Muon hyperfine coupling Aμ (MHz)"
        metadata["y_label"] = "Radical correlation (a.u.)"
    return MuonDataset(time=freq, asymmetry=amp, error=err, metadata=metadata)


def _panel(qapp: QApplication) -> PlotPanel:
    panel = PlotPanel(domain="frequency")
    if not getattr(panel, "_has_mpl", False):
        panel.deleteLater()
        pytest.skip("matplotlib not available")
    return panel


def _build(panel: PlotPanel, ds: MuonDataset, tmp_path: Path, name: str = "figure"):
    export_dir = tmp_path / f"{name}.gleplot"
    export_dir.mkdir()
    gle_path = export_dir / f"{name}.gle"
    payloads = panel.get_current_plot_export_data([ds])
    panel._build_gle_export(glp, gle_path, payloads)
    return gle_path, export_dir


def _data_rows(dat_text: str) -> list[list[float]]:
    return [
        [float(tok) for tok in ln.split()]
        for ln in dat_text.splitlines()
        if ln and not ln.startswith("!")
    ]


def test_frequency_gauss_export_window_labels_and_line(qapp, tmp_path):
    panel = _panel(qapp)
    try:
        ds = _spectrum()
        panel.plot_dataset(ds)
        panel._frequency_x_unit_combo.setCurrentText("Field (G)")
        panel._x_min.setValue(0.0)
        panel._x_max.setValue(6000.0)

        gle_path, export_dir = _build(panel, ds, tmp_path)
        gle = gle_path.read_text(encoding="utf-8")

        # Axis window converted to the absolute display (Gauss) axis, and the
        # real x/y titles — not "Time (µs)" / "Asymmetry (%)".
        assert 'xtitle "Field (G)"' in gle
        assert 'ytitle "FFT Magnitude (a.u.)"' in gle
        assert "xaxis min 0 max 6000" in gle
        # Line idiom, not the time-domain errorbar dots — and piecewise-linear:
        # GLE's ``smooth`` spline overshoots on sharp resonance lines.
        assert " line " in gle
        assert " smooth" not in gle
        assert "errup" not in gle and "errdown" not in gle

        # Digit-led run label is prefixed to survive the gleplot parser.
        dat = (export_dir / "run_20_main.dat").read_text(encoding="utf-8")
        assert "! field_G  amplitude  error  frequency_MHz" in dat
        rows = _data_rows(dat)
        x_display = panel._convert_frequency_axis_for_display(ds.time)
        np.testing.assert_allclose([r[0] for r in rows], x_display, rtol=1e-6)
        # Trailing canonical-MHz column preserves the spectrum's own axis.
        np.testing.assert_allclose([r[3] for r in rows], ds.time, rtol=1e-6)
    finally:
        panel.close()
        panel.deleteLater()


def test_frequency_mhz_export_has_no_trailing_mhz_column(qapp, tmp_path):
    panel = _panel(qapp)
    try:
        ds = _spectrum(run_number=21)
        panel.plot_dataset(ds)  # default unit = MHz
        gle_path, export_dir = _build(panel, ds, tmp_path)

        dat = (export_dir / "run_21_main.dat").read_text(encoding="utf-8")
        assert "! frequency_MHz  amplitude  error\n" in dat
        rows = _data_rows(dat)
        assert all(len(r) == 3 for r in rows)
        np.testing.assert_allclose([r[0] for r in rows], ds.time, rtol=1e-6)
    finally:
        panel.close()
        panel.deleteLater()


def test_frequency_relative_mode_window_adds_reference(qapp, tmp_path):
    panel = _panel(qapp)
    try:
        ds = _spectrum(run_number=22, field=3000.0)
        panel.plot_dataset(ds)
        panel._frequency_x_unit_combo.setCurrentText("Field (G)")
        panel._frequency_axis_relative_check.setChecked(True)
        panel._x_min.setValue(0.0)
        panel._x_max.setValue(500.0)

        reference_g = panel._display_frequency_reference(unit="field_gauss")
        assert reference_g == pytest.approx(3000.0, abs=1e-6)

        gle_path, _ = _build(panel, ds, tmp_path)
        gle = gle_path.read_text(encoding="utf-8")
        # Exported window is the absolute display value: control + reference.
        assert f"xaxis min {reference_g:g} max {reference_g + 500.0:g}" in gle
    finally:
        panel.close()
        panel.deleteLater()


def test_frequency_band_present_with_positive_errors(qapp, tmp_path):
    panel = _panel(qapp)
    try:
        ds = _spectrum(run_number=23, error=0.02)
        panel.plot_dataset(ds)
        gle_path, export_dir = _build(panel, ds, tmp_path)
        gle = gle_path.read_text(encoding="utf-8")

        # The band is drawn as a fill referencing its own data file, in a light
        # GLE tint — GLE has no fill alpha, so the series color itself would
        # render as a solid block that swallows the spectrum line.
        assert "fill d" in gle
        band_fills = [ln for ln in gle.splitlines() if ln.strip().startswith("fill d")]
        assert band_fills and all("LIGHTGRAY" in ln for ln in band_fills)
        assert "fill d1,d2 color BLACK" not in gle
        # The spectrum line itself stays piecewise-linear (no GLE spline).
        series_lines = [ln for ln in gle.splitlines() if " line " in ln and "key" in ln]
        assert series_lines and all(" smooth" not in ln for ln in series_lines)
        from asymmetry.gui.utils.gle_export import extract_gle_data_dependencies

        deps = extract_gle_data_dependencies(gle_path)
        assert "run_23_main_band.dat" in deps
        for name in deps:
            assert (export_dir / name).exists()
    finally:
        panel.close()
        panel.deleteLater()


def test_frequency_no_band_with_zero_errors(qapp, tmp_path):
    panel = _panel(qapp)
    try:
        ds = _spectrum(run_number=24, error=0.0)
        panel.plot_dataset(ds)
        gle_path, export_dir = _build(panel, ds, tmp_path)
        gle = gle_path.read_text(encoding="utf-8")

        assert "fill d" not in gle
        assert not (export_dir / "run_24_main_band.dat").exists()
    finally:
        panel.close()
        panel.deleteLater()


def test_frequency_annotation_lands_at_display_unit_x(qapp, tmp_path):
    panel = _panel(qapp)
    try:
        ds = _spectrum(run_number=25)
        panel.plot_dataset(ds)
        panel._frequency_x_unit_combo.setCurrentText("Field (G)")
        panel._annotations.append({"x": 2500.0, "y": 0.5, "text": "peak", "artist": None})

        export_dir = tmp_path / "annot.gleplot"
        export_dir.mkdir()
        gle_path = export_dir / "annot.gle"
        payloads = panel.get_current_plot_export_data([ds])

        fig = glp.figure()
        ax = fig.add_subplot(1, 1, 1)
        panel._plot_export_payloads_on_axis(
            ax,
            payloads,
            axis_key=None,
            written_files=[],
            dat_writes=[],
            gle_path=gle_path,
            colors=["black"],
            show_legend=False,
            used_tokens=set(),
        )
        assert ax.texts
        assert ax.texts[0]["x"] == pytest.approx(2500.0)
    finally:
        panel.close()
        panel.deleteLater()


def test_frequency_text_export_x_range_filters_on_display_column(qapp, tmp_path):
    panel = _panel(qapp)
    try:
        ds = _spectrum(run_number=26)
        panel.plot_dataset(ds)
        panel._frequency_x_unit_combo.setCurrentText("Field (G)")

        payload = panel.get_current_plot_export_data([ds])[0]
        x_display = panel._convert_frequency_axis_for_display(ds.time)
        lo, hi = 1000.0, 4000.0  # Gauss window
        dat_path = tmp_path / "clip.dat"
        panel._write_data_file(dat_path, payload, x_range=(lo, hi))

        rows = _data_rows(dat_path.read_text(encoding="utf-8"))
        expected = [float(x) for x in x_display if lo <= x <= hi]
        assert [r[0] for r in rows] == pytest.approx(expected)
        assert rows and all(lo <= r[0] <= hi for r in rows)
    finally:
        panel.close()
        panel.deleteLater()


def test_frequency_correlation_axis_exports_metadata_label_and_mhz(qapp, tmp_path):
    panel = _panel(qapp)
    try:
        ds = _spectrum(run_number=27, field=None, correlation=True)
        panel.plot_dataset(ds)
        assert panel._frequency_axis_is_correlation is True

        gle_path, export_dir = _build(panel, ds, tmp_path)
        gle = gle_path.read_text(encoding="utf-8")
        assert "coupling" in gle  # x-title from the correlation metadata label

        dat = (export_dir / "run_27_main.dat").read_text(encoding="utf-8")
        assert "! coupling_MHz  amplitude  error\n" in dat
        rows = _data_rows(dat)
        # Correlation axis is already MHz — the x column is unconverted and no
        # trailing frequency_MHz column is added.
        assert all(len(r) == 3 for r in rows)
        np.testing.assert_allclose([r[0] for r in rows], ds.time, rtol=1e-6)
    finally:
        panel.close()
        panel.deleteLater()
