from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting import CompositeModel, FitEngine, Parameter, ParameterSet
from asymmetry.core.fitting.spectral import (
    append_frequency_field_derived_parameters,
    default_frequency_model,
    frequency_mhz_to_field_gauss,
)


@pytest.fixture
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _frequency_dataset(
    run_number: int = 1,
    *,
    center: float = 3.2,
    fwhm: float = 0.35,
    height: float = 7.0,
    bg: float = 0.4,
) -> MuonDataset:
    freq = np.linspace(1.0, 5.0, 401)
    values = bg + height * np.exp(-4.0 * np.log(2.0) * ((freq - center) / fwhm) ** 2)
    return MuonDataset(
        time=freq,
        asymmetry=values,
        error=np.full_like(freq, 0.05),
        metadata={
            "run_number": run_number,
            "run_label": str(run_number),
            "plot_domain": "frequency",
            "field": 200.0 + run_number,
        },
    )


def test_gaussian_frequency_peak_fit_recovers_center_and_fwhm() -> None:
    dataset = _frequency_dataset(center=3.4, fwhm=0.42, height=5.5, bg=0.3)
    model = default_frequency_model()
    params = ParameterSet(
        [
            Parameter("height", value=5.0, min=0.0),
            Parameter("nu0", value=3.2, min=0.0),
            Parameter("fwhm", value=0.5, min=1e-6),
            Parameter("bg", value=0.0),
        ]
    )

    result = FitEngine().fit(dataset, model.function, params)

    assert result.success
    assert result.parameters["nu0"].value == pytest.approx(3.4, abs=1e-3)
    assert result.parameters["fwhm"].value == pytest.approx(0.42, rel=0.02)
    assert result.parameters["bg"].value == pytest.approx(0.3, abs=1e-3)


def test_lorentzian_frequency_peak_fit_recovers_background_slope() -> None:
    freq = np.linspace(1.0, 5.0, 401)
    center = 2.8
    fwhm = 0.5
    bg = 0.2
    slope = 0.03
    values = bg + slope * freq + 4.0 / (1.0 + 4.0 * ((freq - center) / fwhm) ** 2)
    dataset = MuonDataset(
        time=freq,
        asymmetry=values,
        error=np.full_like(freq, 0.05),
        metadata={"run_number": 2, "plot_domain": "frequency"},
    )
    model = CompositeModel(["LorentzianPeak", "LinearBackground"], operators=["+"])
    params = ParameterSet(
        [
            Parameter("height", value=3.5, min=0.0),
            Parameter("nu0", value=2.7, min=0.0),
            Parameter("fwhm", value=0.4, min=1e-6),
            Parameter("bg", value=0.0),
            Parameter("slope", value=0.0),
        ]
    )

    result = FitEngine().fit(dataset, model.function, params)

    assert result.success
    assert result.parameters["nu0"].value == pytest.approx(center, abs=1e-3)
    assert result.parameters["fwhm"].value == pytest.approx(fwhm, rel=0.02)
    assert result.parameters["slope"].value == pytest.approx(slope, abs=1e-3)


def test_frequency_fit_appends_field_equivalent_parameters() -> None:
    params = ParameterSet(
        [
            Parameter("nu0", value=2.5),
            Parameter("fwhm", value=0.25),
        ]
    )

    derived, uncertainties = append_frequency_field_derived_parameters(
        params,
        {"nu0": 0.1, "fwhm": 0.02},
    )

    assert derived["B0"].value == pytest.approx(frequency_mhz_to_field_gauss(2.5))
    assert derived["Bwid"].value == pytest.approx(frequency_mhz_to_field_gauss(0.25))
    assert uncertainties["B0"] == pytest.approx(frequency_mhz_to_field_gauss(0.1))


@pytest.mark.gui
def test_fit_panel_frequency_domain_defaults(qapp) -> None:
    from asymmetry.gui.panels.fit_panel import FitPanel

    panel = FitPanel()
    dataset = _frequency_dataset()

    panel.set_domain("frequency")
    panel.set_dataset(dataset)
    panel.set_datasets([dataset, _frequency_dataset(2, center=3.4)])

    assert panel.domain() == "frequency"
    assert panel.single_fit_formula_string() == "height*exp(-4*ln(2)*((nu-nu0)/fwhm)^2) + bg"
    assert "GaussianPeak" in panel._single_tab._composite_model.component_names
    assert panel._single_tab._fit_wizard_btn.isEnabled() is False


def _time_domain_dataset(run_number: int = 1, *, peak_time: float = 4.1) -> MuonDataset:
    """A time-domain dataset whose asymmetry peaks at *peak_time* µs.

    Used to reproduce the stale-seed bug: switching to the frequency domain
    seeds the peak model against whatever dataset is current, so a leftover
    time-domain peak lands ``nu0`` at a time value (µs) far off the MHz axis.
    """
    time = np.linspace(0.0, 10.0, 500)
    asym = np.exp(-((time - peak_time) ** 2) / 0.5)
    return MuonDataset(
        time=time,
        asymmetry=asym,
        error=np.full_like(time, 0.01),
        metadata={"run_number": run_number, "field": 80000.0},
    )


@pytest.mark.gui
def test_frequency_peak_seed_refreshes_from_spectrum_not_carried(qapp) -> None:
    """Switching to a frequency run re-derives nu0 from that spectrum's peak.

    Regression: the peak model was seeded during ``set_domain("frequency")``
    against the still-current time-domain dataset (``nu0`` ≈ the peak *time*),
    and the carry-forward branch replayed that stale seed verbatim — so the
    Gaussian sat far off the MHz axis and a preview showed only the background.
    """
    from asymmetry.gui.panels.fit_panel import FitPanel

    panel = FitPanel()
    panel.set_dataset(_time_domain_dataset(1, peak_time=4.1))
    panel.set_domain("frequency")
    panel.set_dataset(_frequency_dataset(2, center=3.4))

    seeds = panel._single_tab.current_seed_values()
    # The peak is at 3.4 MHz; the stale time-domain seed would be ≈4.1.
    assert float(seeds["nu0"]) == pytest.approx(3.4, abs=0.05)

    # Selecting another frequency run refreshes the field-dependent peak again.
    panel.set_dataset(_frequency_dataset(3, center=2.6))
    assert float(panel._single_tab.current_seed_values()["nu0"]) == pytest.approx(2.6, abs=0.05)


@pytest.mark.gui
def test_frequency_preview_renders_a_visible_peak(qapp) -> None:
    """The refreshed seed makes Preview draw the peak, not just the background."""
    from asymmetry.gui.panels.fit_panel import FitPanel

    panel = FitPanel()
    panel.set_dataset(_time_domain_dataset(1, peak_time=4.1))
    panel.set_domain("frequency")
    panel.set_dataset(_frequency_dataset(2, center=3.4, height=7.0, bg=0.4))

    captured: dict[str, np.ndarray] = {}
    panel.preview_requested.connect(lambda _r, curve, _c: captured.update(y=curve[1]))
    panel._single_tab._on_preview()

    y = captured["y"]
    # Peak (~bg + height) must rise well clear of the background floor.
    assert float(np.max(y)) > float(np.min(y)) + 3.0


@pytest.mark.gui
def test_frequency_adding_a_second_peak_seeds_both(qapp) -> None:
    """Switching to a two-peak model (e.g. via Edit Function) seeds both lines.

    Regression: with duplicate peak components the params are suffixed
    (``nu0_1``/``nu0_2``), so the old single-peak seeder matched only ``bg`` and
    both peaks kept the off-screen ``nu0=1.0`` default.
    """
    from asymmetry.gui.panels.fit_panel import FitPanel

    freq = np.linspace(1.0, 5.0, 401)
    values = (
        0.4
        + 6.0 * np.exp(-4.0 * np.log(2.0) * ((freq - 2.0) / 0.2) ** 2)
        + 3.0 * np.exp(-4.0 * np.log(2.0) * ((freq - 3.5) / 0.3) ** 2)
    )
    dataset = MuonDataset(
        time=freq,
        asymmetry=values,
        error=np.full_like(freq, 0.05),
        metadata={"run_number": 5, "plot_domain": "frequency", "field": 200.0},
    )

    panel = FitPanel()
    panel.set_domain("frequency")
    panel.set_dataset(dataset)
    panel._single_tab._set_composite_model(
        CompositeModel(["GaussianPeak", "GaussianPeak", "ConstantBackground"], operators=["+", "+"])
    )

    seeds = panel._single_tab.current_seed_values()
    assert float(seeds["nu0_1"]) == pytest.approx(2.0, abs=0.05)
    assert float(seeds["nu0_2"]) == pytest.approx(3.5, abs=0.05)

    captured: dict[str, np.ndarray] = {}
    panel.preview_requested.connect(lambda _r, curve, _c: captured.update(x=curve[0], y=curve[1]))
    panel._single_tab._on_preview()

    from scipy.signal import find_peaks

    maxima, _ = find_peaks(captured["y"], prominence=0.5)
    assert len(maxima) == 2


@pytest.mark.gui
def test_frequency_restored_real_fit_is_not_reseeded(qapp) -> None:
    """A genuinely restored fit keeps its recorded parameters (no peak re-seed)."""
    from asymmetry.gui.panels.fit_panel import FitPanel

    panel = FitPanel()
    panel.set_single_fit_restore_provider(
        lambda _ds: {
            "parameters": [
                {"name": "height", "value": 42.0},
                {"name": "nu0", "value": 999.0},
                {"name": "fwhm", "value": 0.5},
                {"name": "bg", "value": 0.0},
            ]
        }
    )
    panel.set_domain("frequency")
    panel.set_dataset(_frequency_dataset(4, center=3.4))

    assert float(panel._single_tab.current_seed_values()["nu0"]) == pytest.approx(999.0)


@pytest.mark.gui
def test_fit_panel_frequency_range_editable_with_placeholder_when_unset(qapp) -> None:
    """D6/F15: the frequency fit-range spins stay editable, never a stale value.

    In the time domain the plot always supplies a fit range (seeded to the
    full dataset extent), so an absent range there still disables the spins.
    In the frequency domain an unset range must not disable the fields or
    leave a leftover time-domain number behind — it shows a "full spectrum"
    placeholder instead. (The plot span is draggable in both domains, but the
    spins remain the keyboard entry point.)
    """
    from asymmetry.gui.panels.fit_panel import FitPanel

    panel = FitPanel()
    dataset = _frequency_dataset()
    panel.set_domain("time")
    panel.set_fit_range_display(0.006, 9.839)
    assert panel._single_tab._fit_range_min_spin.value() == pytest.approx(0.006)

    panel.set_domain("frequency")
    panel.set_dataset(dataset)
    panel.set_fit_range_display(None, None)

    min_spin = panel._single_tab._fit_range_min_spin
    max_spin = panel._single_tab._fit_range_max_spin
    assert min_spin.isEnabled() is True
    assert max_spin.isEnabled() is True
    assert min_spin.text() == ""
    assert max_spin.text() == ""
    assert min_spin.placeholderText() == "full spectrum"
    assert max_spin.placeholderText() == "full spectrum"

    panel.set_fit_range_display(1.5, 42.0)
    assert min_spin.isEnabled() is True
    assert min_spin.value() == pytest.approx(1.5)
    assert max_spin.value() == pytest.approx(42.0)

    panel.set_domain("time")
    panel.set_fit_range_display(None, None)
    assert panel._single_tab._fit_range_min_spin.isEnabled() is False


@pytest.mark.gui
def test_fit_panel_frequency_global_missing_spectra_status(qapp) -> None:
    from asymmetry.gui.panels.fit_panel import FitPanel

    panel = FitPanel()
    panel.set_domain("frequency")
    panel.set_datasets([_frequency_dataset(1), _frequency_dataset(2)])

    panel.set_frequency_missing_spectra_status([3, 4], cached_count=2)

    status = panel._global_tab._result_text.toPlainText()
    assert "2 cached frequency spectra selected" in status
    assert "Compute a Fourier spectrum for run(s) 3, 4" in status


@pytest.mark.gui
def test_frequency_plot_draws_fit_range_span(qapp) -> None:
    """The frequency plot shows the fit-range span (previously drawn only in time)."""
    from types import SimpleNamespace

    from asymmetry.gui.panels.plot_panel import PlotPanel

    panel = PlotPanel(domain="frequency")
    panel.plot_dataset(_frequency_dataset(1, center=3.0))

    # Auto-seeded to the full extent, so a span + two edge handles are drawn.
    assert len(panel._fit_span_artists) == 1
    assert len(panel._fit_min_handles) == 1
    assert len(panel._fit_max_handles) == 1

    window = panel._frequency_fit_range_display()
    assert window is not None
    assert window == pytest.approx((panel._fit_x_min, panel._fit_x_max))

    # A hit-test near the max edge finds the handle when moments are hidden.
    panel._set_fit_range(1.5, 4.5, emit_signal=False, redraw=True)
    panel._active_fit_handle = "max"
    panel._on_canvas_motion_notify(SimpleNamespace(xdata=4.0, inaxes=panel._ax))
    assert panel._fit_x_max == pytest.approx(4.0)
    assert panel._fit_x_min == pytest.approx(1.5)


@pytest.mark.gui
def test_frequency_fit_range_drag_converts_field_units_to_mhz(qapp) -> None:
    """Dragging in a field-unit display converts the cursor back to canonical MHz."""
    from types import SimpleNamespace

    from asymmetry.gui.panels.plot_panel import PlotPanel

    panel = PlotPanel(domain="frequency")
    panel.plot_dataset(_frequency_dataset(1, center=3.0))
    panel._set_fit_range(2.0, 4.0, emit_signal=False, redraw=True)

    combo = getattr(panel, "_frequency_x_unit_combo", None)
    assert combo is not None
    idx = combo.findData("field_gauss")
    assert idx >= 0
    combo.setCurrentIndex(idx)
    assert panel._current_frequency_x_unit == "field_gauss"

    # Drag the min handle to the Gauss position for 2.5 MHz; it must store MHz.
    gauss_2p5 = panel._convert_canonical_mhz_to_display_limit(
        2.5, unit="field_gauss", relative=False
    )
    panel._active_fit_handle = "min"
    panel._on_canvas_motion_notify(SimpleNamespace(xdata=gauss_2p5, inaxes=panel._ax))
    assert panel._fit_x_min == pytest.approx(2.5, abs=1e-6)


@pytest.mark.gui
def test_frequency_stationary_handle_click_opens_no_modal_dialog(qapp, monkeypatch) -> None:
    """A stationary click on a frequency handle must not open a modal editor.

    Regression: enabling frequency fit-range handles made a click-without-drag
    reach ``_prompt_handle_value_edit``, whose modal ``QInputDialog`` blocks the
    event loop — a hang the per-test *thread* timeout cannot interrupt (it hung
    a CI GUI shard for 10 min). Frequency exact-entry is the Fit-dock spinboxes,
    so the stationary click is a no-op; only dragging moves the range.
    """
    from types import SimpleNamespace

    from PySide6.QtWidgets import QInputDialog

    from asymmetry.gui.panels.plot_panel import PlotPanel

    calls: list[int] = []
    monkeypatch.setattr(
        QInputDialog, "getDouble", staticmethod(lambda *a, **k: (calls.append(1), (0.0, False))[1])
    )

    panel = PlotPanel(domain="frequency")
    panel.plot_dataset(_frequency_dataset(1, center=3.0))
    panel._canvas.draw()
    panel._moments_overlay_visible = False
    panel._add_label_btn.setChecked(False)

    px = panel._ax.transData.transform((panel._fit_x_min, 0.0))[0]
    event = SimpleNamespace(
        inaxes=panel._ax, x=px, y=100.0, xdata=panel._fit_x_min, ydata=0.0, button=1, dblclick=False
    )
    panel._on_canvas_button_press(event)
    panel._on_canvas_button_release(event)

    assert calls == []  # no modal dialog was opened


@pytest.mark.gui
def test_frequency_fit_handle_defers_to_visible_moments_overlay(qapp) -> None:
    """With the moments overlay visible, its handles win the click (shared grammar)."""
    from types import SimpleNamespace

    from asymmetry.gui.panels.plot_panel import PlotPanel

    panel = PlotPanel(domain="frequency")
    panel.plot_dataset(_frequency_dataset(1, center=3.0))
    panel._moments_overlay_visible = True

    hit = panel._detect_handle_hit(SimpleNamespace(inaxes=panel._ax, x=10.0, y=10.0))
    assert hit is None
