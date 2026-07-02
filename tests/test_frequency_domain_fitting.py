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


@pytest.mark.gui
def test_fit_panel_frequency_range_editable_with_placeholder_when_unset(qapp) -> None:
    """D6/F15: the frequency fit-range spins stay editable, never a stale value.

    In the time domain the plot always supplies a fit range (seeded to the
    full dataset extent), so an absent range there still disables the spins.
    In the frequency domain there is no draggable selector, so an absent
    range must not disable the fields or leave a leftover time-domain number
    behind — it shows a "full spectrum" placeholder instead.
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
