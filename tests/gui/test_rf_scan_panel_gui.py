"""GUI tests for the RF-µSR resonance surface on the integral-scan path.

The RF surface extends the existing ALC integral-scan mode (see
``test_integral_scan_gui.py``): a "RF resonance (Green − Red)" build toggle on
:class:`ALCFitPanel`, and an "RF resonance (A_µ, A_p)" fit group on
:class:`ALCScanView` driving ``RFResonanceMuP``. These tests exercise the wiring
end to end on synthetic two-period runs and guard the Round-2 picker-visibility
finding. Numerical recovery on the benzene corpus lives in
``test_rf_scan_builder.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

pyside6 = pytest.importorskip("PySide6")
from PySide6.QtCore import QSettings  # type: ignore
from PySide6.QtWidgets import QApplication  # type: ignore

import asymmetry.gui.mainwindow as mw_module
from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.fitting.muon_proton import rf_resonance_mup
from asymmetry.gui.mainwindow import MainWindow
from asymmetry.gui.panels.alc_panel import ALCFitPanel, ALCScanView

_A_MU, _A_P, _NU_RF = 515.0, 124.0, 218.5


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def mainwindow(qapp: QApplication) -> MainWindow:
    settings = QSettings()
    settings.setValue(mw_module._UI_SCALE_SETTINGS_KEY, 1.0)
    return MainWindow()


def _gmr_value(field: float) -> float:
    """(Green − Red) integral value (%) at *field* from the verified RF model."""
    return float(
        rf_resonance_mup(
            np.array([field], dtype=float),
            A_mu=_A_MU,
            A_p=_A_P,
            nu_RF=_NU_RF,
            ampl1=2.0,
            wid1=30.0,
            ampl2=2.0,
            wid2=30.0,
            BG=0.1,
        )[0]
    )


def _two_period_ds(run_number: int, field: float) -> MuonDataset:
    """A two-period (red/green) dataset whose Green − Red curve encodes the model."""
    time = np.linspace(0.1, 8.0, 48)
    red = np.zeros_like(time)
    green = np.full_like(time, _gmr_value(field))
    err = np.full_like(time, 0.01)
    meta = {"run_number": run_number, "field": float(field), "temperature": 293.0}
    run = Run(
        run_number=run_number,
        # Forward/backward detectors so the *plain* integral path works too
        # (used by the "RF fit requires an RF scan" guard test); the RF builder
        # itself reads the period_reduced cache below and ignores these.
        histograms=[
            Histogram(counts=np.full(8, 110.0), bin_width=0.1),
            Histogram(counts=np.full(8, 90.0), bin_width=0.1),
        ],
        metadata=dict(meta),
        grouping={
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": 7,
            "bunching_factor": 1,
            "period_count": 2,
            "period_reduced": [
                (time.copy(), red, err.copy()),
                (time.copy(), green, err.copy()),
            ],
        },
    )
    return MuonDataset(
        time=time,
        asymmetry=np.zeros_like(time),
        error=err.copy(),
        metadata=dict(meta),
        run=run,
    )


def _rf_datasets() -> list[MuonDataset]:
    return [
        _two_period_ds(56400 + i, float(b)) for i, b in enumerate(np.arange(560.0, 1081.0, 20.0))
    ]


def _enter_alc(mw: MainWindow) -> None:
    mw._plot_workspace.set_active_view("integral_scan")


# --- widget presence ----------------------------------------------------------


def test_rf_widgets_present(qapp: QApplication) -> None:
    panel = ALCFitPanel()
    assert panel.rf_difference_enabled() is False  # off by default
    view = ALCScanView()
    assert view.rf_nu() == pytest.approx(218.5)
    assert view.rf_a_mu_seed() == pytest.approx(515.0)
    assert view.rf_a_p_seed() == pytest.approx(124.0)


# --- picker-visibility regression (Round-2 finding) ---------------------------


def test_rf_model_listed_for_field_trend_not_time_domain() -> None:
    """RFResonanceMuP is offered on the field trend picker, absent from time-domain."""
    from asymmetry.core.fitting.domain_library import components_for_domain
    from asymmetry.core.fitting.parameter_models import component_names_for_x

    assert "RFResonanceMuP" in component_names_for_x("field")
    assert "RFResonanceMuP" not in component_names_for_x("temperature")
    # The time-domain fit-function registry is separate and must NOT list it
    # (RF resonance is a field-scan observable, not a per-run time fit).
    assert "RFResonanceMuP" not in components_for_domain("time")


# --- build + fit end to end ---------------------------------------------------


def test_rf_mode_builds_difference_scan(mainwindow: MainWindow) -> None:
    mw = mainwindow
    _enter_alc(mw)
    mw._fit_panel.set_datasets(_rf_datasets())
    mw._alc_fit_panel._rf_difference_check.setChecked(True)

    mw._on_scan_requested()

    assert mw._alc_rf_mode is True
    series = next(s for s in mw._project_model.batches.values() if s.batch_id.startswith("scan-"))
    assert series.label.startswith("RF scan")
    assert mw._alc_scan_view.point_count() == len(_rf_datasets())


def test_rf_fit_populates_couplings(mainwindow: MainWindow) -> None:
    mw = mainwindow
    _enter_alc(mw)
    mw._fit_panel.set_datasets(_rf_datasets())
    mw._alc_fit_panel._rf_difference_check.setChecked(True)
    mw._on_scan_requested()

    view = mw._alc_scan_view
    assert view.x_key() == "field"  # RF model fits vs field
    mw._on_fit_rf()

    assert mw._alc_rf_fitted is True
    text = view._rf_results.text()
    assert "A_µ" in text and "A_p" in text
    assert view._fit_curve is not None  # resonance overlay drawn


def test_rf_fit_requires_rf_scan(mainwindow: MainWindow, monkeypatch) -> None:
    # Clicking "Fit RF resonance" on a plain integral scan is refused with a
    # message rather than fitting the RF model to the wrong observable.
    mw = mainwindow
    _enter_alc(mw)
    # Headless (offscreen) routes guidance to the log instead of a modal
    # QMessageBox (which would block the event loop); a modal here is a bug.
    monkeypatch.setattr(
        mw_module.QMessageBox,
        "information",
        staticmethod(lambda *a, **k: pytest.fail("modal QMessageBox fired headless")),
    )
    monkeypatch.setattr(
        mw_module.QMessageBox,
        "warning",
        staticmethod(lambda *a, **k: pytest.fail("modal QMessageBox fired headless")),
    )
    mw._alc_fit_panel._rf_difference_check.setChecked(False)
    mw._fit_panel.set_datasets(_rf_datasets())
    mw._on_scan_requested()  # a normal integral scan
    assert mw._alc_rf_mode is False

    mw._on_fit_rf()
    assert mw._alc_scan_view._rf_results.text() == ""  # nothing fitted
    assert "RF resonance" in mw._log_panel.to_plain_text()  # the user was told why


def test_rf_scan_persists_and_restores(mainwindow: MainWindow) -> None:
    mw = mainwindow
    _enter_alc(mw)
    datasets = _rf_datasets()
    mw._fit_panel.set_datasets(datasets)
    mw._alc_fit_panel._rf_difference_check.setChecked(True)
    mw._on_scan_requested()
    mw._on_fit_rf()
    assert mw._alc_rf_fitted is True

    mw._sync_alc_series_extra()
    saved = mw._project_model.batch(mw._alc_scan_series_id).to_dict()
    assert saved["extra"]["rf_mode"] is True
    assert saved["extra"]["rf_fitted"] is True

    settings = QSettings()
    settings.setValue(mw_module._UI_SCALE_SETTINGS_KEY, 1.0)
    mw2 = MainWindow()
    for ds in datasets:
        mw2._data_browser.add_dataset(ds)
    from asymmetry.core.representation.series import FitSeries

    mw2._project_model.add_batch(FitSeries.from_dict(saved))
    mw2._restore_alc_scan()

    assert mw2._alc_rf_mode is True
    assert mw2._alc_scan_view._fit_curve is not None  # RF fit re-run on load
    assert "A_µ" in mw2._alc_scan_view._rf_results.text()
