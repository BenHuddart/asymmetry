"""B6: the ALC integral-scan fit panel must reset on File -> New Project.

Live-testing finding (Round-3): after building an ALC scan and configuring a
Cubic baseline + fit regions + Lorentzian peaks, starting a new project (and
building a different scan on a different field span) left the previous scan's
baseline model, regions and peaks in place. The stale regions/peaks also
distorted the new scatter's auto-range. ``_clear_all_state`` never touches the
ALC scan view, so its analysis state leaks across projects.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

pyside6 = pytest.importorskip("PySide6")
from PySide6.QtCore import QSettings  # type: ignore  # noqa: E402
from PySide6.QtWidgets import QApplication  # type: ignore  # noqa: E402

import asymmetry.gui.mainwindow as mw_module  # noqa: E402
from asymmetry.core.data.dataset import Histogram, MuonDataset, Run  # noqa: E402
from asymmetry.gui.mainwindow import MainWindow  # noqa: E402


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


def _ds(run_number: int, fwd: float, bwd: float, field: float) -> MuonDataset:
    n = 4
    meta = {"run_number": run_number, "field": field, "temperature": 10.0}
    run = Run(
        run_number=run_number,
        histograms=[
            Histogram(counts=np.full(n, fwd), bin_width=0.01),
            Histogram(counts=np.full(n, bwd), bin_width=0.01),
        ],
        metadata=dict(meta),
        grouping={
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": n - 1,
            "bunching_factor": 1,
        },
    )
    t = np.arange(n) * 0.01
    return MuonDataset(
        time=t,
        asymmetry=np.zeros(n),
        error=np.full(n, 0.01),
        metadata=dict(meta),
        run=run,
    )


def test_alc_analysis_state_clears_on_new_project(mainwindow: MainWindow):
    mw = mainwindow
    mw._plot_workspace.set_active_view("integral_scan")

    # Build a scan and configure a Cubic baseline + regions + a Lorentzian peak.
    mw._fit_panel.set_datasets([_ds(11, 110.0, 90.0, 100.0), _ds(12, 120.0, 80.0, 200.0)])
    mw._on_scan_requested()
    assert mw._alc_scan_view.point_count() == 2

    view = mw._alc_scan_view
    idx = view._baseline_model_combo.findText("Cubic")
    view._baseline_model_combo.setCurrentIndex(idx)
    view._add_region()
    view._add_region()
    view._add_peak("Lorentzian")

    assert view._regions_table.rowCount() == 2
    assert view._peaks_table.rowCount() == 1
    assert view.baseline_model() == "Cubic"

    # File -> New Project clears all session state.
    mw._clear_all_state()

    assert view._regions_table.rowCount() == 0, "ALC fit regions leaked across New Project"
    assert view._peaks_table.rowCount() == 0, "ALC peaks leaked across New Project"
    assert view.baseline_model() == "Linear", "ALC baseline model not reset on New Project"
    assert view.point_count() == 0, "ALC scan data leaked across New Project"
