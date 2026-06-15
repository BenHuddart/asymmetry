"""RED target for branch ``fix/global-wizard-series-overview``.

Round-3 GUI finding (Al-LLZ, ``_findings/windows-gui/Round3_progress.md``): the
Global Fit Wizard's **"1. Series Overview"** tab shows only the header — zero rows —
even though the wizard reports "N datasets" and lists them in the subtitle. The
Run / Field / Temperature columns are known at ``set_analysis_context`` time yet the
``_overview_table`` is left empty (``set_analysis_context`` ends with
``_set_empty_state()`` and never populates the overview rows).

Contract: after ``set_analysis_context([N datasets])`` the Series Overview table has
one row per dataset (at minimum Run / Field / Temperature filled). RED today
(rowCount == 0).
"""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

import numpy as np
from PySide6.QtWidgets import QApplication

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.gui.windows.global_fit_wizard_window import GlobalFitWizardWindow


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def _dataset(run_number: int, field: float, temperature: float) -> MuonDataset:
    run = Run(
        run_number=run_number,
        histograms=[
            Histogram(np.array([10.0, 20.0, 30.0, 40.0]), 0.1, 0),
            Histogram(np.array([8.0, 16.0, 24.0, 32.0]), 0.1, 0),
        ],
        metadata={"field": field, "temperature": temperature},
        grouping={
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": 3,
        },
    )
    return MuonDataset(
        np.array([0.0, 0.1, 0.2, 0.3]),
        np.array([0.1, 0.1, 0.1, 0.1]),
        np.array([0.01, 0.01, 0.01, 0.01]),
        {"run_number": run_number, "field": field, "temperature": temperature},
        run,
    )


def test_series_overview_lists_the_selected_datasets(app):
    datasets = [
        _dataset(51341, field=0.0, temperature=160.0),
        _dataset(51342, field=5.0, temperature=160.0),
        _dataset(51343, field=10.0, temperature=160.0),
    ]
    wizard = GlobalFitWizardWindow()
    try:
        wizard.set_analysis_context(datasets)
        assert wizard._overview_table.rowCount() == 3, (
            "Series Overview tab is empty — it must list one row per selected "
            "dataset (Run / Field / Temperature) after set_analysis_context"
        )
    finally:
        wizard.close()
        wizard.deleteLater()
