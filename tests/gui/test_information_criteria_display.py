"""Unavailable criteria have one display and remain outside numeric rankings."""

import pytest
from PySide6.QtCore import Qt

from asymmetry.core.fitting.information_criteria import InformationCriteria, SelectionMetric
from asymmetry.gui.utils.information_criteria import criterion_item

pytestmark = [pytest.mark.gui]


def test_aicc_pole_cell_has_exact_reason(qapp):
    item = criterion_item(InformationCriteria(1.0, 3, 4), SelectionMetric.AICC)
    assert item.text() == "—"
    assert item.toolTip() == "AICc undefined: n − k − 1 = 0 (n = 4 points, k = 3 free parameters)"


def test_available_and_unavailable_items_sort_numerically(qapp):
    items = [
        criterion_item(InformationCriteria(statistic, 3, n), SelectionMetric.AICC)
        for statistic, n in [(1, 4), (100, 5), (1, 5)]
    ]
    ordered = sorted(items)
    assert [item.data(Qt.ItemDataRole.UserRole) for item in ordered] == [31.0, 130.0, float("inf")]
    assert ordered[-1].text() == "—"


def test_single_wizard_renders_unavailable_aicc_in_both_columns(qapp):
    from dataclasses import replace

    import numpy as np

    from asymmetry.core.data.dataset import MuonDataset
    from asymmetry.gui.windows.fit_wizard_window import FitWizardWindow
    from tests.gui.test_fit_wizard_window import _fake_recommendation

    time = np.linspace(0, 8, 120)
    dataset = MuonDataset(time=time, asymmetry=np.zeros_like(time), error=np.ones_like(time))
    rec = _fake_recommendation(dataset)
    row = replace(rec.assessments[0], sample_count=rec.assessments[0].parameter_count + 1)
    window = FitWizardWindow()
    window._recommendation = replace(rec, assessments=(row,))
    window._populate_compare_table()
    for column in (1, 3):
        item = window._compare_table.item(0, column)
        assert item.text() == "—"
        assert "n − k − 1 = 0" in item.toolTip()
    window.close()


def test_global_wizard_renders_unavailable_screen_and_optimised_aicc(qapp):
    from dataclasses import replace

    import numpy as np

    from asymmetry.core.data.dataset import MuonDataset
    from asymmetry.gui.windows.global_fit_wizard_window import GlobalFitWizardWindow
    from tests.gui.test_global_fit_wizard_window import _fake_recommendation

    time = np.linspace(0, 8, 120)
    datasets = [
        MuonDataset(
            time=time,
            asymmetry=np.zeros_like(time),
            error=np.ones_like(time),
            metadata={"run_number": run, "field": float(run)},
        )
        for run in (1, 2)
    ]
    rec = _fake_recommendation(datasets)
    row = replace(rec.assessments[0], sample_count=rec.assessments[0].parameter_count + 1)
    window = GlobalFitWizardWindow()
    window._recommendation = replace(rec, assessments=(row,))
    window._populate_screening_table([row])
    window._populate_optimised_table([row])
    for table in (window._screening_table, window._optimised_table):
        for column in (1, 3):
            assert table.item(0, column).text() == "—"
            assert "n − k − 1 = 0" in table.item(0, column).toolTip()
    window.close()
