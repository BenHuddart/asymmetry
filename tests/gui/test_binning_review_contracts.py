"""Wizard cache IDs remain JSON integers across metadata representations."""

import json

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.gui.panels.fit.single_tab import SingleFitTab
from asymmetry.gui.windows.fit_wizard_window import FitWizardWindow

pytestmark = pytest.mark.gui


@pytest.mark.parametrize("run_number", [101, "101", np.int64(101), None])
@pytest.mark.parametrize("consumer", ["tab", "window"])
def test_single_wizard_signature_preserves_canonical_run_ids(qapp, consumer, run_number):
    time = np.linspace(0, 4, 80)
    dataset = MuonDataset(time, np.exp(-time), np.full(80, 0.1), {"run_number": 101})
    if consumer == "tab":
        owner = SingleFitTab()
        owner.set_dataset(dataset)
    else:
        owner = FitWizardWindow()
        owner.set_analysis_context(dataset)
    dataset.metadata["run_number"] = run_number
    signature = (
        owner._wizard_context_signature() if consumer == "tab" else owner._analysis_signature()
    )
    expected = None if run_number is None else 101
    assert type(signature["run_number"]) is type(expected)
    assert signature["run_number"] == expected
    restored = json.loads(json.dumps(signature))
    if consumer == "tab":
        restored["run_number"] = expected
        assert owner._wizard_base_signature_matches(restored, signature)
    assert dataset.metadata["run_number"] is run_number
    owner.close()
