"""Fit-workflow guards: stop users silently doing the wrong thing.

Covers three "silent wrong thing" workflow gaps found in the full-corpus GUI
evaluation:

(a) fitting a time-domain model against the FFT spectrum when the workspace is
    in the frequency domain (CdS),
(b) the default ``Exponential + Constant`` splitting the amplitude during
    amplitude calibration with no affordance to drop the background (Photo),
(c) a buried 2nd period (light ON/OFF) the data browser never surfaced (Photo).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run  # noqa: E402
from asymmetry.core.fitting.composite import CompositeModel  # noqa: E402
from asymmetry.gui.panels.data_browser import (  # noqa: E402
    _PERIOD_ROLE,
    _TITLE_COLUMN,
    DataBrowserPanel,
)
from asymmetry.gui.panels.fit_panel import (  # noqa: E402
    GlobalFitTab,
    SingleFitTab,
    _dataset_representation_domain,
    _fit_domain_mismatch_message,
    _model_without_trailing_background,
)


def _time_dataset(run_number: int = 1) -> MuonDataset:
    t = np.linspace(0.0, 8.0, 200)
    return MuonDataset(
        time=t,
        asymmetry=0.2 * np.exp(-0.5 * t),
        error=np.full_like(t, 0.01),
        metadata={"run_number": run_number},
    )


def _frequency_dataset(run_number: int = 1) -> MuonDataset:
    f = np.linspace(0.0, 32.0, 200)
    return MuonDataset(
        time=f,
        asymmetry=np.abs(np.cos(f)),
        error=np.full_like(f, 0.01),
        metadata={"run_number": run_number, "plot_domain": "frequency"},
    )


# ── (a) domain-mismatch guard ────────────────────────────────────────────────


def test_representation_domain_reads_plot_domain_metadata() -> None:
    assert _dataset_representation_domain(_frequency_dataset()) == "frequency"
    assert _dataset_representation_domain(_time_dataset()) == "time"
    assert _dataset_representation_domain(None) == "time"


def test_domain_mismatch_message_only_on_disagreement() -> None:
    # Matching domains → no refusal.
    assert _fit_domain_mismatch_message("time", _time_dataset()) is None
    assert _fit_domain_mismatch_message("frequency", _frequency_dataset()) is None
    # Time fit against an FFT spectrum, and a frequency fit against time data.
    assert _fit_domain_mismatch_message("time", _frequency_dataset()) is not None
    assert _fit_domain_mismatch_message("frequency", _time_dataset()) is not None


def test_single_fit_refuses_time_fit_against_frequency_data(qapp: QApplication) -> None:
    """A time-domain fit must refuse the FFT spectrum, not fit garbage against it."""
    tab = SingleFitTab()
    # _domain stays "time" (the workspace switched to frequency without the fit
    # tab following); the dataset is the FFT spectrum.
    assert tab._domain == "time"
    tab.set_dataset(_frequency_dataset())

    captured: dict = {}
    tab._fit_engine = type(
        "E", (), {"fit": lambda self, *a, **k: captured.setdefault("ran", True)}
    )()
    tab._run_fit()

    assert "ran" not in captured  # the engine was never invoked
    assert tab._result_label.text().startswith("ERROR")
    assert "frequency-domain spectrum" in tab._result_label.text()


def test_single_fit_allows_matching_time_data(qapp: QApplication) -> None:
    """The guard must not block the normal time-domain flow."""
    tab = SingleFitTab()
    tab.set_dataset(_time_dataset())
    assert _fit_domain_mismatch_message(tab._domain, tab._current_dataset) is None


def test_global_fit_refuses_domain_mismatch(qapp: QApplication) -> None:
    tab = GlobalFitTab(member_kind="runs")
    tab.set_datasets([_frequency_dataset(1), _frequency_dataset(2)])
    assert tab._domain == "time"
    tab._run_global_fit()
    assert tab._result_text.toPlainText().lower().startswith("error")
    assert "frequency-domain spectrum" in tab._result_text.toPlainText()


# ── (b) drop-background affordance ───────────────────────────────────────────


def test_model_without_trailing_background_drops_only_a_plain_constant() -> None:
    exp_const = CompositeModel(["Exponential", "Constant"], operators=["+"])
    assert _model_without_trailing_background(exp_const).component_names == ["Exponential"]

    osc = CompositeModel(["Oscillatory", "Exponential", "Constant"], operators=["*", "+"])
    assert _model_without_trailing_background(osc).component_names == ["Oscillatory", "Exponential"]

    # No removable background: bare Exponential, or a Constant that is not the
    # final additive term.
    assert _model_without_trailing_background(CompositeModel(["Exponential"])) is None
    multiplied = CompositeModel(["Exponential", "Constant"], operators=["*"])
    assert _model_without_trailing_background(multiplied) is None


def test_drop_background_button_yields_single_exponential(qapp: QApplication) -> None:
    """The calibration affordance turns the default Exp+Const into a bare Exp."""
    tab = SingleFitTab()
    tab.set_dataset(_time_dataset())
    # The default model carries a free background, so the affordance is offered.
    assert tab._composite_model.component_names == ["Exponential", "Constant"]
    assert tab._drop_background_action.isEnabled()

    tab._on_drop_background()

    assert tab._composite_model.component_names == ["Exponential"]
    # Nothing left to drop → the affordance disables itself.
    assert not tab._drop_background_action.isEnabled()


def test_drop_background_disabled_in_frequency_domain(qapp: QApplication) -> None:
    tab = SingleFitTab()
    tab.set_domain("frequency")
    assert not tab._drop_background_action.isEnabled()


# ── (c) multi-period browser cue ─────────────────────────────────────────────


def _two_period_dataset(run_number: int = 5) -> MuonDataset:
    t = np.linspace(0.0, 8.0, 400)
    counts = np.full((400,), 100.0)
    run = Run(
        run_number=run_number,
        histograms=[Histogram(counts=counts, bin_width=0.02, t0_bin=0)],
        grouping={
            "period_reduced": [
                (t, np.cos(t), np.ones_like(t)),
                (t, np.sin(t), np.ones_like(t)),
            ]
        },
    )
    return MuonDataset(
        time=t,
        asymmetry=np.cos(t),
        error=np.ones_like(t),
        metadata={"run_number": run_number, "period_count": 2, "title": "Photo light"},
        run=run,
    )


def test_browser_surfaces_period_state_for_two_period_run(qapp: QApplication) -> None:
    browser = DataBrowserPanel()
    browser.add_dataset(_two_period_dataset(run_number=5))

    assert browser.multi_period_run_numbers() == {5}

    title_item = None
    for row in range(browser._table.rowCount()):
        item = browser._table.item(row, _TITLE_COLUMN)
        if item is not None and item.data(_PERIOD_ROLE):
            title_item = item
            break
    assert title_item is not None, "the 2-period run carries no browser cue"
    note = str(title_item.data(_PERIOD_ROLE))
    assert "2 periods" in note
    assert "Red" in note  # the default-active period a user silently fits
    # The cue is also explained on hover, with how to switch.
    assert "RG Mode" in title_item.toolTip()


def test_browser_no_period_cue_for_single_period_run(qapp: QApplication) -> None:
    """Single-period runs must stay uncluttered — no false multi-period cue."""
    browser = DataBrowserPanel()
    browser.add_dataset(_time_dataset(run_number=7))

    assert browser.multi_period_run_numbers() == set()
    for row in range(browser._table.rowCount()):
        item = browser._table.item(row, _TITLE_COLUMN)
        if item is not None:
            assert not item.data(_PERIOD_ROLE)
