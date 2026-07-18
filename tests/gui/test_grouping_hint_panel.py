"""The transverse-field grouping nudge surfaced on the plot / Fourier panels.

The grouping dialog already nudges the user away from a grouping that washes out
a transverse precession, but that hint is buried inside a dialog the user may
never open — a corpus study loaded a HiFi TF run, saw an empty plot, and wrongly
concluded the dataset was unusable in-app. These tests pin the fix: the same
recommendation (``recommend_grouping_preset_for_run``) is surfaced on both plot
panels, is dismissable per run, and clears itself once the run is regrouped onto
the recommended preset.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

pytest.importorskip("PySide6")
from PySide6.QtCore import QSettings  # type: ignore
from PySide6.QtWidgets import QApplication  # type: ignore

import asymmetry.gui.mainwindow as mw_module
from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.gui.mainwindow import MainWindow


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


def _hifi_dataset(
    run_number: int = 91491,
    *,
    field_direction: str = "Transverse",
    grouping_preset: str | None = None,
) -> MuonDataset:
    """A minimal HiFi-flavoured run on the longitudinal (forward/backward) grouping.

    ``detect_instrument`` keys off the ``instrument`` metadata, so two histograms
    are enough to exercise the recommendation without building all 64 detectors.
    """
    counts = np.array([100.0, 96.0, 92.0, 88.0], dtype=float)
    grouping: dict = {
        "groups": {1: [1], 2: [2]},
        "forward_group": 1,
        "backward_group": 2,
        "alpha": 1.0,
        "first_good_bin": 0,
        "last_good_bin": 3,
        "bunching_factor": 1,
        "deadtime_correction": False,
    }
    if grouping_preset is not None:
        grouping["grouping_preset"] = grouping_preset
    metadata = {
        "run_number": run_number,
        "instrument": "HIFI",
        "field": 160.0,
        "field_direction": field_direction,
    }
    run = Run(
        run_number=run_number,
        histograms=[
            Histogram(counts=counts, bin_width=0.01),
            Histogram(counts=counts * 0.8, bin_width=0.01),
        ],
        metadata=dict(metadata),
        grouping=grouping,
    )
    t = np.array([0.0, 0.01, 0.02, 0.03])
    return MuonDataset(
        time=t,
        asymmetry=np.zeros_like(t),
        error=np.full_like(t, 0.01),
        metadata=dict(metadata),
        run=run,
    )


def _time_bar(window: MainWindow):
    return window._plot_panel._grouping_hint_bar


def _freq_bar(window: MainWindow):
    return window._frequency_plot_panel._grouping_hint_bar


def test_transverse_run_shows_hint_on_both_panels(mainwindow: MainWindow) -> None:
    mainwindow._data_browser.add_dataset(_hifi_dataset())
    mainwindow._on_dataset_selected(91491)

    assert _time_bar(mainwindow).isVisibleTo(mainwindow._plot_panel)
    assert _freq_bar(mainwindow).isVisibleTo(mainwindow._frequency_plot_panel)
    # The nudge names the concrete preset to apply.
    text = mainwindow._plot_panel._grouping_hint_label.text()
    assert "Transverse (Vector)" in text
    assert "Open Grouping" in text


def test_longitudinal_run_shows_no_hint(mainwindow: MainWindow) -> None:
    mainwindow._data_browser.add_dataset(
        _hifi_dataset(run_number=91000, field_direction="Longitudinal")
    )
    mainwindow._on_dataset_selected(91000)

    assert not _time_bar(mainwindow).isVisibleTo(mainwindow._plot_panel)
    assert not _freq_bar(mainwindow).isVisibleTo(mainwindow._frequency_plot_panel)


def test_run_already_on_recommended_preset_shows_no_hint(mainwindow: MainWindow) -> None:
    mainwindow._data_browser.add_dataset(
        _hifi_dataset(run_number=91492, grouping_preset="Transverse (Vector)")
    )
    mainwindow._on_dataset_selected(91492)

    assert not _time_bar(mainwindow).isVisibleTo(mainwindow._plot_panel)


def test_dismiss_hides_hint_for_that_run_across_reselection(mainwindow: MainWindow) -> None:
    mainwindow._data_browser.add_dataset(_hifi_dataset(run_number=91491))
    mainwindow._data_browser.add_dataset(_hifi_dataset(run_number=91493))
    mainwindow._on_dataset_selected(91491)
    assert _time_bar(mainwindow).isVisibleTo(mainwindow._plot_panel)

    # Dismiss by clicking the real ✕ button (exercises its clicked wiring).
    mainwindow._plot_panel._grouping_hint_dismiss_btn.click()
    assert not _time_bar(mainwindow).isVisibleTo(mainwindow._plot_panel)
    assert not _freq_bar(mainwindow).isVisibleTo(mainwindow._frequency_plot_panel)

    # A different transverse run still nudges…
    mainwindow._on_dataset_selected(91493)
    assert _time_bar(mainwindow).isVisibleTo(mainwindow._plot_panel)

    # …but the dismissed run stays quiet when reselected.
    mainwindow._on_dataset_selected(91491)
    assert not _time_bar(mainwindow).isVisibleTo(mainwindow._plot_panel)


def test_hint_link_opens_grouping_dialog(mainwindow: MainWindow, monkeypatch) -> None:
    opened: list[bool] = []
    monkeypatch.setattr(mainwindow, "_on_grouping_current", lambda: opened.append(True))

    mainwindow._data_browser.add_dataset(_hifi_dataset(run_number=91491))
    mainwindow._on_dataset_selected(91491)
    # Activating the label's link (its real linkActivated wiring) must reach the
    # grouping dialog opener.
    mainwindow._plot_panel._grouping_hint_label.linkActivated.emit("#grouping")

    assert opened == [True]


def test_regrouping_onto_recommended_preset_clears_hint(mainwindow: MainWindow) -> None:
    dataset = _hifi_dataset(run_number=91491)
    mainwindow._data_browser.add_dataset(dataset)
    mainwindow._on_dataset_selected(91491)
    assert _time_bar(mainwindow).isVisibleTo(mainwindow._plot_panel)

    # Applying the recommended preset records it on the run's grouping; the next
    # render must drop the nudge.
    dataset.run.grouping["grouping_preset"] = "Transverse (Vector)"
    mainwindow._render_current_selection_plot()

    assert not _time_bar(mainwindow).isVisibleTo(mainwindow._plot_panel)


def test_multi_run_overlay_suppresses_hint(mainwindow: MainWindow) -> None:
    mainwindow._data_browser.add_dataset(_hifi_dataset(run_number=91491))
    mainwindow._data_browser.add_dataset(_hifi_dataset(run_number=91493))
    # Single run first: the hint shows and _current_dataset is 91491.
    mainwindow._on_dataset_selected(91491)
    assert _time_bar(mainwindow).isVisibleTo(mainwindow._plot_panel)

    # Extend to a two-run overlay and re-render. _current_dataset stays 91491
    # (the overlay branch never reassigns it), so the hint must be suppressed by
    # the run-count gate rather than attributed to a run that isn't singled out.
    mainwindow._data_browser.select_runs([91491, 91493])
    mainwindow._render_current_selection_plot()
    assert not _time_bar(mainwindow).isVisibleTo(mainwindow._plot_panel)


def test_panel_clear_hides_hint(mainwindow: MainWindow) -> None:
    mainwindow._data_browser.add_dataset(_hifi_dataset(run_number=91491))
    mainwindow._on_dataset_selected(91491)
    assert _time_bar(mainwindow).isVisibleTo(mainwindow._plot_panel)

    mainwindow._plot_panel.clear()
    assert not _time_bar(mainwindow).isVisibleTo(mainwindow._plot_panel)


def test_dismissal_does_not_leak_across_projects(mainwindow: MainWindow) -> None:
    # Run numbers are per-experiment counters, so a dismissal must not silently
    # pre-suppress the nudge on a same-numbered run in the next project.
    mainwindow._data_browser.add_dataset(_hifi_dataset(run_number=3366))
    mainwindow._on_dataset_selected(3366)
    mainwindow._plot_panel._grouping_hint_dismiss_btn.click()
    assert (3366, "Transverse (Vector)") in mainwindow._dismissed_grouping_hints

    mainwindow._clear_all_state()
    assert mainwindow._dismissed_grouping_hints == set()
    assert mainwindow._active_grouping_hint_key is None

    mainwindow._data_browser.add_dataset(_hifi_dataset(run_number=3366))
    mainwindow._on_dataset_selected(3366)
    assert _time_bar(mainwindow).isVisibleTo(mainwindow._plot_panel)
