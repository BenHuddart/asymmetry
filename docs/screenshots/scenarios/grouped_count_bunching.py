"""Raw detector counts at toolbar Bunch 2."""

from __future__ import annotations

from ..data import make_ybco_knight_grouped
from ._base import Scenario, register


class GroupedCountBunchingScenario(Scenario):
    name = "grouped_count_bunching"
    description = "Raw counts for two detector groups at toolbar Bunch 2."
    size = (1600, 740)

    def build(self):
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        dataset = make_ybco_knight_grouped()
        dataset.run.grouping["included_groups"] = {3: False, 4: False}
        window._data_browser.add_dataset(dataset)
        window._data_browser.select_runs([dataset.run_number])
        window._dock_data_browser.hide()
        window._dock_log.hide()
        return window

    def settle(self, window):
        super().settle(window)
        window._raw_counts_action.trigger()
        window._view_bunch_spin.setValue(2)
        window._dock_fit.hide()
        window._dock_fourier.hide()
        window._dock_fit_parameters.hide()
        super().settle(window)
        axes = window._plot_panel._figure.axes
        assert len(axes) == 2 and all(axis.get_ylabel() == "Counts" for axis in axes)


register(GroupedCountBunchingScenario())
