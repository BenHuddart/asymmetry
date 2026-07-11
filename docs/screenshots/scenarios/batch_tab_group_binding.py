"""Batch tab bound to a data group, with one member unticked.

Forms a data group over four runs of the EuO ZF temperature scan and drives
it through the real **"Fit this group..."** flow
(``MainWindow._on_fit_group_requested``) rather than an ad-hoc selection, so
the Batch tab shows the group-binding banner and the per-member **Batch
members** checkbox list this creates. One member is unticked exactly as a
user would via the checkbox, exercising the same
``itemChanged``/``_on_member_check_changed`` path -- excluding it from this
particular analysis without touching the group's own membership. See
*Reference > GUI usage > Fitting a group directly*.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from ..data import make_euo_tf_tscan
from ._base import Scenario, register


class BatchTabGroupBindingScenario(Scenario):
    name = "batch_tab_group_binding"
    description = "Batch tab bound to a data group, with one member excluded."
    size = (1500, 920)

    def build(self) -> QWidget:
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        window._on_fit()
        window._dock_log.hide()
        window.resizeDocks(
            [window._dock_data_browser], [300], Qt.Orientation.Horizontal
        )

        datasets = make_euo_tf_tscan()[:4]  # 30, 50, 65, 69 K -- below/at Tc
        for dataset in datasets:
            window._data_browser.add_dataset(dataset)
        run_numbers = [int(ds.run_number) for ds in datasets]

        gid = window._data_browser.create_data_group(run_numbers, name="T scan — EuO")
        # Drive the real "Fit this group..." handler so the Batch tab binds
        # to the group and its member checkboxes populate exactly as they
        # would for a user-triggered context-menu action.
        window._on_fit_group_requested(gid)

        global_tab = window._fit_panel._global_tab
        window._fit_panel._tabs.setCurrentWidget(global_tab)

        # Untick the highest-temperature member (69 K, right at Tc, the run
        # most likely to be excluded from a below-Tc order-parameter batch).
        members_list = global_tab._members_list
        for row in range(members_list.count()):
            item = members_list.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == run_numbers[-1]:
                item.setCheckState(Qt.CheckState.Unchecked)
                break

        window._on_dataset_selected(run_numbers[0])
        return window


register(BatchTabGroupBindingScenario())
