"""Data Browser with a user group and an auto-created group side by side.

Populates a standalone Data Browser panel with the six-run EuO ZF
temperature scan and forms two data groups over it: a **user** group ("T <
Tc — EuO", named explicitly, so it paints in the blue ramp) covering the
three ordered-state runs, and an **auto** group ("Runs 3003–3005", the name
and red-grey palette a real batch fit would mint automatically for an ad-hoc
run selection) that overlaps the user group by one run. The shared run
(3003) therefore renders twice: once under its primary membership (the user
group) and once as a marked copy row under the auto group, with a
circled-digit marker and an "Also in:" tooltip naming the other membership
-- the multi-group presentation documented in
*Reference > GUI usage > Data groups*.

A standalone panel (rather than a full ``MainWindow``) is used so the row
widths are governed only by the panel's own size, not by ``MainWindow``'s
dock-splitter layout.
"""

from __future__ import annotations

from PySide6.QtWidgets import QWidget

from ..data import make_euo_tf_tscan
from ._base import Scenario, register


class DataBrowserGroupsScenario(Scenario):
    name = "data_browser_groups"
    description = (
        "Data browser with a user (blue) group and an auto (red-grey) group "
        "sharing a marked duplicate row."
    )
    size = (720, 400)

    def build(self) -> QWidget:
        from asymmetry.gui.panels.data_browser import DataBrowserPanel

        browser = DataBrowserPanel()
        for dataset in make_euo_tf_tscan():
            browser.add_dataset(dataset)

        # A user-named group over the three runs below Tc=69 K (ordered
        # state) -- kind="user" is the default, painting the blue ramp.
        browser.create_data_group([3001, 3002, 3003], name="T < Tc — EuO")
        # An auto-created group (as a real ad-hoc batch fit would mint,
        # mainwindow._on_fit_group_requested/_record_fit_series naming
        # convention: "Runs <range>") sharing run 3003 with the user group
        # above, so 3003 renders once under each -- the primary row under
        # its first (user) group, a marked copy row under this one.
        browser.create_data_group([3003, 3004, 3005], name="Runs 3003–3005", kind="auto")

        browser.select_runs([3002])
        # The Run column's automatic width cap (150 px) elides the auto-group's
        # full name; widen it as a user would drag it, so the "Runs <range>"
        # naming convention the docs describe is actually legible.
        browser._table.horizontalHeader().resizeSection(0, 190)
        return browser


register(DataBrowserGroupsScenario())
