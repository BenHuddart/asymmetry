"""A malformed owned project is rejected beside an intact unsaved Ag analysis."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PySide6.QtWidgets import QMessageBox

from ..data import make_ag_zf_gkt
from ._base import Scenario, register


class ProjectRejectionScenario(Scenario):
    name = "project_rejection"
    description = "Malformed project rejection with unsaved synthetic Ag data retained."
    size = (1200, 760)

    def build(self):
        from asymmetry.core.project import CURRENT_SCHEMA_VERSION
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        dataset = make_ag_zf_gkt()
        window._data_browser.add_dataset(dataset)
        window._on_dataset_selected(dataset.run_number)
        window._mark_dirty()
        errors = []
        with TemporaryDirectory() as directory:
            path = Path(directory) / "malformed.asymp"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": CURRENT_SCHEMA_VERSION,
                        "datasets": [{"run_number": 12, "source_file": []}],
                    }
                )
            )
            with patch.object(
                QMessageBox, "critical", side_effect=lambda *args: errors.append(args)
            ):
                window._open_project_file(str(path))
        assert window._data_browser.get_dataset(dataset.run_number) is dataset
        assert window._dirty
        assert "source_file: expected str" in errors[0][-1]
        return window


register(ProjectRejectionScenario())
