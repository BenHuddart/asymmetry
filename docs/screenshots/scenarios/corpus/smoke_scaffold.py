"""Scaffold smoke test: load one Basics run into the MainWindow and grab it."""

from __future__ import annotations

from PySide6.QtWidgets import QWidget

from ._corpus import CorpusScenario, load_corpus_datasets, register


class ScaffoldSmokeScenario(CorpusScenario):
    name = "corpus_scaffold_smoke"
    description = "Scaffold check: EMU00018850.nxs loaded into the main window."
    example = "Basics"
    size = (1400, 860)

    def build(self) -> QWidget:
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        datasets = load_corpus_datasets(["Basics/data/EMU00018850.nxs"])
        self.add_to_browser(window, datasets)
        window._on_dataset_selected(datasets[0].run_number)
        return window


register(ScaffoldSmokeScenario())
