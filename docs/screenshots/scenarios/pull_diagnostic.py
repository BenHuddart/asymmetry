"""Conditional ideal-count pulls with seeds, assumptions and full denominators."""

import numpy as np
from PySide6.QtCore import QEventLoop, QTimer

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.pull_refit import EngineRefit
from asymmetry.core.pull_request import make_pull_request
from asymmetry.gui.windows.pull_diagnostic_window import PullDiagnosticWindow

from ._base import Scenario, register


class PullDiagnosticScenario(Scenario):
    name = "pull_diagnostic"
    description = "Conditional ideal-count constant-fit pull experiment."
    size = (850, 650)
    requires_fit = True

    def build(self):
        template = Run(
            446,
            [Histogram(np.zeros(500), 0.016, good_bin_end=499) for _ in range(2)],
            grouping={
                "groups": {1: [1], 2: [2]},
                "alpha": 1.2,
                "first_good_bin": 0,
                "last_good_bin": 499,
            },
        )
        model = CompositeModel(["Constant"])
        refit = EngineRefit(model, ParameterSet([Parameter("A_bg", 20)]))
        request = make_pull_request(
            template,
            model,
            {"A_bg": 20},
            total_events=1e6,
            n_seeds=100,
            seed_start=71,
            refit_configuration=refit.configuration(),
        )
        window = PullDiagnosticWindow(request=request, model=model, refit=refit)
        window._run_button.click()
        return window

    def settle(self, widget):
        loop = QEventLoop()
        poll = QTimer()
        poll.timeout.connect(lambda: loop.quit() if widget._worker is None else None)
        poll.start(10)
        QTimer.singleShot(10000, loop.quit)
        loop.exec()
        poll.stop()
        if widget.last_result is None:
            widget.close()
            raise RuntimeError("Conditional pull screenshot did not complete")
        super().settle(widget)


register(PullDiagnosticScenario())
