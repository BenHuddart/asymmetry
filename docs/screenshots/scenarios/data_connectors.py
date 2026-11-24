"""Short-dashed data guides and a real weighted fit to synthetic EuO precession."""

import numpy as np

from asymmetry.core.fitting.engine import FitEngine
from asymmetry.core.fitting.models import MODELS
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.gui.panels.plot_panel import PlotPanel

from ..data import make_euo_tf_tscan
from ._base import Scenario, register


class DataConnectorsScenario(Scenario):
    name = "data_connectors"
    description = (
        "Short-dashed source-sample connectors and a solid weighted fit to EuO precession."
    )
    size = (1000, 600)
    requires_fit = True

    def build(self):
        dataset = make_euo_tf_tscan()[0]
        function = MODELS["Oscillatory"].function
        parameters = ParameterSet(
            [
                Parameter("A0", 22, min=0, max=40),
                Parameter("frequency", 22.28, min=20, max=24),
                Parameter("phase", 0),
                Parameter("Lambda", 0.1, min=0),
                Parameter("baseline", 0.4),
            ]
        )
        result = FitEngine().fit(dataset, function, parameters)
        if not result.success:
            raise RuntimeError("Synthetic connector screenshot fit must converge")
        time = np.linspace(0, 6, 12001)
        panel = PlotPanel()
        panel.plot_dataset(dataset)
        panel.plot_fit(
            time,
            function(time, **{p.name: p.value for p in result.parameters}),
            label="Weighted oscillatory fit",
            run_number=dataset.run_number,
        )
        panel.set_connect_points_enabled(True)
        panel.set_view_limits(0, 0.8, -26, 26)
        return panel


register(DataConnectorsScenario())
