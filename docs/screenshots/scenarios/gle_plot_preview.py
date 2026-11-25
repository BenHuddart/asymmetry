"""Vector preview with synthetic observations outside a component stack."""

from io import BytesIO

import numpy as np
from matplotlib.figure import Figure

from asymmetry.gui.windows.gle_preview import GlePlotPreview

from ._base import Scenario, register


class GlePreviewScenario(Scenario):
    name = "gle_plot_preview"
    description = "Vector preview with complete outlying observations and component fills"
    size = (1000, 720)

    def build(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([0.2, 0.4, 0.3])
        figure = Figure(figsize=(6.4, 4.8))
        axes = figure.add_subplot(111)
        axes.errorbar(
            x,
            y,
            yerr=np.full_like(y, 0.01),
            marker="o",
            linestyle="none",
            color="black",
            markersize=4,
            capsize=2,
        )
        model_x = np.linspace(1.0, 3.0, 200)
        constant = np.full_like(model_x, 0.05)
        total = 0.05 * model_x + 0.15
        for lower, upper, color in [(0, constant, "lightblue"), (constant, total, "lightgreen")]:
            axes.fill_between(model_x, lower, upper, color=color, alpha=0.3)
            axes.plot(model_x, upper, linestyle="--", color=color, linewidth=0.8)
        axes.plot(model_x, total, color="red")
        axes.set_xlabel("Temperature (K)", fontfamily="serif")
        axes.set_ylabel("Relaxation rate (µs⁻¹)", fontfamily="serif")
        for label in axes.get_xticklabels() + axes.get_yticklabels():
            label.set_fontfamily("serif")
        figure.tight_layout()
        output = BytesIO()
        figure.savefig(output, format="pdf")
        preview = GlePlotPreview(None)
        preview.status.setText("GLE plot exported — synthetic parameter fit with components")
        preview.set_pdf(output.getvalue())
        return preview


register(GlePreviewScenario())
