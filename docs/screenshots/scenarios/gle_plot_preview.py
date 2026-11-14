"""Read-only vector preview of a deterministic synthetic Ag spectrum."""

from io import BytesIO

from matplotlib.figure import Figure

from asymmetry.gui.windows.gle_preview import GlePlotPreview

from ..data.archetypes import make_ag_zf_gkt
from ._base import Scenario, register


class GlePreviewScenario(Scenario):
    name = "gle_plot_preview"
    description = "Vector plot preview with fit and zoom controls"
    size = (1000, 720)

    def build(self):
        data = make_ag_zf_gkt()
        figure = Figure(figsize=(6.4, 4.8))
        axes = figure.add_subplot(111)
        axes.errorbar(data.time, data.asymmetry, yerr=data.error, fmt=".", markersize=2)
        axes.set_xlabel("Time (µs)")
        axes.set_ylabel("Asymmetry (%)")
        figure.tight_layout()
        output = BytesIO()
        figure.savefig(output, format="pdf")
        preview = GlePlotPreview(None)
        preview.status.setText("GLE plot exported — synthetic ZF Ag spectrum")
        preview.set_pdf(output.getvalue())
        return preview


register(GlePreviewScenario())
