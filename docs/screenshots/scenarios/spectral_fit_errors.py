"""Frequency-fit result caveat from the actual engine and shared result card."""

import numpy as np
from PySide6.QtWidgets import QWidget

from ._base import Scenario, _process_events_for, register


class SpectralFitErrorsScenario(Scenario):
    name = "spectral_fit_errors"
    description = "Frequency-fit result qualifies approximate errors and statistics."
    size = (1050, 210)
    requires_fit = True

    def build(self) -> QWidget:
        from asymmetry.core.data.dataset import MuonDataset
        from asymmetry.core.fitting import FitEngine, Parameter, ParameterSet
        from asymmetry.core.fourier.spectrum import (
            GroupSpectrumConfig,
            compute_average_group_spectrum,
        )
        from asymmetry.gui.panels.fit.single_tab import SingleFitTab
        from docs.screenshots.data import make_ybco_vortex_lattice

        source = make_ybco_vortex_lattice()
        spectrum = compute_average_group_spectrum(
            source.run, GroupSpectrumConfig(padding=4, estimate_average_error=True)
        )
        assert spectrum is not None
        selected = (spectrum.time >= 26) & (spectrum.time <= 30)
        dataset = MuonDataset(
            time=spectrum.time[selected],
            asymmetry=spectrum.asymmetry[selected],
            error=spectrum.error[selected],
            metadata=dict(spectrum.metadata),
        )
        height = float(np.max(dataset.asymmetry))
        center = float(dataset.time[np.argmax(dataset.asymmetry)])

        def model(x, **p):
            return p["amplitude"] * np.exp(-0.5 * ((x - p["center"]) / p["width"]) ** 2)

        result = FitEngine().fit(
            dataset,
            model,
            ParameterSet(
                [
                    Parameter(name="amplitude", value=height),
                    Parameter(name="center", value=center),
                    Parameter(name="width", value=0.3, min=0.01),
                ]
            ),
        )
        assert any("Approximate spectral fit" in text for text in result.warnings)
        tab = SingleFitTab()
        tab._show_fit_summary(result, tag="Fit ⚠", tone="warn", headline="frequency spectrum")
        card = tab._results_card
        card.setParent(None)
        tab.close()
        tab.deleteLater()
        _process_events_for(milliseconds=80)
        return card


register(SpectralFitErrorsScenario())
