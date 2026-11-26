"""Fit cards produced by a default FFT and an FFT with estimated errors."""

import numpy as np
from PySide6.QtWidgets import QVBoxLayout, QWidget

from ._base import Scenario, _process_events_for, register


class SpectralFitErrorsScenario(Scenario):
    name = "spectral_fit_errors"
    description = "Frequency-fit results show their effective spectral weights."
    size = (1280, 345)
    requires_fit = True

    def build(self) -> QWidget:
        from asymmetry.core.fitting import FitEngine, Parameter, ParameterSet
        from asymmetry.core.fourier.spectrum import (
            GroupSpectrumConfig,
            compute_average_group_spectrum,
        )
        from asymmetry.gui.panels.fit.single_tab import SingleFitTab
        from docs.screenshots.data import make_ybco_vortex_lattice

        source = make_ybco_vortex_lattice()
        display = QWidget()
        layout = QVBoxLayout(display)

        def model(x, **p):
            return p["amplitude"] * np.exp(-0.5 * ((x - p["center"]) / p["width"]) ** 2)

        for estimated, title in ((False, "Default FFT"), (True, "FFT with Average errors")):
            spectrum = compute_average_group_spectrum(
                source.run, GroupSpectrumConfig(padding=4, estimate_average_error=estimated)
            )
            assert spectrum is not None
            dataset = spectrum.time_range(26, 30)
            height = float(np.max(dataset.asymmetry))
            center = float(dataset.time[np.argmax(dataset.asymmetry)])
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
                error_oversampling=4,
            )
            assert result.success
            assert any(
                warning.startswith("Diagonal spectral errors" if estimated else "Unit weights")
                for warning in result.warnings
            )
            tab = SingleFitTab()
            tab._show_fit_summary(result, tag="Fit ✓", tone="ok", headline=title)
            layout.addWidget(tab._results_card)
            tab.close()
            tab.deleteLater()
        layout.addStretch()
        _process_events_for(milliseconds=80)
        return display


register(SpectralFitErrorsScenario())
