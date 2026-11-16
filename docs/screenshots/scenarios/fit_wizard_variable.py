"""A variable-binned count curve with a separately labelled native FFT view."""

from PySide6.QtWidgets import QVBoxLayout, QWidget

from ._base import Scenario, _process_events_for, register


class FitWizardVariableScenario(Scenario):
    name = "fit_wizard_variable"
    description = "Fit Wizard with Variable science bins and a fixed native spectral diagnostic."
    size = (1180, 660)
    requires_fit = True

    def build(self) -> QWidget:
        from dataclasses import replace

        import numpy as np

        from asymmetry.core.fitting.component_tags import PhysicsClass
        from asymmetry.core.fitting.fit_wizard import build_fit_wizard_recommendation
        from asymmetry.core.fitting.wizard_scope import WizardScope
        from asymmetry.core.simulate import BUILTIN_TEMPLATES, reduce_run_to_dataset, simulate_run
        from asymmetry.gui.windows.fit_wizard_window import FitWizardWindow

        template = replace(
            BUILTIN_TEMPLATES["ideal_continuous_fb"],
            n_detectors=2,
            n_bins=2000,
            bin_width_us=0.01,
            t0_bin=0,
            forward_detectors=(1,),
            backward_detectors=(2,),
        ).build()
        run = simulate_run(
            template,
            lambda t: 16 * np.exp(-0.15 * t) * np.cos(2 * np.pi * 1.3 * t) + 1,
            total_events=1e9,
            seed=465,
            run_number=465,
        )
        run.grouping.update(binning_mode="variable", bin0_us=0.02, bin10_us=0.08)
        run.metadata.update(title="Synthetic 1.3 MHz precession", field_direction="TF")
        dataset = reduce_run_to_dataset(run)
        window = FitWizardWindow()
        window.set_analysis_context(dataset)
        recommendation = build_fit_wizard_recommendation(
            dataset,
            max_workers=4,
            refine_top_candidates=0,
            scope=WizardScope(physics=frozenset({PhysicsClass.MAGNETISM})),
            diagnostic=window._spectral_preview.observation,
        )
        window.set_cached_recommendation(recommendation)
        _process_events_for(milliseconds=150)
        panel = QWidget()
        panel._wizard_window = window
        QVBoxLayout(panel).addWidget(window._fingerprint_plot_widget)
        window._fft_ax.set_xlim(0, 3)
        return panel

    def teardown(self, widget: QWidget) -> None:
        widget._wizard_window.close()
        super().teardown(widget)


register(FitWizardVariableScenario())
