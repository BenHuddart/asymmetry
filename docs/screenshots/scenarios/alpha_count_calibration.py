"""Forward/backward count calibration with explicit promotion controls.

A pulsed TF simulation has α = 1.25 and A cos(2π f t + φ), with f in MHz
and A in percent. The matched Oscillatory fit reports α and its uncertainty.
The Poisson fit reads integer raw counts without applied deadtime correction.
Its result says Poisson deviance/ν; sparse tail bins have no χ² quality verdict.
The Calibration section shows Promote DT₀, which writes a captured total table.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QWidget

from ._base import Scenario, _process_events_for, register


def _tf_asymmetry(t, A=20.0, f=1.5, phi=0.3):  # noqa: N803 — A is the asymmetry symbol
    """Transverse-field precession asymmetry (percent): A·cos(2πft + φ)."""
    return A * np.cos(2.0 * np.pi * f * np.asarray(t, dtype=float) + phi)


class AlphaCountCalibrationScenario(Scenario):
    name = "alpha_count_calibration"
    description = (
        "Count-domain forward/backward free-α fit on a TF calibration run, "
        "with the Count-fit options and Calibration sections expanded."
    )
    size = (640, 1080)
    requires_fit = True

    def build(self) -> QWidget:
        from asymmetry.core.fitting.composite import CompositeModel
        from asymmetry.core.fitting.seeding import Seed
        from asymmetry.core.simulate import (
            build_builtin_template,
            reduce_run_to_dataset,
            simulate_run,
        )
        from asymmetry.gui.windows.multi_group_fit_window import MultiGroupFitWindow

        window = MultiGroupFitWindow()

        # A pulsed TF calibration run with a known detector balance α = 1.25 and
        # a clean single-frequency precession — the count-domain unit-test recipe
        # (tests/core/test_count_domain_fits.py) promoted into a screenshot.
        template = build_builtin_template("ideal_pulsed_fb")
        run = simulate_run(
            template,
            _tf_asymmetry,
            {"A": 20.0, "f": 1.5, "phi": 0.3},
            total_events=40e6,
            alpha=1.25,
            seed=1,
        )
        run.grouping["good_frames"] = 1.0e6
        run.run_number = 3520
        run.metadata["title"] = "TF calibration (α = 1.25)"
        # Reduce the run to its F-B asymmetry so the surface has a non-empty
        # active dataset (the count fit reads the run's raw histograms, but the
        # fit-range preview needs the reduced time axis).
        dataset = reduce_run_to_dataset(run)
        window.set_dataset(dataset)

        # Fit target → F + B (free α). Item data drives the mode key, but the
        # index is stable (All groups / F + B / Single group).
        window._target_combo.setCurrentIndex(1)
        _process_events_for(milliseconds=60)

        # Expand the two collapsed advanced sections so their real controls — the
        # Poisson/Gaussian Cost selector, the skip window, the nuisance toggles,
        # the double-pulse field, and the α/t₀/background/DT₀ promotes — are the
        # subject of the shot. These are the exact surfaces the page documents.
        window._count_options_section.setExpanded(True)
        window._calibration_section.setExpanded(True)
        _process_events_for(milliseconds=60)

        # Match the model to the generator's signal: a single Oscillatory
        # component is a bare A·cos(2πft + φ) (the composite exposes only the
        # amplitude/frequency/phase — no damping or baseline), so seeding the
        # frequency and phase to the run's own 1.5 MHz / 0.3 rad makes the fit
        # model identical to the synthesised precession and the fit lands at
        # χ²ᵣ ≈ 1 rather than a spurious poor minimum.
        tab = window._single_fit_tab
        tab._set_composite_model(CompositeModel(["Oscillatory"]))
        _process_events_for(milliseconds=60)
        tab._group_model_table.reseed({"frequency": Seed(1.5), "phase": Seed(0.3)})
        _process_events_for(milliseconds=40)

        # Run the real forward/backward free-α count fit synchronously (worker
        # thread, blocked on with a live event loop for a deterministic capture).
        tab._run_count_domain_fit()
        tab.wait_for_fit()
        _process_events_for(milliseconds=80)
        return window


register(AlphaCountCalibrationScenario())
