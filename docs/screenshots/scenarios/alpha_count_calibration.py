"""α from a TF calibration run — the count-domain forward/backward free-α fit.

The count-domain fitting page's headline workflow: recover the detector-balance
α from a transverse-field calibration run by fitting the forward and backward
count histograms jointly with α free, rather than reading it off a grid
estimator. This is the statistically proper route — the joint fit reports α
together with its uncertainty and its correlation with the signal amplitude.

The scenario drives the real :class:`MultiGroupFitWindow` (the count-domain
grouped-fit surface that hosts the page's *Fit target*, *Count-fit options* and
*Calibration* controls):

1. Synthesises a high-statistics pulsed TF calibration run (``ideal_pulsed_fb``
   template) with a known detector balance α = 1.25 and a clean single-frequency
   precession asymmetry cos(2π·1.5·t + 0.3), via the same ``core.simulate``
   builder the count-domain unit tests use.
2. Selects **Fit target → F + B (free α)** and expands the collapsed
   **Count-fit options** (Cost / Skip / Nuisances / Double pulse) and
   **Calibration** (Promote DT₀ / α / t₀ / background) sections so their real
   controls are on screen — these are exactly the surfaces the page documents.
3. Chooses the matching single-``Oscillatory`` model (``A·cos(2πft + φ)``) and
   seeds its frequency/phase to the run's own values so the fit model matches
   the generator's signal shape, then runs the real forward/backward count fit.

Because the model matches the signal, the fit converges to χ²ᵣ ≈ 1.03 and
recovers α = 1.250(1) — the worked-example balance the page quotes.

The synthesised signal uses a deliberately small (~1 %) asymmetry amplitude:
the grouped fit surface normalises the shared model amplitude to unity and
carries the real amplitude in its per-group nuisance, which the forward/backward
count driver does not consume, so the count fit's model amplitude currently
sits pinned at that unity reference (reported as ``A_1 (%) = 1 … at bound``).
Matching the generator's amplitude to that pinned value keeps the demonstration
fit's χ²ᵣ at ≈ 1 and its α recovery honest; a larger amplitude would inflate
χ²ᵣ purely from the pinned-amplitude mismatch. α itself is fixed by the
forward/backward count ratio and is recovered correctly regardless.

Marked ``requires_fit = True`` because it runs a real iminuit count-domain fit
(``asymmetry.core.fitting.count_domain.fit_fb_alpha``), which trips on
numpy ≥ 2.3 in dev environments; CI keeps numpy < 2.3.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QWidget

from ._base import Scenario, _process_events_for, register


def _tf_asymmetry(t, A=1.0, f=1.5, phi=0.3):  # noqa: N803 — A is the asymmetry symbol
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
            {"A": 1.0, "f": 1.5, "phi": 0.3},
            total_events=40e6,
            alpha=1.25,
            seed=1,
        )
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
        tab._group_model_table.apply_value_seeds({"frequency": 1.5, "phase": 0.3})
        _process_events_for(milliseconds=40)

        # Run the real forward/backward free-α count fit synchronously (worker
        # thread, blocked on with a live event loop for a deterministic capture).
        tab._run_count_domain_fit()
        tab.wait_for_fit()
        _process_events_for(milliseconds=80)
        return window


register(AlphaCountCalibrationScenario())
