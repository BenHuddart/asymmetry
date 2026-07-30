.. _asymmetry-units:

Asymmetry units across the API
==============================

The μSR asymmetry is a ratio, so it has no physical unit — but it does have a
*scale*, and Asymmetry's public API uses two of them. The low-level primitives
return the dimensionless **fraction** :math:`A \in [-1, 1]`, the textbook
definition; the loaders, the reductions, and every built-in fit model work **in
percent** (:math:`0`–:math:`100`), the WiMDA-style convention in which a 16 %
asymmetry is the number ``16.0``. Both conventions are internally consistent, and
each is the natural one where it is used: a fraction composes cleanly with
polarisation functions, while percent is what a fitted amplitude is quoted as in
a paper.

The cost of having two is that mixing them is a silent factor of 100. It does not
raise, and it rarely looks like an error: a 20 % transverse-field amplitude fitted
against fraction-scale data reads as 0.2 %, which is a perfectly plausible number
for a paramagnet. This page is the map of which scale each part of the API speaks,
so no call site has to guess.

The short version
-----------------

* **Counts in, fraction out.** Anything that forms an asymmetry *from
  histograms* — :func:`~asymmetry.core.transform.compute_asymmetry`,
  :func:`~asymmetry.core.transform.binned_fb_asymmetry`,
  :func:`~asymmetry.core.transform.integrate_asymmetry` — returns a fraction.
* **Anything that hands you a dataset or a reduction is percent.**
  :attr:`~asymmetry.core.data.dataset.MuonDataset.asymmetry`,
  :func:`~asymmetry.core.transform.reduce_grouped_asymmetry`, every loader.
* **Fit amplitudes follow the data.** The engine never rescales, and the
  built-in models seed ``A0`` at ``25`` because they expect percent.
* **The field-scan / ALC surface is the exception**: a
  :class:`~asymmetry.core.transform.FieldScan` is fraction-scale, and so are the
  amplitudes fitted to it.

Naming a scale in code
----------------------

:mod:`asymmetry.core.transform.units` gives the distinction a name, so a call
site can state a scale rather than leave a reader to infer it:

.. code-block:: python

   from asymmetry.core.transform import (
       ASYMMETRY_FRACTION,
       ASYMMETRY_PERCENT,
       AsymmetryUnit,
       convert_asymmetry,
       to_fraction,
       to_percent,
   )

   asym, err = compute_asymmetry(forward, backward, alpha)   # a fraction
   result = FitEngine().fit_arrays(                          # percent expected
       time, to_percent(asym), to_percent(err), model.function, params
   )

   reduction.units is ASYMMETRY_PERCENT   # the reduction states its own scale
   scan.units is ASYMMETRY_FRACTION       # and so does a field scan

:class:`~asymmetry.core.transform.units.AsymmetryUnit` has exactly two members,
``FRACTION`` and ``PERCENT``, with the module-level aliases
:data:`~asymmetry.core.transform.units.ASYMMETRY_FRACTION` and
:data:`~asymmetry.core.transform.units.ASYMMETRY_PERCENT`. There is deliberately
no "automatic" member: a scale is a property of the function that produced the
values, never something to be guessed from their magnitude, since guessing is the
failure this distinction exists to prevent. Use
:func:`~asymmetry.core.transform.units.to_percent` and
:func:`~asymmetry.core.transform.units.to_fraction` for the common direction, or
:func:`~asymmetry.core.transform.units.convert_asymmetry` when both ends are
named explicitly.

None of this changes any existing return value. The two result containers that sit
on opposite sides of the divide now *record* their scale —
``GroupedAsymmetryReduction.units`` is always ``ASYMMETRY_PERCENT`` and
``FieldScan.units`` always ``ASYMMETRY_FRACTION`` for a scan built by
:func:`~asymmetry.core.transform.build_field_scan` — so downstream code can assert
the convention it assumed instead of trusting a comment.

The table
---------

"Preserves" means the function is linear in the asymmetry and returns whatever
scale it was given: it is safe on either, and never converts.

.. list-table:: Asymmetry-valued surfaces in ``core.transform``
   :header-rows: 1
   :widths: 44 14 42

   * - Function or attribute
     - Scale
     - Notes
   * - :func:`~asymmetry.core.transform.compute_asymmetry`
     - fraction
     - The textbook primitive; the error is on the same scale.
   * - :func:`~asymmetry.core.transform.compute_asymmetry_with_count_errors`
     - fraction
     - As above, with supplied count uncertainties.
   * - :func:`~asymmetry.core.transform.binned_fb_asymmetry`
     - fraction
     - Counts are binned first, then the ratio formed.
   * - :func:`~asymmetry.core.transform.reduce_grouped_asymmetry`
     - **percent**
     - The one step on the reduction path that applies the ``×100``;
       ``units`` records it.
   * - :func:`~asymmetry.core.transform.integrate_asymmetry`
     - fraction
     - Integral (ALC) observable, both reduction methods.
   * - :func:`~asymmetry.core.transform.integrate_run`
     - fraction
     - Reduces the run's *counts*, so a ``MuonDataset`` argument's own
       percent-scale curve is never involved.
   * - :func:`~asymmetry.core.transform.integrate_curve`
     - preserves
     - Averages a curve you supply; percent in, percent out.
   * - :func:`~asymmetry.core.transform.build_field_scan`,
       :class:`~asymmetry.core.transform.FieldScan`,
       :class:`~asymmetry.core.transform.FieldScanPoint`
     - fraction
     - ``units`` records it; the ALC parameter models expect this scale.
   * - :func:`~asymmetry.core.transform.differentiate_scan`
     - preserves
     - Dividing by :math:`\Delta x` leaves the asymmetry scale alone.
   * - :func:`~asymmetry.core.transform.rebin`,
       :meth:`~asymmetry.core.data.dataset.MuonDataset.rebin`
     - preserves
     - A mean of values with errors in quadrature.
   * - :func:`~asymmetry.core.transform.rrf_demodulate`,
       :func:`~asymmetry.core.transform.rrf_demodulate_values`,
       :class:`~asymmetry.core.transform.RRFCurve`
     - preserves
     - Demodulation is linear, so the components carry the input's scale.
   * - :func:`~asymmetry.core.transform.estimate_alpha`,
       :func:`~asymmetry.core.transform.estimate_alpha_detailed`,
       :func:`~asymmetry.core.transform.corrected_grouped_counts`, the background
       and deadtime corrections
     - n/a
     - Count domain, or dimensionless ratios — no asymmetry scale.

.. list-table:: Asymmetry-valued surfaces in ``core.data`` and ``core.fitting``
   :header-rows: 1
   :widths: 44 14 42

   * - Function or attribute
     - Scale
     - Notes
   * - :attr:`~asymmetry.core.data.dataset.MuonDataset.asymmetry`,
       :attr:`~asymmetry.core.data.dataset.MuonDataset.error`
     - **percent**
     - The stored convention; the loaders apply the ``×100``.
   * - :attr:`~asymmetry.core.data.dataset.MuonDataset.asymmetry_percent`,
       :attr:`~asymmetry.core.data.dataset.MuonDataset.error_percent`
     - **percent**
     - Explicit views, for when the scale should be stated at the call site.
   * - :attr:`~asymmetry.core.data.dataset.MuonDataset.asymmetry_fraction`,
       :attr:`~asymmetry.core.data.dataset.MuonDataset.error_fraction`
     - fraction
     - As above, on the other scale.
   * - :meth:`~asymmetry.core.fitting.FitEngine.fit`,
       :meth:`~asymmetry.core.fitting.FitEngine.global_fit`
     - **percent**
     - Take a ``MuonDataset``, so percent by its convention. A global fit's
       datasets must all be on one scale.
   * - :meth:`~asymmetry.core.fitting.FitEngine.fit_arrays`
     - **percent** expected
     - Takes bare arrays and rescales nothing, so the scale is the caller's to
       state; percent matches the built-in models' seeds.
   * - :class:`~asymmetry.core.fitting.FitResult` — ``residuals``, and the
       amplitude entries of ``parameters`` / ``uncertainties``
     - follows the data
     - Percent for a normal time-domain fit. Fit fraction-scale arrays and the
       amplitudes come back as fractions.
   * - The built-in models' ``A0`` / ``A`` / ``baseline``
       (:data:`~asymmetry.core.fitting.models.MODELS`, the composite
       ``COMPONENTS``)
     - **percent**
     - Why ``A0`` seeds at ``25``, and why ``PARAM_INFO_REGISTRY`` labels ``A``
       with ``%``.
   * - :func:`~asymmetry.core.fitting.count_domain.fit_single_histogram`,
       :func:`~asymmetry.core.fitting.count_domain.fit_fb_alpha`
     - **percent** parameters
     - The data is raw counts; the physics amplitude is still percent, divided
       down internally.
   * - :func:`~asymmetry.core.fitting.fit_mu_relaxation_series`,
       ``MuRelaxationSeriesResult.shared_amplitude``,
       :func:`~asymmetry.core.fitting.mu_relaxation_from_amplitude`
     - **percent**
     - Including ``reference_amplitude``.
   * - :func:`~asymmetry.core.fitting.field_scan.fit_scan_model`,
       :func:`~asymmetry.core.fitting.field_scan.fit_scan_baseline`,
       :func:`~asymmetry.core.fitting.field_scan.rf_resonance_seeds`
     - fraction
     - They fit a ``FieldScan``, so amplitudes come back fraction-scale. The one
       fraction-scale corner of the fitting layer.
   * - The polarisation kernels ``P(t)`` — ``risch_kehr``,
       ``bessel_oscillation``, and the muonium, muon-fluorine and
       nuclear-dipole modules
     - scale-free
     - Unit-normalised: ``P(0) = 1``, with the amplitude supplied by whichever
       component scales them.

When it bites, and how it is caught
-----------------------------------

The two scales meet most often at the boundary between a custom reduction and a
fit. Forming an asymmetry from a non-default detector pairing gives a fraction;
handing it to :meth:`~asymmetry.core.fitting.FitEngine.fit_arrays` with the
models' default seeds fits percent-scale amplitudes against it.

Asymmetry guards that specific case at run time. When the data and the seeded
model curve straddle the fraction/percent boundary — one peak at or below 1.5,
where a true :math:`|A|` must lie, and the other necessarily percent — the engine
emits an :class:`~asymmetry.core.fitting.AsymmetryScaleWarning` before minimising.
It is advisory only: it never raises and never changes the fit, and the message is
carried on ``FitResult.warnings`` and shown in the GUI fit panel's result box (see
:ref:`fit-advisory-warnings`). A merely badly-guessed amplitude on the right scale
is deliberately not flagged — the guard is about scale confusion, not seed quality.

The guard cannot catch everything, since a genuinely small amplitude on the
percent scale is indistinguishable from a large one on the fraction scale. The
reliable habit is the explicit one: use
:attr:`~asymmetry.core.data.dataset.MuonDataset.asymmetry_percent` /
:attr:`~asymmetry.core.data.dataset.MuonDataset.asymmetry_fraction` when reading
from a dataset, and :func:`~asymmetry.core.transform.units.to_percent` /
:func:`~asymmetry.core.transform.units.to_fraction` when moving a primitive's
output across the boundary.

See also :ref:`cookbook-asymmetry-scale` for the worked seeding recipe, and
:doc:`fitting` for the fit engine's own entry points.
