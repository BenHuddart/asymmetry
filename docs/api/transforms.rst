Data transforms
===============

.. currentmodule:: asymmetry.core.transform

The transform layer is the sequence of operations applied to raw
per-detector histograms before any model fit sees them: deadtime
correction
(:func:`~asymmetry.core.transform.deadtime`), background subtraction
(:func:`~asymmetry.core.transform.background`), grouping into forward
and backward detector sums
(:func:`~asymmetry.core.transform.grouping.align_forward_backward`),
asymmetry calculation with the chosen :math:`\alpha` calibration constant
(:func:`~asymmetry.core.transform.asymmetry.compute_asymmetry`,
:func:`~asymmetry.core.transform.asymmetry.estimate_alpha`), and
count binning before the asymmetry ratio is formed
(:class:`~asymmetry.core.transform.rebin.BinningSpec`). The order is fixed:
deadtime → grouping → background → count binning → asymmetry. The user-facing
documentation of these steps and of Asymmetry's conventions (alpha
applied to the backward group; Mantid-style uncertainty handling at
zero denominator) is in :doc:`/reference/data_processing`.

Validated grouped counts
------------------------

:func:`~asymmetry.core.transform.grouping.align_forward_backward` aligns the
selected detectors and records their shifts and terminal crop in an
:class:`~asymmetry.core.transform.fb_counts.AlignmentCrop`.
:func:`~asymmetry.core.transform.reduce.corrected_grouped_counts` returns a
:class:`~asymmetry.core.transform.fb_counts.ForwardBackwardCounts` containing
both groups, their independent variances and shared components, an inclusive
:class:`~asymmetry.core.transform.fb_counts.BinWindow`, and the time grid.
Count arrays are owned, read-only snapshots. Background provenance records
constant levels; exported standard errors derive from the shared components.

.. automodule:: asymmetry.core.transform.fb_counts
   :members: BinWindow, GroupCounts, SharedComponent, AlignmentCrop, ForwardBackwardCounts, good_window, linear_combination_variance

.. code-block:: python

   counts = corrected_grouped_counts(
       histograms=run.histograms, grouping=run.grouping,
       forward_idx=[0], backward_idx=[1],
       use_deadtime=False, deadtime_mode="off", use_background=False,
   )
   time, asymmetry, error = binned_fb_asymmetry(
       counts, grouping=run.grouping, alpha=1.0,
   )

``time`` is in μs; ``asymmetry`` and ``error`` are fractions. The good-bin
window is selected before binning, and counts are summed before forming the
ratio. See :doc:`/reference/data_reduction/binning` for rejection messages.

Asymmetry calculation
----------------------

.. automodule:: asymmetry.core.transform.asymmetry
   :members:
   :undoc-members:

.. autofunction:: asymmetry.core.transform.asymmetry.compute_asymmetry
   :no-index:

.. autofunction:: asymmetry.core.transform.asymmetry.estimate_alpha
   :no-index:

Grouping
--------

.. automodule:: asymmetry.core.transform.grouping
   :members:
   :undoc-members:

.. autofunction:: asymmetry.core.transform.grouping.align_forward_backward
   :no-index:

Background
----------

.. automodule:: asymmetry.core.transform.background
   :members:
   :undoc-members:

Deadtime
--------

.. automodule:: asymmetry.core.transform.deadtime
   :members:
   :undoc-members:

Rebinning
---------

Replay a selected histogram observation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The toolbar and scripts share
:func:`~asymmetry.core.transform.preparation.prepare_histogram_asymmetry`.
The caller supplies the selected run and resolves any live reference acquisition
before calling. The operation reads the run's grouping, preserves its base
binning, and merges complete output bins before forming the percent asymmetry.
It returns correction outcomes for each requested observation as well as the
arrays. G+R is one pooled prepared-count observation with group background once;
G−R differences separately reduced sets on common native support. Both use
one explicitly selected α/β. Final ratio usability follows the requested statistic.

``integrate_run`` honours Red, Green, pooled G+R, and independent G−R for both
existing native-bin estimators. Extracted sets and mapped originals use the same
count preparation; held inspection clocks do not override acquired timing.

This synchronous operation consumes complete selected good-bin support. It does
not recover the provenance or selected membership of an existing processed curve.
Interactive callers retain worker, cancellation, and cache ownership; process
workers that only fit the resulting arrays can use
:meth:`~asymmetry.core.data.dataset.MuonDataset.fit_record` afterwards.

Known reduction producers attach a transient ``HistogramObservation`` to a
dataset. ``prepare_wizard_dataset`` composes complete-group merges with its
recorded selections; ``time_range`` retains crop membership. Capture owns read-only
copies of sample, held-period, and reference counts and supplied variances,
deduplicating repeated uses of the same array within that capture. Settings and
timing are copied too. Later writes through the caller's arrays or their aliases
do not alter the captured values or identity; the caller's arrays remain usable.
A fresh explicit capture uses the edited values and identifies that observation.
Unknown custom or historical curves have no such context and use independent means. Explicit
``merge_curve_bins`` drops histogram replay context. ``fit_record`` strips the
context and period arrays before candidate process fan-out. No recipe is inferred
from an attached Run or a median time step. Simulation reductions record the actual uncorrected settings and selected
period policies, including an implicit acquired-window policy when used.

.. automodule:: asymmetry.core.transform.preparation
   :members: prepare_histogram_asymmetry, histogram_observation, prepare_wizard_dataset, HistogramObservation

.. automodule:: asymmetry.core.transform.rebin
   :members:
   :undoc-members:

.. autofunction:: asymmetry.core.transform.rebin.merge_curve_bins
   :no-index:
