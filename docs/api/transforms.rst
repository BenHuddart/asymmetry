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
optional rebinning of the resulting asymmetry trace
(:func:`~asymmetry.core.transform.rebin.rebin`). The order is fixed
and matches musrfit's ``PRunAsymmetry`` pipeline:
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

.. automodule:: asymmetry.core.transform.rebin
   :members:
   :undoc-members:

.. autofunction:: asymmetry.core.transform.rebin.rebin
   :no-index:
