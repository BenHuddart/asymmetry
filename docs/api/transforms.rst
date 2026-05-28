Data Transforms
===============

.. currentmodule:: asymmetry.core.transform

The transform layer is the sequence of operations applied to raw
per-detector histograms before any model fit sees them: deadtime
correction
(:func:`~asymmetry.core.transform.deadtime`), background subtraction
(:func:`~asymmetry.core.transform.background`), grouping into forward
and backward detector sums
(:func:`~asymmetry.core.transform.grouping.apply_grouping`),
asymmetry calculation with the chosen :math:`\alpha` balance factor
(:func:`~asymmetry.core.transform.asymmetry.compute_asymmetry`,
:func:`~asymmetry.core.transform.asymmetry.estimate_alpha`), and
optional rebinning of the resulting asymmetry trace
(:func:`~asymmetry.core.transform.rebin.rebin`). The order is fixed
and matches musrfit's ``PRunAsymmetry`` pipeline:
deadtime → background → grouping → asymmetry. The user-facing
documentation of these steps and of Asymmetry's conventions (alpha
applied to the backward group; Mantid-style uncertainty handling at
zero denominator) is in :doc:`/user_guide/data_processing`.

Asymmetry Calculation
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

.. autofunction:: asymmetry.core.transform.grouping.apply_grouping
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
