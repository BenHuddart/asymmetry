Negative-Muon Analysis API (Experimental)
==========================================

.. warning::

   **Experimental — work in progress.** The :mod:`asymmetry.core.negmu`
   package is unvalidated against real μ⁻ data; see
   :doc:`/user_guide/negative_muon_analysis` for the full disclaimer,
   physics background, and worked example.

.. currentmodule:: asymmetry.core.negmu

The :mod:`asymmetry.core.negmu` package provides a scriptable API for
μ⁻ capture-lifetime elemental analysis on raw single-histogram counts.
It is **not registered in the GUI fit builders** and has no plot-mode or
panel exposure.

Element lifetimes
-----------------

.. automodule:: asymmetry.core.negmu.lifetimes
   :members:
   :undoc-members:

Multi-exponential count model
-----------------------------

.. automodule:: asymmetry.core.negmu.model
   :members:
   :undoc-members:

Fitting
-------

.. automodule:: asymmetry.core.negmu.fit
   :members:
   :undoc-members:

Capture-ratio report
--------------------

.. automodule:: asymmetry.core.negmu.ratio
   :members:
   :undoc-members:

Background subtraction
----------------------

.. automodule:: asymmetry.core.negmu.background
   :members:
   :undoc-members:

Polarisation multipliers
------------------------

.. automodule:: asymmetry.core.negmu.polarisation
   :members:
   :undoc-members:
