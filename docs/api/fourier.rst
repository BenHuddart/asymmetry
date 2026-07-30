Fourier analysis
================

.. currentmodule:: asymmetry.core.fourier

The Fourier subsystem exposes :mod:`asymmetry.core.fourier.fft` for
direct FFT-based spectra of any time-domain dataset *or* bare
``(t, A, σ)`` triple, with the WiMDA-compatible filter / apodisation
model documented in :doc:`/reference/fourier_analysis`, and
:mod:`asymmetry.core.fourier.window` for the windowing primitives the
FFT path uses. :mod:`asymmetry.core.fourier.apodisation` holds the
matched-filter suggester and the early-signal guard that warns when an
apodisation has deleted the beginning of the record.
:mod:`asymmetry.core.fourier.maxent` re-exports the grouped-count
maximum-entropy engine that lives in :mod:`asymmetry.core.maxent`. The
GUI Fourier panel does not call this API directly — it computes
grouped-count FFTs from the current grouping payload — but the two
share the same windowing primitives.

FFT
---

.. automodule:: asymmetry.core.fourier.fft
   :members:
   :undoc-members:

Apodisation
-----------

.. automodule:: asymmetry.core.fourier.apodisation
   :members:
   :undoc-members:

Maximum entropy
---------------

.. automodule:: asymmetry.core.fourier.maxent
   :members:
   :undoc-members:

Windowing
---------

.. automodule:: asymmetry.core.fourier.window
   :members:
   :undoc-members:
