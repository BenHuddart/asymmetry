Fourier analysis
================

.. currentmodule:: asymmetry.core.fourier

The Fourier subsystem exposes :mod:`asymmetry.core.fourier.fft` for
direct FFT-based spectra of any time-domain dataset, with the
WiMDA-compatible filter / apodisation model documented in
:doc:`/reference/fourier_analysis`, and
:mod:`asymmetry.core.fourier.window` for the windowing primitives the
FFT path uses. :mod:`asymmetry.core.fourier.maxent` is a placeholder
for the maximum-entropy reconstruction currently on the roadmap;
calling it raises ``NotImplementedError`` in the current release. The
GUI Fourier panel does not call this API directly — it computes
grouped-count FFTs from the current grouping payload — but the two
share the same windowing primitives.

FFT
---

.. automodule:: asymmetry.core.fourier.fft
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
