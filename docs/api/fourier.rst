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

Automatic phase estimation
--------------------------

The ``GroupSpectrumConfig`` recipe identifier ``phaseOptReal`` selects the
GUI's **Auto Phase** display. Direct estimators use the typed acquisition
and result objects below.

.. automodule:: asymmetry.core.fourier.phase_fit
   :members: PhaseAcquisition, PhaseFitConfig, PhaseEstimate, PhaseRejection, PhaseFitCancelled, PhaseFitRejectedError, fit_fft_phase

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

Pulse request contract
----------------------

``GroupSpectrumConfig(pulse_compensation=False)`` leaves pulse settings unused.
Active compensation requires an explicit finite positive ``pulse_half_width_us``
(parabolic support half-width). ``pulse_n_pulses=0`` is the explicit identity;
``1`` consumes width only, and ``2`` also consumes a finite nonnegative
``pulse_separation_us``. No width is inferred from arbitrary run metadata.
Invalid consumed inputs raise ``PulseRequestError``, a ``ValueError`` subclass.
MaxEnt uses ``pulse_mode="ignore"``, ``"single"``, or ``"double"`` with the same
response. Applied result metadata retain the consumed settings and convention.

.. automodule:: asymmetry.core.maxent.pulse
   :members: PulseRequestError

Quantity and sampled support
----------------------------

``compute_average_group_spectrum`` resolves canonical display semantics before
conditioning: amplitudes and existing standard errors receive ``g=1/R``;
Power and its standard errors receive ``g**2``. Phase angles receive no gain.
The amplitude limit ``pulse_max_gain`` defaults to 25, so Power can receive up
to 625. This preserves mean-of-displayed-quantity group averaging.

The first positive frequency with ``R < 1/pulse_max_gain`` and all following
bins are omitted from the returned dataset, including its companion
``fourier_imag`` channel. Baseline estimation and unit-area normalisation use
retained support. The cutoff is a grid-dependent numerical safeguard.
``fourier_pulse_response`` metadata record quantity, exponent, amplitude limit,
model, consumed parameters, units, source, origin, sign, and cutoff policy;
``fourier_compensation_cutoff_mhz`` records the sampled cutoff when reached.

Direct ``apply_spectrum_conditioning`` calls accept ``quantity="amplitude"``
(default), ``"power"``, or ``"angle"``; unknown quantity tokens raise
``ValueError`` at this public input boundary. Limits below unity or NaN retain
the unity amplitude limit, which applied metadata report. Ignore leaves that
limit unused (recorded as ``None``). This lower-level result preserves array
shape, marking pulse-removed values/errors as NaN and exposing the amplitude
``gain`` (zero for removed support). The public spectrum operation selects the
retained bins before returning a dataset. Errors remain conditional on the
supplied response and existing estimator; no response uncertainty is added.
