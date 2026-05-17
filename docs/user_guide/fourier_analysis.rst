Fourier Analysis
================

Transform time-domain data to frequency domain to identify oscillation frequencies.

Implementation Summary
----------------------

Asymmetry currently exposes two Fourier workflows:

* the core Python API can transform a supplied time-domain dataset directly
* the desktop GUI uses a WiMDA-style grouped-detector workflow that rebuilds a
  grouped count signal for each included detector group before taking the FFT

In the GUI, the frequency-domain plot always shows the average of the
currently included groups rather than separate per-group spectra. Each loaded
dataset keeps its own Fourier phase-table state, so changing runs restores the
phase values, included groups, and auto-estimated markers associated with that
run.

Notation
--------

In this guide, frequency-domain quantities use standard symbols:

* :math:`A(t)` for asymmetry in the time domain
* :math:`\nu` for frequency (MHz)
* :math:`\lvert\mathcal{F}\rvert` for FFT magnitude

Fast Fourier Transform (FFT)
-----------------------------

The simplest method for frequency analysis:

.. code-block:: python

   from asymmetry.core.fourier.fft import fft_asymmetry
   from asymmetry.core.io import load
   
   dataset = load("data.nxs")
   
   # Compute FFT — returns (frequencies, real_part, magnitude)
   frequencies, real_part, magnitude = fft_asymmetry(dataset)
   
   # Plot
   import matplotlib.pyplot as plt
   plt.plot(frequencies, magnitude)
   plt.xlabel(r"$\\nu$ (MHz)")
   plt.ylabel(r"$|\\mathcal{F}|$")
   plt.show()

Filter / Apodisation
~~~~~~~~~~~~~~~~~~~~

The main Fourier workflow now follows WiMDA's filter model rather than the
older whole-trace window presets. The FFT panel exposes three apodisation
modes:

* ``None``
* ``Lorentzian``
* ``Gaussian``

along with:

* ``Filter start`` in microseconds
* ``Tau`` (the filter time constant) in microseconds

For a Lorentzian filter, Asymmetry follows WiMDA's two cases:

.. math::

   A_f(t) = e^{-t/\tau} A(t) \qquad \text{when } t_\mathrm{start} = 0

.. math::

   A_f(t) = \frac{1 + e^{-t_\mathrm{start}/\tau}}{1 + e^{(t - t_\mathrm{start})/\tau}} A(t)
   \qquad \text{when } t_\mathrm{start} > 0

The Gaussian mode uses the same WiMDA-style softened start with squared
arguments. ``Subtract average signal`` is applied before the filter, matching
WiMDA's default preprocessing path.

How The GUI FFT Is Built
~~~~~~~~~~~~~~~~~~~~~~~~

The docked Fourier workflow in the desktop application is intentionally more
specific than the low-level API:

* the FFT input is the grouped detector-count signal, not the forward/backward
  asymmetry trace
* the grouped signal is rebuilt from the current grouping and current bunching
  factor every time the transform is computed
* grouped-count inputs are corrected for the muon lifetime envelope before the
  later mean-subtraction and apodisation stages
* the active time-domain fit range is used as the FFT time window when a fit
  range is present
* zero padding is applied after the selected time window and preprocessing are
  fixed

This means the Fourier spectrum shown in the GUI is tied directly to the
current grouped analysis state, rather than being a generic transform of
whatever line is currently plotted in the time-domain view.

FFT Phase Modes
~~~~~~~~~~~~~~~

The GUI computes one complex grouped FFT and then derives the displayed curve
from that spectrum:

* ``(Power)^1/2``: magnitude :math:`|F|`
* ``Phase Spectrum``: raw spectral angle
* ``Cos``: real / cosine component of the uncorrected FFT
* ``Sin``: imaginary / sine component of the uncorrected FFT
* ``Phase``: WiMDA-style phase-corrected real projection
* ``phaseOptReal``: entropy-optimized real projection using musrfit's
  two-parameter optimizer backend

Only ``Phase`` uses the manual phase value, per-group phase table, and
``t0 Offset (μs)`` entry. ``phaseOptReal`` ignores those manual controls and
computes its own optimizer-driven correction instead.

Maximum Entropy Method
-----------------------

.. note::

   The Maximum Entropy (MaxEnt) spectral reconstruction method is planned for a
   future release. The ``maxent()`` function exists in the codebase as a stub and
   currently raises ``NotImplementedError``.

When implemented, MaxEnt will offer advantages over FFT including:

* Better frequency resolution with noisy data
* Handling of unevenly sampled data
* Proper incorporation of error bars for weighting

Choosing FFT vs MaxEnt
-----------------------

Use **FFT** when:

* You have evenly sampled data
* Fast computation is needed
* Data has good signal-to-noise ratio

Use **MaxEnt** (when available) when:

* Data is noisy
* You need better frequency resolution
* Data sampling is irregular
* You have reliable error estimates

Practical Example
-----------------

Complete workflow for frequency analysis:

.. code-block:: python

   from asymmetry.core.io import load
   from asymmetry.core.fourier.fft import fft_asymmetry
   import matplotlib.pyplot as plt
   
   # Load data
   dataset = load("data.nxs")
   
      # Compute FFT with WiMDA-style Gaussian apodisation and 4× zero-padding
   freq, real_part, magnitude = fft_asymmetry(
       dataset,
         window="gaussian",
         filter_start_us=0.2,
         filter_time_constant_us=1.5,
       padding_factor=4,
       t_min=0.1,
       t_max=10.0,
   )
   
   # Plot
   fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
   
   # Time domain
   ax1.errorbar(dataset.time, dataset.asymmetry, yerr=dataset.error, fmt='.')
   ax1.set_xlabel("Time (μs)")
   ax1.set_ylabel("Asymmetry")
   ax1.set_title("Time Domain")
   
   # Frequency domain
   ax2.plot(freq, magnitude)
   ax2.set_xlabel(r"$\\nu$ (MHz)")
   ax2.set_ylabel(r"$|\\mathcal{F}|$")
   ax2.set_title("Frequency Domain (FFT with Gaussian apodisation)")
   ax2.set_xlim(0, 50)  # Focus on 0–50 MHz
   
   plt.tight_layout()
   plt.show()

GUI Fourier Workflow
--------------------

The docked Fourier panel in the desktop application now supports the current
WiMDA-first FFT workflow directly:

* choose an ``FFT Phase Mode`` first. The ``Info`` button beside it opens an
  in-app explanation of the formulas and interpretation of each mode
* keep ``Subtract average signal`` enabled to remove the WiMDA-style pre-FFT
  baseline term before filtering and transformation; this is on by default
* grouped Fourier inputs are also corrected for the muon decay envelope before
  the later mean-subtraction and apodisation steps, matching WiMDA's grouped
  ``Freq`` preprocessing more closely
* choose ``None``, ``Lorentzian``, or ``Gaussian`` apodisation, then tune the
  filter start and tau values to match WiMDA-style FFT shaping
* include or exclude detector groups in the ``Groups`` table before computing
  the transform
* in ``Phase`` mode, either type a single manual phase or enable
  ``Use per-group phase table`` and edit the per-group values
* choose ``Auto method`` (``Peak`` or ``Average``), then press
  ``Fill Phase Estimates`` to estimate one phase per detector group for the
  current dataset
* for averaged grouped spectra, optionally estimate WiMDA-style error bars from
  the spread across the selected group spectra

Phase Estimation Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~

Phase estimation in the GUI is explicit rather than always-on:

* estimates are only generated when you press ``Fill Phase Estimates``
* the selected ``Auto method`` determines how the single-group FFT phase is
  measured within the current phase-estimation window
* when a field value is available, that window is narrowed around the expected
  precession frequency; otherwise the positive-frequency FFT range is used
* the estimated values are written into the per-group phase table for the
  current dataset only

The phase color cues in the table are also meaningful:

* blue: the phase value is currently active and will be used
* grey: the phase value is present but not currently active
* green: the phase value was auto-estimated for the current dataset via
  ``Fill Phase Estimates``

Asymmetry now makes two explicit design decisions for this workflow:

* Fourier transforms always use grouped detector-count signals rather than the
  FB asymmetry trace as the input source.
* The frequency-domain plot always shows the average of the currently included
  groups rather than separate per-group spectra.

The FFT therefore always uses the current grouped/bunched data definition. The
grouped detector-count signal is rebuilt with the current bunching factor
before the transform, and that grouped count signal is lifetime-corrected
before the later subtract-average and filter stages. The FFT bandwidth and
frequency spacing therefore track the effective binned timestep in the same way
they do in WiMDA.

The Fourier panel is scrollable so these controls remain usable when the dock
is narrow.

Frequency-Domain Plot Behavior
------------------------------

FFT output is shown on the dedicated ``Frequency Domain`` tab in the central
plot workspace. Each loaded run keeps its own frequency-domain view state in the
session, and runs with no computed FFT remain empty on that tab. The current
x-range is preserved when switching onto a run whose Fourier spectrum has not
yet been computed.

The frequency viewer keeps canonical FFT x-data in absolute MHz or Gauss on the
axis itself. The main toolbar also provides an ``FFT X relative to field``
toggle, which recenters the x-limit controls around the applied
field/frequency of the selected run while leaving the plotted tick labels in
absolute units.
