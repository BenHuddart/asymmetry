Fourier Analysis
================

Transform time-domain data to frequency domain to identify oscillation frequencies.

Fast Fourier Transform (FFT)
-----------------------------

The simplest method for frequency analysis:

.. code-block:: python

   from asymmetry.core.fourier.fft import fft_asymmetry
   from asymmetry.core.io import load
   
   dataset = load("data.wim")
   
   # Compute FFT — returns (frequencies, real_part, magnitude)
   frequencies, real_part, magnitude = fft_asymmetry(dataset)
   
   # Plot
   import matplotlib.pyplot as plt
   plt.plot(frequencies, magnitude)
   plt.xlabel("Frequency (MHz)")
   plt.ylabel("|FFT|")
   plt.show()

Windowing
~~~~~~~~~

Apply a window function to reduce spectral leakage:

.. code-block:: python

   from asymmetry.core.fourier.window import apply_window
   
   # Apply window before FFT
   windowed = apply_window(dataset.asymmetry, name="hann")
   frequencies, power = compute_fft(dataset.time, windowed)

Available window types:

* ``'gaussian'`` — Gaussian apodization
* ``'hann'`` — Hann (Hanning) window
* ``'cosine'`` — Raised-cosine window
* ``'lorentzian'`` — Lorentzian apodization

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
   dataset = load("data.wim")
   
   # Compute FFT with Hann window and 4× zero-padding
   freq, real_part, magnitude = fft_asymmetry(
       dataset,
       window="hann",
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
   ax2.set_xlabel("Frequency (MHz)")
   ax2.set_ylabel("|FFT|")
   ax2.set_title("Frequency Domain (FFT with Hann window)")
   ax2.set_xlim(0, 50)  # Focus on 0–50 MHz
   
   plt.tight_layout()
   plt.show()
