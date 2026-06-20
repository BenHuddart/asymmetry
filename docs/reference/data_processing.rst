Data Processing
===============

The processing tools in this chapter act on a loaded dataset before it
reaches the fit engine: rebinning to trade time resolution for statistics,
restricting the time range, applying the grouping defined in the Grouping
dialog, estimating the asymmetry-balance parameter :math:`\alpha`, and the
specific deadtime / background corrections that PSI BIN-style raw
histograms typically require. The order is always deadtime first, then
background, then grouping, then asymmetry — skipping deadtime when it is
needed inflates the fitted amplitudes, and applying background subtraction
before grouping mis-weights the per-detector contributions.

Rebinning is the most common operation. Combining :math:`N` adjacent bins
reduces the per-bin uncertainty by :math:`\sqrt{N}` at the cost of time
resolution, so the new bin width should stay well below the period of any
precession you intend to fit (rule of thumb: at least eight samples per
cycle). NeXus files from ISIS arrive pre-corrected for deadtime; PSI BIN
and ROOT files generally require deadtime correction to be enabled in the
Grouping dialog before any analysis. Visually compare the raw and
processed traces in the plot panel before committing to a fit — the
preview is faster than diagnosing a misweighted asymmetry after the fact.

.. image:: /_generated/screenshots/bunching_comparison.png
   :alt: ×1, ×4, and ×16 bunching comparison on a 1.4 MHz TF signal
   :width: 100%

For an end-to-end example that exercises rebinning in context, see
:doc:`/workflows/temperature_scan_magnetism`.

Rebinning
---------

.. image:: /_generated/screenshots/data_processing_rebin.png
   :alt: Side-by-side raw vs ×4 rebinned 100 G TF asymmetry signal
   :width: 100%

*Same synthetic 100 G TF dataset before (left) and after (right) a* ×4 *bin*
*merge. Combining* N *adjacent bins reduces the per-bin uncertainty by* √N
*at the cost of time resolution — the precession is still clearly resolved*
*at* ×4 *because the new bin width (~67 ns) is well below the precession*
*period (~740 ns).*

Reduce noise by combining adjacent time bins:

.. code-block:: python

   from asymmetry.core.transform.rebin import rebin
   import numpy as np
   
   # Original data
   time = np.linspace(0, 10, 1000)
   asymmetry = np.random.randn(1000) * 0.01
   error = np.ones(1000) * 0.01
   
   # Rebin by factor of 10
   t_rebinned, a_rebinned, e_rebinned = rebin(time, asymmetry, error, factor=10)
   
   print(f"Original: {len(time)} points")
   print(f"Rebinned: {len(t_rebinned)} points")
   # Errors are propagated: σ_new = √(Σσ²) / N

Time Range Selection
--------------------

Extract a subset of the data:

.. code-block:: python

   dataset = load("data.nxs")
   
   # Select data between 0.5 and 5.0 μs
   subset = dataset.time_range(t_min=0.5, t_max=5.0)
   
   print(f"Original: {dataset.n_points} points")
   print(f"Subset: {subset.n_points} points")

Grouping Histograms
-------------------

Combine detector histograms into logical groups:

.. code-block:: python

   from asymmetry.core.data import Histogram
   from asymmetry.core.transform import apply_grouping
   import numpy as np

   # Synthetic detector histograms for the example
   histograms = [Histogram(counts=np.random.poisson(1000, 200), bin_width=0.01) for _ in range(8)]
   
   # Define grouping: forward and backward detectors
   grouping = {
       "forward": [0, 1, 2, 3],
       "backward": [4, 5, 6, 7]
   }
   
   forward_counts = apply_grouping(histograms, grouping["forward"])
   backward_counts = apply_grouping(histograms, grouping["backward"])

In the GUI, grouping is configured from the Grouping dialog. For supported
instrument layouts, including ISIS HiFi/MuSR/EMU and the PSI FLAME and HAL-9500
spectrometers, the
**Detector Layout...** editor provides an interactive detector schematic,
preset groupings, and named group slots that are saved with project state and
``.grp`` files.

For detector-by-detector instrument layouts and grouping workflows, see
:doc:`detector_grouping`.

Alpha Estimation and Asymmetry
------------------------------

Estimate ``alpha`` from a good-bin range, then compute asymmetry and errors.

.. code-block:: python

   from asymmetry.core.transform import compute_asymmetry, estimate_alpha

   alpha = estimate_alpha(
       forward_counts,
       backward_counts,
       first_good_bin=5,
       last_good_bin=150,
   )
   asymmetry, error = compute_asymmetry(forward_counts, backward_counts, alpha=alpha)

Asymmetry Definition and Error Model
------------------------------------

Asymmetry uses the standard pair formula:

.. math::

   A(t) = \frac{F(t) - \alpha B(t)}{F(t) + \alpha B(t)}

where :math:`F` and :math:`B` are grouped forward/backward counts. Asymmetry's
alpha convention applies :math:`\alpha` to the backward group. This convention
is retained consistently in the GUI, loaders, grouping tools, and fitting
inputs, even when comparing against packages that place the balance factor on
the forward group.

In the asymmetry expression, **Forward** and **Backward** are defined relative
to the initial muon spin direction. PSI detector labels and FLAME detector
layout presets use the PSI instrument convention, where forward/backward are
defined relative to the beam direction. For PSI data, the Grouping dialog
therefore swaps the analysis dropdown defaults: the analysis **Forward Group**
is the detector-layout **Backward** group, and the analysis **Backward Group**
is the detector-layout **Forward** group. ISIS data keep the instrument
forward/backward convention already used by their detector layouts.

Asymmetry and uncertainty handling follows Mantid-style behaviour:

* If :math:`F + \alpha B = 0`, asymmetry is set to 0 for that bin.
* The default uncertainty for that bin is 1.0 (fractional), i.e. 100% in the
  GUI percentage display.
* For non-zero denominator, uncertainty is computed as:

.. math::

   \sigma_A = \frac{\sqrt{(F + \alpha^2 B)\left(1 + \left(\frac{F-\alpha B}{F+\alpha B}\right)^2\right)}}{|F+\alpha B|}

In low-count late-time tails, this naturally produces large fluctuations and
large uncertainties.

For musrfit-style PSI background correction, the background is subtracted from
the grouped count histograms before this pair formula is evaluated. The
asymmetry value still uses the same alpha convention above, but uncertainty is
propagated from the corrected count histograms using musrfit-style count
errors: fixed-background subtraction keeps Poisson errors from the raw counts,
while estimated-background subtraction adds the uncertainty of the estimated
constant background before propagating through the pair formula.

In vector polarization mode, axis-specific alpha values (``alpha_x``,
``alpha_y``, ``alpha_z``) are used for ``P_x``, ``P_y``, and ``P_z``
respectively. See :doc:`vector_polarization` for setup and UI behaviour.

Co-adding Datasets
------------------

Average multiple datasets with proper error propagation:

.. code-block:: python

   import numpy as np
   from asymmetry.core.io import load
   
   # Load multiple datasets
   datasets = [load(f"run_{i}.nxs") for i in range(1, 4)]
   
   # Ensure same time grid (or interpolate if needed)
   time_grid = datasets[0].time
   
   asymmetries = [ds.asymmetry for ds in datasets]
   errors = [ds.error for ds in datasets]
   
   # Average
   avg_asymmetry = np.mean(asymmetries, axis=0)
   
   # Error propagation: σ_avg = √(Σσ²) / N
   avg_error = np.sqrt(np.sum([e**2 for e in errors], axis=0)) / len(datasets)

The GUI provides automatic co-adding via the context menu: select 2+ datasets,
right-click, and choose "Co-add Selected" to combine them with proper error
propagation.

In the GUI, co-add is intentionally strict:

* every selected dataset must already have equivalent grouping settings,
* the resulting combined dataset mirrors that shared grouping state.

When you later edit grouping on a combined dataset, Asymmetry applies the
change to the hidden source datasets and rebuilds the combined row from those
updated sources. Separating the combined dataset restores those source runs
with the same grouping that was active on the combined entry.

Windowing for Fourier Analysis
------------------------------

Apply window functions before FFT:

.. code-block:: python

   from asymmetry.core.fourier.window import apply_window
   
   dataset = load("data.nxs")
   
   # Apply Hann window
   windowed_data = apply_window(dataset.asymmetry, window_type="hann")
   
   # Available windows: 'hann', 'hamming', 'blackman', 'bartlett'

Runnable Example
----------------

See ``examples/transform_workflow.py`` for a complete executable script.
