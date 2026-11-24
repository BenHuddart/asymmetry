Data processing
===============

F/B reduction and grouped count/spectral preparation resolve detector membership,
alignment, and required output and
active-background samples before deadtime availability and correction. Unused
File coefficients and saturated observations outside that union do not affect
the result. In scripted reduction, explicit time limits select final bin
centres; only their contributing native bins enter correction, together with
active estimator samples. Ordinary merging performs no spectral analysis.
Grouped fits select merged-bin centres before resolving
the contributing raw bins, including disjoint fit exclusions. Spectral preparation
also retains active baseline, conditioning, phase, and background dependencies;
a final display crop alone does not remove those observations. Shared preparation
is bound to the requested cohort and actual sample/reference sources.
See :ref:`comparison-deadtime-scope` for the workflow convention.

The processing tools in this chapter act on a loaded dataset before it
reaches the fit engine: rebinning to trade time resolution for statistics,
restricting the time range, applying the grouping defined in the Grouping
dialog, estimating the calibration constant :math:`\alpha`, and the
deadtime and background corrections for raw detector histograms. The order is
deadtime, grouping, background subtraction, then asymmetry. The nonlinear
deadtime inverse acts on each raw detector before any count subtraction.

The result's captured effective recipe retains its resolved processing settings
independently of later source edits. Export headers distinguish source settings
captured for inspection from that result history; see :doc:`workflow_conveniences`
and :doc:`/explanation/reduction_pipeline`.

Rebinning trades time resolution for counting precision. For independent,
comparable observations of a slowly varying signal, averaging :math:`N` bins
reduces the error on the mean approximately by :math:`\sqrt{N}`. Raw-count
reduction instead pools counts and retained variance before taking the ratio;
shared background terms do not follow an independent-bin rule. The new bin
width should stay well below the period of any
precession you intend to fit (rule of thumb: at least eight samples per
cycle). ISIS NeXus files carry raw counts and may supply per-detector deadtime
coefficients; enable **File** in the Grouping window to apply them. PSI BIN/MDU
and ROOT can use separately calibrated coefficients in their whole-acquisition
normalisation. See :doc:`detector_grouping` for the correction's domain and
signed coefficients. Compare the raw and processed traces before fitting.

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
*merge. For comparable independent bins, the asymmetry error falls approximately*
*as* 1/√N *at the cost of time resolution — the precession is still clearly resolved*
*at* ×4 *because the new bin width (~67 ns) is well below the precession*
*period (~740 ns).*

The array helper below averages independent processed curve points and adds
their supplied errors in quadrature. It does not replay detector-count
reduction or shared-background covariance; see :doc:`data_reduction/binning`
for ordinary count-level binning.

.. code-block:: python

   from asymmetry.core.transform.rebin import merge_curve_bins
   import numpy as np
   
   # Original data
   time = np.linspace(0, 10, 1000)
   rng = np.random.default_rng(413)
   asymmetry = rng.normal(0.0, 0.01, 1000)
   error = np.ones(1000) * 0.01
   
   # Rebin by factor of 10
   t_rebinned, a_rebinned, e_rebinned = merge_curve_bins(time, asymmetry, error, factor=10)
   
   print(f"Original: {len(time)} points")
   print(f"Rebinned: {len(t_rebinned)} points")
   # Errors are propagated: σ_new = √(Σσ²) / N

Time range selection
--------------------

This example requires your own single-period NeXus file ``data.nxs`` and
the appropriate HDF5 or HDF4 reader extra. Extract a subset of its curve:

.. code-block:: python

   from asymmetry.core.io import load

   dataset = load("data.nxs", period=1)
   
   # Select data between 0.5 and 5.0 μs
   subset = dataset.time_range(t_min=0.5, t_max=5.0)
   
   print(f"Original: {dataset.n_points} points")
   print(f"Subset: {subset.n_points} points")

Grouping histograms
-------------------

Combine detector histograms into logical groups:

.. code-block:: python

   from asymmetry.core.data import Histogram
   from asymmetry.core.transform import apply_grouping_aligned
   import numpy as np

   # Synthetic detectors share t0 and bin width; indices are zero-based.
   rng = np.random.default_rng(707)
   histograms = [Histogram(counts=rng.poisson(1000, 200), bin_width=0.01) for _ in range(8)]
   
   # Define grouping: forward and backward detectors
   grouping = {
       "forward": [0, 1, 2, 3],
       "backward": [4, 5, 6, 7]
   }
   
   forward_counts = apply_grouping_aligned(histograms, grouping["forward"])
   backward_counts = apply_grouping_aligned(histograms, grouping["backward"])

In the GUI, grouping is configured from the Grouping dialog. For supported
instrument layouts, including ISIS HiFi/MuSR/EMU and the PSI FLAME and HAL-9500
spectrometers, the
**Detector Layout...** editor provides an interactive detector schematic,
preset groupings, and named group slots that are saved with project state as
grouping profiles.

For detector-by-detector instrument layouts and grouping workflows, see
:doc:`detector_grouping`.

Alpha estimation and asymmetry
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

Asymmetry definition and error model
------------------------------------

Asymmetry uses the standard pair formula:

.. math::

   A(t) = \frac{F(t) - \alpha B(t)}{F(t) + \alpha B(t)}

where :math:`F` and :math:`B` are grouped forward/backward counts. Asymmetry's
alpha convention applies :math:`\alpha` to the backward group. This convention
is retained consistently in the GUI, loaders, grouping tools, and fitting
inputs, even when comparing against packages that place :math:`\alpha` on
the forward group.

In the asymmetry expression, **Forward** and **Backward** are defined relative
to the initial muon spin direction. PSI detector labels and FLAME detector
layout presets use the PSI instrument convention, where forward/backward are
defined relative to the beam direction. For PSI data, the Grouping dialog
therefore swaps the analysis dropdown defaults: the analysis **Forward Group**
is the detector-layout **Backward** group, and the analysis **Backward Group**
is the detector-layout **Forward** group. ISIS data keep the instrument
forward/backward convention already used by their detector layouts.

Point errors use first-order propagation at fixed applied :math:`\alpha`
and :math:`\beta`; they are not exact finite-count ratio variances or confidence
intervals. Applied calibration uncertainty and full between-bin covariance are
not added to ordinary fit weights. See :ref:`asymmetry-counting-errors` and
:ref:`conditional-calibration-errors` for the assumptions and fit eligibility.
The following :math:`\beta=1` formulas use that convention:

* If :math:`F + \alpha B = 0`, asymmetry is set to 0 for that bin.
* The default uncertainty for that bin is 1.0 (fractional), i.e. 100% in the
  GUI percentage display.
* For independent raw Poisson counts, a non-zero denominator and positive
  propagated variance, the uncertainty at β = 1 is:

.. math::

   \sigma_A = \frac{2|\alpha|\sqrt{FB(F+B)}}{(F+\alpha B)^2}

In low-count late-time tails, this naturally produces large fluctuations and
large uncertainties.

With independent count variances :math:`v_F` and :math:`v_B`, the same
ratio derivatives give

.. math::

   \sigma_A = \frac{2|\alpha|\sqrt{B^2v_F+F^2v_B}}{(F+\alpha B)^2}.

Deadtime correction changes the count values while retaining measured counting
variance, or explicitly supplied independent variance. Coefficients and frame
exposure are fixed; their uncertainty and physical deadtime-induced correlations
are excluded. See :doc:`detector_grouping` for the convention and limitations.

Background is subtracted from the grouped count histograms before the pair
formula is evaluated. The value retains the same alpha convention. Error
propagation uses raw independent count variances, the shared background-level
variance and its covariance with any overlapping signal bins. A merged block
of m bins carries the shared level variance multiplied by m². Fixed levels
are exact by declaration, and independent background-run spectra add their
frame-scaled variances. See :ref:`background-count-uncertainty` for the formula
and the tail-fit approximation. Degenerate first-order asymmetry variance
uses the unit error sentinel.

In vector-polarisation mode, axis-specific alpha values (``alpha_x``,
``alpha_y``, ``alpha_z``) are used for ``P_x``, ``P_y``, and ``P_z``
respectively. See :doc:`vector_polarization` for setup and UI behaviour.

Co-adding datasets
------------------

Co-add original detector observations before forming the asymmetry, rather
than averaging already reduced curves. For independent inputs the count
coefficients act linearly and their squares act on variance. An unweighted
mean of reduced asymmetries generally gives a different result, even when
each input uses the same time grid.

With your own single-period files ``run_1.nxs`` to ``run_3.nxs`` and the
appropriate reader extra installed:

.. code-block:: python

   from asymmetry.core.data.combine import combine_runs, reduce_combined_run
   from asymmetry.core.io import load

   datasets = [load(f"run_{i}.nxs", period=1) for i in range(1, 4)]
   combined_run = combine_runs([dataset.run for dataset in datasets])
   combined = reduce_combined_run(combined_run)

Preparation retains each original acquisition's timing, supported exposure,
file coefficients and counting noise. Enabled nonlinear detector corrections
act on those original observations before pooling; the requested grouped
background operation then acts on the pooled set. This is not a prescription
to correct one total with an averaged deadtime coefficient. Errors remain
conditional on the supplied weights, calibration and correction settings.
See :doc:`run_arithmetic` for compatibility, overlap, signed-input and
project-replay limits.

In the Data Browser select two or more runs and choose **Co-add Selected**.
The inputs must share authored grouping intent; the combined row retains
source-specific file facts. Editing its grouping rebuilds it from the retained
originals. **Separate Combined** restores those source rows with the shared
authored settings. Neither action discards the detector observations.

Windowing for Fourier analysis
------------------------------

This self-contained example applies a symmetric Hann taper to synthetic
values. For a loaded curve, pass ``dataset.asymmetry`` in place of ``signal``.
See :doc:`fourier_analysis` for the preparation choices before an FFT:

.. code-block:: python

   from asymmetry.core.fourier.window import apply_window
   
   import numpy as np

   signal = np.cos(np.linspace(0.0, 8.0 * np.pi, 200))
   windowed_data = apply_window(signal, name="hann")

   # Supported names: gaussian, hann, cosine, lorentzian

Runnable example
----------------

See ``examples/transform_workflow.py`` for a complete executable script.
