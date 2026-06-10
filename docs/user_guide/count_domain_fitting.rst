Count-Domain Fitting (α calibration & single histograms)
========================================================

Count-domain fitting works on raw detector counts rather than the reduced
asymmetry. Two modes are available from the **Fit target** selector at the top
of the **Multi-Group Fit** window, alongside the existing *All groups* grouped
fit:

* **Forward + Backward (free α)** — fit a forward and a backward count
  histogram simultaneously with the detector-balance parameter α as a free
  fit parameter. This is the statistically proper way to obtain α from a
  transverse-field calibration run, superseding the grid estimators in the
  Grouping dialog.
* **Single group** — fit one detector histogram to the count model
  N₀·exp(−t/τ_μ)·(1 + A·P(t)) + bg. This is the musrfit single-histogram
  (*fittype 0*) analogue, used for calibration and diagnostic work, and it is
  the natural mode for continuous-source data where a single detector already
  carries the full decay envelope.

When to use this
----------------

Reach for the **free-α** fit whenever you need a defensible α from a
calibration measurement and want its uncertainty and — importantly — its
correlation with the signal amplitude. In a transverse field the count balance
can trade against the asymmetry amplitude, so α and A are strongly correlated;
the joint fit reports that correlation, which a grid estimate cannot. Reach for
the **single-histogram** fit when one detector is the object of study: a
continuous-source run where you want the bare envelope and background, or a
diagnostic check that one detector's N₀ and background behave as expected. For
ordinary physics extraction from balanced data, the regular F–B asymmetry fit
remains the faster path — count-domain fitting is a calibration and diagnostic
tool, not a replacement for asymmetry fitting.

What these modes fit
--------------------

Both modes fit **raw counts**. The count model is

.. math::

   N(t) = N_0\, e^{-t/\tau_\mu}\left[1 + s\,A\,P(t)\right] + \mathrm{bg}

where :math:`P(t)` is the chosen fit function's polarization, :math:`A` its
amplitude, :math:`\tau_\mu` the fixed physical muon lifetime, and :math:`s` a
sign that is ``+1`` for a forward histogram and ``-1`` for a backward one. The
muon lifetime is held fixed at the physical value.

For the **free-α** fit the forward and backward histograms share one
normalization split by the balance,

.. math::

   N_F(t) &= N_0\sqrt{\alpha}\,e^{-t/\tau_\mu}\left[1 + A\,P(t)\right] + \mathrm{bg}_F \\
   N_B(t) &= \frac{N_0}{\sqrt{\alpha}}\,e^{-t/\tau_\mu}\left[1 - A\,P(t)\right] + \mathrm{bg}_B

so that :math:`\alpha = N_{0,F}/N_{0,B}`. The shared amplitude :math:`A`, the
physics parameters and α are fitted together; each side keeps its own
background. Forward and backward here are the two detector groups designated in
the **Grouping** dialog; sum multi-detector banks into those two groups
upstream.

Choosing the cost
-----------------

A **Cost** selector chooses how the counts are weighted:

* **Poisson** (default) — the Cash statistic, the correct treatment for the
  low-count bins at late time and on continuous-source data, where the count
  distribution is visibly skewed and a Gaussian σ underweights the constraint.
* **Gaussian √N** — ordinary least squares with σ = √N. Faster and adequate
  when every fitted bin has high counts; it matches the historical WiMDA
  weighting.

The two agree where counts are high and diverge where they are sparse; when in
doubt, the Poisson cost is the safe default. The reduced statistic reported for
a Poisson fit is the Cash value per degree of freedom, which behaves like a
reduced χ² near the minimum.

Reading the result
-------------------

The result panel reports the fitted parameters with uncertainties — for
example a recovered balance of α = 1.250(1) — together with the per-degree
statistic. For the free-α fit, the forward result carries the full covariance,
so the α–amplitude correlation is available for inspection.

Worked example — α from a TF calibration run
--------------------------------------------

1. Load the calibration run and open its **Grouping** dialog; confirm the
   forward and backward groups.
2. Open the **Multi-Group Fit** window and choose a transverse-field model
   (an oscillation with a free amplitude and frequency) in the model builder.
3. Set **Fit target** to *Forward + Backward (free α)* and leave **Cost** at
   *Poisson*.
4. Fit. The recovered α is the calibration balance; cross-check it against the
   Grouping dialog's diamagnetic / general / ΣF/ΣB estimators — they should
   agree to a few percent, with differences explained by the estimators
   flattening an asymmetry window while the fit weights the whole count trace.

References
----------

* *Muon Spectroscopy: An Introduction* (S. J. Blundell, R. De Renzi,
  T. Lancaster, F. L. Pratt, eds.), Oxford University Press — the count
  equation N(t) = N₀·e^(−t/τ)(1 + A·P(t)) + B, the detector-balance parameter
  α, and the Poisson nature of detector counts.
* A. Suter and B. M. Wojek, *Phys. Procedia* **30**, 69 (2012) — musrfit, the
  single-histogram (*fittype 0*) count fit and its cost options.
