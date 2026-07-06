Count-domain fitting (α calibration & single histograms)
========================================================

Count-domain fitting works on raw detector counts rather than the reduced
asymmetry. Two modes are available from the **Fit target** selector at the top
of the **Multi-Group Fit** window, alongside the existing *All groups* grouped
fit:

* **Forward + Backward (free α)** — fit a forward and a backward count
  histogram simultaneously with α as a free
  fit parameter. This is the statistically proper way to obtain α from a
  transverse-field calibration run, superseding the grid estimators in the
  Grouping dialog.
* **Single group** — fit one detector histogram to the count model
  N₀·exp(−t/τ_μ)·(1 + A·P(t)) + bg. This is the musrfit single-histogram
  (*fittype 0*) analogue, used for calibration and diagnostic work, and it is
  the natural mode for continuous-source data where a single detector already
  carries the full decay envelope.

*When to use this.* Reach for the **free-α** fit whenever you need a defensible α from a
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

where :math:`P(t)` is the chosen fit function's polarisation, :math:`A` its
amplitude, :math:`\tau_\mu` the fixed physical muon lifetime, and :math:`s` a
sign that is ``+1`` for a forward histogram and ``-1`` for a backward one. The
muon lifetime is held fixed at the physical value.

For the **free-α** fit the forward and backward histograms share one
normalisation split by the balance,

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

A successful fit also draws its model curve over the data in the
**Individual groups** plot — the single-histogram fit over its one group, the
free-α fit over both the forward and backward banks. The fit minimises a
raw-count model, but the plot shows lifetime-corrected counts, so the overlay
is scaled by exp(t/τ_μ) to land on the displayed data; a good fit traces the
group exactly, and a poor one shows where it departs.

Window and nuisance flexibility
-------------------------------

Three optional controls refine the fit window without splitting the run; each
is off by default and, when off, leaves the fit numerically unchanged.

* **Skip window (μs)** — drop an interior window of bins from the fit. Set the
  two fields so the upper bound exceeds the lower; the bins inside are removed
  (endpoints inclusive). Use it to reject a laser or RF artefact, a spike, or
  any localised corruption — the fitted parameters then match what the clean
  data alone would give, where leaving the artefact in pulls the amplitude. The
  label distinguishes this *drop* from the MaxEnt *de-weight* window of the same
  units (see :doc:`exclusions`).
* **Fit t₀ offset** — add a free time-zero offset that shifts the model time
  axis. Enable it when a small time-zero error is suspected; on clean data it
  recovers an offset consistent with zero and changes nothing else.
* **Fit baseline drift** — add a stretched-exponential damping
  exp(−(λ_b·t)^β_b) on the polarisation (β_b held at 1, simple exponential, by
  default), for a slowly relaxing non-precessing baseline.

  .. note::

     WiMDA applies its baseline drift only to a dedicated constant-offset
     component; Asymmetry has no single privileged offset parameter, so the
     drift multiplies the whole polarisation. The term is off by default.

* **Free muon lifetime** — the core API exposes an optional ``tau`` parameter
  that frees the muon lifetime (musrfit-style), defaulting to the physical value
  τ_μ = 2.197 μs. Free it to fit the bare decay of a single histogram or as a
  detector-time diagnostic; with it fixed the fit is byte-identical to the
  standard fixed-lifetime fit. It is not combined with the double-pulse model.

Count loss and double pulse
---------------------------

* **Fit deadtime DT₀** — add a non-paralysable count-loss term to the fit. The
  observed counts are the true counts damped by 1 − DT₀·r, with r the
  per-frame, per-detector count rate; DT₀ is the detector deadtime in
  microseconds, the same quantity the Grouping dialog's deadtime correction
  applies. A high-rate run is needed for DT₀ to be well determined.

  The core API offers WiMDA's full set of count-loss forms via the
  ``deadtime_model`` argument — *simple* (DT₀ only), *linear*
  ((DT₀ + DT₁·evfr)·r, using the group event fraction evfr), *polynomial*
  (adding C₂·10³·r², C₃·10⁶·r³, C₄·10⁹·r⁴), and *power-law*
  ((evfr·DT₀)^C₂·exp(−(C₄·λ_μ·t)^C₃)). The loaders carry no ISIS event-fraction
  block, so evfr defaults to 1 unless the grouping supplies it. The GUI fits the
  *simple* DT₀ term, which is the dominant and best-determined coefficient.

  Once a deadtime fit converges, **Promote DT₀ → grouping** writes the fitted
  value into the grouping's per-detector deadtime (WiMDA's Send-to-Group),
  reporting the before/after values; tick **accumulate** to add to the existing
  value rather than replace it. Re-reduce the run to apply the promoted
  correction. This closes the calibration loop: fit the deadtime, promote it,
  and the reduced data is corrected. A polynomial or power-law promotion also
  records the model name and higher-order coefficients with the run, while the
  reduction still applies the dominant DT₀.

* **Double pulse (μs)** — set the pulse separation for an ISIS double-pulse
  source; 0 leaves the single-pulse model. The two pulses each carry the
  polarisation, evaluated at t ± dpsep/2 and weighted by exp(∓dpsep/2τ_μ). This
  applies to both the single-histogram and the free-α (F+B) targets — for the
  latter the two pulses ride the same √α-tied forward/backward model, so α and
  the double-pulse structure are recovered together. The separation defaults to
  a fixed instrument value; tick **fit** to refine it.

  .. note::

     The separation enters the model through a non-smooth pulse-onset gate, so
     gradient (migrad) fitting of dpsep is unreliable. With **fit** ticked the
     separation is instead located by a coarse→fine grid scan bracketing the
     instrument value — at each grid point migrad refines the other parameters,
     and the best χ² wins. This recovers dpsep robustly without depending on a
     near-truth start. With the separation at its true value the model fits
     cleanly (χ²ᵣ ≈ 1); a wrong separation visibly degrades the fit.

Promoting α, t₀ and the background
----------------------------------

The deadtime promote above is one of a family: the same suggest-only
Send-to-Group pattern promotes the other fitted calibrations into the grouping,
each reporting before/after and a *Re-reduce the run to apply* message. Like
deadtime, α and t₀ are per-sample, per-setup calibrations — α "needs to be
determined for each sample", and the analysis time-zero "is the beginning of
the spin dynamics" — so a value fitted from the run's own counts is a
legitimate calibration to persist.

* **Promote α** writes the free-α forward/backward balance into the grouping
  with ``alpha_method="count_fit"`` provenance. This is the statistically best
  of the four α routes — see :doc:`data_reduction/alpha_calibration`. Available
  after a *Forward + Backward (free α)* fit.
* **Promote t₀** converts the fitted continuous time-zero offset (μs) to the
  nearest integer ``t0_bin`` via the bin width and **discloses the sub-bin
  residual** the integer index cannot represent. The fitted t₀ is per group but
  ``t0_bin`` is a single run-level index, so the promote applies the fitted
  group's value run-wide and says so. Available after a fit with **Fit t₀
  offset** enabled.
* **Promote background** writes the fitted flat count background into the
  grouping's *fixed* background mode as a ``[forward, backward]`` pair. Because
  the count fit reads **raw counts**, this background term measures the *full*
  flat rate — if the grouping already corrects the background, the fit still
  sees it, so do not fix the fit's background to zero (the panel notes this when
  a grouping background correction is active).

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

* S. J. Blundell, R. De Renzi, T. Lancaster, and F. L. Pratt,
  *Muon Spectroscopy: An Introduction* (Oxford University Press, Oxford,
  2022) —
  the count equation N(t) = N₀·e^(−t/τ)(1 + A·P(t)) + B, the detector-balance
  parameter α, and the Poisson nature of detector counts.
* A. Suter and B. M. Wojek, Phys. Procedia **30**, 69 (2012) — musrfit, the
  single-histogram (*fittype 0*) count fit and its cost options.
