.. _alpha-calibration:

Alpha Calibration
=================

The measured forward/backward asymmetry

.. math::

   A(t) = \frac{N_F(t) - \alpha N_B(t)}{N_F(t) + \alpha N_B(t)}

depends on the balance parameter :math:`\alpha = N_F^0/N_B^0`, the ratio of
the two detector groups' efficiency × solid-angle products. With the correct
:math:`\alpha` the asymmetry equals :math:`a_0 P_z(t)`; with the wrong one,
a spurious baseline and a distortion proportional to the muon decay envelope
leak into every fit. Alpha is a property of the instrument configuration
*and* the sample position, so it must be recalibrated per mounting — and,
because detector solid angles change with sample environment, ideally per
cooldown.

The grouping dialog's **Estimate α** control determines :math:`\alpha` from
the reference run and applies it to all selected datasets (the same
reference-run convention as deadtime estimation). Three methods are offered;
all report the estimate with a statistical uncertainty, e.g. 1.3702(45),
obtained from a seeded Poisson bootstrap of the histogram counts. WiMDA
reports a bare number for the same operations.

Choosing a method
-----------------

============================  ====================================================
Data in hand                  Method
============================  ====================================================
Weak-TF calibration run       **Diamagnetic** (preferred whenever TF data exist)
Only relaxing LF/ZF runs      **General** — needs *visible* relaxation
Quick TF cross-check          **Count ratio** (Mantid ``AlphaCalc`` convention)
============================  ====================================================

The standard workflow is the WiMDA/ISIS one: take a weak transverse-field
calibration run (a "T20", around 20 G) on the *same sample and mounting*,
estimate :math:`\alpha` there with the diamagnetic method, and hold it fixed
for the longitudinal- and zero-field runs that follow. The General method
exists for the case where no TF calibration was recorded.

Diamagnetic method
------------------

Minimises the weighted asymmetry power

.. math::

   S(\alpha) = \sum_i \left( \frac{A_i(\alpha)}{\sigma_i(\alpha)} \right)^2,

where :math:`\sigma_i` is the Poisson error propagated to the asymmetry. In
a transverse field the muons in a diamagnetic environment precess, so the
true asymmetry oscillates symmetrically about zero — any imbalance shows up
as a net offset that inflates :math:`S`. The minimum of :math:`S(\alpha)`
is the balanced point.

*When to use this.* The default for any weak-TF run on a sample in a
diamagnetic (non-magnetic, non-relaxing) state — the standard calibration
measurement. It is the most precise of the three methods: the oscillation
provides a zero-mean reference that makes the imbalance directly visible.
Avoid it on magnetically ordered samples (spontaneous fields shift the
oscillation) and on data where the diamagnetic fraction is small.

This is WiMDA's diamagnetic estimate with the grid search replaced by a
bounded continuous minimisation, run on internally packed equal-statistics
bins so the result does not depend on the display bunching factor.

General method
--------------

Uses the lifetime-corrected balanced count

.. math::

   N(\alpha, t) = \left( \frac{N_F(t)}{\sqrt{\alpha}}
                  + \sqrt{\alpha}\, N_B(t) \right) e^{t/\tau_\mu},

which is constant in time *exactly* when :math:`\alpha` is correct: the
polarisation term cancels between the two groups regardless of the shape of
:math:`P_z(t)`. Asymmetry solves this flatness condition in closed form by
equating the mean corrected count density between an early and a late
equal-statistics window — everything is linear in the counts, so the
estimate is unbiased and fails loudly (rather than returning a number) when
the two windows show no polarisation contrast.

*When to use this.* Relaxing LF or ZF data when no TF calibration run
exists — the situation the plain count ratio cannot handle. The polarisation
must visibly relax within the time window: on non-relaxing (or fully
decoupled) data, *no* method can extract :math:`\alpha` from flatness,
because every :math:`\alpha` gives a flat corrected count. Expect a larger
uncertainty than the diamagnetic method at the same statistics; if the
reported uncertainty is comparable to the value itself, treat the result as
a consistency check rather than a calibration. Background and deadtime
corrections should be applied first — a residual flat background grows as
:math:`e^{t/\tau_\mu}` after lifetime correction and biases the late window.

WiMDA's General estimate minimises a weighted relative-scatter functional of
the same :math:`N(\alpha, t)`; that functional loses its interior minimum at
realistic counting statistics, so Asymmetry replaces it with the closed-form
flatness condition above (study record: divergence D14).

Count ratio
-----------

.. math::

   \alpha = \frac{\sum_i N_F(t_i)}{\sum_i N_B(t_i)}

over the good-bin window — the Mantid ``AlphaCalc`` convention, kept for
cross-checks against Mantid workflows and older analyses.

*When to use this.* Only on transverse-field data spanning many precession
cycles, where the oscillating polarisation integrates to zero and the ratio
of summed counts is the efficiency ratio. On relaxing LF/ZF data the
positive polarisation does **not** integrate to zero and the ratio is biased
upward by approximately :math:`a_0 \langle P_z \rangle` — typically several
percent, which is an order of magnitude larger than the statistical
uncertainty of a good calibration. The reported bootstrap uncertainty does
not include this bias.

Uncertainties
-------------

All three methods report a statistical uncertainty from a Poisson bootstrap:
the histogram counts are resampled 200 times from their observed values, the
estimator is re-run on each replica, and the spread of the replicas (a
percentile-based standard error, robust against the heavy tails the General
method develops near its identifiability limit) is quoted. The bootstrap is
seeded, so repeating an estimate gives identical digits. The uncertainty is
statistical only — it does not include the count-ratio method's polarisation
bias, nor systematic effects such as a mis-set t0 or an uncorrected
background.

For the most demanding work, :math:`\alpha` can instead be treated as a free
parameter of a count-level fit (the WiMDA manual's "most accurate way");
that fitting mode is planned as part of the count-domain fit modes and is
not part of the reduction-time estimators described here.

**References**

- S. J. Blundell, R. De Renzi, T. Lancaster, and F. L. Pratt, *Muon
  Spectroscopy: An Introduction* (Oxford University Press, Oxford, 2022).
- F. L. Pratt, Physica B **289–290**, 710 (2000).
- A. Yaouanc and P. Dalmas de Réotier, *Muon Spin Rotation, Relaxation, and
  Resonance: Applications to Condensed Matter* (Oxford University Press,
  Oxford, 2011).
