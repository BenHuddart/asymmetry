.. _alpha-calibration:

Alpha calibration
=================

The measured forward/backward asymmetry

.. math::

   A(t) = \frac{N_F(t) - \alpha N_B(t)}{N_F(t) + \alpha N_B(t)}

depends on the calibration constant :math:`\alpha = N_F^0/N_B^0`, the ratio of
the two detector groups' efficiency × solid-angle products. With the correct
:math:`\alpha` the asymmetry equals :math:`a_0 P_z(t)`; with the wrong one,
a spurious baseline and a nonlinear distortion of the polarisation enter
every fit. Alpha is a property of the instrument configuration
*and* the sample position, so it must be recalibrated per mounting — and,
because detector solid angles change with sample environment, ideally per
cooldown.

The grouping dialog's **Estimate α** control determines :math:`\alpha` from
the reference run and applies it to all selected datasets (the same
reference-run convention as deadtime estimation). The selector offers
**Diamagnetic (TF)**, **General (LF/ZF)**, and **Count ratio ΣF/ΣB**.
Diamagnetic and General return alpha without an added uncertainty,
following WiMDA's quick estimates. **Count ratio** reports a statistical
uncertainty from first-order propagation of the detector-group sums.

Choosing a method
-----------------

============================  ====================================================
Data in hand                  Method
============================  ====================================================
Weak-TF calibration run       **Diamagnetic** (preferred whenever TF data exist)
Only relaxing LF/ZF runs      **General** — needs *visible* relaxation
Quick TF cross-check          **Count ratio** (Mantid ``AlphaCalc`` convention)
Most demanding work           **Count-fit α** (free count-fit parameter)
============================  ====================================================

The standard workflow is the WiMDA/ISIS one: take a weak transverse-field
calibration run (a "T20", around 20 G) on the *same sample and mounting*,
estimate :math:`\alpha` there with the diamagnetic method, and hold it fixed
for the longitudinal- and zero-field runs that follow. The General method
exists for the case where no TF calibration was recorded.

Corrections and the estimate
----------------------------

An **Estimate** result is a stored, one-shot calibration. Changing groups or
corrections preserves its value, method, source and uncertainty; a staleness
warning invites an explicit new estimate. It does not enable automatic fitting.
The separate profile policy ``per_run_estimate`` recomputes only the integral
ratio on the target run's current corrected counts and good window. It requires
finite positive integrals on both sides and a finite positive ratio. Failure
makes the reduction unavailable instead of substituting a coefficient of one.
Fixed unity remains an explicit valid choice. See :doc:`../detector_grouping`.

:math:`\alpha` is estimated on the *corrected* forward/backward counts — the
deadtime correction and the background subtraction configured in the grouping
dialog are applied to the histograms **before** the estimate reads them, so the
balance is measured on exactly the spectra the reduction forms the asymmetry
from. This matters most for the background: a flat background pedestal does not
track the detectors' efficiency ratio, so estimating :math:`\alpha` from the raw
(unsubtracted) counts balances the *totals* and leaves the reduced weak-TF
asymmetry off-centre once the background is subtracted. Estimating on the
background-subtracted counts avoids this pedestal-induced imbalance; the quick
objective can still retain finite-window and weighting biases. Asymmetry estimates on counts after
deadtime correction, grouping,
and background subtraction. This preparation does not imply that every
reference program uses the same estimator or automatic policy. See
:ref:`comparison-calibration-policies` and :doc:`/explanation/reduction_pipeline`.

The calibration dialog shows which corrections the estimate reflects beneath the
result, and warns when a *requested* correction could not be applied to the
selected calibration run — for example a reference-run background whose reference
could not be resolved — so a flattened "after" curve never misrepresents a
calibration the reduction will not reproduce.

Diamagnetic method
------------------

Minimises the weighted asymmetry power

.. math::

   S(\alpha) = \sum_i \left( \frac{A_i(\alpha)}{\sigma_i(\alpha)} \right)^2,

where :math:`\sigma_i` is a nominal first-order Poisson weight computed from
the prepared counts. For corrected input this objective does not model the
full correction covariance. In
a transverse field the muons in a diamagnetic environment precess, so the
true asymmetry oscillates symmetrically about zero — any imbalance shows up
as a net offset that inflates :math:`S`. The minimum of :math:`S(\alpha)`
is a quick balance estimate; finite windows and the alpha-dependent weights
can displace it from the efficiency ratio.

*When to use this.* The default for any weak-TF run on a sample in a
diamagnetic (non-magnetic, non-relaxing) state — the standard calibration
measurement. A resolved oscillation provides a useful balance reference;
precision depends on its amplitude, window, corrections, and count statistics.
Avoid it on magnetically ordered samples (spontaneous fields shift the
oscillation) and on data where the diamagnetic fraction is small.

This is WiMDA's diamagnetic estimate with the grid search replaced by a
bounded continuous minimisation, run on internally packed equal-statistics
bins so the result does not depend on the display bunching factor.

General method
--------------

The General method uses the lifetime-corrected balanced count

.. math::

   N(\alpha, t) = \left( \frac{N_F(t)}{\sqrt{\alpha}}
                  + \sqrt{\alpha}\, N_B(t) \right) e^{t/\tau_\mu}.

At the correct alpha, the polarisation cancels under a common muon-decay
envelope, equal forward/backward polarisation magnitudes (beta = 1), and
correct background subtraction. Flatness alone does not uniquely identify
alpha: constant polarisation makes every alpha give a flat envelope.
Asymmetry equates weighted lifetime-corrected densities in an early and a late
equal-statistics window. With densities :math:`F_1,F_2,B_1,B_2`, this gives
:math:`\alpha=(F_1-F_2)/(B_2-B_1)`. The sums are linear, but their ratio is
nonlinear and need not be statistically unbiased.

Use **General** on relaxing LF/ZF data when no TF calibration is available,
provided the two windows have different mean polarisation. Time variation can
also come from TF precession. A finite window can give equal window means even
for a varying signal. Weak or noisy contrast can give an unstable estimate;
no uncertainty or confidence test is supplied by this quick method.
Background and deadtime corrections should be applied first. A residual flat
background grows after lifetime correction and can bias the late window.

WiMDA's **General** minimises relative scatter of the lifetime-corrected
balanced counts. Asymmetry uses the two-window condition above. These are
different numerical objectives; neither a universal absence of an interior
minimum nor universal agreement is implied.

Asymmetry also checks for numerical degeneracy before dividing the window
differences. Differences no larger than a floating-point accumulation bound
scaled by the absolute summands of the densities are treated as zero contrast. The bound scales with
the number of terms and double-precision machine epsilon [roundoff]_; it is not a
statistical significance threshold. Passing it does not establish physical
identifiability on noisy or weakly relaxing data.

.. _alpha-general-goodness-gate:

*Goodness gate: the lifetime assumption.* The flatness condition holds only if
the counts decay at exactly :math:`\tau_\mu`. Two windows give one equation for
one unknown, so counts whose *effective* lifetime differs slightly from
:math:`\tau_\mu` — through pile-up rejection, or a background subtracted a
little too generously — are indistinguishable from a detector imbalance, and the
whole mismatch is absorbed into :math:`\alpha`. A 0.5 % lifetime error is enough
to shift :math:`\alpha` by 10 % at a 20 % asymmetry, and by tens of per cent
when the asymmetry is small.

Asymmetry also checks the flatness after solving, across eight
equal-statistics sub-windows and with :math:`\alpha` re-freed, and reads the
surviving time trend back as an effective lifetime. When that lifetime differs
from :math:`\tau_\mu` by more than 0.15 % *and* the deviation is resolved at
2.5 standard errors, the estimate comes back with ``ok`` false and a message
naming the deviation, so a wrong number is not presented as a good one. The
value itself is never altered by the gate, and ``objective_value`` carries the
flatness :math:`\chi^2` of the sub-window levels for inspection either way. If
your data trips the gate, calibrate :math:`\alpha` from a fully depolarised
late-time window with the count ratio instead, if enough counts remain.
Where :math:`P_z(t) \rightarrow 0`, the ratio of expected counts equals alpha
without a lifetime assumption; a ratio of measured sums still has sampling bias.
The lifetime diagnostic uses approximate Poisson weights on the prepared counts;
it does not propagate all correction uncertainty or certify a valid calibration.
If it cannot form enough usable sub-windows or resolve its fitted parameters,
the result message says the check was not applied.

Count ratio
-----------

.. math::

   \alpha = \frac{\sum_i N_F(t_i)}{\sum_i N_B(t_i)}

over the good-bin window — the Mantid ``AlphaCalc`` convention, kept for
cross-checks against Mantid workflows and older analyses.

*When to use this.* Only on transverse-field data spanning many precession
cycles, provided the decay-weighted mean polarisation is close to zero.
The ratio of expected summed counts then approaches the efficiency ratio.
On relaxing LF/ZF data a positive weighted mean polarisation produces an
upward relative bias of approximately :math:`2a_0 \langle P_z \rangle`
at small amplitude, potentially much larger than the statistical uncertainty. The reported uncertainty does not include
this bias.

.. _alpha-ratio-subtracted-background:

*Uncertainty when a background has already been subtracted.* The ratio uses
first-order propagation of the two independent detector-group sums:

.. math::

   \left(\frac{\sigma_\alpha}{\alpha}\right)^2
   = \frac{V_F}{S_F^2}+\frac{V_B}{S_B^2},
   \qquad
   V = \sum_S v_j+n^2\sigma_\theta^2-2n\sum_S\kappa_j.

Here :math:`S_F,S_B` are corrected sums over the :math:`n` selected bins,
and the variance and covariance terms are defined in
:ref:`background-count-uncertainty`. For a range mean the overlap term is
computed from the actual raw-bin coefficients. A disjoint estimation window
has :math:`\kappa_j=0` on the signal window and retains the full
:math:`n^2\sigma_\theta^2` shared contribution.

The typed groups carry their independent variances and shared components::

    corrected = corrected_grouped_counts(..., use_background=True)
    estimate = estimate_alpha_detailed(
        corrected.forward,
        corrected.backward,
        method="ratio",
        first_good_bin=lo,
        last_good_bin=hi,
    )

The grouping dialog supplies these groups automatically. The uncertainty is
propagated from the corrected counts and their recorded noise model; the
background is not inferred from the corrected values alone.

Count-fit α
------------

The three estimators above are *reduction-time* methods on the grouping
dialog. A fourth route lives in the **Multi-Group Fit** window: the
forward/backward count fit with the balance free (*Forward + Backward
(free α)* target) recovers :math:`\alpha` as a parameter of a full count-level
fit. The model fits the raw forward and backward histograms simultaneously
with :math:`\alpha` parameterised as :math:`\sqrt{\alpha}`, under the Poisson
likelihood, and reports the full :math:`(\alpha, \text{amplitude})`
covariance — so the strong α–amplitude correlation of a TF run is estimated
*with* the physics rather than before it. This is the WiMDA manual's "most
accurate way" to determine :math:`\alpha`.

*When to use this.* The most demanding calibrations, and any case where α and
the oscillation amplitude are strongly correlated (precessing TF data): the
joint fit propagates that correlation into the reported uncertainty, which the
reduction-time estimators cannot. It needs a physics model for the run, so it
is heavier than the one-click dialog estimators — reach for it when the extra
rigour matters, and use the diamagnetic estimator for routine T20
calibrations.

Once the fit converges, **Promote α** writes the fitted balance into the
grouping with ``alpha_method="count_fit"`` provenance and a reference-run
record, the same keys the dialog estimators write — so the reduction picks it
up on the next re-reduce. The promote is suggest-only and reports the
before/after value; see :doc:`../count_domain_fitting` for the count-fit
controls and the sibling t₀/background promotes.

Uncertainties
-------------

The diamagnetic and General quick estimates return ``alpha_error=None``.
Their result display shows the alpha value alone. Repeating **Estimate α**
applies a new result explicitly; existing saved and manual values, including
historically recorded errors, remain intact until replaced.

The count ratio reports first-order propagation of the window sums, including
recorded shared background components (see
:ref:`background uncertainty <alpha-ratio-subtracted-background>`).
This statistical uncertainty excludes polarisation bias, a mis-set t0, and
unmodelled correction errors. It is not an exact sampling distribution for a
ratio, particularly when the denominator is poorly determined.

The diamagnetic estimate additionally carries a small **method** systematic that
no amount of statistics removes. Its stationary point is
:math:`\alpha^4 = \sum F^3/(B(F+B)) \big/ \sum B^3/(F(F+B))`; expanded in the
oscillation amplitude :math:`u = a_0 P_z(t)`, the estimating equation is
:math:`\sum s\,u = \frac{\alpha - 1}{\alpha + 1}\sum s\,u^2 + O(u^3)`. The
second-order term survives even for a perfectly zero-mean oscillation, so the
estimate is displaced by roughly :math:`a_0^2 (\alpha-1)/(\alpha+1)` — a few
tenths of a percent at a typical 20 % asymmetry, growing quadratically with the
asymmetry and reversing sign for :math:`\alpha < 1`. In resolved, many-cycle
controls this bias need not track frequency. Finite
windows, packing, and unresolved oscillations can introduce further dependence;
a field-dependent estimate does not by itself locate an instrumental defect.

For the most demanding work, :math:`\alpha` can instead be treated as a free
parameter of a count-level fit (the WiMDA manual's "most accurate way"). That
route is the **Count-fit α** section above; its uncertainty comes from the fit
covariance, including the α–amplitude correlation.

Compared with other programs
----------------------------

WiMDA 1.326 quick **Diamagnetic** and **General** return alpha without an error.
On matched rounded synthetic beta = 1 histograms (16 ns bins, raw bins 4–498,
F/B channels 1/2, background and deadtime off, bunch one), its displayed TF
Diamagnetic alpha is 1.237, compared with 1.236610 in Asymmetry. General gives
1.250 on both TF and relaxing controls; Asymmetry gives 1.249827 and 1.250073.
Agreement is within 0.001 absolute alpha, the native search/display step.
These are point checks, not uncertainty coverage or acquired-data validation.
See :doc:`/explanation/comparison` and the `WiMDA overview/manual
<https://shadow.nd.rl.ac.uk/wimda/>`_. Mantid ``AlphaCalc`` is the integral-ratio
counterpart; musrfit model fits use a different objective from these quick
estimates.

**References**

- S. J. Blundell, R. De Renzi, T. Lancaster, and F. L. Pratt, *Muon
  Spectroscopy: An Introduction* (Oxford University Press, Oxford, 2022).
- F. L. Pratt, Physica B **289–290**, 710 (2000).
- A. Yaouanc and P. Dalmas de Réotier, *Muon Spin Rotation, Relaxation, and
  Resonance: Applications to Condensed Matter* (Oxford University Press,
  Oxford, 2011).

.. [roundoff] N. J. Higham, *Accuracy and Stability of Numerical Algorithms*,
   2nd ed. (SIAM, Philadelphia, 2002), Sec. 3.1.
