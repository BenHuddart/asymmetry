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

Shared core TF calibration fit
------------------------------

For scripts, :func:`~asymmetry.core.fitting.tf_calibration.fit_tf_calibration`
fits alpha with beta fixed at one, or alpha and beta jointly, through one
count-model solver. Its method identity is ``tf_calibration_fit``, distinct from
quick ``diamagnetic`` and the beta adapter's historical ``count_fit`` identity.
The grouping dialog's quick-method choices and automatic per-run ratio policy
retain their separate meanings.

The supported polarisation is
:math:`q(t)=(A_0/100)\exp(-\Lambda |t|)\cos(2\pi f t+\phi)`.
Both banks share this exponential envelope, frequency, and phase, with opposite
signs and the count equations in :doc:`beta_calibration`. Time is in μs,
frequency in MHz, phase in radians, and :math:`A_0` in percent. The muon lifetime
is fixed at the cited physical constant. A single common phase is a model
assumption; a successful minimum does not establish its adequacy.

For a dataset with an established grouping::

    from asymmetry.core.fitting.tf_calibration import (
        TFCalibrationRequest, fit_tf_calibration,
    )

    request = TFCalibrationRequest(
        forward_group=1, backward_group=2,
        fit_beta=False, t_min=0.1, t_max=8.0,
    )
    result = fit_tf_calibration(dataset, request)
    if result.ok:
        alpha, conditional_error = result.alpha, result.alpha_error
        assert result.beta_fixed and result.beta_error is None

``fit_beta=True`` frees beta in the same model and solver. Parameter overrides
use immutable ``CountFitParameter`` values. Free-parameter scalar overrides
retain the default fit bounds; supplied finite endpoints may change the search interval while
preserving positive alpha/beta, non-negative count scales/backgrounds, frequency,
and damping, and 0–100 percent amplitude. Invalid domain bounds or fixed values
raise an input error. Fixed parameters do not acquire the free-parameter search
box as a physical limit. Backgrounds can be fixed to an independently established
value or fitted separately for the two banks.
The estimator fits frequency by default. An explicitly fixed finite positive
frequency is honoured as an independently supplied model assumption; no fitted
frequency error is attached. Joint beta calibration rejects a nonpositive or
nonfinite fixed frequency.
The default free-background seeds are one count per bin, avoiding an optimiser
start exactly at the non-negative boundary; this is a starting value, not a
fixed background or prior. The :doc:`/api/fitting/calibration` reference describes
selection and result fields.

The operation consumes the supplied acquisition and its existing grouping,
exclusions, native timing, good-bin support, and bunching. An explicit ``period``
uses the existing acquired-period selector. It requires disjoint detector banks
from acquired integer counts. With deadtime correction off or skipped, it uses
the raw-count Poisson objective. An applied File/manual deadtime correction
selects the existing weighted count fit automatically, retaining measured-count
variance or explicitly supplied independent variance. No cost selector is needed.
Corrections act on each required native detector bin before grouping and packing;
variances sum over the same observations. The weighting uses a variance floor of
one count squared per packed bin, matching the existing count-fit preparation.
This does not propagate uncertainty through the deadtime correction formula or
include coefficient/exposure uncertainty.

Asymmetry also checks the selected acquired counts before admitting a corrected
fit, so correction does not admit fractional or negative source observations.
Arbitrary supplied-variance observations without an applied deadtime correction
and explicit pulse-response requests are unsupported; loading and inspection
remain available. Integer values alone do not establish
raw-Poisson provenance. Asymmetry also rejects selected channels explicitly
labelled ``PostPileUp`` (using grouping labels when the metadata copy is absent)
and a selected period marked as lacking acquired histograms; unselected processed
channels remain loadable alongside raw banks.
For example, "TF calibration cannot use declared PostPileUp channels [1]"
identifies an unsupported selected channel independently of the fit objective.
Unknown or untagged processing remains a caller precondition: supply acquired
raw histograms, rather than processed ROOT channels or a mapped difference.
Count-background settings retain the count fitter's availability preflight.
The two constant backgrounds are fitted model terms even when reduction
background subtraction is off. Grouping fixed values, range estimates, and
automatic-tail estimates neither subtract the calibration observations nor fix
or seed those model terms. Independently established background values can be
fixed through explicit count-model parameter overrides. Reference-run
subtraction retains its unsupported-calculation diagnostic.

``result.fit`` owns every fitted and fixed parameter, both backgrounds, the
unique free-parameter covariance order, units, and the effective native domains.
``result.fit.objective`` identifies Poisson deviance or weighted chi-square, with
:math:`\nu=N_F+N_B-k`, counting each unique free parameter once.
Poisson deviance per degree of freedom is a diagnostic, not an exact chi-square
calibration or an instruction to rescale errors.

Errors are conditional symmetric HESSE curvature standard errors, using a unit
change in the fitted objective, without residual scaling. Weighted errors are
conditional on the retained/supplied variances. They include the fitted nuisance correlations, but not
model selection, uncertain timing/corrections held fixed, or propagated
cross-bin reduction covariance. Explicit HESSE follows minimisation; optional
MINOS intervals are retained separately. ``fit.success`` describes the fit;
``fit.uncertainty_status`` separately reports ``available``, ``unavailable``, or
``unreliable`` covariance. Missing or unreliable errors are ``None``. Fixed
beta has no fitted error or alpha–beta correlation.

Asymmetry also requires a declared fixed alpha to be finite and positive, and
checks the existing fitted scale against the observed count
ratio (factor three) and forward count level (factor 100) to reject spurious
minima. Free-beta calibration additionally retains the amplitude/frequency
significance and finite-beta-error safeguards described in
:doc:`beta_calibration`. Those beta-identifiability checks are not imposed on
alpha-only calibration with beta fixed. Neither set certifies model adequacy.
Cancellation is polled during the actual objective, HESSE/MINOS work, and at
the final result boundary; cancellation raises ``FitCancelledError`` and
publishes no result.

Bounded envelope selection in scripts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a single exponential does not describe the calibrant, use
:func:`~asymmetry.core.fitting.tf_calibration.select_tf_calibration` with the same
``TFCalibrationRequest``. It prepares the selected F/B observations once and
compares undamped, exponential, Gaussian, and stretched-exponential envelopes in
that declared order. Each has common F/B frequency, phase, and envelope, opposing
signs, automatic fitted constant backgrounds, and beta fixed at one or free.

The envelopes are :math:`1`, :math:`\exp(-\Lambda |t|)`,
:math:`\exp[-(\sigma t)^2]`, and :math:`\exp[-(\Lambda |t|)^p]`.
The stretching exponent ``stretch`` (:math:`p>0`, dimensionless) is distinct
from backward-amplitude ``beta``. Rates ``Lambda`` and ``sigma`` are in μs⁻¹;
:math:`p=1` is exponential and :math:`p=2` is Gaussian with :math:`\sigma=\Lambda`.
The undamped candidate has no damping parameter. Rate seeds are 0.1 μs⁻¹;
``stretch`` starts at one with a numerical lower search bound of 0.05 and no
upper bound. The floor avoids the singular zero-exponent limit, where damping
loses its time dependence. It is a search default, not a physical lower limit:
explicit positive bounds below it and exponents above two are accepted.

The recommendation is the usable candidate with the lowest **summed fit cost**:
Poisson deviance on raw observations, or weighted chi-square after applied
deadtime correction with the same retained variances for every candidate.
There is no complexity penalty, reduced-statistic ranking, or additional
near-tie rejection threshold. Exact cost ties follow declared candidate order.
The stretched model contains the other shapes as limits and will often win
when its extra freedom is resolved; selection does not identify a physical
relaxation mechanism or certify model adequacy.

For example::

    from asymmetry.core.fitting.tf_calibration import select_tf_calibration

    selection = select_tf_calibration(dataset, request)
    for candidate in selection.candidates:
        print(candidate.model, candidate.ok, candidate.message,
              candidate.fit.statistic, candidate.fit.uncertainty_status)
    if selection.selected is not None:
        alpha = selection.selected.alpha
        conditional_error = selection.selected.alpha_error

``selection.candidates`` retains all four full fits, coefficients, nuisance
covariance, effective requests, and failure messages. A failed candidate does
not discard usable alternatives; no usable fit gives ``selected=None``. Fit
success, calibration verdict, and covariance quality remain separate states.
As for the single-model operation, an alpha-only fit may be usable with
unavailable covariance; its error is then ``None``. Free beta retains its
precession and finite-error safeguards. Errors and MINOS intervals are
conditional on the individual model, without model-selection uncertainty.
Local minima and numerical failures remain possible, including in the flexible
stretched candidate; no extra retry procedure is applied. Cancellation during
fitting or between candidates raises ``FitCancelledError`` without a partial
recommendation.

Common request overrides constrain every candidate. Candidate-specific damping
constraints use ``envelope_parameters`` so settings cannot be silently dropped::

    from asymmetry.core.fitting.count_results import CountFitParameter
    from asymmetry.core.fitting.tf_calibration import TFEnvelope

    selection = select_tf_calibration(
        dataset, request,
        envelope_parameters={
            TFEnvelope.STRETCHED: (
                CountFitParameter("stretch", 1.5, min=0.01, max=4.0),
            ),
        },
    )

Use ``Lambda`` for exponential, ``sigma`` for Gaussian, and ``Lambda``/``stretch``
for stretched; undamped accepts no damping overrides. Asymmetry also checks
that overrides name parameters actually used by their candidate, with positive
stretching exponents, before minimisation. Fixed values must lie within any
explicitly supplied finite bounds; omitted endpoints stay unbounded for fixed
values, independently of the numerical defaults used for free searches. Invalid requests raise ``ValueError``.
The selected result's ``model`` identifies its effective envelope, such as
``gaussian_common_phase`` or ``stretched_common_phase``. Calling
``fit_tf_calibration`` still fits only the exponential model. Selection is a
separate core operation; it does not change grouping-dialog choices, apply
coefficients, or alter saved/manual calibrations.

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

The closest raw-count counterpart is musrfit's global histogram likelihood fit,
not its type-2 asymmetry fit. Matched musrfit 1.12.1/ROOT 6.40.04 fixed/free-beta
controls agree within 0.0001 absolute alpha/beta, after reciprocal alpha and
fraction/percent conversion. The installed likelihood route's differing error
scale is documented in :doc:`/explanation/comparison`; Asymmetry retains the
independently checked conventional standard-error scale.

For applied deadtime correction, matched musrfit 1.12.1 type-0 weighted
histogram controls agree within 0.0001 absolute alpha/beta only after matching
the error convention. Its compared export rebuilds errors from corrected counts;
Asymmetry retains measured/supplied variance, consistent with Mantid's muon
correction and WiMDA's ordinary counting-error basis. These are distinct weights,
not full propagation through deadtime correction. Conditions and limits are in
:ref:`comparison-count-calibration`.

Envelope selection is an Asymmetry convenience, not a shared default of these
programs. The `musrfit THEORY documentation
<https://lmu.pages.psi.ch/musrfit-docu/user-manual.html#the-theory-block>`_
defines ``simpleGss`` as
:math:`\exp[-(\sigma_{\mathrm{native}}t)^2/2]`, requiring
:math:`\sigma_{\mathrm{native}}=\sqrt{2}\sigma`; ``generExpo`` uses the
stretched envelope above. Matched musrfit 1.12.1/ROOT 6.40.04 component fits
on rounded undamped/exponential/Gaussian/stretched controls, with zero fixed
backgrounds, phases 0/180 degrees, 16 ns bins, raw bins 4–498, and fixed/free
beta, agree within 0.0001 absolute alpha/beta after these unit/orientation
conversions. These checks establish component point agreement, not automatic
selection equivalence or uncertainty coverage. No native WiMDA model fit or
Mantid numerical stretched-fit comparison is claimed.

On an owned rounded beta-one TF control, TF alpha is 1.249995 against generating
1.25, while the quick values above are about 1.237. The mean reduced asymmetry
changes by about −0.53 percentage points on those same counts. On the original
Ag calibration example, matched raw support gives quick alpha 0.974 in WiMDA
1.326 and 0.974190 in Asymmetry, versus TF beta-one 0.966447 (zero backgrounds,
no deadtime or pulse response). The mean reduced asymmetry changes by about
+0.39 percentage points; deviance/dof is about 1.30. No independent physical
alpha truth is asserted for that acquisition, and pulse/phase/model suitability
remains a limitation. These are bounded method comparisons, not evidence for a
universal default or uncertainty coverage.

**References**

- `PSI MuSRFit interface <https://musruser.psi.ch/?page_id=148>`_, exponential,
  Gaussian, and Stretch Cosine components.
- `Mantid StretchExpMuon documentation
  <https://docs.mantidproject.org/nightly/fitting/fitfunctions/StretchExpMuon.html>`_,
  stretched relaxation convention.
- `musrfit Single Histograms tutorial
  <https://musruser.psi.ch/Tutorial/Tutorial2.html>`_, global histogram
  calibration and shared geometric parameters.
- S. Baker and R. D. Cousins, Nucl. Instrum. Methods Phys. Res. **221**, 437 (1984),
  `doi:10.1016/0167-5087(84)90016-4 <https://doi.org/10.1016/0167-5087(84)90016-4>`_.
- `CERN Minuit2 manual <https://root.cern.ch/root/htmldoc/guides/minuit2/Minuit2.html>`_,
  function normalisation, HESSE, and MINOS.
- S. J. Blundell, R. De Renzi, T. Lancaster, and F. L. Pratt, *Muon
  Spectroscopy: An Introduction* (Oxford University Press, Oxford, 2022).
- F. L. Pratt, Physica B **289–290**, 710 (2000).
- A. Yaouanc and P. Dalmas de Réotier, *Muon Spin Rotation, Relaxation, and
  Resonance: Applications to Condensed Matter* (Oxford University Press,
  Oxford, 2011).

.. [roundoff] N. J. Higham, *Accuracy and Stability of Numerical Algorithms*,
   2nd ed. (SIAM, Philadelphia, 2002), Sec. 3.1.

Stored calibration meaning
--------------------------

A TF result applied through the core grouping-profile API retains its request,
effective model and fitted constraints, native support, correction assumptions,
and full covariance in the project. Alpha and beta share a canonical fit record.
A usable alpha-only fit with unavailable or unreliable covariance retains alpha
without an error. Loading uses stored coefficients even when the calibrant file
is absent; it does not fit again. Manual policies and recorded historical quick
errors remain authored values, without inferred TF or HESSE provenance. See
:doc:`../project_files` for the record and conditional uncertainty contract.

Explicit workflow and CLI calibration
-------------------------------------

The :doc:`../agent_workflow` ``alpha`` command and ``--alpha-from`` accept
``--calibration-method tf_calibration_fit`` for the shared TF operation.
``--fit-beta`` enables joint alpha/beta; omission fixes beta at one.
``--calibration-envelope`` constrains the requested model, while omission
selects the minimum finite summed cost from all four usable candidates.
Explicit ``diamagnetic``, ``general``, and ``ratio`` requests use the same
core estimators as scripts. No-option CLI and survey calculations retain
the distinct per-run integral-ratio policy.

An applied TF result carries both coefficients and the complete canonical
record into saved workflow settings. Errors are derived from its full nuisance
covariance, conditional on the selected model and supported observations;
no model-selection or reduced-signal covariance is added. Quick methods retain
no added uncertainty; ratio retains first-order propagation. Missing/unreliable
errors and fixed-beta errors are absent. Requested/applied corrections, native
support, timing, and acquired source period remain in the record.
Explicit quick workflow settings also retain the shared typed result and its
method, source period, correction outcomes, and timing/window context.
Asymmetry also checks that calibration has identified source-period histograms;
a cached curve or an unidentified multi-period parent cannot supply another
period's counts. TF sources must name an acquired period rather than a
nontrivial mapped sum, so a saved ordinal does not misidentify its observations.
Routing equivalence establishes the same calculation across entry points;
it does not extend the reference comparisons or physical-accuracy claims above.
