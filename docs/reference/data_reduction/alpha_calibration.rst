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

The grouping dialog's **Estimate calibration** control uses **TF calibration fit**
for a fresh explicit calibration. It compares undamped, exponential, Gaussian,
and stretched common-phase F/B envelopes and shows the selected model. The
minimum finite summed cost among usable candidates wins; model selection does
not establish physical truth. **Calibrate β** fits alpha and beta jointly in a
single-pair layout; unchecked fixes beta at one. Vector layouts calibrate alpha
for each projection with beta fixed at one. **Estimate All α** publishes the
requested projections together after every fit succeeds. Failure or **Stop**
discards the incomplete batch and leaves the draft coefficients unchanged.

**Diamagnetic**, **Decay balance** (stored method ``general``), and
**Count ratio ΣF/ΣB** remain explicit quick choices. Quick methods disable and
uncheck **Calibrate β**. Diamagnetic and Decay balance return no added uncertainty;
ratio reports first-order errors. TF errors are conditional on the selected
count model, observations, and corrections. An unavailable error is not zero.

The compact result summary identifies the period or mapped set and F/B pair.
TF fits show ``Poisson deviance/ν`` or ``χ²/ν`` and the number of degrees of
freedom, with ``conditional errors`` marked when unavailable or unreliable.
These are diagnostics of the selected fit, without a model-truth verdict.
Quick methods show the bins used and ``first-order errors`` or
``no error estimate``. The calibration-run selector identifies the source;
the complete source context remains in the captured record. Quick records retain
the resolved forward/backward channel members, and quick and TF records retain
authored channel exclusions. These are frozen at execution, independently of
later regrouping or target settings. Older records without these captured facts
leave them unknown rather than infer them from the current groups.

A completed calibration updates the draft alpha and beta spinboxes and the
preview immediately. Fixed-beta operations show beta one. The coefficient's
method replaces ``manual`` until its value is changed by hand. **Apply** commits
both coefficients and their captured record to the run/project; estimating does
not write saved values. **Stop** and input changes cancel pending work, and an
obsolete worker cannot overwrite the displayed values. Completed measurements
retain their captured source and assumptions when editing the draft. Loading a
project never starts a calibration. A zero or nonfinite coefficient
cannot be applied as a detector calibration; the source and diagnostic remain
available for inspection.

While running, the status reads **Computing calibration…**. **Stop** clears the
pending request and restores **No calibration result**; an error reads
**Calibration failed:** followed by its diagnostic. An input rejected before work
starts reads **Calibration unavailable:** followed by the input diagnostic. A
source edit observed during preview refresh also cancels the captured request.
Source and reference identity are checked again before queued results update the
draft. An obsolete success, error, or cancellation cannot overwrite a newer request.

Cancellation is cooperative: quick Diamagnetic work polls before each objective
evaluation, Decay balance before each diagnostic sub-window, and count preparation
between detector and bank stages. TF minimisation, HESSE, and optional MINOS retain
the solver's eight-cost-evaluation polling interval; envelope transitions and the
final result boundary also check cancellation. A single array operation or native
call finishes before its next check, so there is no universal time deadline.
Cancellation returns no partial calibration or vector batch. Committed manual
coefficients and completed calibration history remain available.

Choosing a method
-----------------

Use a weak-TF calibration on the same sample and mounting for TF calibration fit
or the quick Diamagnetic cross-check. Use Decay balance for visibly relaxing
LF/ZF data when no TF calibrant is available. The ratio provides a separate
integral-balance check (Mantid ``AlphaCalc`` convention). Different objectives
and finite windows can produce different alpha values; their uncertainties do
not describe the same estimator.

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

For acquired-run arithmetic, an explicit ``per_run_estimate`` policy uses one
ratio of the pooled corrected native-good sums. Output packing and time crops
retain the complete calibration dependency. A G−R asymmetry uses the Red
estimate on both conditions; explicit projection coefficients take precedence.

Quick estimates use the corrected forward/backward counts: configured deadtime
correction, grouping, and background subtraction precede the estimate. Their
native good-bin window is independent of science output packing and time crops.
A flat background pedestal need not track the detector efficiency ratio, so
background subtraction can change the quick estimate; finite-window and
weighting biases can remain. TF calibration instead fits its own count model
with separate constant backgrounds, using retained counting variances when
supported deadtime correction is applied. The grouping background subtraction
does not fix or subtract those fitted model backgrounds. See
:ref:`comparison-calibration-policies` and :doc:`/explanation/reduction_pipeline`.

The calibration dialog shows which corrections the estimate reflects beneath the
result, and warns when a *requested* correction could not be applied to the
selected calibration run — for example a reference-run background whose reference
could not be resolved — so a flattened "after" curve never misrepresents a
calibration the reduction will not reproduce.

A fresh count request prepares the retained acquired histograms with its current
grouping and corrections. A failed cached asymmetry curve does not block that
request after its grouping is repaired; the cached diagnosis remains history.
Current unavailable corrections, missing acquired observations, and unsupported
count likelihoods still receive their own diagnostics.

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

*When to use this.* A quick cross-check for a weak-TF run on a sample in a
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

The grouping dialog's default **TF calibration fit** already fits both count
histograms with the full fitted-nuisance covariance. It measures alpha together
with amplitude, backgrounds, and the other free model parameters, and can fit
beta in the same operation. Its uncertainty remains conditional on the model
and fixed preparation assumptions.

The **Multi-Group Fit** window also offers *Forward + Backward (free α)* for a
specified count model. This is a separate modelling workflow; its result is not
the quick Diamagnetic or Decay balance estimate. Agreement and precision depend
on the observations, model, and fitted constraints, rather than on a universal
ranking of these methods.

Once the fit converges, **Promote α** writes the fitted balance into the
grouping with ``alpha_method="count_fit"`` provenance and a reference-run
record, the same keys the dialog estimators write — so the reduction picks it
up on the next re-reduce. The promote is suggest-only and reports the
before/after value; see :doc:`../count_domain_fitting` for the count-fit
controls and the sibling t₀/background promotes.

Uncertainties
-------------

The diamagnetic and General quick estimates return ``alpha_error=None``.
Their result display shows the alpha value alone. Repeating **Estimate calibration**
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

A standard error describes a local uncertainty scale. Interpreting estimate
plus or minus one standard error as an interval uses an approximately normal
sampling distribution, with an approximately 68.27 % reference probability,
not an exact confidence guarantee [gum]_. Boundary, weak-contrast, and skewed
cases can violate that approximation. The standard deviation of repeated
estimates and half the 15.865–84.135 percentile range are different summaries
for a skewed distribution; neither automatically defines an interval for a
particular observed calibration.

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

Fixed supplied pulse response
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a pulsed calibration, choose **Time response** → **Fixed double pulse**
and supply **Pulse separation** in µs. **Off** is the default. These TF-only
controls are inactive for Diamagnetic, Decay balance, and Count ratio ΣF/ΣB.
The shared setting applies to every TF candidate and vector alpha projection;
it adds no fitted parameter or model-selection candidate. **Calibrate β** keeps
its single-pair scope. Changing the response invalidates pending calibration.

Scripts use ``TFCalibrationRequest.pulse_separation_us`` or
``CalibrationOptions.pulse_separation_us``; ``None`` means off. A supplied finite
nonnegative separation :math:`d` selects two instantaneous births at
:math:`b_j=\pm d/2`, with equal incident strengths :math:`q_j=1/2`.
Age zero is live, and each complete fractional polarisation :math:`P(u)` is
evaluated only at its own live age :math:`u=t-b_j\geq0`. The count means are

.. math::

   F(t)&=N_0\sqrt{\alpha}\sum_{j:u_j\geq0}q_j e^{-u_j/\tau_\mu}
       [1+P(u_j)]+C_F,\\
   B(t)&=\frac{N_0}{\sqrt{\alpha}}\sum_{j:u_j\geq0}q_j e^{-u_j/\tau_\mu}
       [1-\beta P(u_j)]+C_B.

The backgrounds occur once, outside the birth sum; a birthless observation has
background-only predicted counts. The polarisation baseline is fixed at zero.
Input amplitude is percent and is converted once to fractional polarisation;
frequency is MHz, phase radians, and relaxation rates inverse µs. Detector beta
scales only backward polarisation, independently of the stretched exponent.
The absolute surviving population fixes count normalisation; the normalised
ordinary-asymmetry response alone cannot predict these counts.

The midpoint is on the resolved acquisition clock. No acquired t0, good bins,
arrays, facility shift, pion delay, or finite source width is inferred or changed.
Active zero separation is the causal single-birth limit, including age zero;
Off preserves the ordinary no-pulse domain. Asymmetry also checks that supplied
separation is finite and nonnegative and cannot become a free count nuisance.

The count fitter evaluates packed-bin means at retained centres, with
:math:`N_0,C_F,C_B` in counts per packed bin. It does not integrate rates across
bin edges. A bin straddling a birth can therefore differ substantially from an
integrated count expectation; finer post-arrival bins converge to the centre
approximation. This is an observation-model limitation, without an automatic
onset veto, cropping rule, or facility restriction. Choose an observation model
and explicitly selected support suitable for the intended accuracy.
HESSE/MINOS and alpha–beta correlation include all fitted nuisances but are
conditional on the fixed response, selection, source, and envelope; they do not
include source-setting or model-selection uncertainty.

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
are unsupported; loading and inspection
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

MINOS stores signed lower and upper offsets from the fitted value at the
unit-objective profile criterion, separately from HESSE standard errors. Its
usual approximately 68.27 % interpretation requires regular likelihood
conditions [minuit-intervals]_; it is not a physical detector-coverage certificate. Missing valid
MINOS endpoints remain absent even if HESSE is available. Fit success,
calibration usability, covariance quality, a parameter at a boundary, and
interval availability answer different questions.

For applied deadtime correction, the retained-variance convention supplies
conditional weighted curvature. It omits nonlinear variance propagation,
coefficient/exposure uncertainty, and physical event-loss dependence. A Poisson
experiment with a rational-loss *mean* is a sensitivity approximation, not a
physical detector-process coverage study. Fixed-response and packed-bin-centre
assumptions likewise remain outside the reported fitted errors.

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
``fit_tf_calibration`` fits one explicit envelope, exponential by default. The explicit workflow and grouping dialog use selection when no envelope is
supplied. Selection itself does not apply coefficients or alter saved/manual
calibrations.

Reduced-curve errors remain conditional on the applied alpha/beta, even when
the calibration record carries errors and covariance. Shared correction terms
in marginal point errors do not make an ordinary fit a correlated likelihood;
see :ref:`conditional-calibration-errors` and :ref:`asymmetry-counting-errors`.

Validation scope
~~~~~~~~~~~~~~~~

Independent acquired-Poisson controls exercise the public preparation and
calibration routes with 16 ns bins, raw bins 4–498, generating alpha 1.25,
forward normalisation 20000 counts/bin, amplitude 20 %, frequency 1.3 MHz,
phase 0.17 rad, and separate backgrounds 70/110 counts/bin. Free-beta
controls have generating beta 0.8; the selected-envelope control has
exponential truth. Exponential rate
0.15 μs⁻¹ and Gaussian rate 0.25 μs⁻¹ controls fit all default nuisances.
The fixed-envelope controls and actual default selector use separate seed
streams. Each row has 2500 attempts, usable fits, and available HESSE intervals;
the covering yield over all attempts therefore equals coverage conditional on
availability in these rows.

.. list-table:: HESSE ±SE hit counts and 95 % Wilson intervals
   :header-rows: 1
   :widths: 40 30 30

   * - Fitted operation
     - Alpha hits
     - Beta hits
   * - Exponential, fixed beta
     - | 1684/2500
       | (65.50–69.17 %)
     - Fixed; no fitted interval
   * - Exponential, free beta
     - | 1695/2500
       | (65.94–69.60 %)
     - | 1691/2500
       | (65.78–69.45 %)
   * - Gaussian, free beta
     - | 1706/2500
       | (66.39–70.04 %)
     - | 1678/2500
       | (65.25–68.93 %)
   * - Selected, free beta
     - | 1711/2500
       | (66.59–70.23 %)
     - | 1709/2500
       | (66.51–70.15 %)

Monte Carlo standard errors are about 0.93–0.94 percentage points, distinct
from fitted calibration errors [simulation]_. In the exponential free-beta
cell, optional MINOS has 2500 valid intervals for each coefficient and gives
1695 alpha hits and 1694 beta hits at its unit-deviance criterion. The selector
chooses stretched in 2486 attempts and exponential in 14; its reported errors
remain conditional on that selected envelope. These controls do not establish
selection-adjusted confidence intervals, equivalence between methods, or
coverage under other signals, count levels, constraints, or fixed responses.

Separate 10000-attempt ratio controls use 60 signal bins and either an
independent 20-bin measured pedestal, a 30-bin estimation window overlapping
ten signal bins, or fixed exact subtraction. Their first-order ±SE hit counts
are 6825, 6857, and 6805, respectively, with all intervals available (Monte Carlo
SE about 0.47 percentage points). The target is the ratio of expected prepared
sums; proportional bank signals make it 1.25 in these controls. Shared and
overlap terms are retained through actual preparation. This does not establish
inference with unsupported cross-bank dependence or a poorly determined denominator.

Smaller 200-attempt stress controls expose the limits. Weak and absent TF
contrast give only 13 and five usable free-beta calibrations, respectively;
96 attempts in each return a covariance-based interval. Those populations
are different, and the generating beta label is unidentifiable at zero
amplitude. A short window gives 160 alpha intervals with 110 hits, a covering
yield of 110/200 rather than conditional coverage 110/160. Deliberately fitting
an undamped envelope to Gaussian-damped counts gives only one alpha hit in
200 intervals. Biased background subtraction and nonzero window-average
polarisation also displace the ratio from efficiency alpha. Such cases do
not satisfy the correct-model normal reference.

The applied-deadtime rational-loss-mean Poisson sensitivity control gives
120 alpha and 114 beta hits in 200 available intervals (Monte Carlo SE about
3.5 percentage points). Its retained measured-count weighting is verified
independently by closed-form conditional curvature, but the generating noise
law is an approximation: these hit fractions do not certify physical detector
coverage or justify altering the accepted variance convention. Quick-method
point estimates can be biased or unstable under these stress conditions;
they return no interval and have no interval coverage to report. In particular,
Decay balance can return a finite positive, diagnostically usable estimate on
noisy weak/absent contrast; its validity diagnostics do not certify
identifiability or accuracy.

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

WiMDA 1.326 **Double Pulse** F/B/alpha fitting is the beta-one precedent,
with supplied separation and equal incident populations. Beta calibration is
not verified WiMDA functionality. Mantid's `DoublePulseFit v1
<https://docs.mantidproject.org/nightly/algorithms/DoublePulseFit-v1.html>`_
uses supplied shifted functions and weights, a forward-model analogue rather
than the same TF count estimator or covariance calculation. musrfit's authored
count models require reciprocal count-balance and amplitude/phase conventions
to be matched. A post-arrival shifted-TF polarisation check does not establish
onset integration or calibration uncertainty agreement.

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

.. [gum] Joint Committee for Guides in Metrology, *Evaluation of measurement
   data — Guide to the expression of uncertainty in measurement*, JCGM 100:2008,
   Secs. 5.1–5.2 and Annex G,
   `doi:10.59161/JCGM100-2008E <https://doi.org/10.59161/JCGM100-2008E>`_.

.. [minuit-intervals] iminuit documentation, `Hesse and Minos
   <https://scikit-hep.org/iminuit/notebooks/hesse_and_minos.html>`_,
   curvature, profile intervals, and asymptotic coverage.

.. [simulation] T. P. Morris, I. R. White, and M. J. Crowther,
   Stat. Med. **38**, 2074 (2019),
   `doi:10.1002/sim.8086 <https://doi.org/10.1002/sim.8086>`_.

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

.. _calibration-count-packing:

Explicit workflow and CLI calibration
-------------------------------------

Scripts can choose calibration count packing independently of science reduction
through ``CalibrationOptions.count_packing``:

.. code-block:: python

   from asymmetry import load
   from asymmetry.core.workflow import (
       CalibrationOptions, ReductionSettings, calibrate_run, reduce_run,
   )

   source = load("calibration.nxs")
   science_settings = ReductionSettings(rebin=2, t_min=0.5, t_max=5.0)
   calibration = calibrate_run(
       source, science_settings,
       CalibrationOptions(
           method="tf_calibration_fit", fit_beta=True,
           count_packing=4, period=1,
       ),
   )
   # Apply the captured joint coefficients to a separate science run.
   from dataclasses import replace
   science = load("science.nxs")
   reduced = reduce_run(science.run, replace(science_settings, alpha=calibration.record))

The factor must be a positive integer; booleans and non-integral values are
rejected. ``None`` (the default) preserves the source profile's fixed bunching
factor. An explicit factor replaces that factor, with one requesting native
bins; it does not multiply source binning. TF count fitting uses fixed packing,
including when the source has a nonuniform display-binning mode.
``ReductionSettings.rebin`` controls only science reduction and does not compose
with calibration packing. The same option is supported by ``calibrate_source``.
Quick methods, including automatic ratio, reject explicit ``count_packing`` and
retain their native-window estimators.

Counts are summed from the selected acquired period's inclusive good-bin window
before fitting. Incomplete trailing groups are dropped. Times are the mean of
the contributing native centres; ``t_min``, ``t_max``, and ``exclude`` select
these packed centres. Correction preflight and fitted-background treatment are
unchanged. The captured domains record the actual native windows and times,
and alpha and beta retain their joint covariance. Different packing can change
a nonlinear fit; compare fitted values and covariance only on identical
observations. Saved results load as captured, without refitting.

The :doc:`../agent_workflow` ``alpha`` command and ``--alpha-from`` accept
``--calibration-method tf_calibration_fit`` for the shared TF operation.
Omitting options in ``calibrate_run`` selects this fresh explicit default;
passing ``CalibrationOptions()`` deliberately retains automatic ratio.
``--fit-beta`` enables joint alpha/beta; omission fixes beta at one.
``--calibration-pulse-separation US`` supplies the fixed TF separation;
omission means Off.
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
method, source period, correction outcomes, and timing/window context. A mapped
quick source retains its contributor recipe and selected mapped-set ordinal
separately from an acquired-period identity.
Asymmetry also checks that calibration has identified source-period histograms;
a cached curve or an unidentified multi-period parent cannot supply another
period's counts. TF sources must name an acquired period rather than a
nontrivial mapped sum, so a saved ordinal does not misidentify its observations.
Routing equivalence establishes the same calculation across entry points;
it does not extend the reference comparisons or physical-accuracy claims above.

Run arithmetic and TF calibration
---------------------------------

TF calibration requires an original acquisition with its own count model.
A co-added or signed run cannot supply one surrogate deadtime/exposure model
for its retained acquisitions. Calibrate a source run and apply the resulting
shared alpha/beta to the combination. Quick corrected-count preparation and
ordinary asymmetry reduction retain each source's own correction facts.

Matched musrfit 1.12.1/ROOT 6.40.04 global-histogram likelihood controls on
rounded double-pulse counts agree within 0.0001 absolute alpha/beta for beta
fixed at one and free beta. The comparison uses 16 ns bins, raw bins 13–498,
zero fixed backgrounds, fixed lifetime, a shared free phase (native degrees
with a 180-degree backward offset), and reciprocal native count balance.
The native post-arrival damped-cosine representation absorbs the supplied
source into its amplitude, phase, and normalisation. This checks point estimates
after both births; it does not test native onset, bin integration, or error
equivalence. Independent count and full-nuisance information controls provide
the separate scientific targets.
