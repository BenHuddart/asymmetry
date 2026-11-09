.. _beta-calibration:

Beta calibration
================

The corrected asymmetry

.. math::

   A(t) = \frac{F(t) - \alpha B(t)}{\beta F(t) + \alpha B(t)}

carries a second detector-balance constant, :math:`\beta = A_{0,b}/A_{0,f}`,
alongside :math:`\alpha` — musrfit's asymmetry-fit (fit type 2) correction pair.
Where :math:`\alpha` balances the two groups' *count rates* (efficiency and
solid angle), :math:`\beta` balances their *asymmetry amplitudes*: a detector
pair can see the same number of counts (:math:`\alpha = 1`) yet report
different oscillation amplitudes because of geometric or absorption effects
that scale the observable polarisation without touching the count total. See
:doc:`/explanation/conventions` for the full convention and its :math:`\alpha`
companion, :doc:`alpha_calibration`.

:math:`\beta = 1` is the standard formula and the default; a project that
never measures :math:`\beta` behaves exactly as before.

Why beta needs a fit, not a count ratio
----------------------------------------

Beta describes the relative polarisation magnitudes in the two detector
groups. Integrated counts alone cannot generally separate it from alpha and
the physical polarisation. The implemented calibration fits a resolved weak-TF
signal on the same sample and mounting, recovering the two amplitudes.
It requires precession and reports failure when that signal cannot be resolved.
This is a limitation of this fit, not a statement that beta cannot affect LF/ZF
observations: for fixed F, B, and alpha, the corrected asymmetry above generally
changes with beta even without precession. Other models require their own
identifiability assessment.

In the grouping dialog select **TF calibration fit** and check **Calibrate β**
in the shared calibration section. **Estimate calibration** updates both draft
spinboxes and the preview, showing the selected envelope and conditional errors.
The method beside beta stays until its value is edited, then reads ``manual``.
**Apply** commits the pair through one captured calibration record. Leave the
checkbox unchecked for beta fixed at one. Quick methods disable and uncheck it;
vector layouts retain one alpha per projection and beta fixed at one. A manual
beta can still be entered in the beta card. Loading saved values never fits them.

For TF calibration with beta fixed at one, the spinbox label reads ``fixed``,
including after reopening it. A fitted beta of one retains its calibration
method; changing the coefficient by hand changes the label to ``manual``.

The API also retains the explicit single-histogram ratio cross-check described
below; the GUI uses the shared TF operation with automatic envelope selection.

Count fit
---------

*Count fit (recommended)* (method id ``count_fit``) fits the forward and
backward count histograms **simultaneously**, with a single :math:`\beta`
scaling only the backward
polarisation term:

.. math::

   N_f(t) &= N_0\sqrt{\alpha}\, e^{-t/\tau_\mu}\left[1 + A P(t)\right] + b_f \\
   N_b(t) &= \frac{N_0}{\sqrt{\alpha}}\, e^{-t/\tau_\mu}\left[1 - \beta A P(t)\right] + b_b

with :math:`P(t)` a damped-cosine precession. :math:`N_0`, :math:`\alpha`,
:math:`\beta`, the amplitude, and the precession frequency/damping/phase are
shared free parameters; each side keeps its own background. Because the
physics is shared between the two histograms, this is the statistically
strong protocol: it reports :math:`\beta` together with the :math:`\alpha`
fitted alongside it and their correlation, so the two balances are measured
*with* their mutual dependence rather than in isolation. It is the
**endorsed measurement protocol** and the only one of the two that yields an
:math:`\alpha`–:math:`\beta` correlation.

Raw counts use a shared Poisson likelihood. Applied File/manual deadtime
correction selects weighted count fitting with retained measured or supplied
independent variances, through the same model and solver. A skipped optional
File correction retains raw Poisson observations and its warning/provenance.
Backgrounds remain automatic model terms; grouping background subtraction is
not applied to these observations. See :doc:`alpha_calibration` for selection,
supported observations, and conditional-error limits.

The scriptable :func:`~asymmetry.core.fitting.tf_calibration.fit_tf_calibration`
operation owns this joint fit. Its ``TFCalibrationRequest(fit_beta=False)`` fixes
beta at one in the same physical model for alpha-only calibration; ``fit_beta=True``
frees it. The convenience ``estimate_beta_detailed(method="count_fit")`` delegates
to that operation while retaining its historical result fields and method key.

Asymmetry also checks that free-beta calibration has a resolved amplitude and
frequency: each must reach the existing three-standard-error significance gate
(the amplitude helper retains its small nonzero fallback when an error is absent).
For fitted frequency, a missing frequency error makes calibration unavailable.
An explicitly fixed finite positive frequency is instead an independently
supplied model assumption: only its fitted-frequency significance test is
omitted, while the amplitude and reliable-beta-error checks remain. The estimator
never fixes frequency automatically. A missing reliable beta error makes
calibration unavailable, with a diagnostic. These are safeguards for this weak-TF protocol,
not universal identifiability tests or thresholds attributed to musrfit/WiMDA.
They do not apply automatically to alpha-only calibration with beta fixed.

Single-histogram ratio
-----------------------

*Single-histogram ratio* (method id ``single_histogram``) instead fits the
forward and backward histograms **independently** — each its own
:math:`N_0`, background, amplitude, frequency, damping and phase, with no
physics shared between them — and forms

.. math::

   \hat\beta = \frac{\hat A_{0,b}}{\hat A_{0,f}}, \qquad
   \sigma_\beta = \hat\beta \sqrt{\left(\frac{\sigma_{A_f}}{\hat A_{0,f}}\right)^2
                  + \left(\frac{\sigma_{A_b}}{\hat A_{0,b}}\right)^2}

by ratio (the fitted :math:`\hat\alpha = \hat N_{0,f}/\hat N_{0,b}` is
reported the same way). Because the two fits share no physics, this protocol
is statistically weaker than the count fit — it does not correlate
:math:`\alpha` and :math:`\beta`, and reports no such correlation (``None``
in the underlying :class:`~asymmetry.core.fitting.beta_calibration.BetaEstimate`)
— but it is a useful independent cross-check: two fits that never see each
other's data landing on compatible :math:`\beta` values is good evidence the
count-fit result is not an artefact of the shared model.

Applying the joint result
-------------------------

Alpha and beta refer to the same count fit and shared covariance. **Apply** uses
both coefficients from that record. A fixed-beta calibration applies beta one,
even when a manual or historical nonunity beta is present. The result displays
fixed, fitted, and unavailable-error states distinctly; missing covariance does
not by itself make an otherwise usable alpha-only fit fail.

Instrument guidance
--------------------

:math:`\beta` is broadly an **instrument-dependent** property of a detector
pair — a fact about the geometry and absorption of that pair, not about the
sample — so, unlike :math:`\alpha`, it does not need to be remeasured for
every sample mounting. Typical values sit close to but below 1: around
0.8–0.9 on PSI's FLAME, and just under 1 on GPS. A fitted :math:`\hat\beta`
outside :math:`[0.5, 1.5]` is flagged as suspicious in the result row —
worth checking the fit before applying it, since typical instruments sit
comfortably inside that range.

Uncertainties
-------------

Both protocols report symmetric curvature standard errors from the underlying
count fit(s). The joint operation explicitly evaluates HESSE after minimisation;
the independent single-histogram cross-check retains its optimiser covariance. For the count fit, :math:`\sigma_\beta` and
:math:`\sigma_\alpha` come directly from the joint fit's covariance, together
with their correlation; for the single-histogram ratio, they are propagated
from the two independent fits' amplitude (and :math:`N_0`) errors by the
standard ratio-error formula above, and no correlation is available. As with
:math:`\alpha`, the provenance recorded on the profile — method, source run,
and error — lets a re-opened project show exactly how its :math:`\beta` was
obtained, and the estimate goes **stale** (the same staleness convention as
:math:`\alpha`) when the deadtime or background corrections change after it
was measured, since :math:`\beta` was fitted on the counts those corrections
produced.

The full joint parameter/covariance owner includes both backgrounds and every
other nuisance. Errors and alpha–beta correlation derive from that matrix;
unavailable/unreliable covariance supplies no coefficient error. The legacy
``reduced_chi2`` field for ``count_fit`` contains the selected objective divided
by joint degrees of freedom, counting shared free parameters once: Poisson
deviance for raw data or weighted chi-square after applied deadtime correction.
Neither is used to rescale parameter errors; Poisson deviance is not an exact
chi-square test. See
:doc:`alpha_calibration` for supported observations, fixed-beta error
state, cancellation, and conditional-error limits.

Compared with other programs
----------------------------

musrfit's `Single Histograms tutorial
<https://musruser.psi.ch/Tutorial/Tutorial2.html>`_ describes simultaneous
histogram calibration, including shared alpha/beta parameters. Its examples
also illustrate Gaussian damping and separate detector phases; those choices
are distinct from this exponential/common-phase model. Matched global
histogram likelihood controls in musrfit 1.12.1 (``708f777b``, ROOT 6.40.04)
agree in fixed/free-beta point estimates within 0.0001 absolute alpha/beta.
Here alpha is :math:`N_F/N_B`, reciprocal to the compared native parameter,
while beta is the same backward/forward amplitude ratio. The native type-2
asymmetry chi-square route has a different objective.

The installed native likelihood route reports errors approximately
:math:`1/\sqrt{2}` of the independently checked conventional Poisson standard
errors. Asymmetry does not copy that scale; see :doc:`/explanation/comparison`
for the one-normalisation control and its limits. WiMDA quick Diamagnetic is
a different point objective. Native WiMDA paired-count model-fit errors are
not verified by these checks. Mantid AlphaCalc is the separate ratio comparator.


The separate script operation
:func:`~asymmetry.core.fitting.tf_calibration.select_tf_calibration` compares four
common-phase envelopes with beta fixed or free. Its dimensionless ``stretch``
exponent is distinct from detector beta; candidate errors retain all fitted
nuisance correlations and are conditional on that envelope. See
:doc:`alpha_calibration` for summed-cost ranking, constraints, candidate failures,
and reference component comparisons. The historical ``count_fit`` adapter keeps
its exponential model.

**References**

- A. Suter and B. M. Wojek, Phys. Procedia **30**, 69 (2012) — musrfit, whose
  asymmetry fit (fit type 2) defines :math:`\beta`.
- musrfit user manual, "Asymmetry Fit (fit type 2)"
  (<https://lmu.pages.psi.ch/musrfit-docu/user-manual.html>).

Stored joint calibration
------------------------

When a shared TF result is applied through the core grouping-profile API, beta
and alpha refer to the same frozen fit and nuisance covariance. Fixed beta has
no fitted error or correlation. Replacing beta with a manual policy removes its
active joint link while keeping the captured fit in profile history. Stored
independent scalar alpha and beta records do not imply a joint measurement,
even when their source runs match. See :doc:`../project_files`.

Workflow application
---------------------

``asymmetry alpha --calibration-method tf_calibration_fit --fit-beta`` and
``reduce --alpha-from RUN`` with those options use the same joint TF solver.
Without ``--fit-beta`` the request fixes beta at one, without a fitted error or
alpha/beta correlation. Applying the captured record replaces both coefficient
policies, including fixed beta one, and retains their common full covariance
and conditional error meaning in the workflow workdir. See
:doc:`../agent_workflow` for source-period and envelope options. Reduced-signal
errors retain the existing conditional counting convention.
