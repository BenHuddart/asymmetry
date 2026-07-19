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

:math:`\beta` is invisible to integrated counts: it enters only through the
size of the oscillating term, not the total number of events in either
detector, so no count-ratio estimator — the ones :math:`\alpha` uses on a
diamagnetic or count-ratio basis — can see it. Measuring :math:`\beta`
therefore needs a *resolved precession signal*: the same weak-TF calibration
run already used to calibrate :math:`\alpha` (a T20-style run, around 20 G, on
the same sample and mounting), fitted so that the forward and backward
amplitudes are recovered separately. On non-precessing (relaxing LF/ZF or
fully decoupled) data every :math:`\beta` gives the same corrected asymmetry,
so the estimator reports failure rather than a number.

The grouping dialog's β card grows a **Measure from run** section for this:
a **Calibration run** picker (weak-TF candidates highlighted, the same
convention :doc:`alpha_calibration` uses), a **Protocol** selector, and an
**Estimate β** button. Two protocols are offered.

Choosing a protocol
--------------------

============================  ====================================================
Situation                     Protocol
============================  ====================================================
Routine calibration           **Count fit (recommended)** — the default
Independent cross-check       **Single-histogram ratio**
============================  ====================================================

Count fit
---------

*Count fit (recommended)* (method id ``count_fit``) fits the forward and
backward raw count histograms **simultaneously** under a shared Poisson
likelihood, with a single :math:`\beta` scaling only the backward
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

Fitted alpha and "Also update alpha"
--------------------------------------

Because both protocols are count fits, they recover :math:`\alpha` alongside
:math:`\beta`. The result row shows both — e.g.
"β = 0.8732 ± 0.0041 · fitted α = 1.1520 ± 0.0081 · Count fit (recommended) ·
run 7101" — as a **consistency readout**: the fitted :math:`\alpha` is
warn-tinted when it disagrees with the card's currently applied
:math:`\alpha` by more than roughly :math:`3\sigma`, which is a sign the two
calibrations should agree and currently do not (a stale :math:`\alpha`, a
mismatched calibration run, or a poor fit).

**Apply β** writes only :math:`\beta`; the fitted :math:`\alpha` is not
applied automatically. An explicit **Also update α** checkbox next to it
additionally applies the fitted :math:`\alpha` through the same path the
:math:`\alpha` card's own estimator uses, stamped with method ``count_fit``
provenance — for *both* protocols, since Protocol B's :math:`\alpha` is still
recovered from a pair of count fits and ``single_histogram`` is not part of
:math:`\alpha`'s method vocabulary.

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

Both protocols report symmetric (HESSE) standard errors from the underlying
count fit(s) by default. For the count fit, :math:`\sigma_\beta` and
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

**References**

- A. Suter and B. M. Wojek, Phys. Procedia **30**, 69 (2012) — musrfit, whose
  asymmetry fit (fit type 2) defines :math:`\beta`.
- musrfit user manual, "Asymmetry Fit (fit type 2)"
  (<https://lmu.pages.psi.ch/musrfit-docu/user-manual.html>).
