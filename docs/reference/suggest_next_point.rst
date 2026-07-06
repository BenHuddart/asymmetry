Suggest next point
===================

.. image:: /_generated/screenshots/suggest_next_point.png
   :alt: The model-fit dialog's Suggest next point section, showing the
         utility band peaking near Tc on a synthetic order-parameter trend
   :width: 100%

*A c-optimal suggestion for* :math:`T_c` *on a synthetic order-parameter*
*trend (generic* :math:`T_c = 100` *K). The shaded band is the utility*
*curve; the dashed line marks the suggested x. The result line reports the*
*events factor and the Monte-Carlo-calibrated post-fit sigma.*

Once a trend model is fitted (:doc:`parameter_trending`), the natural next
question at the instrument is where to put the *next* run, and how long to
count it, to constrain that model the most. The **Suggest next point**
section of the model-fit dialog answers this from the fit you already have —
its parameter values and their covariance — rather than from a rule of
thumb.

How it works
------------

Treating the fit's parameters as approximately Gaussian around their fitted
values (the Laplace approximation), a hypothetical new measurement at some
:math:`x` reduces the uncertainty in the model parameters by an amount that
depends on how sensitive the model is to each parameter there, and how
precisely the new point would be measured. Two standard design criteria turn
that into a single number per candidate :math:`x`:

- **c-optimal** (the default) — minimise the posterior variance of *one*
  parameter of interest, e.g. "pin down :math:`T_c`". You choose the target
  parameter from the model's free parameters.
- **D-optimal** ("all parameters") — shrink the whole parameter covariance
  at once, without favouring any single one.

Both are evaluated as an *expected information gain*, using the fit's
covariance matrix and the model's sensitivity to each parameter (a numerical
derivative at the fitted values) — full derivations are not needed to use
the feature; what matters in practice is that the result is a **curve over
:math:`x`**, not a bare number, so you see *why* one region is more
informative than another before committing beam time to it. The whole curve
is shown, together with its maximum, so you retain judgement over the
suggestion rather than being handed a single instruction.

A worked example
-----------------

Fit a trend model as usual (here the ``OrderParameter`` form on a synthetic
:math:`\nu(T)`-like trend through a generic :math:`T_c = 100` K), then open
the **Suggest next point** section beneath the fit result:

1. **Target.** Choose the parameter you most want to pin down — ``Tc``, in
   the screenshot above — or **All parameters (D-optimal)** to reduce the
   whole covariance ellipsoid instead.
2. **Candidate range.** Defaults to the measured x span; widen it if you are
   willing to measure outside the runs you already have (extrapolated
   candidates are drawn in a visually distinct style, see below).
3. Click **Suggest**. The utility curve appears as a shaded band on the
   preview with a dashed line at the suggested x, and the result line reads,
   for example:

   .. code-block:: text

      Measure at x = 97 × 0 of a typical run's statistics
      → σ(Tc) ≈ 0.603 (MC-calibrated)

   For this order-parameter trend the suggestion lands just below the
   fitted :math:`T_c`, where :math:`\partial y/\partial T_c` is steepest —
   exactly where classical optimal-design theory places the information for
   a critical-point model. The "× 0" here means the precision goal below is
   already met at the existing statistics — see the next step.

4. **Precision goal.** With a single-parameter target, type a target sigma
   (e.g. ``1.0`` for :math:`\sigma(T_c) \le 1` K) in **Precision goal**. The
   result line's events factor is how many times a *reference* run's event
   count (the existing runs' own statistics) you would need at the
   suggested x to reach that goal, found by solving the posterior variance
   for :math:`N` in closed form. A factor of ``0`` means the goal is
   already met without a new point at all; a factor below 1 means less
   counting than a typical run is enough; above 1, more.
5. **Typical run / rate.** Fill in your instrument's typical run size
   (Mevents) to convert the events factor into an absolute Mevents figure,
   and optionally a count rate (Mevents/h) to additionally show the
   equivalent counting time. Both fields are **display-only**: they convert
   the recommendation into units you can act on at the instrument, but
   never feed back into the suggestion itself — events, not wall-clock
   time, are what set the statistics, and beam conditions vary too much
   for a time estimate to be part of the calculation.

Why the sigma is "MC-calibrated"
---------------------------------

The rank-one update used to rank candidates is exact for a linear model,
but for the curved models trending most often fits — order parameters,
critical divergences — it can *underestimate* the realised post-fit
uncertainty near the critical point, because the model's local-linear
approximation breaks down exactly where the sensitivity is largest. Rather
than show that optimistic figure, the dialog runs a short Monte Carlo pass
off-thread the moment you request a suggestion: it simulates adding the
proposed point (drawn from the fitted model plus the empirical noise at the
target statistics), refits, and repeats a few dozen times. The median
realised sigma is what the result line reports, labelled **(MC-calibrated)**;
if that pass has not finished yet the line briefly reads "calibrating…" and
falls back to the analytic figure (labelled **(approximate)**) if the
calibration itself fails. The *ranking* of candidates — which :math:`x` is
better than which — is reliable either way; it is only the absolute
predicted sigma and event count that the calibration corrects.

Warnings you may see
---------------------

The section always shows the utility curve when one can be computed, but
flags the cases where you should trust it less rather than hiding it:

- **"Fit is barely constrained… consider a coarse scan."** Too few points
  for the number of free parameters means the fitted covariance itself is
  unreliable, and a model-driven suggestion can confidently point at the
  wrong place. This is the single strongest lesson from the autonomous
  scattering literature the method is grounded in: model-based suggestion
  only helps once a feature has been *localised* by an ordinary scan: it is
  not a substitute for one.
- **"Fit covariance is ill-conditioned (strong parameter correlations);
  utilities are approximate."** Two or more parameters are strongly
  correlated (a large condition number on the covariance matrix), which
  inflates the numerical sensitivity of the suggestion to the exact fitted
  values.
- **"The suggested point sits near a model domain boundary; its utility is
  step-sensitive and approximate."** Some models change behaviour sharply
  at a boundary (an order parameter vanishing above :math:`T_c`); right at
  that edge, the numerical derivative the acquisition relies on is
  sensitive to the step size used to compute it, so treat the ranking there
  as approximate.
- **"The precision goal cannot be reached with a single new point…"** The
  posterior variance of the target parameter has a floor set by the *other*
  parameters' uncertainty — no amount of counting at one point drives it
  below that floor. The dialog reports the floor sigma rather than
  suggesting an absurd event count.
- Extrapolated candidates (outside the measured x span) are drawn in a
  visually distinct style on the preview, so a suggestion that reaches
  beyond your data is obvious at a glance rather than a warning you have to
  read.

Compare against: which model fits better?
-------------------------------------------

.. image:: /_generated/screenshots/suggest_next_point_compare.png
   :alt: The Suggest next point section with a PowerLaw alternative fitted
         and the discrimination overlay showing the best discriminating point
   :width: 100%

*The same trend with an alternative* ``PowerLaw`` *model fitted for*
*comparison. The AIC evidence line reports the alternative's weight is*
*decisively lower than the order-parameter leader's, and the second*
*(tan) overlay shows where a new point would best tell the two models*
*apart.*

Pinning down one model's parameters is a different question from asking
*whether that model is even the right one*. The **Compare against** row
lets you fit an alternative model over the same masked data as the active
range — pick a component from the list, or **Edit…** to build a composite
the same way you build the primary model — and click **Fit & compare**.
Each successful alternative is added to a running list beneath the button,
with:

- **AIC evidence.** Each candidate's Akaike weight
  :math:`w_i \propto \exp(-\mathrm{AIC}_i/2)`, and its **ratio** against the
  leading model. A ratio above 100 is called **decisive** — the standard
  Jeffreys-scale convention for "the data overwhelmingly prefer one model
  over the other". In the screenshot, an alternative ``PowerLaw`` fit is
  decisively disfavoured against the ``OrderParameter`` leader for this
  trend, as expected of data actually generated from an order-parameter
  form.
- **Best discriminating point.** A second overlay, drawn in a visually
  distinct style from the refinement band, showing where a new measurement
  would best separate the *worst-agreeing* alternative from the current
  leader — the point of maximum disagreement between the two curves,
  weighted by the expected measurement noise there. This is a genuinely
  different question from "where pins down :math:`T_c` best", and can
  suggest a different :math:`x`.

With two or more candidate models that already agree with each other
everywhere in the candidate range, the dialog reports that no discriminating
point exists rather than pointing at an arbitrary location — there is
nothing in the current range that would tell the models apart.

Cost weighting
--------------

**Weight by measurement cost** (off by default) accounts for the fact that
moving the instrument and counting both take time, and that time is not the
same in every direction — cooling and warming a cryostat, or ramping a
magnet up and down, rarely take equally long. Supplying a counting time per
point, a move-time rate for increasing and decreasing :math:`x`, and the
instrument's current position re-weights the utility curve by a cost model
(:math:`\mathrm{utility}^{0.7}/\mathrm{time}`, following the exponent used
in the autonomous-scattering literature this method draws on) and moves the
marker to whichever :math:`x` is most informative *per unit time*, not per
measurement.

Cost weighting changes **where** the marker sits, never the underlying
precision figures: because those figures (the predicted sigma, the events
factor) describe the *unweighted* best point, they are dropped from the
result line whenever weighting moves the marker elsewhere, replaced by a
**"(cost-weighted)"** note. Toggling the checkbox or editing the cost fields
re-renders instantly from the last computed suggestion — it never re-runs
the underlying calculation or the Monte Carlo calibration.

Limitations
-----------

- The suggestion assumes the **fitted model form is correct**. It tells you
  where to measure to constrain *this* model best, not whether a different
  model would fit the data better — use **Compare against** for that
  question, and treat a persistently poor :math:`\chi^2_r` as a sign that
  the refinement suggestion itself rests on shaky ground.
- The analytic post-fit sigma is a **rank-one, locally linear**
  approximation, and is known to be optimistic near the sharp features
  (critical points, boundaries) that these models often have — this is
  exactly why the Monte Carlo calibration pass exists, and the calibrated
  figure is the one to trust for planning.
- Errors set to **None** or **Estimate from scatter** disable the section
  entirely (the target selector, goal field, and **Suggest** button are all
  greyed out). Both modes carry no real per-point noise estimate to
  interpolate for a hypothetical new point — a unit-weight fit has no
  absolute noise scale, and a scatter-rescaled fit only recovers one after
  the fact — so there is nothing physical for the acquisition to predict a
  new measurement's precision from. Switch to **Column**, **Percent**, or
  **Absolute** errors to use the feature.
- The section itself is only available once a range has a **successful fit
  with a covariance**: if HESSE did not run or did not converge, there is no
  covariance to compute a sensitivity-weighted suggestion from.

.. seealso::

   :doc:`parameter_trending` for fitting the trend model this feature builds
   on, including the ``OrderParameter`` form used above and the :math:`\chi^2`
   quality verdict for judging whether a fit is trustworthy enough to
   suggest from in the first place.
