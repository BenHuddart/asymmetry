Fit Wizard
==========

The fit wizard is a guided workflow for choosing a sensible single-spectrum
time-domain fit function. Open it, click **Analyze**, and it fingerprints the
active spectrum, screens candidate model families drawn from the component
library, ranks the fitted candidates with an information criterion (AICc by
default), and hands you a plain-language recommendation with a confidence
grade. Accepting it writes the chosen result back into the single-fit tab. The
most natural use is the first pass on an unfamiliar spectrum — temperature
points in the middle of a transition, a sample whose magnetic structure is not
yet known, a survey of multiple compounds in a synthesis batch — but the
ranking is also a useful sanity check when you already suspect a model: if the
wizard does not agree with your guess, digging into the decision trail usually
tells you why. For the very simplest cases (clean single-frequency
transverse-field (TF) precession, an obvious single-exponential decay) building
the model by hand in the fit panel remains faster.

Once the wizard has written a model into the single-fit tab, :doc:`fitting`
covers running and refining it and :doc:`assessing_a_fit` covers judging the
result; for the same guided approach applied across a whole run series, see the
:doc:`global_fit_wizard`.

The wizard opens in a non-modal window from the single-fit tab and does not
start the expensive analysis until you press **Analyze**. It uses the same
dataset, bunching, and fit range that the single-fit tab is using at the time
you open it, so candidates are compared on exactly the same points a manual
fit would use. Completed wizard analyses are cached per dataset in the
fit-panel state, reused when the wizard is reopened on the same run,
persisted in project files, and consumed by the Global Fit Wizard as the
screening table for ordered-series analysis. Reopening the wizard on a run it
has already analysed (with nothing changed) skips straight to the result — no
need to click Analyze again.

.. image:: /_generated/screenshots/fit_wizard_result.png
   :alt: Fit Wizard result page — answer card with the recommended fit above the decision trail
   :width: 100%

*The wizard's result page on a synthetic ZF Ag polycrystal dataset: the answer*
*card gives the plain-language verdict, a confidence grade, and the data with*
*the recommended fit overlaid, above a decision trail whose steps expand to*
*show the reasoning behind the recommendation.*

The two-click path
-------------------

For most spectra the whole interaction is: open the wizard, click **Analyze**,
read the answer. Everything else on the page is there for when you want to
check the wizard's working, not because you have to read it first.

1. **Open.** The wizard opens on a plain explanation of what it is about to
   do — analyse this spectrum, fit a set of physics-motivated candidate
   models, and recommend one with a confidence grade, typically in about a
   minute — followed by a one-line run-context summary (run number, field,
   temperature, sample).
2. **Analyze.** Click the **Analyze** button. There is no surprise
   computation before this point.
3. **Watch the decision trail.** While the analysis runs, a short list of
   stage headlines lights up one at a time as the wizard works through
   reading the run conditions, choosing physics families, searching the
   spectrum, fitting candidates, and weighing the winner. **Cancel** stays
   visible the whole time.
4. **Read the answer card.** When the run finishes, the wizard shows an
   answer card: a plain-language verdict headline, a confidence sentence, a
   plot of the data with the recommended fit overlaid, and a prominent
   **Apply this fit** button. A "no significant structure" result is shown
   here as a legitimate outcome, not an error.
5. **Apply, or dig in.** Click **Apply this fit** to hand the candidate to
   the single-fit tab, or expand any step of the decision trail below the
   card to see the reasoning behind it before deciding.

Guiding the analysis (optional)
--------------------------------

The welcome page has a collapsed **"Guide the analysis (optional)"** section.
Leave it closed and the wizard infers everything it needs from the run
metadata; open it only when you know something the metadata does not:

- **Scope.** A preset menu offers physics-motivated selections — ZF static
  magnetism, TF Knight shift / precession, TF superconductor, LF dynamics,
  fluoride (F-μ-F), muonium / radical, or everything — and the default
  ``Auto`` preset infers a scope from the run metadata: the recorded field
  geometry selects ZF, TF, or LF families, and for TF runs the field
  magnitude excludes muonium components outside their validity regime (the
  low-TF doublet above ~150 G, the Paschen-Back pair below ~1.5 kG; the exact
  four-frequency muonium model is never field-excluded). Field geometry is
  read from the data file only — it is never guessed from the field
  magnitude — and when the metadata does not record a geometry the wizard
  falls back to considering every family. A tree below the preset shows each
  component with the reason it was excluded, and any component can be ticked
  back in (or out); user-registered functions are always offered. A live
  estimate of the candidate and fit counts indicates the cost of the current
  selection.
- **Peak seeding.** Below the scope selector, the same time-domain and FFT
  plots you would see after analysis are already available, so you can seed a
  peak before the first run. Clicking on the FFT plot adds a *user peak* at
  that frequency (dashed red marker); clicking an existing user marker
  removes it. A peaks table lists every seeded and already-detected line and
  supports the same removal from a selected row. User peaks are treated as
  trusted frequencies: they seed oscillatory candidates directly and
  participate in pattern matching, which is the quickest way to steer the
  wizard when you can see a line it underrates.

Changing the scope or the peak seeds after an analysis has already run marks
the displayed result stale — a banner says so, and the **Analyze** button
relabels itself **Re-run Analysis** until you click it again.

The decision trail
-------------------

Below the answer card sits the decision trail: six plain-sentence steps
summarising how the wizard reached its recommendation. Each step expands to
more detail; three of them expand into the same interactive panels you can
reach from the guidance section, now populated with the finished analysis:

1. **Run conditions read** — the scope inferred from run metadata (or a note
   that none could be inferred). Expands to the same scope panel described
   above, now showing the resolved outcome.
2. **Physics families considered** — which candidate families were screened
   and whether each was expanded for detailed fitting.
3. **Spectral search results** — how many spectral lines and recognised
   patterns (a Larmor line, a muonium doublet, an F-μ-F triplet, and similar)
   were found. Expands to the FFT plot and peaks table, with the same
   click-to-seed/click-to-remove peak controls available during guidance.
4. **Candidates fitted, rejections with reasons** — how many candidate models
   were fitted successfully, how many reference baselines were also fitted,
   and, in plain terms, why any candidate was rejected. Expands to the full
   comparison table (score, information-criterion values, residual-gate
   status, reduced chi-squared, and parameter count for every candidate,
   selectable to change the plot overlay).
5. **Winner vs null baseline and checks** — the recommended candidate and
   whether it is decisively better than a plain-relaxation reference.
6. **Confidence statement** — the same confidence sentence shown on the
   answer card, with any caveat spelled out in full underneath.

Expanding a step never re-runs anything; the trail (and every panel it
exposes) is derived directly from the completed analysis.

The comparison table (reached from step 4) also lets you switch the ranking
metric. A **"Ranking metric"** control on the result page reranks the already
computed candidate fits immediately — it does not rerun the expensive fitting
stage. Each shortlisted candidate was fitted with a deterministic multi-start
strategy: five initial parameter sets around heuristic starting values,
including factor-of-two perturbations, keeping the best successful result per
template. The three available metrics are:

.. math::

   \mathrm{AIC} = \chi^2 + 2k

.. math::

   \mathrm{AICc} = \mathrm{AIC} + \frac{2k(k+1)}{n-k-1}

.. math::

   \mathrm{BIC} = \chi^2 + k \ln(n)

Here :math:`k` is the number of free parameters and :math:`n` is the number of
fitted points. Smaller values are preferred. ``AICc`` is the default because
it adds a small-sample correction when :math:`n` is not large compared with
:math:`k` (falling back to ``AIC`` when the correction would not be valid).
``BIC`` applies a stronger penalty to model complexity and therefore usually
favours simpler descriptions.

Alternatives and applying a fit
---------------------------------

When another candidate scored close to the recommended one, the answer card
shows an alternatives strip beneath the plot — a compact chip per candidate,
each carrying a metric-delta badge (``· +1.0``) that says how much worse it
scored than the winner, with its component family in a tooltip. Clicking an
alternative swaps the overlaid curve and becomes the candidate that **Apply
this fit** would hand off, without leaving the card. A **"Show residuals"**
toggle next to the plot switches the overlay to a residuals view for the
currently selected candidate.

Applying a candidate (from the card or from a row selected in the comparison
table) updates the single-fit tab: the composite function is replaced with
the chosen candidate, fitted parameter values are written into the parameter
table, the fit summary is updated with the wizard statistics, and the rest of
the GUI refreshes normally. Even if you do not apply a candidate immediately,
the comparison table is preserved with the analysis; this matters for later
global analysis, because the Global Fit Wizard can reuse the stored per-run
tables instead of recomputing them.

If you want to reconsider from scratch, **Re-analyze** returns to the opening
page so you can adjust the guidance before running again — this is a
different action from **Re-run Analysis**, which is what the Analyze button
relabels itself to after a scope or peak-seed change makes the current result
stale.

**Copy analysis log** renders the full six-step decision trail — headline and
detail bullets for every step — as plain text to the clipboard. This is the
right thing to paste when asking a supervisor to sanity-check a
recommendation, or when reporting an issue with the wizard: it captures
exactly what the wizard considered and why, without requiring a screenshot.

Candidate families
-------------------

The wizard groups the component library into families — simple relaxation,
multi-rate relaxation, static nuclear fields (Kubo-Toyabe), precession signals
(including vortex-lattice line shapes), muonium, and muon-fluorine bonding
(μ-F / F-μ-F) — and screens them in two stages. Stage 1 fits one cheap
representative per in-scope family (both exponential and Gaussian shapes for
the relaxation family). A family is expanded to its full portfolio when its
representative passes the residual checks, scores within a small margin of
the best family, matches a recognised multiplet pattern in the detected
peaks, or is pointed at by a fingerprint hint; expensive members such as the
numerical F-μ-F powder averages are only ever fitted inside an expanded
family, seeded from the match (a hyperfine constant from a muonium pair, a
μ-F distance from a triplet). When several strong spectral lines are
detected, the wizard also constructs multi-cosine candidates with one damped
oscillator per line.

The additive multi-component candidates are especially useful for spectra
whose smoothed semilog envelope changes slope while remaining largely
monotonic. In that situation a very low-frequency FFT peak can be a
by-product of envelope shape rather than genuine precession, so the wizard
distinguishes resolved oscillations from multi-rate monotonic relaxation and
can try mixtures of exponential and Gaussian channels with up to three
relaxing components.

If the single-fit tab already has a different composite model selected, the
wizard fits that function too and keeps it as a baseline comparison, useful
when you already have a hand-built function and want to see whether the
wizard's simpler portfolio explains the same spectrum comparably well.

Recommendation rules
----------------------

The wizard does not claim that there is always one unquestionable winner. Its
default recommendation policy is:

1. Rank successful candidates by the selected metric, excluding null
   baselines and any candidate disqualified for a targeted physical reason
   (see :ref:`fit-wizard-confidence-and-verdicts` below).
2. Break ties by preferring fewer free parameters.
3. Break remaining ties by preferring fewer additive terms.
4. If the top two candidates are within 2 score units, present them as a
   comparable pair (surfaced as an alternative on the answer card) and
   recommend the simpler one.
5. Check the winner against a simpler null baseline. If it does not clear
   that bar, recommend the null instead.
6. Otherwise recommend the winner, with a confidence tier set from the
   residual gate.

This keeps the default behaviour interpretable and favours models that are
good enough without adding unnecessary physical parameters.

.. _fit-wizard-confidence-and-verdicts:

Confidence and verdicts
------------------------

Every recommendation carries a confidence tier and, in the unusual case that
nothing in the portfolio is worth recommending, a different kind of verdict
altogether. Both are stated in plain words on the answer card, and the same
wording appears (with any caveat spelled out in full) in the decision trail's
confidence step.

**High confidence** reads "High confidence — the recommended model describes
the data cleanly." It means the recommended candidate's residuals pass every
check in the residual gate — nothing about the fit's mismatch with the data
looks structured.

**Medium confidence** reads "Medium confidence — this is the best model
tried, but the fit leaves patterns in the residuals. Usable; review before
publishing," followed by a caveat naming which diagnostics still show
structure. It means the candidate is still the clear winner by the selected
metric, but one or more residual diagnostics still show structure. Treat a
Medium-confidence recommendation as usable with that caveat in mind — check
the named diagnostics in the residuals view before trusting the fit for
publication. Either way, High or Medium, the best-scoring candidate is
recommended: the residual gate no longer removes a candidate from
consideration, it only tells you how much to trust the number at the top.

What *does* remove a candidate from consideration is a **targeted
disqualifier** — a check aimed at a specific, physically implausible failure
mode rather than general residual shape:

- a fitted oscillation frequency sitting at the 1/T resolution floor of the
  fit window, or pinned against one of its bounds — both are signs the "line"
  is really an artefact of a too-short window or an unconstrained fit, not a
  resolved frequency
- an oscillation amplitude statistically consistent with zero — the component
  is present in the model but not actually needed by the data
- a free-running oscillation frequency with no supporting line in the
  detected-peaks table and too few cycles inside the statistically
  informative part of the window to stand on its own

A disqualified candidate is still shown in the comparison table (step 4 of
the trail), with its title suffixed "(disqualified)" and the specific reason
available as a tooltip on that row, but the wizard moves on to the next
candidate rather than recommending it.

A recommended oscillation that clears that floor but only just — completing
somewhere around two to three cycles inside the informative part of the
window — keeps its computed confidence tier, but if its spectral line is also
weak or missing from the detected-peaks table, the caveat notes that the
oscillation sits at the edge of what the run can resolve and suggests a
longer or higher-statistics measurement to confirm it. A strong,
well-corroborated line at the same cycle count is not flagged.

Before settling on a winner among the surviving candidates, the wizard also
checks it against two cheap **null baselines** that are always fitted
unconditionally: a flat constant, and a plain exponential decay plus
constant. These appear in the comparison table with their titles suffixed
"(baseline)". If the best candidate does not improve on the simpler of the
two nulls by roughly 10 AICc units or more, the wizard concludes that the
data do not carry enough structure to justify the richer model and
recommends the null baseline instead — this is the wizard's way of saying
"there is nothing here worth a richer model." The answer card shows this as
its own headline, "Your data look like a simple decay — no oscillation worth
fitting," and treats it as a legitimate result rather than a failure: it
usually means the spectrum is well described by a plain relaxation (or is
flat within the noise), and chasing a more elaborate model would be
over-fitting.

Limitations
-----------

- The wizard currently supports one time-domain asymmetry spectrum at a time.
- Frequency-domain fingerprinting uses the standard FFT path only; MaxEnt is
  not part of the wizard workflow.
- Recommendations are limited to models that can already be assembled from the
  supported composite-model components.
- Bayesian model comparison is not part of version 1, although the comparison
  backend is designed so that it can be added later.

The wizard should be treated as a decision aid rather than a replacement for
physical judgement. Inspect the fit overlay, residuals, and parameter values
before accepting the recommendation, especially when two candidates score
similarly.
