Fit wizard
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
dataset and bunching that the single-fit tab is using at the time you open it,
but it analyses the **whole record** rather than the plot's fit range: a
heavily damped line lives in the first few tens of nanoseconds, before most
fit ranges open, and a slow one needs the late tail, so cropping before
looking would answer a different question from the one you asked. Applying a
candidate is unaffected — the model and its fitted parameters go to the
single-fit tab, which goes on fitting your range. Where that range would
exclude a line the recommendation rests on, the wizard says so above the
result:

   The plot's fit range starts at 1.2 µs, after the 240 MHz line
   (1/λ = 0.023 µs) has decayed; widen the range before applying this fit.

Completed wizard analyses are cached per dataset in the
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
  wizard when you can see a line it underrates. Naming a frequency also buys
  you its envelope: the same Δχ² test the damped-line scan uses is run at the
  frequency you clicked, and the rate it measures appears in the peaks table's
  **"Damping (µs⁻¹)"** column and seeds the oscillator's envelope. Because you
  supplied the frequency, the test no longer has to correct for having
  searched a whole band, so a line the blind scan declined can still be
  measured this way; a click on empty spectrum measures nothing and the seed
  stays a bare frequency. A heavily damped line that the windowed FFT plot
  cannot show you no longer needs seeding by hand at all — see
  :ref:`fit-wizard-damped-line-scan`.

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
peaks, or is pointed at by a fingerprint hint — and an accepted
matched-apodisation scan line promotes the oscillatory family the same way a
pattern match does. Expensive members such as the
numerical F-μ-F powder averages are only ever fitted inside an expanded
family, seeded from the match (a hyperfine constant from a muonium pair, a
μ-F distance from a triplet). When several strong spectral lines are
detected, the wizard also constructs multi-cosine candidates with one damped
oscillator per line, up to three.

.. _fit-wizard-damped-line-scan:

Heavily damped lines: the matched-apodisation scan
----------------------------------------------------

An oscillation whose envelope dies in the first few tens or hundreds of
nanoseconds lives entirely in the leading fraction of the record — and every
symmetric apodisation window (``hann``, ``cosine``, ``gaussian``) is zero at
the first sample, so a windowed transform deletes it. That is the same trap
the library's :class:`~asymmetry.core.fourier.apodisation.ApodisationEarlySignalWarning`
exists to flag when you drive the Fourier tools yourself.

The wizard's spectral search therefore runs a second, unwindowed pass: a
**matched-apodisation scan**. The record is multiplied by a decaying
exponential :math:`e^{-t/\tau}` and transformed, once for every :math:`\tau` on
a geometric ladder running from about twenty time bins up to half the
informative record, at roughly four rungs per decade. A line whose own
envelope is :math:`e^{-\lambda t}` reaches its greatest signal-to-noise on the
rung where :math:`\tau = 1/\lambda`, because the apodisation is then a filter
matched to the signal — the classical result that, for a known pulse shape in
additive noise, the filter matched to that shape maximises the output
signal-to-noise ratio. The rung a line peaks on therefore **measures** its
damping instead of inferring it from a crop length somebody chose, and no crop
is asked of you: the ladder replaces it. Detection runs on the whole record,
truncated only at the point where the growing per-point error has made the
remaining data uninformative.

A peak on the ladder is a candidate, not a line. Each candidate is tested by
weighted linear least squares: the record is fitted with a dictionary of slow
decays (:math:`e^{-\lambda_k t}` for :math:`\lambda_k` = 0, 0.3, 1, 3, and 10
μs\ :sup:`-1`), then again with an :math:`e^{-\lambda t}\cos` /
:math:`e^{-\lambda t}\sin` pair added at the candidate's :math:`(f, \lambda)`,
and the χ² improvement Δχ² between the two is maximised over a bounded box
around the seed. Both fits are linear, so Δχ² is exact rather than the outcome
of an iterative search. Acceptance is a **look-elsewhere-corrected** threshold,

.. math::

   \Delta\chi^2 \ge 2\ln(N_\mathrm{trials} / r),

where :math:`r` = 0.01 is the number of false lines tolerated per record and
:math:`N_\mathrm{trials}` counts the independent cells the ladder searched —
the same philosophy as the windowed pass's SNR gate, applied to a statistic
rather than to an amplitude ratio. An accepted line joins the nuisance basis
*and* is subtracted from the record before the ladder is scanned again, so a
strong line's skirt cannot manufacture a second one; up to three lines are
found this way, each finally re-fitted with the others held in the basis so
their reported numbers are comparable.

One accepted line is enough to build candidates the earlier portfolio could
not offer: a sum of damped oscillators, one per scan line and up to three,
**plus** a slowly relaxing term and a constant, with either an exponential or
a Gaussian envelope on the oscillators. A heavily damped line almost always
sits on a relaxing background it decays into, and without a term for that
background the fit spends an oscillator on it. Every oscillator is seeded from
the measurement — frequency, envelope rate, amplitude, and phase — with rate
bounds always containing :math:`\lambda/4` to :math:`4\lambda` and frequency
bounds a few line widths wide. An accepted line also promotes the oscillatory
family for detailed fitting, so these candidates are fitted whenever the scan
finds anything at all; on a record where it finds nothing they never appear.

Three things to know when reading the result:

- Scan lines are labelled ``damped_scan`` in the peaks table, and the envelope
  each one measured appears in its own **"Damping (µs⁻¹)"** column, with the
  amplitude, phase, and Δχ² in the row's tooltip. On the FFT plot they carry a
  dashed marker and name themselves in the legend, "Matched-apodisation scan
  line — not expected to be visible in this windowed transform": the plot is
  the *windowed* transform, so a scan line is legitimately absent underneath
  its own marker. (The older unwindowed crop ladder, whose ``early_fft`` peaks
  used a dotted marker, remains in the library but no longer runs as part of
  the search.)
- Its numbers are not comparable with the windowed pass's. A scan
  signal-to-noise is an excess over one rung's own median-absolute-deviation
  noise floor, not a magnitude ratio, and Δχ² is a third scale again. Rank
  peaks within a pass, never across.
- Slow lines belong to the windowed pass. Where a line is visible to both, the
  windowed pass wins: it looked at the whole informative record, so its
  frequency estimate is the better one, and the scan only ever *adds* lines
  the windowed pass missed. The scan is also deliberately blind to slow, low-Q
  features: a candidate completing fewer than about three cycles inside its own
  fitted lifetime is read as a relaxation shape rather than an oscillation and
  rejected, unless its envelope is faster than any relaxation the slow-decay
  dictionary models. That rule is what keeps a static Kubo-Toyabe minimum out —
  a single dip and recovery that no sum of monotonic decays reproduces, and
  which Δχ² alone would happily call a line.

The scan reaches envelope rates of a few hundred μs\ :sup:`-1` on a finely
binned record. The limit is the short end of the ladder, so what it can reach
depends on the binning: 0.1 ns bins reach far further than 16 ns ones. Faster
than that the oscillation is not recoverable at any apodisation, and the
wizard simply offers no damped candidate. At the other end, a component so
overdamped that it barely completes a cycle is fitted as one broad line if it
is accepted at all; treat the frequency it reports as a description of that
line shape rather than as a precession frequency, and check the fit overlay
before reading physics into it.

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
- a **relaxation component the data do not resolve** — see
  :ref:`fit-wizard-component-resolution` below

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

.. _fit-wizard-component-resolution:

Are the components actually resolved?
-------------------------------------

A composite relaxation model will happily converge with one branch railed to a
rate so fast that its 1/e time falls inside the first few time bins, or collapsed
so slow that it never decays inside the fit window. Neither is a measured rate:
the first is absorbing the leading bins, the second is indistinguishable from a
free constant baseline. Both still buy χ², so AICc will "prefer" the extra
component — which is exactly how an extra exponential ends up in a published
model.

The wizard therefore checks every candidate that carries **two or more relaxation
rates**. For each rate it asks whether the value lies inside the window the data
can resolve:

- the 1/e time must span at least a few time bins (the *fast* edge), and
- the 1/e time must be no longer than the fit window (the *slow* edge).

The check is two-sided on purpose. A fast-edge-only rule silently admits the
collapsed-to-zero branch, which is the same pathology seen from the other end.

Crucially, the verdict is not read off the fitted value alone. Where the
likelihood has a flat ridge in a rate, two optimisers landing in statistically
indistinguishable minima can return opposite answers on the same data — one
"resolved", one "railed". So the wizard judges the whole Δχ² ≤ 1 neighbourhood:
it pins the rate just outside each edge, re-fits everything else, and asks
whether the data can tell the difference. That gives three distinct outcomes:

``resolved``
   The rate, and every value the data cannot distinguish from it, sit inside the
   resolvable window.

``unresolved_fast`` / ``unresolved_slow``
   The fitted rate itself is outside the window — railed into the leading bins,
   or degenerate with the baseline.

``undetermined``
   The fitted rate is inside the window but the admissible interval reaches
   outside it. This is the honest third answer: the data do not determine the
   component either way, and neither "resolved" nor "railed" would be true.

Anything other than ``resolved`` disqualifies the candidate, with the reason
carried on the row exactly like the oscillation disqualifiers above.

The same rule is available on its own for scripted analyses, where you can set
the bin tolerance your binning justifies and supply your own multistart
solutions as the neighbourhood instead of paying for the probe re-fits:

.. code-block:: python

   from asymmetry.core.fitting.resolution import assess_component_resolution

   assessment = assess_component_resolution(
       fit_result,
       model,
       dataset,
       min_bins_per_e_folding=20.0,      # a rebinned corpus can justify far more
   )
   if not assessment.is_resolved:
       for reason in assessment.disqualification_reasons():
           print(reason)

Pass ``solutions=[(chi_squared, {"Lambda_1": ..., "Lambda_2": ...}), ...]`` to
use converged multistart seeds you already have; pass
``probe_neighbourhood=False`` to fall back to the point estimate, which is
cheapest and documented as optimiser-dependent.

.. _fit-wizard-convergence-quality:

Convergence quality
-------------------

Comparing two models is only meaningful when both were searched to comparable
depth. Two things guard that:

- the Stage-2 seed ladder scales with a candidate's **rate dimensionality**, so a
  three-rate family is not judged on the same handful of seeds that fully covers
  a one-rate family's search space; the extra seeds widen the seeded rate
  *separation*, which is the dimension a multi-component fit exists to determine;
- after ranking, the top few candidates are **re-fitted with the full seed
  ladder**. Whichever fit is better is the one reported, and the χ² gained is
  recorded on the assessment as ``refinement_delta_chi_squared``, with
  ``under_converged`` set when it exceeds one χ² unit.

``under_converged`` is not a warning that the reported number is wrong — the
reported number is the deeper fit. It is the measured statement that this
candidate's original ranking was search-limited, and therefore that the whole
table around it should be read with that in mind. Pass
``refine_top_candidates=0`` to skip the pass.

A third thing matters on a long record: **the candidates are fitted on a
rebinned copy of it**. A dozen multi-component fits on 10\ :sup:`5` points is
minutes of arithmetic for information a value-rebinned record already carries,
so above roughly 8 000 points the wizard rebins — never so far that the
highest seeded frequency drops below eight samples per cycle — and says which
record it used:

   Candidates were fitted on a ×8 rebinned copy (11250 points); information
   criteria refer to that record.

Line detection is *not* rebinned: it runs on the full record, where the
bandwidth a heavily damped line needs still exists. Only the fitting stage
sees the copy. Every number in the comparison table — ``AIC``, ``AICc``,
``BIC``, χ²ᵣ, and the residual diagnostics — is computed on that copy, which
is worth remembering when comparing a wizard score against one from a
hand-built fit on the raw record. The recommendation records both the factor
and the point count, as ``rebin_factor`` and ``analysed_points``.

.. _fit-wizard-scripting-and-parallelism:

Calling the wizard from a script
--------------------------------

``build_fit_wizard_recommendation`` evaluates its candidate models across a
process pool. Python starts those worker processes with the ``spawn`` method,
and every spawn worker begins by **re-importing the calling script's**
``__main__`` **module**. Put your analysis behind a guard so the workers do not
re-run it:

.. code-block:: python

   from asymmetry.core.fitting.fit_wizard import build_fit_wizard_recommendation

   def main():
       recommendation = build_fit_wizard_recommendation(dataset)
       print(recommendation.recommended_key)

   if __name__ == "__main__":
       main()

Without the guard the wizard detects the situation before starting any worker,
runs **serially** instead, and warns once with ``SpawnUnsafeWarning`` — the
results are identical, only slower. Interactive sessions, ``python -c ...`` and
notebooks are unaffected: there is no module for a worker to re-import, so they
keep full parallelism.

Passing ``max_workers=1`` is always a safe escape hatch — it never starts a
worker process at all, and it also makes runs bit-for-bit reproducible when you
need to compare two invocations exactly. The same rules apply to
:doc:`global_fit_wizard`, which uses the same machinery.

The returned ``FitWizardRecommendation`` carries the record the candidates were
actually fitted on as ``rebin_factor`` and ``analysed_points`` (both described
under :ref:`fit-wizard-convergence-quality`), and its ``peak_analysis`` carries
the matched-apodisation scan's own result — the accepted lines with their
rates, amplitudes, phases, and Δχ², together with the threshold and trial count
they were judged against — as ``peak_analysis.damped_lines``. The scan is also
usable on its own:

.. code-block:: python

   from asymmetry.core.fitting.damped_line_scan import detect_damped_lines

   analysis = detect_damped_lines(dataset.time, dataset.asymmetry, dataset.error)
   for line in analysis.lines:
       print(line.frequency_mhz, line.damping_rate_per_us, line.delta_chi_squared)

``measure_line_at_frequency`` in the same module answers the narrower question
the GUI's click-to-seed asks: given a frequency you already trust, what is its
envelope?

Limitations
-----------

- The wizard currently supports one time-domain asymmetry spectrum at a time.
- Frequency-domain fingerprinting uses the standard FFT path only; MaxEnt is
  not part of the wizard workflow.
- Analysis and fitting run on the record's **statistically informative
  window**, not necessarily on all of it: where the per-point error has grown
  to several times its early-time value, the remaining data are noise that
  would only whiten the transform, so the window is cut there. A run whose
  errors do not grow that way is analysed whole.
- The fit range set in the plot panel governs the fit you *apply*, not the
  analysis. A range that would exclude a detected damped line is flagged above
  the result, not corrected for you.
- Information criteria refer to the analysed record, which on a long run is a
  rebinned copy (see :ref:`fit-wizard-convergence-quality`).
- ``max_workers`` bounds how many fits run at once, not how much CPU the run
  uses: the vortex-lattice line shapes reach a multi-threaded BLAS from inside
  a single fit, so even ``max_workers=1`` can saturate the machine. Set
  ``OMP_NUM_THREADS`` / ``OPENBLAS_NUM_THREADS`` in the environment before
  starting Python if you need to bound it.
- Recommendations are limited to models that can already be assembled from the
  supported composite-model components.
- Bayesian model comparison is not part of version 1, although the comparison
  backend is designed so that it can be added later.

The wizard should be treated as a decision aid rather than a replacement for
physical judgement. Inspect the fit overlay, residuals, and parameter values
before accepting the recommendation, especially when two candidates score
similarly.
