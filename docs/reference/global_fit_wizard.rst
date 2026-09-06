Global fit wizard
=================

The global fit wizard is the analogue of :doc:`fit_wizard` for an ordered
series of runs — a longitudinal-field decoupling series, a temperature scan, a
fallback run-order sweep — where the experiment is set up so that one common
composite model should describe every dataset, with each parameter either
shared globally across the series or free per run. The textbook use is a
longitudinal-field (LF) decoupling series where the field-distribution width
:math:`\Delta` is shared across runs and the applied field :math:`B_L` is
local to each run (:doc:`/workflows/lf_decoupling_dynamics`), but the same
workflow applies to any ordered sweep you expect a single model family to fit
— an ``Oscillatory * Exponential + Constant`` precession signal followed
through one magnetic phase, say. Where the model qualitatively *changes* across
the series — a paramagnetic component appearing through a transition,
oscillations collapsing into a relaxation — the wizard partitions the series
into **phases** at the transition and fits each phase under its own template
and global/local assignment, instead of forcing one model across a break it
was never going to describe; see `Transitions: phases and the penalty path`_
below.

Like the single-spectrum wizard, the global wizard is now an answer-first,
three-state window: a **Setup** page where you review the series and choose
scope, a **Running** page that streams its progress, and a **Result** page
that leads with a plain recommendation and the fitted series before exposing
the supporting detail. It differs from the single wizard in one important way:
it drives a two-phase screening-then-optimisation workflow rather than a single
recommendation, so the Result page carries an explicit screening shortlist from
which you launch the expensive coupled fits.

The reason for the two phases is cost. Screening builds a ranked table from
independent single-dataset fits across the whole series — fast, and enough to
see at a glance which candidate families look promising. The coupled global
optimisation, which actually enforces the shared-parameter constraints, then
runs only for the candidates you select. Keeping the stages separate makes it
obvious which rows are still only single-fit screening results and which have
been optimised under parameter sharing. The coupled step is where the wizard
pays for itself: sharing a parameter usually tightens the uncertainties on the
common quantities (typically the field-distribution widths and amplitudes)
below what any single-run fit can achieve, and it cleans up the per-run trends
in the local parameters by suppressing the noise that arises when each run
independently re-optimises an otherwise common quantity. It is also a useful
cross-check on a series you have already fit by hand — the screening phase
should recover the same model family you converged on.

Once the wizard has applied a model, :doc:`fitting` covers running and
refining the coupled fit and :doc:`assessing_a_fit` covers judging the result.
For the single-spectrum version of the same guided approach, see
:doc:`fit_wizard`.

The guided journey
------------------

Open the wizard from the global-fit tab with a run series selected. It uses the
datasets, bunching, and fit range the tab is using at the time you open it, so
candidates are compared on exactly the points a manual global fit would use.
Completed wizard states are cached with the tab context and persisted in
project files, so reopening the wizard on an unchanged series skips straight to
the last result rather than rebuilding an unchanged screening table or rerunning
finished optimisations.

Setup: review the series and choose scope
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: /_generated/screenshots/global_fit_wizard_setup.png
   :alt: Global Fit Wizard Setup page — series overview, scope selector, and Run screening button
   :width: 100%

*The Setup page on a four-field Ag longitudinal-field decoupling series. The*
*Series table lists each run as soon as the context arrives; the classification*
*columns stay* ``—`` *until screening runs. Below it, the scope selector, the*
*collapsed "Guide the search (optional)" section, the search settings, and the*
*primary "Run screening" button.*

The **Series** table lists one row per dataset with its **Run**,
**Field (G)**, and **Temperature (K)** filled immediately — no need to run
screening first. Three further columns summarise the same deterministic
fingerprint hints the single-spectrum wizard uses — whether oscillations look
resolved (**Osc.**), whether the shape looks Kubo-Toyabe-like (**KT-like**),
and whether the envelope suggests more than one relaxation rate
(**Multi-rate**) — alongside a per-run **Confidence** grade and
**Recommendation**. These are computed during screening, so they read ``—``
until you run it; afterwards the rows reorder to follow the inferred sweep axis.

The wizard infers one dominant sweep axis from the run metadata: a field sweep,
a temperature sweep, or a fallback run-order series when neither field nor
temperature varies. A temperature scan through a single magnetic phase is
handled exactly like the field series shown here, with **Temperature (K)** as
the axis. If both field and temperature vary materially, the wizard reports
that it cannot make an automatic recommendation for that mixed grid.

The **Scope** selector chooses which candidate families the wizard screens
across the series, resolved over the whole series so that a component is offered
when it is in scope for *any* run — a temperature series crossing a transition
keeps both its ordered-state and paramagnetic families. Start from a preset, or
from ``Auto``, which infers a scope from the recorded field geometry; when the
geometry is not recorded the wizard falls back to screening every family (as in
the screenshot above, where the synthetic runs carry no geometry tag). A live
estimate of the candidate and screening-fit counts beneath the family tree
indicates the cost of the current selection. Changing the scope after screening
has run marks the shown results stale — an amber banner says so — and clears
the screening selection.

The collapsed **Guide the search (optional)** section is where you tell the
wizard what you already know physically before the expensive search starts.
Leave it closed and the defaults apply; open it to review the combined
parameter list and set an expected role and bounds for each parameter:

- amplitude-like parameters start as ``Global`` with positive bounds
- rate-like parameters start as ``Local`` with positive bounds
- background-like terms stay ``Global`` unless you change them

These choices set the initial expectations and the bounds honoured during both
screening seeding and coupled optimisation. They do not force the final
recommendation unless you mark a parameter ``Fixed``; a fixed parameter is left
untouched throughout. Invalid bounds are reported inline and stop the run
before any fitting starts.

The **Search settings** row carries the ranking metric (``AICc`` by default;
see :ref:`global-fit-wizard-metrics`) and a single, honest optimisation mode —
the **separable role search** (see `How the role search works`_ below) —
reached by the primary **Run screening** button.

Running: the streaming decision trail
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: /_generated/screenshots/global_fit_wizard_running.png
   :alt: Global Fit Wizard Running page — a streaming decision trail above the expanded Live log
   :width: 100%

*The Running page part-way through a screening pass: the first steps are marked*
*done, the current step is highlighted, and the Live log is expanded to show*
*every progress message inline.*

While the analysis runs, the Running page streams a short decision trail whose
steps light up as the core reports progress — reading the series conditions,
choosing candidate families, screening each run independently, and ranking the
candidates across the series. A coupled optimisation shows a different set of
steps (preparing the selected candidates, running the coupled optimisation,
scoring the Global/Local roles, and reranking). The collapsible **Live log**
below the trail captures every progress message in full, and **Cancel** stays
visible throughout so a long run can be stopped cleanly. You can work in the
main window while the analysis runs; if it ends up in front of the wizard, the
wizard returns to the front by itself as soon as the analysis finishes.

Result: the answer card and the screening shortlist
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: /_generated/screenshots/global_fit_wizard_result.png
   :alt: Global Fit Wizard Result page — the series answer card
   :width: 100%

*The answer card after a coupled optimisation of the LF Kubo-Toyabe candidate.*
*Every run is overlaid with its global-fit curve, colour-graded along the*
*series axis, beside a panel showing the local parameter (here* :math:`B_L`
*) against the sweep axis. The screening shortlist and the demoted detail*
*tables continue below the card.*

The answer card leads with the recommendation — a headline naming the
recommended candidate and a plain summary line — above the series overlay:
every run drawn with its coupled global-fit curve, colour-graded along the
sweep axis, beside a **Local parameter trend** panel that plots the leading
local parameter against that axis. In the LF decoupling example above, the
0 G run shows the classic Kubo-Toyabe dip and one-third recovery while the
higher-field runs decouple toward a flat line, and the local :math:`B_L` tracks
the applied field — exactly the shared-:math:`\Delta`, local-:math:`B_L`
structure the model expresses.

Beneath the plot, an alternatives strip surfaces other optimised candidates
that scored close to the winner; because several optimised assignments of the
same template differ only in their Global/Local split, each alternative is
labelled with its local-parameter signature to keep them distinct. Clicking an
alternative swaps the overlaid curves and becomes the candidate that **Apply
recommended fit** would hand back to the global-fit tab. Applying a result
updates the tab's composite function, parameter values, bounds, and Global or
Local roles directly, reusing the already-computed fit bundle so the plots and
parameter views refresh immediately without rerunning the fit.

When the series alphabet's per-run scores suggest a structural change partway
through the series, a **Transitions** card appears between the answer card and
the shortlist, offering a second answer — one phase per side of the break
instead of one model for the whole series. See `Transitions: phases and the
penalty path`_ below for what it shows and how to use it.

Below the card sits the **Screening shortlist** — the ranked table of candidate
families from the single-fit screening pass, with **Screening Score**, the
**AIC** / **AICc** / **BIC** values, a **Status** column, and the parameter
counts. This table is deliberately screening-only: a good row means the family
looks promising across the series when each dataset is fit on its own, not that
it has survived coupled global fitting. Its status column reads ``Not
optimized`` for a screening-only row, ``Running`` while a coupled fit is in
flight, ``Optimized`` once a coupled result is available, and ``Optimization
failed`` when a coupled fit was attempted but did not complete. Select one or
more rows and press **Optimize selected (N)** to launch their coupled fits;
when several are selected the wizard optimises them independently and, where it
is safe to do so, in parallel.

The finished decision trail beneath the shortlist expands to the supporting
detail, each step opening the table it summarises:

- **Candidate portfolio** — every candidate family with its model expression,
  category, and rationale.
- **Global optimized fits** — only the candidates that have been through
  coupled optimisation, with their scores and their Global/Local parameter
  split; these are the only rows that can be recommended or applied. This is
  where you switch the recommendation to a different optimised candidate.
- **Parameter sharing diagnostics** — for each non-fixed parameter, the score
  with it kept ``Global``, the score with it made ``Local``, the difference,
  and simple trace diagnostics (normalised total variation and roughness).
  These recommendations discourage overfitting: a model with more local
  parameters usually fits better in raw :math:`\chi^2`, so the wizard only
  recommends ``Local`` when the penalised information criterion improves enough
  to overcome the extra flexibility.
- **Apply to the fit panel** — a summary of the currently selected optimised
  candidate, with buttons to apply either the recommended candidate or the one
  currently selected in the results.

.. _global-fit-wizard-transitions:

Transitions: phases and the penalty path
-----------------------------------------

.. image:: /_generated/screenshots/global_fit_wizard_transitions.png
   :alt: Global Fit Wizard Result page with the Transitions card, penalty path table, and per-phase strip
   :width: 100%

*The Result page after optimising a synthetic two-phase temperature scan: the*
*Transitions card between the answer card and the shortlist shows the penalty*
*path (0 and 1 breaks, the elbow pre-selected) and, once optimised, a chip per*
*phase naming its range, template, Global/Local split, and confidence.*

A temperature or field series can cross one or more transitions, and the
model that describes the runs on one side does not describe the runs on the
other — oscillations collapsing into a relaxation, a paramagnetic component
appearing above an ordering temperature. Rather than fit the whole series
under one template and hope the residuals stay clean, the wizard partitions
the series into **phases** — contiguous runs along the sweep axis, each with
its own template and its own Global/Local assignment — and fits each phase
independently. A **break** between two phases is always a change of *model
family* — damped oscillation, single relaxation, multi-rate relaxation,
Kubo-Toyabe — never a change of template within a family, of Global/Local
split, or of parameter values. Which template a phase uses (two damped lines
or one, an exponential or a Gaussian envelope) and which of its parameters are
shared are decided by the coupled fit *within* the phase; a global parameter
that drifts smoothly along the whole series has exactly two honest
representations, Global or Local, and can never be approximated by inserting
breaks.

An oscillation that dies out is a change of family too. A damped-cosine
template is read as oscillatory on a run only while at least one of its line
amplitudes is measured — larger than twice its own fitted uncertainty — so on
the runs where every line has collapsed into the envelope the template is
describing relaxation, is not offered as an oscillatory phase there, and the
wizard places a break where the lines stopped rather than carrying one
oscillatory phase across them.

A **Transitions** card appears on the Result page, between the series answer
card and the screening shortlist, whenever the series alphabet's per-run
scores support a partition. It states the whole *penalty path* — the best
partition of the series with exactly :math:`0, 1, 2, \dots` breaks — as one
row per solution, in a table with **Breaks**, **Boundaries**, **Gain**, and
**Status** columns; a solution with nothing to show on a column (no boundary
at zero breaks, no gain at the top of the path) reads ``—``. **Gain** is the
drop in total BIC against the solution with one fewer break — its column
header carries the tooltip "ΔBIC against the solution with one fewer break."
The path's own recommendation — the *elbow*, the largest number of breaks
whose marginal gain still clears a fixed penalty floor — is pre-selected and
marked ``elbow`` in **Status**; once a row's phases have been fitted exactly it
additionally reads ``verified``, and a phase too short to fit on its own is
named directly (``excluded: run 706``, or ``excluded: runs 706, 707`` for
more than one). A short summary above the table states the selected row in
plain language — ``"2 transitions found: 16.5 ± 0.5 K and 28.5 ± 0.5 K."``, or
``"No transitions found: one phase describes the whole series."`` when the
elbow sits at zero breaks — with an excluded phase named in the same sentence:
``" Run 706 is excluded from the global fit: it looks like a different
phase."`` A footnote below the table reads "Transitions are scored with BIC;
the ranking metric applies within a phase." — the partition is always scored
with BIC (a structural change between nested model families is nearly free
under AIC's flatter penalty, so AIC/AICc would frequently see no elbow at all)
whatever :ref:`ranking metric <global-fit-wizard-metrics>` you have selected;
that metric still decides which *candidate* wins inside each phase. Selecting
a row recolours the series overlay above by phase instead of by sweep
position.

A phase must span at least three runs. A shorter run of leftover points is
admitted only at either end of the series, where it is scored at its own
per-run cost plus the usual break penalty and reported as excluded rather than
forced into its neighbour's fit; an interior run that does not fit its
neighbours is a per-run gate failure instead (see `When to trust the
recommendation`_ below), never silently dropped from the middle of a phase. A
break's position is reported as the midpoint between the two adjacent runs,
with a half-gap uncertainty of half their separation — a break between runs at
15 K and 18 K reports as :math:`16.5 \pm 1.5` K.

Selecting a row with at least one break enables **Optimize phases**, which
runs the coupled search independently on each phase of that solution (plus
the neighbouring solutions and shifted breaks the wizard checks to confirm the
elbow, at no extra cost to you beyond the wait) — the break-free row is the
ordinary series-wide answer the shortlist's own **Optimize selected** already
produces, so it carries no separate action here. The Running page shows
"Optimizing each phase…", stepping from "Preparing the series screening
table…" through "Optimising each phase…", which becomes "Optimising phase
*i* of *N*…" once individual phases start; the status line beneath reads
"Running the coupled global optimisation once per phase. Progress is streamed
to the live log." Once it finishes, a strip of phase chips appears beneath the
table, one per phase, each naming its ordinal and range, its template, its
Global/Local split (``Global: A_1, A_bg · Local: Lambda``), and a confidence
line ("High confidence", or "Medium confidence — check the warnings"); the
verified row's **Apply phases** button then creates one nested data group per
phase under the series group (see :ref:`phases-within-a-group` in
:doc:`gui_usage`), records one global-fit series per phase (see
:ref:`trend-phase-owned-series` in :doc:`parameter_trending`), and binds the
global-fit tab to the first phase. The main window's status bar confirms what
was created, e.g. "Applied 2 phases under Runs 901-906 (2 transition(s))."

References
~~~~~~~~~~

1. R. Killick, P. Fearnhead, and I. A. Eckley, J. Am. Stat. Assoc. **107**,
   1590 (2012).
2. N. R. Zhang and D. O. Siegmund, Biometrics **63**, 22 (2007).

.. _global-fit-wizard-role-search:

How the role search works
--------------------------

Once a template is selected for coupled optimisation — for the whole series,
or for one phase of a partition — the wizard still has to decide, for every
promotable parameter, whether it is shared (``Global``) or free per run
(``Local``). This is the **role search**, and it is now **separable**: every
effort tier resolves to the same engine, so there is exactly one honest
optimisation mode rather than a slider that trades accuracy for speed.

The separable search never pays to *discover* the all-local answer: :math:`G`
independent per-run fits already *are* the all-local assignment, so it is
assembled directly from the phase's own per-run results — no joint fit. A
full-covariance surrogate (a generalised-least-squares collapse of the per-run
values and covariances) then scores every sharing pattern at once and hands
back a warm start for it — pooled shared values, plus each run's conditional
local values. **Backward elimination** walks from all-local, promoting one
parameter to ``Global`` at a time — cheapest by the surrogate first — with one
exact, warm-started coupled fit per step, and stops as soon as a step no
longer improves the information criterion; a handful of the most promising
templates race through this together, so a template that falls behind early
keeps its free all-local score rather than being fitted further for no
benefit. Once elimination stops, the winner's single-flip neighbourhood (every
parameter toggled once from the winning assignment) is fitted too, so the
per-parameter Global/Local recommendations on the results page are exact
rather than inferred from the path taken to reach them. Every one of these
fits runs at the series' own search resolution (the coarsest rebinning any
run's own analysis chose); in a series-wide search only the winner, and its
flip-neighbourhood, are refitted once more at full resolution for the numbers
you actually see. A *phase* (see `Transitions: phases and the penalty path`_)
is reported at the search resolution instead — every row of the penalty path
is then scored on the same points, and the fit you apply from a phase seeds
the Batch tab's own global fit, which runs on the native record. The whole
search costs on the order of :math:`P` coupled fits per template, where the
previous exhaustive search cost :math:`2^P`.

Each of those coupled fits is solved by a **sparse least-squares** minimiser
rather than by Minuit. A coupled node's Jacobian is arrow-shaped — every
residual depends on the shared parameters and on exactly one run's local ones —
so the solver is told that pattern up front and evaluates one finite-difference
column for the same local across every run at once. The cost of a Jacobian then
grows with the *per-run* parameter count rather than with
:math:`n_\text{global} + n_\text{local}\,G`. On a wide node — a twelve-run phase
with nine free parameters per run over a couple of hundred thousand points —
that is the difference between minutes per fit and seconds, and the equivalent
Minuit problem over ninety-odd parameters may not converge at all. The fitted
values, :math:`\chi^2` and parameter uncertainties are the same quantities
Minuit reports (the uncertainties come from the Gauss–Newton curvature at the
solution, which for a :math:`\chi^2` cost is Minuit's own convention), so
nothing about the ranking or the reported numbers changes — only how long the
search takes. Asymmetric MINOS intervals are the one thing this solver cannot
produce; the wizard does not request them.

The exhaustive "wavefront" search that enumerates every :math:`2^P` role
assignment has not gone away — it is retained behind the lower-level
``search_engine="exhaustive"`` argument as the harness referee the separable
engine is measured against, and it remains reachable through the same
argument for the small set of callers that still want it explicitly. It is
never reached from the GUI or from ``effort_tier``.
``tools/global_wizard_harness.py --engine {separable,exhaustive}`` runs either
engine against the frozen golden-verdict baseline; acceptance for a candidate
is at least 95% verdict agreement with the frozen (exhaustive) baseline, with
every disagreement inside the harness's IC-gap tolerance — measured at 100%
agreement on the harness's case set.

.. _global-fit-wizard-metrics:

Ranking metrics
---------------

Candidates are ranked, and can be reranked, with the same three information
criteria as the single-spectrum wizard:

.. math::

   \mathrm{AIC} = \chi^2 + 2k

.. math::

   \mathrm{AICc} = \mathrm{AIC} + \frac{2k(k+1)}{n-k-1}

.. math::

   \mathrm{BIC} = \chi^2 + k \ln(n)

Here :math:`k` is the total number of free global parameters plus the
run-specific local parameters, and :math:`n` is the total number of fitted
points across the selected datasets. ``AICc`` is the default; ``BIC`` applies a
stronger complexity penalty and usually favours simpler descriptions. Changing
the metric reranks the already-computed rows without rerunning the analysis —
rebuilding the screening table is only required if the selected datasets, model,
bounds, or expected roles change.

When to trust the recommendation
--------------------------------

The wizard states its recommendation plainly, but it is a decision aid, not a
verdict to accept unread. Its confidence is worth calibrating against what the
gate logic actually checks.

The recommended candidate is the best-scoring optimised candidate whose
residuals pass every automatic residual and continuity check across the series.
When the top two are within a small score margin the wizard presents them as a
comparable pair and prefers the simpler one, surfaced as an alternative on the
card. This is the case to trust with least reservation: a clean recommendation
means every run's residuals look unstructured under the shared-parameter fit.

Two softer outcomes deserve a closer look. When *no* candidate passes the
strict series checks but the best coupled fit is nonetheless excellent — every
run clears its own per-run residual gate — the wizard does not veto to nothing.
It surfaces that candidate as a **tentative** recommendation and names the
series-consistency check that flagged (a fingerprint jump across a transition,
a rough local-parameter trace), with the caveat "Review before applying." Treat
a tentative recommendation as a lead: the per-run fits are sound, but something
about how the parameters move across the series is worth understanding before
you rely on it. A per-run gate failure is different and still blocks — it means
the model genuinely does not fit some runs — so a tentative result is
specifically the "fits every run, but the trend looks odd" case, not "fits
badly somewhere."

The per-run readouts on the Setup and Result tables carry the same honesty. A
run whose best single fit shows **no significant structure** — its winner
cannot beat a flat or plain-exponential baseline by a clear margin — is flagged
with an unmissable series-level banner naming the affected runs. This is a
per-run statement, not a series-wide verdict: it is entirely normal for most of
a temperature scan to show clean structure while a handful of runs near a
transition, or at the noisy end of a decoupling series, do not. It usually
means the data there are well described by a plain relaxation, and forcing the
richer global model onto those runs would be over-fitting.

In all three cases the honest move is the same: before applying, open the
optimised candidate's fit overlay and residuals, read the parameter-sharing
diagnostics, and check that the local-parameter trend behaves the way the
physics leads you to expect.

Programmatic global fitting
---------------------------

This wizard is the **asymmetry-domain** shared-parameter workflow, driven from
the GUI. To share fit-function parameters across runs **programmatically**, use
the **count-domain** API ``fit_grouped_series(relationship="global", ...)`` —
see :ref:`grouped-cross-run-global-api` in
:doc:`grouped_time_domain_fitting`.

Calling the global wizard from a script
---------------------------------------

``build_global_fit_wizard_screening_recommendation`` starts by analysing every
dataset in the series with the **single-run Fit Wizard** — the same tiered family
screen, peak analysis and damped-line scan
:func:`~asymmetry.core.fitting.fit_wizard.build_fit_wizard_recommendation`
performs on one spectrum. The series' candidate list is then the union of the
templates those per-run analyses assessed, so a model that describes only part of
a temperature series (a heavily damped pair below a transition, say) is
considered for the whole series instead of being averaged away. A few runs are
analysed side by side, in sweep-axis order, sharing one pool of worker
processes: each analysis spends much of its time in stages that use one core
(spectral detection, the pattern search, the tier gating), so overlapping them
lets one run's fitting use the workers another run's serial stages leave idle.
That pool is opened once for the whole of this phase and serves the scoring pass
below as well.

Each analysis is also started from the fitted values of the nearest run that has
already finished — an extra first attempt per candidate, tried ahead of the seed
ladder. When that warm attempt and the plain seed both converge to the same
minimum, the rest of the ladder is skipped: a fitted answer for the same model on
a neighbouring run agreeing with a cold start leaves the remaining seeds nothing
to find. They are climbed as usual when neither converges or when the two land in
different minima, and the top-ranked candidates are re-fitted from the full
ladder afterwards either way, so the ranking the recommendation rests on is never
the short ladder's.

A candidate no partition of the series could ever select is then dropped from
the list before any scoring: one whose bare χ² on every run already exceeds some
*one* other candidate's information criterion there cannot win a segment under
any sharing, whatever the segment. A candidate a run's own analysis recommended,
or scored comparable, is never dropped this way.

Every remaining run is then scored against every candidate at one common
rebinning factor — the smallest any run's own analysis chose — so the
information criteria of two runs, or of two candidates, may be summed and
compared. The cells a run's own analysis already holds at that factor are kept;
the rest are fitted, warm-started from that run's or a sibling run's values. A
cell started from **that run's own** values for the same model at another
binning is the same data and the same model at a different sampling — the same
minimum by construction — so it is fitted once, from those values. A cell
started from a **sibling** run's values is a different record, so it gets the
plain seed alongside the warm one.

On a long series this legitimately runs for **many minutes** before it returns;
each completed analysis and each completed score row is reported through
``progress_callback`` so you can see it advancing rather than guessing whether it
has stalled. Pass a ``cancel_callback`` if you need to be able to stop it — it is
polled several times a second throughout, including while fits are running in
worker processes.

The scoring pass fans out one task per cell — one run, one candidate — across
the same process pool, so the
``if __name__ == "__main__":`` rule in
:ref:`fit-wizard-scripting-and-parallelism` applies here too: an unguarded
script degrades to serial execution with a ``SpawnUnsafeWarning`` rather than
crashing.

A cached single-run analysis is reused instead of repeated when it answers the
same question — same candidate scope, same user-declared frequencies. That is
what the Fit tabs' own wizard results give the global wizard: analyse a run once
in the Fit Wizard and the series analysis does not pay for it again.

The two stages behind that call are also available separately, for a caller
that wants to reuse phase 1 across several screening calls (a scope sweep,
say) instead of paying for it every time:
:func:`~asymmetry.core.fitting.global_fit_wizard.build_or_complete_single_fit_wizard_recommendations_for_global_portfolio`
returns a ``GlobalFitWizardScreeningTable`` — its ``portfolio``, the completed
``recommendations_by_run``, the runs' own ``single_fit_recommendations_by_run``,
the ``generated_run_numbers``, and the ``series_rebin_factor``. Pass that
``portfolio`` (together with the ``single_fit_recommendations_by_run`` that
covers it) into ``build_global_fit_wizard_screening_recommendation(...,
portfolio=..., single_fit_recommendations_by_run=...)`` and phase 1 is skipped
entirely.

Every screening recommendation also carries a ``partition_path`` — the whole
penalty path over 0, 1, 2, … structural breaks along the series (``None`` on a
series of fewer than six runs, where a partition is not attempted); see
`Transitions: phases and the penalty path`_ above for what the path means.
``partition_path.selected_k`` is the pre-selected elbow, and
``transitions_summary(partition_path.solutions[k], recommendation.series_axis_label)``
renders the same plain sentence the GUI's Transitions card shows for that row.
Passing a ``partition_path`` together with a ``partition_k`` into
``build_global_fit_wizard_recommendation`` switches it from one series-wide
answer to one answer per phase of that solution — the two arguments are
refused unless given together, since a bare index does not name a path. The
result's ``phase_assessments`` then maps ``(partition_k, segment_index)`` to a
``GlobalCandidateAssessment``, and ``recommended_partition_k`` records which
solution was optimised.

``tools/global_wizard_harness.py --engine {separable,exhaustive}`` is the
regression check for the role search itself; see `How the role search
works`_ above for what the two engines are and the acceptance bar between
them.

Screening never sets ``recommended_key``. That is deliberate — a pre-screen score
comes from independent per-dataset fits, which are not evidence about a *coupled*
global fit — but it means a script that reads ``recommended_key`` alone gets
``None`` from a perfectly good screen. Read the ranked table instead
(``sorted_prescreen_assessments()``), or read ``summary``, which now names how
many candidates scored and which one ranks first, and says explicitly when a
screen scored *nothing* — the failure case that used to be indistinguishable
from an ordinary one.

.. _global-fit-wizard-effort-tiers:

Buying a cheaper answer: effort tiers
-------------------------------------

Scoring cost is (candidates × datasets × per-fit cost), and the candidate list a
broad scope produces includes numerically integrated dynamic Kubo-Toyabe models
that can cost an order of magnitude more than the rest of the list combined. On a
fourteen-dataset series that is the difference between a screen that returns and
one that does not.

``effort_tier`` controls it. It narrows the series' candidate list after the
per-run analyses have produced it, so it buys a cheaper *scoring* pass, not a
cheaper per-run analysis:

``EffortTier.LOW``
   Scores a small set of the cheapest, most parsimonious candidates. Coarse, and
   fast enough to be interactive.

``EffortTier.BALANCED``
   Drops only the numerically expensive candidates.

``EffortTier.THOROUGH`` / ``EffortTier.EXHAUSTIVE`` (the default)
   Score the whole candidate list, exactly as before.

Candidates a run's peak search positively identified — a multiplet pattern match,
or a multiplet model built from that run's detected lines — are never dropped:
those come from the data, so removing them would change the answer rather than
coarsen it. Whatever *is* skipped is announced through ``progress_callback`` and
listed in ``instrumentation["screening_skipped_template_keys"]``, so a coarser
answer always says what it was coarsened by.

This tier only ever narrows the *screening* candidate list. The coupled
optimisation that follows — deciding which parameters of the surviving
candidate are Global versus Local — is a separate stage with its own, single
engine at every tier; see `How the role search works`_ below.

.. _global-fit-wizard-timing:

Timing: telling "slow" apart from "hung"
----------------------------------------

Both wizards fill in a standard timing block when you pass an ``instrumentation``
dict, and emit structured per-stage events to a ``stage_callback``:

.. code-block:: python

   from asymmetry.core.fitting.global_fit_wizard import (
       build_global_fit_wizard_screening_recommendation,
   )
   from asymmetry.core.fitting.wizard_scope import EffortTier

   def main():
       instrumentation = {}
       events = []
       recommendation = build_global_fit_wizard_screening_recommendation(
           datasets,
           effort_tier=EffortTier.BALANCED,
           instrumentation=instrumentation,
           stage_callback=events.append,
       )
       timing = instrumentation["timing"]
       print(timing["elapsed_seconds"], timing["cpu_seconds"], timing["cpu_cores"])
       for stage in timing["stages"]:
           print(stage["stage"], stage["elapsed_seconds"], stage["cpu_cores"])

   if __name__ == "__main__":
       main()

``cpu_seconds`` covers this process **and its reaped pool workers**, so
``cpu_cores`` is what distinguishes a slow computation (several cores busy) from
a stalled one (near zero) — the question that otherwise sends a caller to the
process table. Each ``stage_callback`` event carries the stage name, a
``start``/``item``/``end`` marker, items done and total, and the elapsed and CPU
time so far, which is what you want a timeout to watch: absence of *progress*
rather than total runtime.
