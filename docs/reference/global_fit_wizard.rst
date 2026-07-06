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
oscillations collapsing into a relaxation — fit each run individually rather
than forcing a single global model.

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
the exact bounded search — reached by the primary **Run screening** button.

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
visible throughout so a long run can be stopped cleanly.

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
