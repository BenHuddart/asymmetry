Global Fit Wizard
=================

The global fit wizard is the analogue of :doc:`fit_wizard` for an ordered
series of runs — a longitudinal-field decoupling series, a temperature
scan, a fallback run-order sweep — where the experiment is set up so that
one common composite model should describe every dataset, with parameters
either shared globally or free per run. The textbook use is an LF
decoupling series where :math:`\Delta` is shared across runs and
:math:`B_L` is fixed per run (:doc:`workflows/lf_decoupling_dynamics`),
but the same workflow applies whenever you expect a single model family
(e.g. ``Oscillatory + Constant`` throughout an ordered magnetic phase) to
fit every run. When the model qualitatively *changes* across the series
— a paramagnetic component appearing through a transition, oscillations
collapsing into a relaxation — fit each run individually rather than
forcing a single global model.

The wizard runs as an explicit two-phase process. It first builds a
ranked **screening table** from independent single-dataset fits across
the whole series, which is fast and lets you see at a glance which
candidate families look promising. You then select one or more rows and
trigger the expensive **coupled global optimisation** only for those
candidates. Keeping the two stages separate makes it obvious which rows
are still only single-fit screening results and which have actually been
optimised under shared-parameter constraints.

The typical session: select the runs in the global-fit tab and verify
the starting model, parameter values, and bounds; open
``Global Fit Wizard...``; in the parameter-setup dialog, review the
combined parameter list and adjust expected roles
(``Global`` / ``Local`` / ``Fixed``) and bounds; click
``Build Screening Table``; pick one or more promising rows and
``Optimize Selected``; inspect the coupled-fit results in the
``Global Optimized Fits`` tab; apply the recommended or selected result
back into the global-fit tab. Completed wizard states are cached with the
tab context and persisted in project files, so reopening the wizard does
not force a rebuild of an unchanged screening table or a rerun of
finished optimisations.

The coupled-fit step is where the global wizard pays for itself: the
shared-parameter constraint usually tightens the uncertainties on the
common parameters (typically the field-distribution widths and
amplitudes) below what any single-run fit can achieve, and it cleans up
the per-run trends in the local parameters by suppressing the noise that
arises when each run independently re-optimises an otherwise common
quantity. It is also a useful cross-check on a series you have already
fit individually — the wizard's screening phase should recover the same
model family you converged on by hand.

Before Analysis Starts
----------------------

Before the screening table is built, the wizard opens a
``Global Fit Wizard Parameter Setup`` dialog. This dialog is the place to tell
the wizard what you already know physically before it starts the expensive
search.

For each combined parameter name across the candidate families you can set:

- the expected role: ``Global``, ``Local``, or ``Fixed``
- the numerical bounds used during both screening seeding and coupled
  optimization
- which candidate families use that parameter

The defaults are intentionally conservative:

- amplitude-like parameters start as ``Global`` with positive bounds
- rate-like parameters start as ``Local`` with positive bounds
- background-like terms stay ``Global`` unless you change them

These choices do not force the final recommendation unless you mark a parameter
``Fixed``. They set the initial expectations and bounds that the global search
will honour when a candidate is later optimized.

The wizard also makes sure that every selected run has a matching single-fit
wizard comparison table for the same shared candidate portfolio. Existing tables
are reused when they already match; missing tables are generated automatically
and cached back into the normal per-run single-fit state.

Workflow
--------

The wizard is organised into six pages.

Series Overview
~~~~~~~~~~~~~~~

The first page lists the ordered datasets and summarizes the same deterministic
fingerprint hints used by the single-spectrum fit wizard:

- whether oscillations appear resolved
- whether the shape looks Kubo-Toyabe-like
- whether the envelope suggests more than one relaxation rate

The wizard infers one dominant sweep axis. Version 1 supports:

- field sweeps
- temperature sweeps
- fallback run-order series when neither field nor temperature varies

If both field and temperature vary materially, the wizard reports that it
cannot make an automatic recommendation for that mixed grid.

Candidate Portfolio
~~~~~~~~~~~~~~~~~~~

The second page shows the curated model portfolio that will be compared. It
reuses the same version-1 candidate family as the single-spectrum fit wizard,
including the current global-fit function as a baseline when one is already
selected in the tab.

The always-available candidates are:

- ``Exponential + Constant``
- ``Exponential + Exponential + Constant``
- ``Exponential + Gaussian + Constant``
- ``Gaussian + Constant``
- ``Gaussian + Gaussian + Constant``
- ``Exponential + Exponential + Exponential + Constant``
- ``Exponential + Exponential + Gaussian + Constant``
- ``Exponential + Gaussian + Gaussian + Constant``
- ``Gaussian + Gaussian + Gaussian + Constant``
- ``Stretched Exponential + Constant``

When the ordered series looks Kubo-Toyabe-like, the wizard also considers:

- ``StaticGKT_ZF + Constant``
- ``StaticGKT_ZF * Exponential + Constant``

When the fingerprints suggest oscillations, the wizard also considers:

- ``Oscillatory * Exponential + Constant``
- ``Oscillatory * Gaussian + Constant``

Single-Fit Screening
~~~~~~~~~~~~~~~~~~~~

The third page ranks every candidate family using the independent single-fit
wizard results for each dataset. For each candidate the wizard sums the per-run
``AIC``, ``AICc``, and ``BIC`` values across the whole series and sorts the
shared portfolio by the currently selected metric.

If a candidate family succeeds on some datasets but fails on others, the wizard
now performs one repair pass before excluding that family. It retries the
failed datasets using fitted parameters from successful runs of the same family
as new initial guesses, which often rescues otherwise incomplete screening rows
for smooth field or temperature series.

This table is deliberately labeled as screening only. A good row here means the
function family looks promising across the series when each dataset is fit on
its own. It does **not** yet mean the candidate has survived coupled global
fitting or parameter-sharing tests.

The table also shows status information:

- ``Not optimized`` means the row is still only a screening result
- ``Running`` means the row is currently being optimized
- ``Optimized`` means the row has a coupled global-fit result available
- ``Optimization failed`` means the coupled fit was attempted but did not
  complete successfully

You can select one row or multiple rows. This lets you optimize candidates one
at a time or in batches.

Global Optimized Fits
~~~~~~~~~~~~~~~~~~~~~

The fourth page lists only candidates that have already been run through the
coupled global optimization stage. These results are the only rows that can be
recommended or applied back into the global-fit tab.

When several candidates are selected from the screening table, the wizard runs
their coupled optimizations independently. Where it is safe to do so, these can
run in parallel. On Windows the wizard currently falls back to a safer serial
path for batch optimization because threaded native fitting back-ends have been
unstable in this workflow.

Candidates can be reranked with:

.. math::

   \mathrm{AIC} = \chi^2 + 2k

.. math::

   \mathrm{AICc} = \mathrm{AIC} + \frac{2k(k+1)}{n-k-1}

.. math::

   \mathrm{BIC} = \chi^2 + k \ln(n)

Here :math:`k` is the total number of free global parameters plus the
run-specific local parameters, and :math:`n` is the total number of fitted
points across the selected datasets.

Changing the ranking metric reranks the already computed rows without rerunning
the analysis. Rebuilding the screening table is only required if the selected
datasets, model, bounds, or expected roles change.

Parameter Sharing
~~~~~~~~~~~~~~~~~

The fifth page explains the recommended role for each non-fixed parameter of
the currently selected optimized candidate. For each parameter the wizard
reports:

- the score with that parameter kept ``Global``
- the score with that parameter made ``Local``
- the absolute score difference
- simple trace diagnostics such as normalized total variation and roughness

These recommendations are intended to discourage overfitting. A model with many
local parameters will usually fit better in raw :math:`\chi^2`, so the wizard
only recommends ``Local`` when the penalized information criterion improves
enough to overcome the extra flexibility.

Apply
~~~~~

The final page summarizes the currently selected optimized candidate and lets
you apply either:

- the recommended optimized candidate
- the optimized candidate currently selected in the results tab

Applying a result updates the global-fit tab's composite function, parameter
values, bounds, and ``Global`` or ``Local`` roles directly. The wizard also
reuses the already computed fit bundle so that plots and parameter views can
refresh immediately without rerunning the fit.

Notes
-----

- The screening table is intentionally more permissive than the final optimized
  results table. Treat it as a work queue, not as the final answer.
- You do not need to optimize every row. The intended workflow is to inspect
  the screening ranking, optimize the most promising families, then compare the
  coupled results.
- Completed wizard states are cached with the current global-fit tab context,
  so reopening the wizard does not force you to rebuild an unchanged screening
  table or rerun finished candidate optimizations.
