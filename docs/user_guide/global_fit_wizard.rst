Global Fit Wizard
=================

The global fit wizard is a guided workflow for choosing one common time-domain
fit function across an ordered field or temperature series. It opens in a
separate non-modal window from the global-fit tab, compares a curated set of
candidate composite models, and recommends whether each fitted parameter should
remain ``Global`` or become ``Local`` across the selected runs.

The window opens immediately and waits for you to press ``Start Analysis``.
The expensive analysis then runs in the background so the main GUI stays
responsive while the candidate fits are prepared.

The toolbar also lets you choose the search strategy before you start:

- ``legacy`` keeps the original exact forward-search and backward-pruning flow
- ``staged_v1`` keeps the same candidate scoring and output format, but uses
   relaxed proposals, per-run prefits, bounded multi-local branching, and exact
   rescue fits to handle harder large-series assignments more efficiently
- ``staged_v2`` keeps the same final exact scoring and recommendation format,
   but adds penalty continuation, relaxed active-set freezing, cheap probe
   screening, and bounded beam refinement to cut down expensive exact fits

In practice, use ``legacy`` for the smallest and simplest series, ``staged_v1``
as the default robustness-oriented choice, and ``staged_v2`` when the series is
large enough that repeated exact role tests are becoming expensive.

The wizard always uses the same dataset selection that is active in the
global-fit tab at the moment you open it, including any inherited single-fit
seeds that are already reflected in the tab's parameter values and bounds.

Quick Start
-----------

1. Select the runs you want in the global-fit tab and verify the starting model,
   parameter values, and bounds.
2. Open ``Global Fit Wizard...`` and choose the search strategy before you run
   the analysis.
3. In the parameter-setup dialog, review the combined parameter list, adjust
   expected roles, and loosen or tighten bounds where needed.
4. Let the analysis run, then compare the shortlisted families and parameter
   sharing assignments.
5. Apply either the recommended result or a manually selected alternative.

If you know that a parameter may legitimately go negative, set that bound before
the expensive search starts. For example, if :math:`A_{\mathrm{bg}}` can be
negative for your experiment, change its lower bound in the parameter-setup
dialog instead of leaving the default positive-only range.

Before Analysis Starts
----------------------

After you press ``Start Analysis``, the wizard first builds the candidate
portfolio for the currently selected series and then opens a
``Global Fit Wizard Parameter Setup`` dialog. This dialog is the place to tell
the wizard what you already know physically before it starts the expensive
role-search stage.

For each combined parameter name across the candidate families you can set:

- the expected role: ``Global``, ``Local``, or ``Fixed``
- the numerical bounds used during the search
- which candidate families use that parameter

The defaults are intentionally conservative:

- amplitude-like parameters start as ``Global`` with positive bounds
- rate-like parameters start as ``Local`` with positive bounds
- background-like terms stay ``Global`` unless you change them

These choices do not force the final recommendation unless you mark a parameter
``Fixed``. They set the initial role expectations and bounds that the wizard
will honour while comparing candidate families.

Workflow
--------

The wizard is organised into five pages.

Series Overview
~~~~~~~~~~~~~~~

The first page lists the ordered datasets and summarizes the same deterministic
fingerprint hints used by the single-spectrum fit wizard:

- whether oscillations appear resolved
- whether the shape looks Kubo-Toyabe-like
- whether the envelope suggests more than one relaxation rate

The wizard tries to infer one dominant sweep axis. Version 1 supports:

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

Version 1 keeps the portfolio intentionally conservative. It does not
auto-suggest specialist muon-fluorine or superconductivity model families, and
it does not try to recommend segmented multi-model descriptions.

If you already built a custom global function in the tab, that current model is
included as a baseline candidate when possible. This gives you a direct way to
compare your hand-built starting point against the curated portfolio.

Compare Fits
~~~~~~~~~~~~

Each candidate first receives a fast all-shared pass so the wizard can
shortlist the most plausible model families. The shortlist always keeps the
best-scoring candidates, the current baseline when present, and the simplest
anchor candidate for each major family.

For shortlisted candidates, the wizard then searches parameter-role
assignments. Both strategies start from an exact all-shared fit, unless the
current model is being used as a baseline, in which case the current
``Global`` and ``Local`` roles are used as the starting point.

The ``legacy`` strategy then performs the original exact forward search: it
localizes one parameter at a time when the penalized metric improves enough to
justify the extra degrees of freedom, then runs one backward simplification
pass to merge weakly justified local parameters back to ``Global``.

The ``staged_v1`` strategy keeps the same final exact scoring, but inserts a
stronger seed-and-refine path before the final exact assignments:

1. Build per-run single-fit prefits to improve the initial parameter seeds.
2. Run a relaxed staged search to identify the most plausible ``Global`` and
   ``Local`` role proposals.
3. For shortlisted multi-local assignments, explore a bounded set of partial
   localization branches instead of following only one ordered path.
4. Attempt exact completion fits from the best staged seeds and branch-derived
   warm starts.
5. Run the same backward simplification pass to remove weakly justified local
   parameters.

In practice this keeps the public recommendation logic unchanged while making
three-or-more local-parameter assignments much more robust on larger ordered
series.

The ``staged_v2`` strategy keeps that staged structure, but shifts the emphasis
from raw robustness toward search efficiency:

1. Run the relaxed optimizer across an increasing penalty schedule instead of a
   single penalty weight.
2. Freeze nearly shared relaxed parameters back into an active shared set before
   extracting discrete role assignments.
3. Generate multiple discrete alternates when more than one role pattern looks
   plausible.
4. Use cheap stage-budget probes to screen role changes before paying for a
   full exact assignment.
5. Refine the discrete search with a bounded beam instead of a strictly greedy
   single-path expansion.

This version is intended for the harder cases where ``staged_v1`` is still too
slow or too brittle on larger series with several genuinely local parameters.

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

Changing the ranking metric reranks the already computed assessments without
rerunning the expensive analysis. Changing the search strategy, bounds, fixed
status, or starting role expectations does require a fresh analysis.

``AICc`` is the default because it remains more conservative than plain
``AIC`` when the total point count is not very large compared with the number
of fitted degrees of freedom. ``BIC`` applies a stronger complexity penalty and
often favours more heavily shared descriptions.

The wizard also opens a read-only log window while the background analysis is
running. This is useful for long series because it shows which candidate family
or staged search step is currently being evaluated.

Parameter Sharing
~~~~~~~~~~~~~~~~~

The fourth page explains the recommended role for each non-fixed parameter.
For each parameter the wizard reports:

- the score with that parameter kept ``Global``
- the score with that parameter made ``Local``
- the absolute score difference
- simple trace diagnostics such as normalized total variation and roughness

These recommendations are intended to discourage overfitting. A model with many
local parameters will usually fit better in raw :math:`\chi^2`, so the wizard
only recommends ``Local`` when the penalized information criterion improves
enough to overcome the extra flexibility.

For runtime reasons, the expensive per-parameter ``Global`` versus ``Local``
retests are only generated for the final candidate set that matters most for
decision making. If you click on another candidate and the table is empty, that
does not mean the candidate has no sharing assignment; it means the wizard kept
the exact assignment but skipped the extra explanatory retests for that row.
Applying that candidate still uses its actual fitted ``Global`` and ``Local``
roles.

Apply
~~~~~

The final page summarizes the selected candidate and any warnings. Applying the
recommended or selected fit updates the global-fit tab directly:

- the composite function is replaced with the chosen candidate
- parameter values and bounds are written back into the global-fit table
- the ``Global`` and ``Local`` comboboxes are updated from the wizard
  recommendation
- the normal global-fit update path is emitted using the wizard's computed fit
  results, so plots and fitted-parameter views refresh immediately

The completed recommendation is cached against the current dataset selection,
model, values, roles, bounds, and search strategy. If you reopen the wizard
without changing that context, it can reuse the finished result instead of
starting over.

Worked Example
--------------

Suppose you have a temperature series where the asymmetry amplitude looks fairly
stable, but the relaxation rate broadens steadily on cooling.

1. Select the full ordered series in the global-fit tab and start from a simple
   model such as ``Exponential + Constant``.
2. Open ``Global Fit Wizard...`` and leave the search strategy at
   ``staged_v1`` unless the series is large enough that runtime is already a
   concern.
3. In the parameter-setup dialog, keep the amplitude parameter ``Global``, keep
   ``A_bg`` global unless you have a physical reason not to, and leave
   ``Lambda`` as ``Local`` with a physically sensible positive bound.
4. Run the analysis and inspect the compare page. If the simplest shared model
   fails continuity or residual checks, look at whether a one-local-parameter
   assignment resolves the issue without forcing a more complicated model
   family.
5. On the parameter-sharing page, confirm that the wizard is keeping amplitude
   terms shared while localizing the rate-like parameter you expected to vary
   across the series.
6. Apply the recommended fit, then use the fitted-parameter tools in the main
   GUI to inspect the resulting parameter trace against temperature or field.

This is the common pattern the global wizard is meant to accelerate: choosing a
single family for the whole series and deciding which parameters should vary
run-by-run without paying for repeated manual global fits.

Warnings
--------

The wizard combines per-run residual checks with ordered-series continuity
diagnostics. Warnings can be raised when:

- residual failures cluster across neighbouring runs
- fingerprint features jump abruptly between adjacent spectra
- a recommended local parameter trace changes too sharply across the series

These warnings do not automatically invalidate a fit. Instead, they tell you
that a single continuous parametrization may be under strain and that a
segmented interpretation or a different physical model may need inspection.

Recommendation Rules
--------------------

The default recommendation policy is:

1. Prefer successful candidates that pass the residual and continuity checks.
2. Rank them by the selected metric.
3. If the top two candidates are within 2 score units, recommend the simpler
   one and flag the other as a similarly scoring alternative to inspect.
4. Within one candidate, recommend ``Local`` only when the score improvement is
   large enough to overcome the complexity penalty or when it resolves a
   diagnostic failure without materially worsening the score.

Limitations
-----------

- The current implementation supports one dominant ordered axis at a time, not mixed field and
- The current implementation supports one dominant ordered axis at a time, not mixed field and
  temperature grids.
- For non-fixed parameters, the wizard only compares ``Global`` versus
   ``Local``. ``Fixed`` is treated as a user-supplied constraint from the
   parameter-setup dialog or the global-fit tab.
- It does not fit explicit parameter-vs-field or parameter-vs-temperature
  trend models inside the wizard.
- It does not automatically split the selected series into different fit
  functions, even when warnings suggest a possible regime change.

Caching and Persistence
-----------------------

Finished global-wizard results are stored with the global-fit tab state. In
practice this means:

- reopening the wizard with the same context can reuse the previous result
- saving the project preserves the finished recommendation, selected strategy,
  and analysis log text
- restoring the project lets you inspect or apply the last completed wizard
  result without rerunning the search immediately

Treat the wizard as a decision aid. It is designed to save time on sensible
first-pass global fits, not to replace physical judgement about the correct
model family for a given experiment.
