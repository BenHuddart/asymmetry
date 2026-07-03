Fit Wizard
==========

.. image:: /_generated/screenshots/fit_wizard_portfolio.png
   :alt: Fit Wizard Candidate Portfolio page populated on the Ag ZF GKT dataset
   :width: 100%

*Candidate Portfolio page of the Fit Wizard run on the same synthetic Ag*
*polycrystal ZF dataset used in* :doc:`fitting` *— the wizard ranks the*
*curated candidate models by an information-theoretic metric (AICc by*
*default) and correctly identifies* ``StaticGKT_ZF + Constant`` *as the*
*best fit, in line with Ag being the textbook nuclear-dipolar reference*
*sample (Blundell et al. Ch 5.2).*

The fit wizard is a guided workflow for choosing a sensible single-spectrum
time-domain fit function. It fingerprints the active spectrum, screens
candidate model families drawn from the component library (restricted to the
physics scope you choose), ranks the fitted candidates with an information
criterion (AICc by default), and writes the chosen result back into the
single-fit tab when you accept it. The most natural
use is the first pass on an unfamiliar spectrum — temperature points in
the middle of a transition, a sample whose magnetic structure is not yet
known, a survey of multiple compounds in a synthesis batch — but the
ranking is also a useful sanity check when you already suspect a model: if
the wizard does not place your guess at the top, the residual plots on the
compare page usually tell you why. For the very simplest cases (clean
single-frequency TF precession, an obvious single-exponential decay)
building the model by hand in the fit panel remains faster.

The wizard opens in a non-modal window from the single-fit tab and does
not start the expensive analysis until you press ``Start Analysis``.
Fingerprinting and model comparison then run in the background with a
progress indicator, so the main GUI stays responsive. The wizard uses the
same dataset, bunching, and fit range that the single-fit tab is using at
the time you open it, so candidates are compared on exactly the same
points a manual fit would use. Completed wizard analyses are cached per
dataset in the fit-panel state, reused when the wizard is reopened on the
same run, persisted in project files, and consumed by the Global Fit
Wizard as the screening table for ordered-series analysis.

The typical session is: open the wizard with the single-fit tab already
pointing at the right dataset and fit range; click ``Start Analysis``;
review the fingerprint summary and the candidate portfolio; switch
between ``AIC``, ``AICc``, and ``BIC`` on the compare page to see how the
ranking depends on the complexity penalty; inspect overlays and residuals
on the candidates that interest you; and click ``Apply Recommended Fit``,
or highlight a different row and click ``Apply Selected Fit``. If you
later change the fit range, bunching, or the baseline composite already
selected in the tab, reopen the wizard or click ``Refresh Analysis`` so
the recommendation is rebuilt from the current context.

Workflow
--------

The wizard is organised into five pages.

Scope
~~~~~

The first page decides which candidate families the wizard may consider.
A preset menu offers physics-motivated selections — ZF static magnetism,
TF Knight shift / precession, TF superconductor, LF dynamics, fluoride
(F-µ-F), muonium / radical, or everything — and the default ``Auto``
preset infers a scope from the run metadata: the recorded field geometry
selects ZF, TF, or LF families, and for TF runs the field magnitude
excludes muonium components outside their validity regime (the low-TF
doublet above ~150 G, the Paschen–Back pair below ~1.5 kG; the exact
four-frequency ``MuoniumTF`` is never field-excluded). Field geometry is
read from the data file only — it is never guessed from the field
magnitude — and when the metadata does not record a geometry the wizard
falls back to screening every family. The tree below the preset shows
each component with the reason it was excluded, and any component can be
ticked back in (or out); user-registered functions are always offered. A
live estimate of the candidate and fit counts indicates the cost of the
current selection. Changing the scope after an analysis marks the results
stale — rerun to rebuild them.

Fingerprint
~~~~~~~~~~~

The first page summarises deterministic features extracted from the active
spectrum:

- background or late-time tail estimate
- initial amplitude estimate
- raw and smoothed zero crossings
- smoothed turning-point count
- dominant FFT peak frequency, signal-to-noise ratio, and cycle count across the fitted window
- monotonic decay fraction in the smoothed trace
- early-time polynomial curvature
- semilog slope ratio for the relaxation envelope
- late-time dip or recovery score

These features only *prioritise* the candidate families — they never
exclude one. Every family in scope gets at least its cheap Stage-1
representative fitted, so a misleading fingerprint cannot hide the right
model; it can only delay it to a lower priority.

Below the feature table, a peak list shows every line found by the
multi-peak spectral search (frequency, signal-to-noise ratio, width, and
any recognised multiplet pattern — a Larmor line at the applied field, a
low- or high-TF muonium pair, or the characteristic µ-F and F-µ-F
triplets). Clicking on the FFT plot adds a *user peak* at that frequency
(dashed red marker); clicking an existing user marker, or selecting its
row and pressing ``Remove Selected Peak``, removes it. User peaks are
treated as trusted frequencies: they seed oscillatory candidates directly
and participate in pattern matching, which is the quickest way to steer
the wizard when you can see a line it underrates. Peaks can be placed
before the first analysis run.

Candidate Portfolio
~~~~~~~~~~~~~~~~~~~

The second page shows the candidate families that will be screened. The
wizard groups the component library into families — relaxation,
multi-rate relaxation, Kubo-Toyabe, precession (including vortex-lattice
line shapes), muonium, and fluorine dipolar (µ-F / F-µ-F) — and screens
them in two stages. Stage 1 fits one cheap representative per in-scope
family (both exponential and Gaussian shapes for the relaxation family).
A family expands to its full Stage-2 portfolio when its representative
passes the residual checks, scores within a small margin of the best
family, matches a recognised multiplet pattern in the detected peaks, or
is pointed at by a fingerprint hint; expensive members such as the
numerical F-µ-F powder averages are only ever fitted inside a promoted
family, with match-derived seeds (a hyperfine constant from a muonium
pair, a µ-F distance from a triplet). When several strong spectral lines
are detected, the wizard also constructs multi-cosine candidates with one
damped oscillator per line. Families that are screened but not promoted
are listed with the concrete reason, so nothing disappears silently.

If the single-fit tab already has a different composite model selected, the
wizard includes that function as a baseline comparison as well.

The additive multi-component candidates are especially useful for spectra whose
smoothed semilog envelope changes slope while remaining largely monotonic. In
that situation a very low-frequency FFT peak can be a by-product of envelope
shape rather than genuine precession, so the wizard now distinguishes
``resolved oscillations`` from ``multi-rate monotonic relaxation`` and can try
mixtures of exponential and Gaussian channels with up to three relaxing
components.

Compare Fits
~~~~~~~~~~~~

Each shortlisted candidate is fitted with a deterministic multi-start strategy.
The wizard seeds five initial parameter sets around heuristic starting values,
including factor-of-two perturbations, and keeps the best successful result for
each template.

Candidates can then be ranked with one of three information criteria:

.. math::

   \mathrm{AIC} = \chi^2 + 2k

.. math::

   \mathrm{AICc} = \mathrm{AIC} + \frac{2k(k+1)}{n-k-1}

.. math::

   \mathrm{BIC} = \chi^2 + k \ln(n)

Here :math:`k` is the number of free parameters and :math:`n` is the number of
fitted points. Smaller values are preferred.

Changing the ranking metric inside the wizard reranks the already computed
candidate fits immediately. It does not rerun the expensive fit stage unless
you explicitly refresh the analysis.

``AICc`` is the default recommendation metric because it adds a small-sample
correction when :math:`n` is not large compared with :math:`k`. If
:math:`n \le k + 1`, the correction is not valid and the wizard falls back to
``AIC`` for that candidate. ``BIC`` applies a stronger penalty to model
complexity and therefore usually favours simpler descriptions.

The compare page also shows residual plots. A candidate is not recommended
automatically unless it passes a lightweight residual gate:

- standardised residual RMS
- runs-test z score
- low-lag autocorrelation magnitude
- residual FFT peak signal-to-noise ratio

Candidates that fail the residual gate are still listed, because they may still
be scientifically useful, but the wizard will flag the warning and avoid making
them the default recommendation.

The current model from the single-fit tab is kept as a baseline candidate when
it differs from the standard curated portfolio. This is useful when you already
have a hand-built function and want to see whether the wizard's simpler
portfolio explains the same spectrum comparably well.

Apply
~~~~~

The final page shows the currently selected candidate, the fitted parameter
values, and any residual warnings. Applying the recommended or selected fit
updates the single-fit tab:

- the composite function is replaced with the chosen wizard candidate
- fitted parameter values are written back into the parameter table
- the fit summary label is updated with the wizard statistics
- the standard ``fit_completed`` update path is emitted so plots and downstream
  views refresh normally

Even if you do not apply a candidate immediately, the comparison table itself
is still preserved. This matters for later global analysis, because the Global
Fit Wizard can reuse those stored per-run tables instead of recomputing them.

Worked Example
--------------

Suppose you open a moderately noisy zero-field spectrum that appears mostly
monotonic, but the semilog envelope bends noticeably at intermediate times.

1. Keep the fit range and bunching you would normally use for a manual first
   pass.
2. Open ``Fit Wizard...`` and run the analysis.
3. On the fingerprint page, check whether the wizard reports a strong
   multi-rate hint but no convincing resolved oscillation.
4. On the candidate portfolio and compare pages, inspect whether
   ``Exponential + Exponential + Constant`` or
   ``Exponential + Gaussian + Constant`` improves the penalised score without
   triggering residual warnings.
5. If the top two candidates are close, switch between ``AICc`` and ``BIC``.
   ``AICc`` often keeps the slightly richer model; ``BIC`` often shows whether
   the extra relaxation channel is really justified.
6. Apply the recommended fit, then inspect the returned parameter values in the
   single-fit tab before committing to a physical interpretation.

This is the intended use case for the wizard: a quick, defensible first pass on
which composite family deserves closer manual inspection.

Recommendation Rules
--------------------

The wizard does not claim that there is always one unquestionable winner. Its
default recommendation policy is:

1. Rank successful candidates that pass the residual gate by the selected
   metric.
2. Break ties by preferring fewer free parameters.
3. Break remaining ties by preferring fewer additive terms.
4. If the top two candidates are within 2 score units and both pass the gate,
   present them as a comparable pair and recommend the simpler one.

This keeps the default behaviour interpretable and favours models that are good
enough without adding unnecessary physical parameters.

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
