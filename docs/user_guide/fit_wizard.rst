Fit Wizard
==========

The fit wizard is a guided workflow for choosing a sensible single-spectrum
time-domain fit function when you do not want to start from a blank function
builder. It opens in a separate non-modal window from the single-fit tab and
walks through fingerprinting the spectrum, comparing a curated set of candidate
models, and applying the selected result back to the fit controls.

The window opens immediately and does not start the expensive analysis until
you press ``Start Analysis`` inside the wizard. The fingerprinting and model
comparison step then runs in the background with a progress indicator so the
main GUI stays responsive.

The wizard always uses the same analysis dataset that the single-fit tab is
using at the time you open it. That means the current bunching state and fit
range are preserved, so the wizard compares models on exactly the same points
that a manual fit would use.

Completed wizard analyses are cached per dataset in the normal fit-panel state.
That cache is reused when you reopen the wizard for the same run and model
context, persisted in project files, and can also be consumed by the Global Fit
Wizard as the first-stage comparison table for ordered series analysis.

Quick Start
-----------

1. Choose the dataset, fit range, and bunching you want in the single-fit tab.
2. Open ``Fit Wizard...`` and click ``Start Analysis``.
3. Review the fingerprint summary and the candidate portfolio.
4. Use the compare page to inspect overlays, residual warnings, and alternative
   rankings under ``AIC``, ``AICc``, or ``BIC``.
5. Click ``Apply Recommended Fit`` for the default choice, or highlight a
   different row and click ``Apply Selected Fit``.

If you later change the fit range, bunching, or current baseline model, reopen
the wizard or click ``Refresh Analysis`` so the recommendation is rebuilt from
the current context.

Workflow
--------

The wizard is organised into four pages.

Fingerprint
~~~~~~~~~~~

The first page summarizes deterministic features extracted from the active
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

These features are used only to decide which broad model families should be
considered. They do not replace the actual fit comparison step.

Candidate Portfolio
~~~~~~~~~~~~~~~~~~~

The second page shows the curated model portfolio that will be fitted. Version
1 intentionally uses a small set of plausible, interpretable composite models
that can already be built from the supported fit components instead of trying
every possible combination.

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

When the spectrum looks Kubo-Toyabe-like, the wizard also considers:

- ``StaticGKT_ZF + Constant``
- ``StaticGKT_ZF * Exponential + Constant``

When the fingerprint suggests oscillations, the wizard also considers:

- ``Oscillatory * Exponential + Constant``
- ``Oscillatory * Gaussian + Constant``

If the single-fit tab already has a different composite model selected, the
wizard includes that function as a baseline comparison as well.

The automatic portfolio is deliberately conservative. Specialized muon-fluorine
functions such as ``MuF`` and ``FmuF_*`` are not auto-suggested in version 1;
build those manually in the function builder when the experiment calls for
them.

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
complexity and therefore usually favors simpler descriptions.

The compare page also shows residual plots. A candidate is not recommended
automatically unless it passes a lightweight residual gate:

- standardized residual RMS
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
   ``Exponential + Gaussian + Constant`` improves the penalized score without
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
