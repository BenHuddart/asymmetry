.. _assessing-a-fit:

Assessing a Fit
===============

A converged fit is not yet a trustworthy one. Three checks, in increasing
strength, tell you whether to believe a result — and they answer different
questions. This page is the entry point that ties them together; each step
links to where it is documented and driven in full.

1. Is the χ² in its band?
-------------------------

The first, cheapest check is the reduced :math:`\chi^2`. For a correct model
with correctly estimated Gaussian errors, the fit :math:`\chi^2` follows the
chi-squared distribution with :math:`\nu = N - N_\mathrm{free}` degrees of
freedom, so a *good* fit lands :math:`\chi^2_r` inside a band around 1 that
tightens as :math:`\nu` grows. Asymmetry turns the fitted :math:`\chi^2` into a
two-sided verdict at a confidence level :math:`R` (default 0.95):

* **good** — :math:`\chi^2_r` inside the band;
* **poor** (upper tail) — the model is missing physics, or the input errors are
  underestimated;
* **overdone** (lower tail) — the fit reproduces the data *better* than the
  errors allow, usually meaning overestimated errors or an over-flexible model.

See :ref:`fit-statistics` for how to read the numbers and what each tail
implies. The verdict assumes the :math:`\chi^2` was computed against *real*
error estimates; with unit weights or scatter-estimated errors (which force
:math:`\chi^2_r` toward 1 by construction) it carries no goodness information
and is suppressed (see :doc:`parameter_trending`). Inspecting the
:ref:`residual time series <fit-statistics>` is the companion structural check:
a good χ² with coherent residual structure still means a missing component.

2. Is the verdict recorded and consistent across the series?
------------------------------------------------------------

A single number is recorded per fit as a compact, JSON-serialisable summary —
success flag, :math:`\chi^2`, reduced :math:`\chi^2`, and the fitted parameter
values with uncertainties. Both the run-batch and grouped-series recording
paths write this *same* shape, so every fit in a series carries a comparable
verdict into the parameter-trending tables (:doc:`parameter_trending`; the
shared data model is described under :ref:`trending-data-model`). Reading the
χ² verdict *across* a run series is often more informative than any one fit: a
χ²ᵣ that drifts upward through a temperature scan flags a model that fits the
low-temperature spectra but not the high-temperature ones, and a background
parameter that wanders usually signals a changing beam spot rather than sample
physics.

3. Are the error bars themselves honest?
----------------------------------------

The strongest test does not look at χ² at all; it asks whether the *reported
uncertainties* are calibrated. The :ref:`pull diagnostic <pull-diagnostic>`
re-simulates the fitted run many times at matched statistics, refits each copy,
and forms the pull :math:`(\hat\theta - \theta_\mathrm{true})/\sigma_{\hat\theta}`
for every free parameter. For a sound analysis chain the pulls are standard
normal: the **mean** sits at zero (a non-zero mean is a bias) and the **width**
sits at one (a width below one means the errors are too large; above one, too
small). This is the definitive answer to "can I quote this uncertainty?" — and
the one a low χ² alone cannot give, because a fit can land a perfect χ² while
its Hessian errors are mis-scaled.

Use the three together: the χ² band is a fast triage, the recorded verdict
tracks quality across a series, and the pull diagnostic validates the error
bars before you quote them.

See also
--------

* :ref:`fit-statistics` — reading χ², reduced χ², and residuals.
* :doc:`parameter_trending` — the recorded per-fit summary and how it feeds
  trending, including the χ² quality band's confidence level.
* :ref:`pull-diagnostic` — the pull-distribution error-bar validation.
* :doc:`fit_wizard` — choosing a starting model when the χ² says the current
  one is wrong.
