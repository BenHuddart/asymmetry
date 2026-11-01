.. _assessing-a-fit:

Assessing a Fit
===============

A converged fit is not yet a trustworthy one. Three checks, in increasing
strength, tell you whether to believe a result — and they answer different
questions. This page is the entry point that ties them together; each step
links to where it is documented and driven in full.

1. Is the χ² in its band?
-------------------------

For independent Gaussian observations with fixed, correctly stated errors and a
linear model, the minimized :math:`\chi^2` has :math:`\nu=N-N_\mathrm{free}`
degrees of freedom. Nonlinear or bounded fits use this law approximately;
the tooltip states "χ² laws are approximate for nonlinear or bounded fits."
Asymmetry reports a two-sided band at confidence :math:`R` (default 0.999):

* **good** — the reduced statistic lies inside the band;
* **poor** — it lies in the upper tail;
* **overdone** — it lies in the lower tail, often indicating overestimated
  errors or excess model flexibility.

A Poisson fit reports **Poisson deviance/ν**, not a Gaussian χ² statistic.
Its likelihood-ratio distribution approaches a χ² law at large expected
counts [pdg_statistics]_. As a display policy, a χ² quality verdict is shown
only when every fitted expected bin count is at least five. This threshold
does not guarantee an exact χ² law. For coupled fits, the minimum includes
all members of the minimized objective. With a sparse fitted bin, the tooltip
says "Poisson deviance has no χ² quality verdict when any fitted expected count
is below 5." If expected counts are unknown, it says "Poisson expected counts
were not recorded; no χ² quality verdict is available." The deviance and
convergence status remain visible. **Fit ✓** denotes convergence, independently
of whether a quality verdict is available.

See :ref:`fit-statistics` for residual checks. Unit weights or scatter-estimated
errors do not supply an independent Gaussian quality test; see
:doc:`parameter_trending`. Coherent residual structure still signals a missing
component even when the scalar statistic looks acceptable.

2. Is the objective recorded and consistent across the series?
--------------------------------------------------------------

Fit summaries record the objective (``gaussian`` or ``poisson``), total and
reduced statistic, fitted parameters and uncertainties. Poisson summaries also
record the minimum fitted expected count needed by the display policy. A
missing quality verdict carries a ``quality_reason``. Run-batch and grouped
series use the same summary format; comparisons require the same likelihood
and data. A mixed-objective series is labelled "Mixed fit objectives." rather
than assigned one averaged statistic.

An unknown saved objective is explicitly ``null``: it has no χ² verdict and
is labelled "Fit objective was not recorded." Known Gaussian producers retain
their objective when loaded. A likelihood cannot be inferred from a numeric
field named ``chi_squared`` alone. See :doc:`project_files` for saved results.

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

Before the full pull diagnostic, a quick read on a *single* parameter's error is
its :ref:`MINOS asymmetric interval <minos-asymmetric-errors>`: when the upper
and lower excursions differ markedly, the symmetric Hessian error is already
suspect — the likelihood is skewed or the parameter is against a bound — and the
pull diagnostic is worth running.

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

Model-selection criteria
-------------------------

Information-criterion comparisons require the same data and likelihood.
AICc is unavailable when its sample-size correction is undefined; a smaller
AIC value cannot replace it in an AICc ranking. See :doc:`fit_wizard` for
formulas, availability and assumptions, and :doc:`global_fit_wizard` for
series totals and partition BIC.

.. [pdg_statistics] G. Cowan and L. Lista, "Statistics", Particle Data Group
   (2025), sections 40.2 and 40.3.3.1,
   `Statistics review <https://pdg.lbl.gov/2025/reviews/rpp2025-rev-statistics.pdf>`_.
