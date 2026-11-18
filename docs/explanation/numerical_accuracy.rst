Numerical accuracy
==================

Numerical evaluation error should be negligible beside the scientific
uncertainty for which a calculation is intended. For revised polarisation
models, Asymmetry targets an absolute error of at most :math:`10^{-5}` in
normalised polarisation, before amplitude and baseline are applied. This
is an evaluation target: it does not specify fit convergence, parameter
uncertainty, statistical coverage, or the adequacy of a physical model.
It does not certify every unchanged model.

A stable numerical method may replace another method for the same physical
model. A different physical approximation must be explicitly selected;
numerical difficulty never silently substitutes another model. Where the
selected evaluation cannot be supported, a meaningful diagnostic should
explain the limitation and an explicitly selected alternative may be used.

Accuracy evidence must be independent of the implementation: analytic
limits, converged field averages, published values, or equivalent native
calculations with recorded versions and settings. Grid or quadrature
refinement must establish reference precision before agreement is claimed.
Checks include interpolation, limiting regimes, dimensionless scaling,
nearby parameter perturbations, and fit recovery where relevant. A model
fitted to its own output alone cannot establish numerical correctness.

Exceptions require independently validated evidence in the owning model's
documentation and tests. Record the domain, measured error, reason for the
exception, and consequences for scientific interpretation. For example,
:ref:`Gaussian-broadened KT <fit-gaussian-broadened-kt>` retains a finite
width quadrature whose error can exceed the static kernel's target for
broad distributions and long windows. Its limitation is distinct from the
accuracy of each static kernel evaluation.

Other quantities require metrics appropriate to their units and scales:
counts, rates, transforms, corrections, and estimated uncertainties do not
share a universal absolute epsilon. State the metric, supported domain,
independent reference precision, and scientific uncertainty that make a
numerical tolerance meaningful.
