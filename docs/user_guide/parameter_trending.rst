Parameter Trending
==================

.. image:: /_generated/screenshots/parameter_trending_mgb2.png
   :alt: σ(T) plot of synthetic MgB₂ data with the two-gap SC fit overlaid
   :width: 100%

*Synthetic MgB₂ σ(T) data (Tc = 36 K) with the SC_TwoGap_SS parametric*
*model fit overlaid (small and large gap ratios 1.1 and 2.3 with weight*
*0.55, the canonical MgB₂ alpha-model decomposition; Niedermayer et al.*
*Phys. Rev. B 65, 094512, 2002). The σ(T) → λ(T) inversion that follows*
*from this fit is discussed in* :doc:`sc_penetration_depth`.

Parameter trending is the second stage of any temperature or field scan
analysis. Once a series of runs has been individually (or globally) fitted
in the time domain, the extracted parameters themselves carry the
physically interesting information: an order parameter as a function of
temperature, a relaxation rate as a function of field, a precession
frequency tracking magnetic ordering. The parameter-trending framework
fits those *derived* curves with parametric models — power laws, Arrhenius
expressions, Redfield denominators, superconducting gap models, transport
forms — and reports the resulting physical quantities (critical
temperature, activation energy, gap ratios, dimensionality of motion)
with uncertainties.

Frequency-domain global fits use the same panel.  Peak centres and widths from
Fourier spectra appear as ``nu0`` and ``fwhm`` in MHz, with derived field
equivalents ``B0`` and ``Bwid`` for plotting or fitting spectral shifts and
broadening directly against field, temperature, or run number.

Representation-Aware Trending
------------------------------

The Fit Parameters panel automatically tracks which *representation* is active
in the central workspace.  The **Showing:** row at the top of the panel
indicates the current representation — ``F-B Asymmetry``, ``Detector Groups``,
``FFT``, or ``MaxEnt`` — and the panel's series buttons below show only the fit
series that belong to that representation.

Switching the central workspace view (e.g. from **F-B asymmetry** to
**Individual Groups**) refreshes the panel instantly to show the grouped-fit
series instead of the asymmetry ones.  Switching back restores the asymmetry
series.  Each representation maintains its own independent set of series, x-axis
selection, and derived composite parameters, so there is no risk of cross-
contamination between a time-domain and a frequency-domain analysis.

When a series button is selected, the **Data Browser** highlights (in amber) the
runs that belong to that series, making it easy to trace which runs contributed
to the current trend.

Typical use cases include extracting the ordering temperature :math:`T_c` and
the critical exponent :math:`\beta` of a magnetic order parameter by fitting a
spontaneous precession frequency :math:`\nu(T)` (or internal field) from a ZF
series with an ``OrderParameter`` form (see `Magnetic Order Parameter`_ below
and :doc:`workflows/temperature_scan_magnetism`); locating a critical
temperature where a relaxation rate *diverges* with a ``CriticalDivergence``
form; inverting the TF
second-moment :math:`\sigma(T)` into a London penetration depth using one
of the ``SC_*`` gap models (:doc:`workflows/superconductor_penetration_depth`
and :doc:`sc_penetration_depth`); fitting an Arrhenius form to
:math:`\lambda(T)` across a temperature scan for motional narrowing;
and field sweeps in either LF or TF geometry, where the trending panel
handles parameter-vs-:math:`B` identically to parameter-vs-:math:`T`. The
GUI panel is the right tool for interactive exploration and quick model
selection; exporting to a Python script (the snippets below) is the right
tool for reproducible, publication-grade analyses and for unusual
parametric models that are not in the built-in registry.

Trending One Parameter Against Another
--------------------------------------

The x-axis need not be a run-level quantity. Below the fixed ``Auto`` / ``B`` /
``T`` / ``Run`` entries, the **X axis** selector lists every fitted parameter
in the active series, so any parameter can be trended against any other — for
example a relaxation rate :math:`\lambda` against a precession frequency
:math:`\nu`, both extracted per run. Internally the choice is the key
``param:<name>``; the core fit functions are x-agnostic, so no API change is
needed — pass the chosen parameter's values as ``x``:

.. code-block:: python

   result = fit_parameter_model(nu_values, lambda_values, lambda_errors,
                                model, params)

When the x-axis is a fitted parameter the field/temperature *scope* no longer
applies (the abscissa is no longer a sample condition), so the component picker
degrades to the **common** basis set — the same components offered for a
run-index x. Field- and temperature-specific forms (penetration-depth gap
models, order parameters) are hidden, because they describe a dependence on a
physical control variable, not on another fitted quantity.

*When to use this.* Reach for parameter-vs-parameter trending when you suspect
a *relationship* between two fitted quantities rather than a dependence on the
external control — correlating a rate with an amplitude across a series, or
plotting an internal field against a measured frequency to check a linear
gyromagnetic relation. For the ordinary "property versus temperature or field"
analysis, keep the ``B`` / ``T`` axes.

Accounting for x Uncertainty
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the abscissa is a fitted parameter it carries its own per-point
uncertainty, which an ordinary least-squares fit (x exact) ignores. Tick
**Account for x uncertainty** in the model-fit dialog — offered only for a
parameter-vs-parameter x — to weight the fit by the *effective variance*

.. math::

   \sigma^2_{\mathrm{eff},i} = \sigma_{y,i}^2
       + \left(\left.\frac{\partial f}{\partial x}\right|_{x_i}\right)^2
         \sigma_{x,i}^2 ,

the first-order errors-in-variables treatment of Orear and York. The slope is
evaluated by a central finite difference, so it needs no analytic derivative,
and because the minimiser re-evaluates the weighting at every step it is
self-consistent at the minimum with no outer iteration. With
:math:`\sigma_x = 0` it reduces exactly to ordinary least squares, so the
toggle is off by default and never changes a field/temperature/run fit.
Programmatically, pass ``xerr``:

.. code-block:: python

   result = fit_parameter_model(nu_values, lambda_values, lambda_errors,
                                model, params, xerr=nu_errors)

The trend plot draws horizontal error bars from :math:`\sigma_x` whenever the
x-axis is a fitted parameter, independent of the toggle, so the spread is
visible even when the fit treats x as exact.

*When to use this.* Switch it on when the x-parameter's relative error is not
small compared with the y-error divided by the local slope; otherwise the
correction is negligible and the simpler exact-x fit suffices. The estimator is
a first-order approximation: it inflates the parameter errors (correctly) and
shifts the estimates slightly — the intended errors-in-variables behaviour.

.. note::

   WiMDA's Model layer treats the x-column as exact and has no x-error concept;
   the effective-variance option is an Asymmetry addition for physical
   correctness in parameter-vs-parameter fits. Total least squares (orthogonal
   distance regression) was considered and rejected: it offers no box
   constraints, and Asymmetry's parameter models rely on bounds.

Available Basis Components
--------------------------

.. code-block:: python

   from asymmetry.core.fitting import component_names_for_x

   print(component_names_for_x("field"))
   print(component_names_for_x("temperature"))

Build a Parameter Composite Model
---------------------------------

In the GUI parameter-trending workflow, the **Edit Model...** action opens the
same expression-oriented builder used for time-domain composite functions.
Each basis model is inserted as a single function token, then combined with
``+``, ``-``, ``*``, ``/``, the quadrature combinator ``⊕`` (below), and
parentheses. The **Info** button reports the documentation for the currently
selected basis model, including superconducting gap-model details when relevant.

The builder validates the expression in real time and shows the expanded
``y(x)`` preview before you accept it.

.. code-block:: python

   from asymmetry.core.fitting import ParameterCompositeModel

   model = ParameterCompositeModel(
       component_names=["DiffusionLF_2D", "Redfield", "Lambda_bg"],
       operators=["+", "+"],
   )
   print(model.formula_string())

Grouped expressions are also supported programmatically and in the GUI, for
example ``Linear + (Arrhenius * Constant)``.

Quadrature Combinator (``⊕``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The parameter-vs-x builder adds one operator beyond ordinary arithmetic: the
*quadrature combinator*

.. math::

   f \oplus g = \sqrt{f^2 + g^2},

which composes two components as the square root of the sum of their squares.
It is the natural composition rule for width-like quantities — relaxation
rates, linewidths, second moments — where independent broadening channels add
in quadrature rather than linearly. ``⊕`` is binary, commutative, and
associative (``a ⊕ b ⊕ c`` :math:`= \sqrt{a^2 + b^2 + c^2}`); it binds at the
same precedence as ``+`` and ``-``, so ``a ⊕ b * c`` evaluates the product
first and ``a ⊕ b + c`` reads as :math:`\sqrt{a^2 + b^2} + c`. Use parentheses
to override the default grouping.

.. code-block:: python

   from asymmetry.core.fitting import ParameterCompositeModel

   model = ParameterCompositeModel.from_expression("PowerLaw ⊕ Constant")
   print(model.formula_string())

The identity ``PowerLaw ⊕ Constant`` :math:`\equiv` ``PowerLawQuadBG`` (with the
power law's additive constant set to zero) is exact — the operator generalises
the fixed :ref:`quadrature-background <quadrature-background>` component to any
pair of basis models.

*When to use this.* Reach for ``⊕`` whenever a fitted quantity is the quadrature
sum of two contributions — most often a signal of interest riding on an
incoherent background floor (a residual linewidth, an instrumental second
moment). When the background is instead a genuine additive offset of the
observable, use ``+`` with a ``Constant``.

.. note::

   ``⊕`` is a parameter-grammar operator only; the time-domain composite
   function builder does not offer it (quadrature of two time-domain muon
   components has no established meaning). In displayed formulae and GLE exports
   the operator is shown as the ``⊕`` glyph rather than expanded to a square
   root. WiMDA has no general quadrature operator — only the fixed
   ``PowerLawQuadBG``-style model — so this is an Asymmetry generalisation.

Single-Series Fit
-----------------

.. code-block:: python

   import numpy as np
   from asymmetry.core.fitting import Parameter, ParameterSet, fit_parameter_model

   field = np.array([20.0, 50.0, 100.0, 200.0, 400.0, 800.0])
   values = np.array([2.7, 2.2, 1.6, 1.1, 0.7, 0.5])
   errors = np.full_like(values, 0.1)

   params = ParameterSet([
       Parameter("A", value=3.0, min=0.0),
       Parameter("D_2D", value=0.7, min=0.0),
       Parameter("D_perp", value=0.0, min=0.0),
       Parameter("D", value=10.0, min=0.0),
       Parameter("nu", value=10.0, min=0.0),
       Parameter("m", value=2.0, min=0.0),
       Parameter("lambda_BG", value=0.0, min=0.0),
   ])

   result = fit_parameter_model(field, values, errors, model, params)
   print(result.success, result.reduced_chi_squared)

Weighting and Error Modes
-------------------------

By default a model fit weights each point by the propagated error of the
trended parameter (**Column** mode). The **Errors** selector in the model-fit
dialog (and the ``error_mode`` argument of ``fit_parameter_model``) offers
four alternatives for when those errors are absent or untrustworthy:

- **Percent of y** — :math:`\sigma_i = (p/100)\,|y_i|`. Points with
  :math:`y_i = 0` carry no error information and are excluded from the fit.
- **Absolute** — one constant :math:`\sigma` for every point.
- **None** — unit weights. The fitted parameters are the unweighted
  least-squares solution; :math:`\chi^2` loses its absolute meaning.
- **Estimate from scatter** — an unweighted fit whose parameter
  uncertainties are rescaled by :math:`\sqrt{\chi^2/\nu}` afterwards. This
  is the standard way to quote errors when the only information about the
  noise is the scatter of the points themselves. It is exactly the
  converged limit of WiMDA's *Estimate* mode (which rescales its constant
  :math:`\sigma` by :math:`\sqrt{\chi^2_r}` after each fit until
  :math:`\chi^2_r = 1`): because a uniform error rescale never moves the
  minimum, the iteration lands on the same answer in one step.

In Column mode (only), errors are floored at half the median positive error
before fitting, so a single near-zero propagated error cannot dominate the
fit; explicit Percent/Absolute choices are honoured verbatim. With unit
weights or scatter-estimated errors, :math:`\chi^2_r` is forced toward 1 by
construction and carries no goodness-of-fit information — the quality
verdict (below) is suppressed in those modes.

.. code-block:: python

   result = fit_parameter_model(
       field, values, None, model, params, error_mode="scatter"
   )

Fit Windows (Union Multi-Range)
-------------------------------

A fit range may be restricted to a union of (min, max) windows: a point
enters the fit if it falls in *any* window, and a single model is fitted
across all of them. Use **+ Window** on a range row in the model-fit dialog,
or pass ``windows`` to ``fit_parameter_model``. The fitted curve is drawn
continuously through the excluded gaps.

The canonical use is excluding a critical region: a relaxation rate
:math:`\lambda(T)` diverging at :math:`T_c` is well described by
``CriticalDivergence`` on both sides of the transition, but the points
nearest :math:`T_c` are dominated by physics the power law does not capture
(and by the singularity itself). Fitting the union
:math:`[T_{\min}, T_c - \delta] \cup [T_c + \delta, T_{\max}]` keeps one
consistent model with the contaminated region excluded:

.. code-block:: python

   result = fit_parameter_model(
       temperature, rate, rate_errors, model, params,
       windows=[(40.0, 64.0), (74.0, 100.0)],
   )

This differs from adding a second *range* in the dialog: separate ranges fit
independent models (piecewise modelling), whereas windows OR-combine into
the mask of one model.

χ² Quality Verdict
------------------

After a successful fit the dialog reports a quality verdict alongside
:math:`\chi^2_r`. For a correct model with correct error bars, :math:`\chi^2`
follows the chi-squared distribution with :math:`\nu = N - N_{\mathrm{free}}`
degrees of freedom, so at 95 % confidence a good fit's :math:`\chi^2_r`
falls inside a band around 1 that tightens as :math:`\nu` grows (for
:math:`\nu = 10` the band is 0.32–2.05; for :math:`\nu = 50`, 0.65–1.43).
The verdict is two-sided:

- **good** — :math:`\chi^2_r` inside the band.
- **poor** — above the band: the model misses real structure, or the errors
  are underestimated.
- **overdone** — below the band: the fit reproduces the data *better* than
  the errors allow, which usually means overestimated errors or too many
  free parameters soaking up noise.

The verdict assumes real (Column-mode) errors and is suppressed for
unit-weight and scatter-estimated fits. Programmatic access is via
``asymmetry.core.fitting.assess_fit_quality(chi_squared, dof)``, which
returns the verdict and the band.

Cross-Group Fitting
-------------------

For jointly fitting multiple groups with shared and local parameters, use
``global_fit_parameter_model`` with ``ParameterGroupData`` entries.

.. code-block:: python

   from asymmetry.core.fitting.parameter_models import (
       ParameterGroupData,
       global_fit_parameter_model,
   )

   groups = [
       ParameterGroupData(
           group_id="g1",
           group_name="Sample A",
           x=field,
           y=values,
           yerr=errors,
           group_variable_value=5.0,
       ),
       ParameterGroupData(
           group_id="g2",
           group_name="Sample B",
           x=field,
           y=values * 1.1,
           yerr=errors,
           group_variable_value=15.0,
       ),
   ]

   result = global_fit_parameter_model(
       groups=groups,
       model=model,
       global_params=["D_2D", "D", "nu", "m"],
       local_params=["A", "lambda_BG"],
       fixed_params={"D_perp": 0.0},
   )
   print(result.success)

The cross-group fit honours the same **error modes** and **fit windows** as a
single-series fit. The cross-group dialog now shows the **Errors** selector and
**+ Window** controls, and ``global_fit_parameter_model`` accepts the matching
``error_mode``, ``error_value``, and ``windows`` arguments; the chosen mode and
window union apply to every group. In *Estimate from scatter* mode the
:math:`\sqrt{\chi^2/\nu}` rescale is applied to **both** the shared global
uncertainties and every group's local uncertainties.

.. code-block:: python

   result = global_fit_parameter_model(
       groups=groups, model=model,
       global_params=["m"], local_params=["b"], fixed_params={},
       error_mode="scatter", windows=[(0.0, 40.0), (60.0, 100.0)],
   )

When the cross-group abscissa is itself a fitted parameter, the same
**effective-variance** x-uncertainty treatment used for single-series fits
(`Accounting for x Uncertainty`_) is available here too. Tick **Account for x
uncertainty** in the cross-group dialog, or pass per-group :math:`\sigma_x`
arrays as ``xerr`` (keyed by ``group_id``); each group's points are weighted by
:math:`\sigma_{y}^2 + (\partial f/\partial x)^2\,\sigma_x^2` with the *same*
estimator as the single-series path. It is off by default, reduces exactly to
ordinary least squares when :math:`\sigma_x = 0`, and is ignored under the
*None* / *Estimate from scatter* modes (whose unit weights carry no scale to
combine with :math:`\sigma_x`).

.. code-block:: python

   result = global_fit_parameter_model(
       groups=groups, model=model,
       global_params=["m"], local_params=["b"], fixed_params={},
       xerr={"g1": nu_errors_g1, "g2": nu_errors_g2},
   )

Recursive Trending (Model-Fit Results as a Series)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A successful cross-group fit is itself recorded as a new trend series, named
**Model fit: <parameter> vs <x>**, alongside the original fit series in the Fit
Parameters panel. Each selected group contributes one row carrying that group's
*local* parameters (with the shared *global* parameters as constant columns),
indexed by the group's orthogonal coordinate — temperature for a field fit, and
vice versa; a final ``globals`` row carries the shared parameters and
:math:`\chi^2_r`. Because this results series is an ordinary trend series, it
can be trended *again* — fit the per-group local rate against temperature, for
instance — so the analysis recurses naturally. This supersedes WiMDA's separate
second-level *Model Fit Table*, which only stored the rows for inspection.

Re-running the same cross-group fit (same parameter, x-axis, and group set)
*replaces* its results series rather than accumulating duplicates.

**Single-fit ranges.** A single-series fit is recorded the same way: each fit
*range* contributes one row to a **Model fit (single): <parameter> vs <x>**
series, carrying that range's fitted parameters and :math:`\chi^2_r`, indexed by
the centre of the range's fitted x-window (in the trend's own x-units, also kept
as a ``range_center`` column). A multi-range fit — for example one model below a
transition and another above — is therefore immediately trendable: plot a fitted
coefficient against the window centre to see how it drifts across the data.

**Trending shared globals across fits.** Every successful cross-group fit also
appends one row to a persistent **Global summary** series, carrying that fit's
shared *global* parameters, :math:`\chi^2_r`, and a monotonic ``fit_index``.
Because the globals from successive fits accumulate in one series, they can be
trended against each other or against ``fit_index`` — the natural way to follow a
shared parameter as you sweep a model choice, a fixed value, or a group set
across a sequence of joint fits. Re-running a given fit updates its row in place
(keyed by the same parameter / x-axis / group set), so the series records one row
per *distinct* fit. These rows sit off the physical field/temperature axes;
select ``fit_index`` or a global parameter as the x-axis (see
`Trending One Parameter Against Another`_) to trend them.

Magnetic Order Parameter
------------------------

For a second-order magnetic transition, the spontaneous muon precession
frequency, the internal field, or an ordered-moment-like asymmetry follows an
order-parameter temperature dependence that rises continuously from zero at the
ordering temperature :math:`T_c` to a saturated value :math:`y_0` at
:math:`T = 0`. The ``OrderParameter`` basis model captures this with a
generalized two-exponent form:

.. math::

   y(T) = y_0 \left[1 - \left(\frac{T}{T_c}\right)^{\alpha}\right]^{\beta}
   \quad (0 \le T < T_c), \qquad y(T) = 0 \quad (T \ge T_c).

Here :math:`y_0` carries the unit of the trended observable (MHz for
:math:`\nu(T)`, G or T for an internal field, % for an asymmetry), :math:`\beta`
is the critical exponent that dominates the near-:math:`T_c` shape (typically
0.33–0.37 for 3D Heisenberg/Ising magnets, 0.5 for mean field), and the shape
exponent :math:`\alpha` controls the departure from a pure power law away from
:math:`T_c`. Fixing :math:`\alpha = 1` recovers the simple near-:math:`T_c`
power law :math:`y_0 (1 - T/T_c)^{\beta}`. The model is exactly zero above
:math:`T_c`, so include the ordered-phase points and let :math:`T_c` fall inside
the fitted range. Use ``CriticalDivergence`` instead for quantities that
*diverge* at :math:`T_c` (such as a relaxation rate), not for an order
parameter that vanishes there.

.. code-block:: python

   import numpy as np
   from asymmetry.core.fitting import (
       Parameter,
       ParameterSet,
       ParameterCompositeModel,
       fit_parameter_model,
   )

   # Spontaneous precession frequency nu(T) through Tc.
   temperature = np.array([1.5, 20.0, 40.0, 55.0, 62.0, 66.0, 68.0])
   nu = np.array([29.9, 29.1, 26.0, 20.6, 15.4, 9.6, 5.7])
   errors = np.full_like(nu, 0.3)

   model = ParameterCompositeModel(["OrderParameter"], operators=[])
   params = ParameterSet([
       Parameter("y0", value=30.0, min=0.0),
       Parameter("Tc", value=69.0, min=0.0),
       Parameter("beta", value=0.36, min=0.0),
       Parameter("alpha", value=1.0, min=0.0),  # fix to fit the near-Tc power law
   ])

   result = fit_parameter_model(temperature, nu, errors, model, params)
   print(result.success, result.reduced_chi_squared)
   for p in result.parameters:
       print(p.name, p.value, result.uncertainties.get(p.name))

Polynomial Trends
-----------------

``Polynomial`` fits an empirical trend or background up to fifth order,

.. math::

   y(x) = c_0 + c_1 x + c_2 x^2 + c_3 x^3 + c_4 x^4 + c_5 x^5,

with all six coefficients available as fit parameters. Fix the unused
high-order coefficients at 0 to fit lower orders — for a quadratic
background, free :math:`c_0`–:math:`c_2` and fix :math:`c_3`–:math:`c_5`.
Use it when no physical model is available, or as the smooth background
under resonance lineshapes (see the LCR recipe below). Two cautions apply
to any polynomial fit: the coefficients are only meaningful inside the
fitted window (polynomials extrapolate badly), and the order should be the
lowest the residuals support — a freed high-order term that the data does
not constrain will soak up noise and inflate the uncertainties of every
other coefficient.

.. _quadrature-background:

Power Law with Quadrature Background
------------------------------------

``PowerLawQuadBG`` combines a power law with a constant background *in
quadrature*,

.. math::

   y(x) = \sqrt{\left(a\,|x|^{n}\right)^2 + \mathrm{BG}^2},

which is the natural composition rule for width-like quantities —
relaxation rates, linewidths, second moments — where independent broadening
channels add as squares. At small :math:`x` the curve saturates smoothly at
:math:`\mathrm{BG}` instead of falling linearly onto an offset; at large
:math:`x` it approaches the bare power law. Use the plain ``PowerLaw``
(with its additive constant) when the background is a genuine offset of
the observable itself rather than an independent broadening channel.

This fixed component is the special case ``PowerLaw ⊕ Constant`` of the general
quadrature combinator (above); reach for ``⊕`` when either side of the
quadrature sum is a richer model than a bare power law or constant.

Muonium Repolarisation
----------------------

``MuRepolarisation`` measures a muonium hyperfine constant from a
longitudinal-field scan, without resolving any precession. In an applied
field :math:`B` along the initial muon polarization, the only muonium
transition that mixes the muon spin states is suppressed as the field
decouples the muon and electron spins; time-averaging the unresolved fast
oscillation leaves the repolarisation curve

.. math::

   y(B) = a_{\mathrm{Mu}}\,
   \frac{\tfrac{1}{2} + (B/B_0)^2}{1 + (B/B_0)^2} + a_{\mathrm{Dia}},
   \qquad B_0 = \frac{A_{\mathrm{hf}}}{\gamma_e + \gamma_\mu},

rising from half the muonium amplitude at :math:`B = 0` (the other half is
lost to the unobserved oscillation) to the full amplitude once
:math:`B \gg B_0`, on top of a field-independent diamagnetic baseline
:math:`a_{\mathrm{Dia}}`. The component is parameterised directly by the
hyperfine constant :math:`A_{\mathrm{hf}}` (MHz) — the quantity the
experiment is designed to extract — with :math:`B_0` derived internally
from CODATA gyromagnetic ratios. For vacuum muonium
(:math:`A_{\mathrm{hf}} = 4463\;\mathrm{MHz}`), :math:`B_0 \approx 1585` G,
and the curve reaches three quarters of the muonium amplitude at exactly
:math:`B = B_0`.

Fit it to an initial-asymmetry or integral-asymmetry LF scan (built with
the integral-asymmetry observable, :doc:`/user_guide/alc_mode`, x-axis
in G). It is the standard method when the hyperfine coupling is too large
for the precession to be resolved directly. The model assumes an isotropic
(vacuum-like) hyperfine interaction observed in time average: anisotropic
muonium, rapid chemical reaction, or spin exchange distort the curve, and
any missing fraction appears as a reduced :math:`a_{\mathrm{Mu}}`.

.. code-block:: python

   import numpy as np
   from asymmetry.core.fitting import (
       Parameter,
       ParameterSet,
       ParameterCompositeModel,
       fit_parameter_model,
   )

   field_G = np.array([0.0, 50.0, 150.0, 400.0, 1000.0, 2500.0, 6000.0, 15000.0])
   asym = np.array([12.6, 12.7, 13.1, 14.2, 16.4, 19.0, 21.0, 21.9])
   errors = np.full_like(asym, 0.2)

   model = ParameterCompositeModel(["MuRepolarisation"])
   params = ParameterSet([
       Parameter("a_Mu", value=10.0),
       Parameter("A_hf", value=4000.0, min=1.0),
       Parameter("a_Dia", value=5.0),
   ])

   result = fit_parameter_model(field_G, asym, errors, model, params)
   print({p.name: p.value for p in result.parameters})

References
~~~~~~~~~~

1. S. J. Blundell, R. De Renzi, T. Lancaster, and F. L. Pratt, *Muon
   Spectroscopy: An Introduction* (Oxford University Press, Oxford, 2022).
2. B. D. Patterson, Rev. Mod. Phys. **60**, 69 (1988).
3. J. Orear, Am. J. Phys. **50**, 912 (1982).
4. D. York, N. M. Evensen, M. L. Martínez, and J. De Basabe Delgado, Am. J.
   Phys. **72**, 367 (2004).

Migrating WiMDA Model Functions
-------------------------------

Every function in WiMDA's Model-layer library ("Standard fit models") has a
direct counterpart or a composite recipe. Parameter names map as follows:

.. list-table::
   :header-rows: 1
   :widths: 28 36 36

   * - WiMDA function
     - Asymmetry equivalent
     - Notes
   * - Polynomial fit up to fifth order
     - ``Polynomial``
     - Identical (coefficients c₀–c₅).
   * - Power law
     - ``PowerLaw``
     - Identical on :math:`x > 0`; Asymmetry evaluates :math:`|x|^n` so
       negative-:math:`x` points cannot fault the fit.
   * - Power law (BG quad)
     - ``PowerLawQuadBG``
     - Identical.
   * - 2 Lorentzians + cubic BG
     - ``LorentzianLCR + LorentzianLCR + Polynomial``
     - Peak terms map exactly (Ampl → f, Pos → B₀, Wid → B_wid; fix
       :math:`c_4, c_5` at 0 for the cubic). **Background coefficients do
       not transfer**: WiMDA's cubic is in powers of :math:`(x - \mathrm{Pos}\,1)`,
       Asymmetry's in absolute :math:`x` — refit rather than copying values.
   * - Thermal activation (2 component)
     - ``Arrhenius + Arrhenius``
     - WiMDA uses eV, Asymmetry meV: :math:`E_a[\mathrm{meV}] = 1000
       \times E_a[\mathrm{eV}]`. WiMDA's hard-coded :math:`e/k_B` is
       0.089 % below the CODATA value, so refitted activation energies are
       expected to come out very slightly higher than WiMDA's.
   * - Internal field vs T for ordered magnet
     - ``OrderParameter + Constant``
     - Identical on the physical domain, including the clamp to the
       background value above :math:`T_c` (B₀ → y₀, B_bg → the Constant).
   * - Divergence of relaxation rate
     - ``CriticalDivergence``
     - Identical away from :math:`T = T_c` (Tc → Tc, alpha → ν,
       Min rate → c, scaling → a). Exclude the critical region from the
       fit window rather than relying on the singular point.
   * - Repolarisation of isotropic Mu
     - ``MuRepolarisation``
     - Same curve; WiMDA fits :math:`B_0` in G, Asymmetry fits
       :math:`A_{\mathrm{hf}}` in MHz with
       :math:`B_0 = A_{\mathrm{hf}}/(\gamma_e + \gamma_\mu)` derived
       (:math:`A_{\mathrm{hf}}[\mathrm{MHz}] = 2.816 \times B_0[\mathrm{G}]`).

Composite Parameters in the Fit Parameters Panel
------------------------------------------------

In the GUI Fit Parameters panel, you can define a derived parameter from
existing fitted parameters using **Create Composite Parameter**.

After selecting a derived parameter in the Y-parameter list, you can also use
**Edit Selected Composite** and **Remove Selected Composite** to manage saved
definitions.

The expression builder supports:

- Arithmetic: ``+``, ``-``, ``*``, ``/``, ``^``
- Parentheses and numeric constants
- Built-in constants: ``pi``, ``e``
- Built-in functions: ``sin``, ``cos``, ``tan``, ``asin``, ``acos``,
  ``atan``, ``sinh``, ``cosh``, ``tanh``, ``exp``, ``log``, ``log10``,
  ``sqrt``, ``abs``

Expressions are parsed safely (no arbitrary code execution) and validated in
real time against available fitted parameter names.

Derived parameters are integrated into the same workflow as ordinary fit
parameters:

- They appear in the Y-parameter selector and fitted-parameter table.
- They can be plotted and used with parameter-model fitting.
- They are saved/restored with project state.
- They are recomputed automatically when source fit parameters change.

Uncertainties are propagated using first-order error propagation:

.. math::

    \sigma_f^2 = J\,\Sigma\,J^T

where :math:`J` is the gradient of the expression with respect to source
parameters and :math:`\Sigma` is the parameter covariance matrix when
available (falling back to diagonal variances otherwise).

Runnable Example
----------------

See ``examples/parameter_trending.py`` for a complete executable script.

Superconducting Gap Models
--------------------------

For TF-muSR superconducting penetration-depth analysis via
temperature-dependent :math:`\sigma(T)`, see
:doc:`sc_penetration_depth`.
