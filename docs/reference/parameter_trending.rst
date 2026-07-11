Parameter trending
==================

.. image:: /_generated/screenshots/parameter_trending_mgb2.png
   :alt: Fit Parameters trending panel showing MgB₂ σ(T) points with the fitted two-gap SC_TwoGap_SS curve
   :width: 100%

*The Fit Parameters trending panel with synthetic MgB₂ σ(T) data (Tc = 36 K)*
*loaded as the* ``σ(T) — MgB₂`` *series and* ``σ (µs⁻¹)`` *selected on the*
*y-axis. The smooth trend curve is a* ``SC_TwoGap_SS`` *two-gap fit (Model Fit\**
*button), an MgB₂ alpha-model decomposition (small and large gap ratios;*
*Niedermayer et al. Phys. Rev. B 65, 094512, 2002); the fit recovers*
*σ₀ ≈ 1.24 µs⁻¹ and Tc ≈ 35.8 K. The σ(T) → λ(T) inversion that follows from*
*this fit is discussed in* :doc:`sc_penetration_depth`.

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

.. _trending-data-model:

The trending data model
-----------------------

Everything trendable is a **FitSeries**. Whatever produced the numbers — a
batch of single-run fits, a grouped multi-detector fit, a global cross-fit, a
frequency-domain peak fit — the result lands in one container type, stored in
the project's batch collection. A FitSeries holds the per-run fit summaries
(:ref:`assessing-a-fit`: success, χ², reduced χ², parameter values and
uncertainties) keyed by run, and that is exactly the structure the
parameter-trending tools consume. There is no separate "trend object": to trend
a quantity is to read a parameter out of a FitSeries across its runs.

Two distinct paths *write* into this one container, by design:

* **Results recursion** records the per-group rows of a model fit — each fitted
  group's parameters become a row, so the series carries a parameter-versus-run
  (or versus-group) curve ready to trend.
* **The global summary** accumulates one row per fit from a cross-group or
  global fit — the shared (global) parameters of each fit, gathered across a
  run series.

Both write FitSeries instances into the same collection; they differ only in
*what* each row represents (per-group values versus per-fit global values), not
in the container. This is why a grouped fit and a global fit both appear as
selectable series in the panel and trend through identical machinery.

The **Global Parameter Fit window** is a *view onto persisted studies*, not a
third container. Each cross-group fit is a named **study** (a
``GlobalFitStudy``) kept in a project-level registry, so several can sit side by
side — rename, duplicate, or delete them from the window's studies sidebar, and
select one to display it. A study records the group snapshot it was fit against
and detects when the live trend data has drifted from it (a *stale* badge, with
a **Refit** action). The window's *decorations* (the per-parameter local model
fits and free-floating plot annotations you add in the window) are stored inside
the FitSeries it is viewing — keyed by batch id, in the series' ``extra`` —
rather than under a separate project key. They
therefore travel with the data they annotate and reappear whenever the window
shows that fit again, including across a project save/reload, and can no longer
be orphaned when the fit is re-run. View *preferences* (log axes, plot mode,
show-components) are not decorations: they are remembered per window, not per
fit. Legacy projects that stored decorations under the old window-state key
still load, and migrate to the new home on the next save.

.. _group-bound-series-staleness:

Group-bound series and staleness
---------------------------------

A run-membered FitSeries produced by a batch or global fit belongs to a Data
Browser group: either the group the fit was launched from (via **Fit this
group…**), or a group auto-created from the ad-hoc run selection. The group
is the batch vehicle — its membership drives the series, not the other way
round. A series' *effective* membership is derived live as the owning group's
members minus any runs the series itself has excluded (untick a run in the
Batch tab's member list without removing it from the group), so the same
group can carry more than one analysis — a field scan fit with one model and,
separately, a subset of it re-fit with another — without duplicating the run
collection.

Because membership is live-derived, adding a run to the group or excluding
one from the series does not retroactively change already-recorded results:
those stay a snapshot of what was actually fit. When the group's current
effective membership no longer matches that snapshot, the series is *stale*,
and its button in the panel's series row grows a **⚠** with the tooltip
"Membership changed since last fit — re-run to refresh."; re-running the
series (from the Batch tab, still bound to the same group) refreshes the
snapshot and clears the marker. Re-running a group-bound series always
replaces its previous results in place rather than accumulating a second
series for the same group and model. A series with no owning group (a legacy
analysis, or one kept as a standalone record after its group was deleted) is
never stale — its membership is a fixed snapshot, exactly as before this
distinction existed. See :doc:`project_files` for the persisted
``group_id`` / ``excluded_run_numbers`` / ``last_fitted_members`` fields and
:doc:`gui_usage` for the Data Browser and Batch tab controls that drive them.

.. _trend-abscissa-coordinate:

How the T / B abscissa is sourced
---------------------------------

When a series is recorded, each member run's temperature and field are stamped
into its summary so the trend plots every point at the right coordinate even
after the dataset leaves the browser or the project is saved and reopened. That
coordinate is the value the **Data Browser displays** for the run — not, in
general, the raw header scalar.

This matters for *parked-setpoint* series. If a scan was run with the cryostat
setpoint left at a single value (e.g. every run logged as 1 K) while the true
sample temperature drifted, the header temperature is identical for every run
and a temperature trend would collapse onto one abscissa point. Enabling
**Options → Use temperature from log** switches the T column to the logged
sample temperature, and the trend X-axis and TSV export now follow: each point
is placed at its logged temperature, so a :math:`T`-trend (or Arrhenius fit)
becomes possible. The field axis behaves the same way under the analogous
*use field from log* option, and per-dataset overrides (set from a single run's
Get Info) are honoured too.

With both toggles off (the default) the abscissa is the header setpoint exactly
as before. A run with no recorded temperature or field stays *off that axis*
(plotted as NaN), never planted at 0. Series recorded before this behaviour
existed re-plot against the browser's currently displayed value when their
runs are still loaded.

Beyond temperature, field, and run number, the **X:** selector also offers any
fitted parameter (parameter-vs-parameter trending) and any **custom data-browser
column** (:ref:`logbook-columns`). Custom columns hold free-form text, so when
one is the x-axis each value is coerced to a number and runs whose value is empty
or non-numeric are dropped, with a note reporting how many were skipped.

.. _trend-axis-transforms:

Axis transforms
---------------

Some of the most common µSR presentations are *linearisations* of a curved
trend: the **Redfield** analysis plots :math:`1/\lambda` against
:math:`(\mu_0 H)^2` so a straight line's slope and intercept give the
fluctuation rate and field width; the **Arrhenius** analysis plots
:math:`\ln\lambda` against :math:`1/T` so the slope is an activation energy.
The collapsible **Axis transforms** section (below the Y-parameter list)
applies such a transform to either axis independently.

.. image:: /_generated/screenshots/parameter_trending_redfield.png
   :alt: The trending panel showing a Redfield linearisation — 1/λ versus B² with a straight-line Linear fit
   :width: 100%

*A Redfield linearisation of a longitudinal-field* :math:`\lambda(B)` *scan:*
*the Y axis transformed to* ``1/x  (reciprocal)`` *and the X axis to*
``x²  (square)`` *turn the three-regime* :math:`\lambda(B)` *falloff into a*
*straight line, and a* ``Linear`` *model fit on the transformed plateau gives*
*the Redfield slope and intercept. The high-field saturated point is excluded*
*from the trend, so it sits off the line.*

Each axis has its own chooser — **X:** and **Y:** — offering ``None``,
``1/x  (reciprocal)``, ``x²  (square)``, ``ln x``, ``log₁₀ x``, ``√x`` and
``Custom…``. Choosing ``Custom…`` opens a small **Custom X transform** /
**Custom Y transform** dialog with one field, *Expression in x:* (placeholder
``e.g. 1000/x``); the expression is validated live and previewed on a
representative data value. The accepted expression then labels the combo item
itself, and the last-used custom expression is remembered per axis.

The transform is applied at the point where the panel assembles its data, so it
governs the plotted points, the propagated error bars **and the trend fit**:
fitting a ``Linear`` model with the axes transformed to :math:`1/\lambda`
versus :math:`(\mu_0 H)^2` *is* the Redfield line, and its slope/intercept are
read straight from the :ref:`model-fit dialog <trend-model-fit-dialog>`. A point
whose transform is undefined (``1/0``, ``ln`` of a non-positive value) is dropped
like any other NaN. Changing a transform marks an existing trend fit for re-fit,
since its curve lives in the previous coordinate.

A transform is distinct from the **log** axis-scale checkbox next to the **X:**
selector (and the per-parameter **log** checkbox in the Y-parameter list): those
change the axis *tick spacing* while leaving the numbers alone, whereas the
transform changes the plotted values (which is what a straight-line Arrhenius fit
needs). To keep the two from compounding, selecting ``ln x`` / ``log₁₀ x`` on an
axis disables that axis's ``log`` checkbox until the transform is cleared.

.. _trend-series-overlay:

Overlaying several series
-------------------------

The series buttons above the plot are multi-select: **Shift+click** a second
series to overlay it on the first (the same selection also arms a joint
`Cross-Group Fitting`_). Overlaid series are
distinguished by colour, with a legend of their names — the natural way to
compare, say, :math:`\sigma(T)` measured at two applied fields, or the same
observable across two samples. When more than one parameter is also selected,
each parameter takes a distinct marker shape so colour stays free to encode the
series; the twin-axis layout is used only for a single series. Model-fit trend
curves are drawn for the active (last-clicked) series, which retains ownership of
the parameter table, composites, and the model-fit controls.

Representation-aware trending
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
spontaneous precession frequency :math:`\nu(T)` (or internal field) from a
zero-field (ZF) series with an ``OrderParameter`` form (see `Magnetic Order Parameter`_ below
and :doc:`/workflows/temperature_scan_magnetism`); locating a critical
temperature where a relaxation rate *diverges* with a ``CriticalDivergence``
form; inverting the transverse-field (TF)
second-moment :math:`\sigma(T)` into a London penetration depth using one
of the ``SC_*`` gap models (:doc:`/workflows/superconductor_penetration_depth`
and :doc:`sc_penetration_depth`); fitting an Arrhenius form to
:math:`\lambda(T)` across a temperature scan for motional narrowing;
and field sweeps in either longitudinal-field (LF) or TF geometry, where the trending panel
handles parameter-vs-:math:`B` identically to parameter-vs-:math:`T`. The
GUI panel is the right tool for interactive exploration and quick model
selection; exporting to a Python script (the snippets below) is the right
tool for reproducible, publication-grade analyses and for unusual
parametric models that are not in the built-in registry.

.. _trend-model-fit-dialog:

Fitting a trend model
---------------------

Selecting a model, seeding it sensibly, and deciding which points it should see
is the slow, error-prone part of trend analysis: a power law seeded far from the
data walks into a bad minimum, and a critical divergence contaminated by the very
region it cannot describe reports a meaningless :math:`T_c`. The **model-fit
dialog** exists to make those decisions visible while you make them. It fits one
parametric model through the selected "parameter versus X" points and shows the
candidate curve against the data as you set it up, so a converging fit is
apparent before you commit to it.

Open it from the **Fit Parameters** panel: choose the abscissa in the **X:**
selector (temperature, field, run number, angle, a fitted parameter, or a custom
logbook column), then click the **Model Fit** button beside the trended quantity
in the Y-parameter list. (The button reads **Model Fit\*** once a fit is active,
and becomes a joint **Global fit** action when two or more group series are
selected — see `Cross-Group Fitting`_.) The panel hands the dialog the included
trend points for that parameter — a point excluded from the trend (via its
context menu or the include checkbox) stays visible on the plot but does not pull
the fit — together with their propagated errors and, when the abscissa is a
fitted parameter, its per-point x-uncertainty.

.. image:: /_generated/screenshots/trend_model_fit_dialog.png
   :alt: The trend model-fit dialog fitting an OrderParameter form to an EuO ν(T) trend
   :width: 100%

*The model-fit dialog fitting the* ``OrderParameter`` *form to a synthetic EuO*
*spontaneous-precession-frequency trend* :math:`\nu(T)` *through the Curie point.*
*The left pane carries the controls — error mode, the range card, the fit-region*
*editor, the seed table and the fit result; the right pane is the live preview,*
*with the candidate curve tracing the in-range points. The converged fit returns*
:math:`T_c = 69.2(6)` *K and* :math:`\beta = 0.41(8)`, *with a* good *χ² verdict.*

The model catalogue
~~~~~~~~~~~~~~~~~~~~

The dialog offers only the basis models that make sense for the current
abscissa: the pool is drawn from ``component_names_for_x`` and filtered by each
component's declared scope, so a temperature axis never lists a field-only form
and vice versa. The registry (grouped by the context that offers them) is:

* **Any axis** (``common``) — ``Constant``, ``Linear``, ``Quadratic``
  (:math:`c_0 + c_1 x + c_2 x^2`, the plain parabola — e.g. a steering-curve
  minimum), and the general fifth-order ``Polynomial``; ``PowerLaw`` and its
  quadrature-background variant ``PowerLawQuadBG``; and ``ExponentialDecay``.
* **Temperature axis** — ``Arrhenius`` (thermal activation), ``OrderParameter``
  (a magnetic order parameter vanishing at :math:`T_c`), ``CriticalDivergence``
  (a rate diverging at :math:`T_c`), and the superconducting gap models
  ``SC_SWave``, ``SC_DWave``, the anisotropic and non-monotonic forms
  (``SC_AnisotropicS_Cos4``, ``SC_NonmonotonicD``, ``SC_PWaveAxial``,
  ``SC_ExtendedS``, ``SC_SPlusG``), the phenomenological ``SC_AlphaModel``, the
  two-gap ``SC_TwoGap_SS`` / ``SC_TwoGap_SD``, and their :math:`q`-weighted
  ``_Q`` counterparts. These feed the :math:`\sigma(T) \to \lambda(T)` inversion
  of :doc:`sc_penetration_depth`.
* **Field axis** — degree-fixed baselines ``Cubic`` / ``Quartic`` / ``Quintic``
  / ``Sextic`` (each a fixed-order restriction of ``Polynomial``, for
  resonance backgrounds of increasing curvature); ``Redfield``; the resonance
  lineshapes ``Lorentzian``, ``GaussianLCR``, ``LorentzianLCR``; the muonium
  repolarisation curve ``MuRepolarisation`` and the exact-diagonalisation RF
  resonance ``RFResonanceMuP``; the longitudinal-field diffusion and ballistic
  transport forms ``DiffusionLF_1D/2D/3D`` and ``BallisticLF_1D/2D/3D`` with the
  ``Lambda_bg`` background (:doc:`diffusion_ballistic_lf`); and the Brandt
  vortex-lattice second-moment models ``SC_Brandt_VortexLattice`` and its powder
  average.
* **Angle axis** — ``KnightAnisotropy``, ``AngularCos2``, and
  ``AngularFourier2`` for a Knight shift mapped out by rotating a single
  crystal (:ref:`knight-shift`).

Any of these can be combined into a composite via **Edit Model**, which opens the
same two-panel builder used for time-domain functions (see `Build a Parameter
Composite Model`_ and the quadrature combinator ``⊕``). The dialog picks a
sensible default for a fresh range — ``Linear`` in general, but
``OrderParameter`` automatically when the trended quantity is a magnetic
order-parameter observable (a precession frequency or internal field) versus
temperature, so the common critical-behaviour fit converges out of the box.

The live preview
~~~~~~~~~~~~~~~~

The right-hand pane redraws as you edit: the data points (with error bars scaled
to the current error mode), the in-range points against the greyed-out excluded
ones, and the candidate model curve. The curve is sampled off the GUI thread —
debounced, with stale results discarded — so dragging an edge or retyping a seed
never freezes the dialog. The **Show residuals** toggle adds a
:math:`(\text{data} - \text{model})/\sigma` strip with a :math:`\pm 1\sigma` guide
band beneath the plot. On a narrow window the preview auto-collapses behind a
**Show preview** toggle so the controls stay usable.

The interactive fit region
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every fit is restricted to a **fit region** — the set of x-intervals a point must
fall inside to enter the fit. A range is defined by a **range card** (a colour
swatch matching its span on the plot, a χ²ᵣ status chip, and its compact bounds)
with **Run Fit** and **Edit Model** actions; **Add Range** adds another. The
region is sculpted, not assembled: it starts as one interval spanning the data
and you *carve gaps out of it* with **Exclude region…** (or by right-dragging
across the plot), leaving a list of included intervals shown in the **Selected
range** editor. This is the natural way to exclude a contaminated critical region
— fit ``CriticalDivergence`` on both sides of a transition while dropping the
points nearest :math:`T_c` — while keeping one consistent model across the gap
(the fitted curve is drawn continuously through it). Directly on the preview you
can drag an interval edge, drag on empty space to add a new range, click a span
to select it, and right-drag to exclude a region.

Adding a second *range* is different from excluding a gap: separate ranges fit
*independent* models to different parts of the abscissa (piecewise modelling),
whereas the intervals of one range OR-combine into the mask of a *single* fit.
Internally the two-or-more-interval case is stored as fit windows and the
single-interval case as a plain range, so a fit region that carves down to one
interval collapses back automatically and saved projects are unaffected.

Seeding and running
~~~~~~~~~~~~~~~~~~~

The seed (starting-value) table lists each model parameter with its start value,
bounds and a Fixed toggle; you can also type bounds to constrain the fit.
Starting values are chosen data-aware where it matters: the critical-temperature
components take :math:`T_c` and amplitude seeds derived from the actual x/y data,
so an ``OrderParameter`` default lands close to the transition rather than at a
generic guess. **Guess seeds** re-derives data-aware starts for the selected
range's *free* parameters on demand (fixed parameters are never touched), running
off the GUI thread. The fit itself is a bounded least-squares minimisation via
iminuit, and the dialog runs it from four extra deterministic starts in addition
to your own seed, keeping the run reproducible while making a bad local minimum
much less likely; when a parameter converges hard against a bound, or a
data-aware restart beats your seed, the dialog says so inline.

The **Errors** selector (`Weighting and Error Modes`_) controls how each point is
weighted, and after a successful fit the result box tints green and reports
:math:`\chi^2`, :math:`\chi^2_r`, and the two-sided quality verdict
(`χ² Quality Verdict`_); the same verdict appears as the range card's status
chip.

Where the fit lands
~~~~~~~~~~~~~~~~~~~

Accepting the dialog with **OK** writes the fit back into the panel: the fitted
curve overlays the trend plot (and any GLE/TSV export of it), the fitted
parameters and uncertainties become readable, and the whole fit is stored with
the project's trend series — the dialog remembers the last-used model per
``(parameter, x-axis)`` within that project, so reopening it restores your
choice. The fit's own outputs are also recorded as a new, trendable results
series (**Model fit (single) · <parameter> vs <x>**): each fit *range*
contributes one row of its fitted parameters and :math:`\chi^2_r`, indexed by the
centre of its fitted x-window, so a multi-range fit is immediately trendable in
turn (see `Recursive Trending (Model-Fit Results as a Series)`_). **Remove Fit**
discards the fit and drops that series.

For jointly fitting several detector groups or samples with shared and local
parameters, the panel routes two or more selected group series to a cross-group
variant of the same dialog — same catalogue, live preview, error modes, and fit
region, described under `Cross-Group Fitting`_.

Trending one parameter against another
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

Accounting for x uncertainty
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

Available basis components
--------------------------

.. code-block:: python

   from asymmetry.core.fitting import component_names_for_x

   print(component_names_for_x("field"))
   print(component_names_for_x("temperature"))

Build a parameter composite model
---------------------------------

In the GUI parameter-trending workflow, the **Edit Model** action opens the
same two-panel builder used for time-domain composite functions: a searchable
basis-model library on the left and the model as structured, reorderable rows
on the right. Each basis model becomes a single row, joined by ``+``, ``-``,
``*``, ``/``, or the quadrature combinator ``⊕`` (below); parentheses group
sub-expressions. The library's ⓘ **Info** button reports the documentation
for the currently selected basis model, including superconducting gap-model
details when relevant. Fraction groups are a time-domain-only concept and are
not offered here — ``⊕`` is the parameter-grammar's dedicated combinator for
sharing a width-like budget between terms.

The builder validates the expression in real time, and the Model Fit dialog
shows the expanded ``y(x)`` preview above the parameter table as you edit.

.. code-block:: python

   from asymmetry.core.fitting import ParameterCompositeModel

   model = ParameterCompositeModel(
       component_names=["DiffusionLF_2D", "Redfield", "Lambda_bg"],
       operators=["+", "+"],
   )
   print(model.formula_string())

Grouped expressions are also supported programmatically and in the GUI, for
example ``Linear + (Arrhenius * Constant)``.

Quadrature combinator (``⊕``)
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

The identity ``PowerLaw ⊕ Constant`` :math:`\equiv` ``PowerLawQuadBG`` is exact —
the operator generalises the fixed
:ref:`quadrature-background <quadrature-background>` component to any pair of
basis models. The parameter mapping is direct: the composite's dedicated
parameters are ``a``, ``n``, ``c_1`` (the power law's own additive constant) and
``c_2`` (the constant term), so the monolith's ``BG`` maps to ``c_2`` while
``c_1`` must be **fixed at 0** for the equivalence to hold — the
``PowerLawQuadBG`` inner term carries no additive offset.

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

Weighting and error modes
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

Fit windows (union multi-range)
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

χ² quality verdict
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

Cross-group fitting
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

Recursive trending (model-fit results as a series)
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

Magnetic order parameter
------------------------

For a second-order magnetic transition, the spontaneous muon precession
frequency, the internal field, or an ordered-moment-like asymmetry follows an
order-parameter temperature dependence that rises continuously from zero at the
ordering temperature :math:`T_c` to a saturated value :math:`y_0` at
:math:`T = 0`. The ``OrderParameter`` basis model captures this with a
generalised two-exponent form:

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

Polynomial trends
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

Power law with quadrature background
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

.. _muonium-repolarisation:

Muonium repolarisation
----------------------

``MuRepolarisation`` measures a muonium hyperfine constant from a
longitudinal-field scan, without resolving any precession. In an applied
field :math:`B` along the initial muon polarisation, the only muonium
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
the integral-asymmetry observable, :doc:`/reference/alc_mode`, x-axis
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
4. D. York, N. M. Evensen, M. L. Martínez, and J. De Basabe Delgado, *Am. J.
   Phys.* **72**, 367 (2004).

RF-μSR resonance (muon + electron + proton)
-------------------------------------------

``RFResonanceMuP`` fits a **field-swept RF-μSR resonance scan** of a muoniated
radical — muon plus electron plus one dominant proton, such as the
cyclohexadienyl radical C₆H₆Mu formed in benzene — and extracts the **muon and
proton hyperfine couplings** :math:`A_\mu` and :math:`A_p` simultaneously. This
is the Asymmetry counterpart of WiMDA's ``RigiWorkshopFit`` exact-diagonalisation
RF model.

At a fixed applied RF frequency :math:`\nu_{\mathrm{RF}}`, the static field is
swept and the (Red − Green) integral asymmetry shows two resonance features.
A resonance occurs where an RF-driven muon-spin-flip transition of the
three-spin Hamiltonian

.. math::

   H = A_\mu\,\mathbf{S}_e\!\cdot\!\mathbf{S}_\mu
     + A_p\,\mathbf{S}_e\!\cdot\!\mathbf{S}_p
     + (\gamma_e S_{e,z} - \gamma_\mu S_{\mu,z} - \gamma_p S_{p,z})\,B

matches :math:`\nu_{\mathrm{RF}}`. The two resonance fields :math:`B_1, B_2` are
found by **exact diagonalisation** of the 8×8 Hamiltonian (the high-field-limit
linear relation :math:`B_\mathrm{res} = (\nu_{\mathrm{RF}} \pm \tfrac{1}{2}A_\mu)/\gamma_\mu`
is inaccurate at the few-hundred-G fields these scans use), and the model is two
Lorentzians on a flat background:

.. math::

   y(B) = \mathrm{BG}
     + \sum_{i=1,2} \mathrm{ampl}_i\,
       \frac{\mathrm{wid}_i^2}{\mathrm{wid}_i^2 + (B - B_i)^2}.

The resonance **mean** tracks :math:`A_\mu` and the **splitting** tracks
:math:`A_p`. Set the amplitudes negative to fit resonance dips (the usual
Red − Green observable); hold :math:`\nu_{\mathrm{RF}}` fixed at the applied
frequency and give :math:`A_\mu, A_p` starting values near the expected couplings
— the resonance condition is nonlinear, so the field-swept curve only constrains
the fit once the trial resonances fall inside the scanned window. :math:`A_\mu`
(the mean) is well determined; :math:`A_p` (the splitting) is the weaker axis and
benefits from a complementary avoided-level-crossing (ALC) measurement.

.. code-block:: python

   import numpy as np
   from asymmetry.core.fitting import (
       Parameter,
       ParameterSet,
       ParameterCompositeModel,
       fit_parameter_model,
   )

   field_G = np.array([580.0, 700.0, 780.0, 820.0, 870.0, 950.0, 1060.0])
   asym = np.array([-1.6, -3.0, -16.0, -10.5, -17.0, -3.2, -1.5])  # Red-Green ×10⁻³
   errors = np.full_like(asym, 0.3)

   model = ParameterCompositeModel(["RFResonanceMuP"])
   params = ParameterSet([
       Parameter("A_mu", value=515.0, min=300.0, max=700.0),
       Parameter("A_p", value=124.0, min=40.0, max=250.0),
       Parameter("nu_RF", value=218.5, fixed=True),
       Parameter("ampl1", value=-18.0), Parameter("wid1", value=25.0, min=1.0),
       Parameter("ampl2", value=-18.0), Parameter("wid2", value=25.0, min=1.0),
       Parameter("BG", value=-1.5),
   ])

   result = fit_parameter_model(field_G, asym, errors, model, params)
   print({p.name: p.value for p in result.parameters})

References
~~~~~~~~~~

1. I. McKenzie, R. Scheuermann, S. P. Cottrell, J. S. Lord, and I. M. Tucker,
   J. Phys. Chem. B **117**, 13614 (2013).
2. E. Roduner, *The Positive Muon as a Probe in Free Radical Chemistry*,
   Lecture Notes in Chemistry Vol. 40 (Springer, Berlin, 1988).

Migrating WiMDA model functions
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
   * - RF resonance Mu+p (exact diag.)
     - ``RFResonanceMuP``
     - Port of ``RigiWorkshopFit``'s ``RFresonanceMuPlusProtonExact`` (the
       analytic ``RFresonanceMuPlusProton`` variant is intentionally not a
       separate component — it is inaccurate at low field). Parameters map
       A → A_mu, Ap → A_p, RF → nu_RF, ampl1/wid1/ampl2/wid2/BG identical.
       WiMDA's bespoke ``Eigenuni.pas`` Hermitian eigensolver is replaced by
       :func:`numpy.linalg.eigvalsh` (basis-independent spectrum, identical
       level differences).

Composite parameters in the Fit Parameters panel
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

.. _knight-shift:

Knight shift
------------

A muon in a metal or paramagnet precesses not at the bare Larmor frequency of the
applied field but at a frequency shifted by the local hyperfine field — the
field-induced polarisation of the conduction electrons (Fermi contact) together
with the dipolar and RKKY-mediated contact fields of any local moments. That
fractional shift is the muon **Knight shift**, the muon analogue of the NMR Knight
shift and a direct, site-resolved probe of the local magnetic response. The panel
converts a fitted precession frequency :math:`\nu` to

.. math::

   K = \frac{\nu - \nu_{\mathrm{ref}}}{\nu_{\mathrm{ref}}}

against one of two references.

The **Knight shift analysis** window is the recommended way to set this up:
it is a non-modal window dedicated to the conversion, so the reference,
unit, and component choices, the resulting branches, and the scan's
crossings are all visible together while you edit them, and nothing is
written back to the trend table until you ask for it. **Analysis → Knight
shift analysis…** is the unconditional entry point, always available
regardless of the active series. The **Knight shift window…** shortcut in
the *Derived parameters* section of the Fit Parameters panel is narrower: it
appears only when the active series' fitted model has at least one
Knight-convertible component (a local precession frequency or field, as
opposed to an applied-field muonium term) — opening it on an unrelated fit
would have nothing to convert. Its sidebar reads top to bottom as the
pipeline:

* **Source** — the fitted series supplying the frequencies (run count,
  component count, and scan axis), with a **Refresh from trend** button that
  rebuilds the snapshot from the trend panel's current rows after a refit or a
  series change.
* **Conversion** — the reference (**Applied field (γ_µ·B)** or **Designated
  component**, with a combo box for the chosen component when the latter is
  selected), the display **Unit** (**Auto (ppm / %)**, **ppm**, **percent**, or
  **fraction**), a checkbox per convertible component to include or exclude it
  from the conversion, and the **Lorentz/demag correction** checkbox with its
  **Shape** combo, **N** field, and **χ (SI)** field (described below) that
  turns the measured shift into the intrinsic :math:`K_\mu`.
* **Branches** — one converted :math:`K` trace per included component, each
  named :math:`K_n` and coloured to match its curve on the plot, together with
  a count of the crossings flagged along the scan (see *Component identity and
  crossings* below).
* **Model fit** — the joint :math:`K(\theta)` fit that resolves those
  crossings (see *Component identity and crossings*, below). The **Info…**
  button beside the model selector opens the shared component-information
  dialog for the selected :math:`K(\theta)` model — its formula, parameters,
  and applicability. Fitted parameters are presented per branch in a table,
  each value quoted to the precision its uncertainty supports, with a **Scale
  errors by √χ²ᵣ** checkbox that inflates the quoted uncertainties when a
  fit's reduced :math:`\chi^2` exceeds one.
* **Suggest next angle** — an optional section, collapsed by default, that
  plans the *next* scan angle from the fit above: see *Suggest next angle*,
  below.

The plot area has two view toggles, **Fold 180°** (overlay symmetry-equivalent
orientations onto one period, for an angle scan — a display choice local to
the window, independent of the trend panel's own **Fold** control described
below) and **Crossing markers** (draw a dashed vertical line at each flagged
crossing; on by default). Every control change re-derives the branches and
redraws immediately, so a bad reference choice or an accidentally excluded
component is visible before anything is published. The footer's **Send K
columns to trend table** button writes the current configuration back to the
trend panel as :math:`K[\ldots]` columns so they can be plotted and exported
alongside the fitted parameters. Until that button is pressed the trend table
is untouched.

.. figure:: /_generated/screenshots/knight_shift_window.png
   :width: 100%
   :align: center
   :alt: The Knight shift analysis window, with the Applied field reference
      selected, two frequency branches converted, a completed joint K(theta)
      fit in the Model fit section, the Suggest next angle section expanded
      with a computed D-optimal refine suggestion, and the fitted curves
      plus the suggestion's utility band overlaid on the K(theta) plot.

   The Knight shift analysis window on a two-site angle scan: **Source**,
   **Conversion**, **Branches**, **Model fit**, and **Suggest next angle** in
   the sidebar — *Model fit* showing a completed joint :math:`K(\theta)` fit
   and *Suggest next angle* a computed **Refine parameters** / **All
   parameters (D-optimal)** suggestion — and the converted branches with
   their fitted curves, **Crossing markers**, and the suggestion's utility
   band overlaid in the plot area.

The window's configuration and view-toggle state — including any joint fit —
persist with the project under the ``knight_shift_analysis_state`` key (the
point snapshot itself is always rebuilt from the source series on load, so a
saved project can never carry stale fitted values). Projects saved before the
window existed stored the conversion, and any joint fit, under the trend
panel's own ``fit_parameters_state`` block; these migrate automatically the
first time the project is opened, so the window reopens already configured
and, where a legacy joint fit is present, with its run-keyed branch
assignment carried over (the migrated fit's curves render as stale — "re-run
to refresh" — until the fit is re-run, since the unit they were originally
fitted in is not always recoverable).

* **Applied field** — the precession is referenced to the bare applied field. For
  a frequency-parameterised component (MHz) the reference is the free-muon Larmor
  frequency :math:`\nu_{\mathrm{ref}} = \gamma_\mu B`
  (:math:`\gamma_\mu/2\pi = 135.5\ \mathrm{MHz\,T^{-1}}`); for a component
  parameterised directly by the local field :math:`B_\mu` (Gauss, as in the
  ``OscillatoryField`` family) the reference is the applied field itself, giving
  the most direct form :math:`K = (B_\mu - B)/B`. This needs no reference line and
  is the default; use it whenever a separate unshifted signal is absent (a
  low-background sample) and the applied field is known.
* **Designated component** — :math:`\nu_{\mathrm{ref}}` is the fitted frequency of
  a chosen oscillation component, such as a co-measured diamagnetic line. Because
  :math:`\nu` and :math:`\nu_{\mathrm{ref}}` then come from the same fit, their
  covariance is carried through the error propagation; the applied-field reference
  treats :math:`B` as exact.

K is dimensionless. It is stored internally as a fraction and shown in a unit you
select, with an **Auto** mode that reads parts per million for a diamagnet
(typically tens of ppm) and per cent for a paramagnet (up to a few per cent). The
conversion itself yields the directly measured shift :math:`K_{\mathrm{exp}}`; the
window's **Lorentz/demag correction** (below) recovers the intrinsic :math:`K_\mu`
from it given the sample geometry and bulk susceptibility.

**Lorentz/demagnetising correction.** The measured shift :math:`K_{\mathrm{exp}}`
includes the Lorentz cavity field and the sample's own demagnetising field, both
of which depend on the sample shape rather than the local hyperfine coupling. The
**Lorentz/demag correction** checkbox in the *Conversion* section removes them,

.. math::

   K_\mu = K_{\mathrm{exp}} - \left(\frac{1}{3} - N\right)\chi
   \qquad\text{(Amato \& Morenzoni Eq. 5.60)}

with :math:`N` the demagnetisation factor along the applied field (SI convention,
:math:`\sum N = 1` over the three principal axes) and :math:`\chi` the sample's
*volume* susceptibility, also SI dimensionless — multiply a CGS value in
emu cm\ :sup:`-3` by :math:`4\pi` before entering it. The **Shape** combo sets
:math:`N` from the sample geometry — **Sphere (N = 1/3)**, for which the
correction vanishes exactly (the Lorentz and demagnetising fields cancel);
**Thin plate, B ∥ plane (N = 0)**; **Thin plate, B ⊥ plane (N = 1)**; **Long
cylinder, B ∥ axis (N = 0)**; **Long cylinder, B ⊥ axis (N = 1/2)**; or
**Custom N**, which enables the **N** field for a value outside these standard
shapes — and **χ (SI)** takes the volume susceptibility. The correction is a
constant offset applied equally to every
branch, so it shifts K but never reorders branches or moves a crossing. It is
exact for an ellipsoidal sample whose orientation relative to the field is fixed;
for a rotating non-spheroidal sample :math:`N` itself varies with angle, which
this scalar form does not capture — a caveat worth remembering for an angle-scan
correction with a non-sphere shape. The K uncertainties reported by the window do
not include a :math:`\chi` uncertainty.

**Angle dependence.** Rotating a single crystal in the applied field maps out
:math:`K(\theta)`. The contact term is isotropic, while the dipolar term carries
the full angular dependence through the dipolar coupling tensor, so :math:`K(\theta)`
constrains the muon stopping site. Add an **Angle (°)** column
(:ref:`logbook-columns`), fit each orientation, and select **Angle (°)** as the
trend x-axis to plot each component's shift against orientation. Because the
dependence is periodic (the dipolar term goes as :math:`\cos^2\theta`, period
180°), the **Fold** control next to the x-axis selector folds the angle into one
period (180° or 360°) so equivalent orientations from a wide or wrap-around scan
overlay — doubling the effective angular sampling and exposing the periodic shape
for a model fit. Folding is a display/analysis choice; the stored angles are
untouched.

**Component identity and crossings.** Across an orientation (or temperature, or
field) scan, each oscillation component keeps its stable label (``frequency``,
``frequency_2``, …) and is converted independently, so the trend follows one
component through the scan. Where two component frequencies approach or cross, that
identification becomes ambiguous — the fit is free to interchange the two — and the
converted trace can jump. The panel detects these points (a frequency crossing, or
a relabelling that better preserves continuity than the raw label order) and marks
them on the trend; it does not silently reassign the components, leaving the
judgement to you. Seeding each fit from its neighbour (chained batch seeding) keeps
the labelling stable through an ordered scan and minimises such crossings in the
first place.

To *resolve* a crossing rather than just flag it, open the Knight shift
analysis window (with at least two branches and **Angle** as the scan axis)
and use the **Model fit** section of its sidebar: choose a **Model**
(``KnightAnisotropy``, ``AngularCos2``, or ``AngularFourier2`` — the same
angular basis models described below) and a **Max iterations** bound, then
press the footer's
**Run joint K(θ) fit** button. The fit runs off-thread — the button reports
progress and re-enables when it finishes — and fits one K(θ) curve per
branch simultaneously, assigning each angle's component points one-to-one to
the curve they best fit (a Hungarian matching) and iterating fit ↔
reassignment to convergence. The plotted branches are realigned so each
follows one physical curve continuously through the crossings, the fitted
model curves overlay in the branch colours, and dashed vertical markers flag
the angles where the assignment swaps (the crossings the fit actually
resolved, a firmer signal than the raw proximity flags). The sidebar reports
each curve's fitted parameters and reduced :math:`\chi^2` beneath the
controls; **Clear fit** discards the fit and returns the branches to their
raw component labels. Changing the display unit or the Lorentz/demag
correction only marks the fitted curves stale (their parameters shift with
either — the assignment does not, since a common offset or scale cannot
reorder branches) — re-run the fit to refresh them. The **Scale errors by
√χ²ᵣ** checkbox inflates every quoted parameter uncertainty by
:math:`\sqrt{\chi^2_r}` when that curve's reduced :math:`\chi^2` exceeds one
(a fit that is already better than the noise is left alone); it is
display-only — the stored fit itself is unchanged, and toggling it appends or
removes an "(errors ×√χ²ᵣ)" note next to the affected curve. The fit
(assignment and per-curve parameters) is saved with the project and restored
on reload, and stays applicable across a **Refresh from trend** as long as
the branch count is unchanged; a different component selection invalidates
it.

In the trend panel, select one or more ``K[...]`` traces and use **Remove**
to delete them: the backing component is dropped from the conversion (via
``set_knight_shift_config``, so the trace does not regenerate). Removing
every component turns the conversion off.

**Fitting the anisotropy** :math:`K(\theta)`. With **Angle (°)** as the trend
x-axis, the model-fit builder offers three angle-only basis models alongside
the usual ones:

.. math::

   K(\theta) = K_{\mathrm{iso}} + K_{\mathrm{ax}}\,\frac{3\cos^2(\theta - \theta_0) - 1}{2}
   \qquad\text{(}\texttt{KnightAnisotropy}\text{)}

for the axial dipolar form — the isotropic contact term :math:`K_{\mathrm{iso}}`
plus the traceless dipolar anisotropy :math:`K_{\mathrm{ax}}` — the general
two-fold modulation

.. math::

   K(\theta) = K_{\mathrm{avg}} + K_{\mathrm{amp}}\cos 2(\theta - \theta_0)
   \qquad\text{(}\texttt{AngularCos2}\text{)}

for a crystal rotated in a plane — algebraically the same curve family as
``KnightAnisotropy`` (:math:`K_{\mathrm{iso}} + K_{\mathrm{ax}}/4 =
K_{\mathrm{avg}}`, :math:`3K_{\mathrm{ax}}/4 = K_{\mathrm{amp}}`, sharing
:math:`\theta_0`) — and a misalignment-sensitive extension

.. math::

   K(\theta) = K_{\mathrm{avg}} + K_1\cos(\theta - \theta_1) + K_{\mathrm{amp}}\cos 2(\theta - \theta_2)
   \qquad\text{(}\texttt{AngularFourier2}\text{)}

which adds a first-harmonic term to ``AngularCos2``. A perfectly aligned
rotation axis gives a pure second harmonic; a tilted axis leaks a first
harmonic whose amplitude grows with the tilt, so a fitted :math:`K_1`
significantly different from zero is fit-level evidence that the rotation
axis is not exactly where the goniometer says it is. This is a
phenomenological Fourier form — it does not itself solve for the physical
tilt angle from :math:`K_1` and :math:`\theta_1` — and, with five free
parameters, needs at least five shared angles per curve to fit. See *Suggest
next angle*, below, for using it to test for misalignment directly rather
than assuming an aligned axis.

``KnightAnisotropy`` and ``AngularCos2`` both carry a :math:`\theta_0` term —
the goniometer/mount misalignment between the zero of the angle scale and the
crystal's principal axis, since a real mount is never perfectly aligned;
without it, that misalignment would otherwise bias :math:`K_{\mathrm{iso}}`
and :math:`K_{\mathrm{ax}}` (or :math:`K_{\mathrm{avg}}` and
:math:`K_{\mathrm{amp}}`) directly. Both forms are invariant under a
:math:`90^\circ` shift of :math:`\theta_0` together with a sign flip of the
anisotropic amplitude, so the joint fit canonicalises the fitted
:math:`\theta_0` into :math:`(-45^\circ, 45^\circ]` — folding to the
small-:math:`|\theta_0|` representation so a mount that is nearly aligned
reads with a small offset and a physically sensible amplitude sign, rather
than an equally valid but larger-offset relabelling of the same curve.
``AngularFourier2``'s second-harmonic phase :math:`\theta_2` folds the same
way; its first-harmonic phase :math:`\theta_1` folds independently, over the
full :math:`360^\circ` period a first harmonic repeats on, into
:math:`(-90^\circ, 90^\circ]` with a sign flip of :math:`K_1`. Every fold is
an exact linear reparameterisation, so the fitted covariance is carried
through it exactly (:math:`\Sigma' = J\Sigma J^{\mathsf{T}}`) rather than
approximated — the quoted uncertainties, and anything downstream that
consumes the covariance (such as *Suggest next angle*, below), see the same
figure a refit started directly in the canonical branch would have produced.
All three models take :math:`\theta` in degrees, so they fit directly against
the Angle axis (folded or not). The diagonal dipolar-tensor components
recovered for rotations about the crystal axes constrain the muon stopping
site. For a worked example end to end, see :doc:`/workflows/knight_shift_angle`.

**Clogston–Jaccarino** (:math:`K` vs :math:`\chi`). Plotting the Knight shift
against the bulk susceptibility with temperature as the implicit parameter gives
a straight line whose slope :math:`\mathrm{d}K/\mathrm{d}\chi` measures the muon
hyperfine coupling and whose intercept is the :math:`\chi`-independent shift. The
muon experiment does not itself measure :math:`\chi`, so this is **API-only** —
pair the exported Knight shift with an independent susceptibility:

.. code-block:: python

   from asymmetry.core.fitting.knight_shift import clogston_jaccarino_fit

   result = clogston_jaccarino_fit(chi, knight, sigma_knight)  # χ, K, σ_K per T
   print(result.slope, result.slope_err)   # dK/dχ ∝ hyperfine coupling
   print(result.intercept)                 # χ-independent (orbital) shift

*When to use this.* Convert to the Knight shift whenever the quantity of interest
is the *shift* of a precession frequency from the applied-field value — tracking a
local susceptibility, determining a muon site from :math:`K(\theta)`, or building a
Clogston–Jaccarino :math:`K`–:math:`\chi` plot. Keep the raw frequency trend when
the absolute precession frequency itself is the observable (an internal field, a
magnetic order parameter).

Suggest next angle
~~~~~~~~~~~~~~~~~~

Once the joint :math:`K(\theta)` fit above has converged, the collapsible
**Suggest next angle** section beneath *Model fit* answers the question
:doc:`suggest_next_point` already answers for a scalar trend — where should
the *next* measurement go? — with three angle-specific flavours, chosen from
a **Mode** selector: **Refine parameters**, **Test misalignment**, and
**Resolve assignment**. All three share one **Candidate range** (seeded from
the measured angle span; widen it to allow an extrapolated suggestion) and
one **Suggest** button, and reuse the Laplace/Fisher acquisition machinery
described on that page rather than a separate derivation — what is genuinely
new here is specific to a rotation scan: one new run yields a value for
*every* curve at once, and which measured value belongs to which physical
curve is itself something the joint fit had to work out.

The section stays inactive, with a muted hint explaining why, until every
prerequisite is met: no joint fit ("Run the joint K(θ) fit first."), one
that no longer matches the branches or predates the current unit/correction
("Re-run the joint K(θ) fit — it no longer matches the branches." / "…the
display unit or correction changed."), or — for a fit saved before this
feature shipped — a missing stored covariance ("Re-run the joint K(θ) fit to
store fit covariance."): the run-keyed assignment of a legacy fit survives a
project reload, but its per-curve covariance was never recorded, so a
next-angle suggestion needs one re-run before it becomes available.

**Refine parameters** sums the expected information gain over every curve —
one new run adds a datum to each of them, so their information gains add.
**Target** chooses what to refine: **All parameters (D-optimal)** shrinks
every curve's whole parameter covariance at once, or pick one curve's
parameter from the list for a c-optimal solve, which enables **Precision
goal** and the **Typical run (Mevents)** / **Rate (Mevents/h)** conversion —
exactly as in :doc:`suggest_next_point`, down to the result line's phrasing:

.. code-block:: text

   Measure at θ = 48.2° × 0.6 of a typical run's statistics
   → σ ≈ 0.018 (approximate)

**Test misalignment** asks a different question: is the scan actually
consistent with a perfectly aligned rotation axis, or is there evidence of a
tilt? On **Suggest** it fits the ``AngularFourier2`` alternative (above)
automatically, off-thread — cached against the joint fit's model, unit, and
Lorentz/demag correction, so repeat clicks do not refit — and ranks candidate
angles by the disagreement between the current model and that alternative,
summed over every branch (the same model-discrimination utility
:doc:`suggest_next_point`'s **Compare against** uses for a scalar trend). The
result line names the currently preferred model and its Akaike weight, e.g.

.. code-block:: text

   Measure at θ = 71.5°; prefers AngularFourier2 (Akaike weight 0.86, evidence ratio 6.1)

and when the two models already agree within noise everywhere in the
candidate range, the section reports that instead of pointing at an
arbitrary angle.

**Resolve assignment** targets the labelling itself rather than either
model's parameters. Near a crossing, the joint fit's classification-EM step
can settle on more than one near-equally-good assignment of which measured
component belongs to which physical curve; this mode ranks candidate angles
by how well a new, unlabelled run there would separate the winning
assignment from its near-degenerate runners-up — a minimum-cost matching
between the two labellings' predicted value sets, zero exactly at a crossing
(where the competing labellings coincide by construction) and largest where
they imply genuinely different curves. The runners-up are kept only in
memory from the last fit run, so if the stored fit came from a project load,
**Suggest** re-runs the joint fit off-thread first to recover them; with
none to compare against, the section reports "No near-degenerate assignments
to discriminate."

**Overlay and risk shading.** A computed suggestion draws the same utility
band used by :doc:`suggest_next_point`, anchored beneath the
:math:`K(\theta)` plot with the best angle marked and any extrapolated
candidates shown at reduced opacity. Refine and Test-misalignment
suggestions can additionally shade a candidate span with a muted hatched
band where two curves' predicted values sit close enough (within about
:math:`2\sigma` of each other) that a new, unlabelled run there risks being
fitted onto the wrong curve — exactly the crossing risk the joint fit's own
Hungarian assignment exists to resolve. A suggested angle that falls inside
a shaded span should be read with that caveat in mind rather than acted on
blindly.

*When to use this.* Reach for **Refine parameters** while extending an
otherwise routine scan; reach for **Test misalignment** once a fit already
looks good but you want to actively rule out a tilted mount rather than
assume alignment; reach for **Resolve assignment** specifically when a
crossing has left two labellings within reach of each other and the next run
should settle which is real.

References
~~~~~~~~~~

1. A. Amato and E. Morenzoni, *Introduction to Muon Spin Spectroscopy:
   Applications to Solid State and Material Sciences*, Lecture Notes in Physics
   Vol. 961 (Springer, Cham, 2024).
2. S. J. Blundell, R. De Renzi, T. Lancaster, and F. L. Pratt, *Muon
   Spectroscopy: An Introduction* (Oxford University Press, Oxford, 2022).
3. W. D. Knight, Phys. Rev. **76**, 1259 (1949).
4. A. M. Clogston, V. Jaccarino, and Y. Yafet, Phys. Rev. **134**, A650 (1964).

Runnable example
----------------

See ``examples/parameter_trending.py`` for a complete executable script.

Downstream trend models
-----------------------

The trend framework feeds several physical models that consume a fitted
parameter series: for TF-μSR superconducting penetration-depth analysis via a
temperature-dependent :math:`\sigma(T)`, see :doc:`sc_penetration_depth`; for
muonium reaction kinetics from relaxation rates linear in reactant
concentration, see :doc:`muonium_kinetics`; and for the field-dependent
transport models fitted to :math:`\lambda(B_\mathrm{LF})`, see
:doc:`diffusion_ballistic_lf`.
