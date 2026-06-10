Composite Models
================

.. image:: /_generated/screenshots/composite_models_builder.png
   :alt: Fit Function Builder dialog with an Oscillatory + Exponential + Constant expression
   :width: 100%

*The Fit Function Builder dialog parses free-form expressions over the*
*registered components. The keypad inserts operators and grouping symbols;*
*the* **Fractions** *button binds two or more additive components into a*
*shared-amplitude fraction group. The preview line at the bottom shows the*
*compiled formula with mangled parameter names (A_1, A_2, …) for each*
*component (Blundell et al.* Muon Spectroscopy *Ch 6.4).*

A realistic μSR asymmetry almost never has the algebraic form of a single
depolarisation function. It is typically the product of a relaxation
envelope with an oscillatory signal, or the sum of contributions from
distinct muon populations (e.g. magnetic and paramagnetic fractions near a
transition, multiple stopping sites in a molecular magnet), with a small
detector-imbalance constant added in. Composite models are how Asymmetry
expresses these combinations: a free-form arithmetic expression over
registered baseline-free components, parsed into a compiled callable that
the fit engine drives like any other model. The two patterns the builder
is designed for are *multiplicative* combinations, where one physical
effect modulates another (a Gaussian envelope multiplying a TF precession
signal in the vortex state; an exponential damping multiplying a Larmor
oscillation), and *additive* combinations, where independent populations
of muons contribute separately. Fraction groups, documented below, let
several additive components share one overall amplitude budget — the
natural representation for two muonium states in a semiconductor whose
fractions must sum to one, or for the magnetic and paramagnetic
fractions of a sample passing through a transition.

For real examples that build composites step-by-step, see
:doc:`workflows/temperature_scan_magnetism` and
:doc:`workflows/superconductor_penetration_depth`.

.. image:: /_generated/screenshots/composite_fractions_dialog.png
   :alt: Fit Function Builder with a fraction-group expression
   :width: 100%

Building a Composite Function
-----------------------------

.. code-block:: python

   from asymmetry.core.fitting import CompositeModel

   model = CompositeModel(
       component_names=["Exponential", "Oscillatory", "Constant"],
       operators=["+", "+"],
   )
   print(model.formula_string())

The time-domain grammar combines components with ``+``, ``-``, ``*``, ``/`` and
parentheses. The quadrature combinator ``⊕`` (:math:`\sqrt{f^2 + g^2}`) is
*not* part of this grammar — it belongs to the parameter-vs-x trend models,
where quadrature composition of width-like quantities is physically meaningful;
see :doc:`parameter_trending`.

Fraction Groups
---------------

Composite models can also share one overall amplitude across several additive
components while fitting normalized fractions inside that group. In Python,
write the grouped sum as ``(...){frac}``:

.. code-block:: python

  fraction_model = CompositeModel.from_expression(
     "( Exponential + Gaussian ){frac} + Constant"
  )

This creates one amplitude for the grouped sum together with fraction
parameters ``fraction_1``, ``fraction_2``, ... that are normalized internally
so the effective fractions always satisfy :math:`\sum_i f_i = 1`.

In the GUI fit-function builder, you do not need to type ``{frac}``. Instead,
select two or more additive components, press ``Fractions``, and the dialog
uses matching colors in the expression editor and preview to show which terms
belong to the same fraction group.

Parameter Naming Rules
----------------------

Composite models generate unique parameter names automatically:

* Additive terms get their own amplitude parameters: ``A_1``, ``A_2``, ...
* Components joined by ``*`` or ``/`` share a single amplitude for that
    multiplicative chain. For example, ``Exponential * Gaussian`` uses only
    ``A_1``.
* Fraction groups share one amplitude across the whole grouped sum and add
  normalized fraction parameters: ``A_1``, ``fraction_1``, ``fraction_2``, ...
* Repeated symbols are indexed: ``Lambda_1``, ``Lambda_2``
* Unique symbols remain unindexed: ``frequency``
* Constant background uses ``A_bg``

This keeps the parameterization closer to the usual physics notation for
products such as an exponentially damped oscillation, where the envelope and
oscillation share one overall asymmetry.

For fraction groups, the final effective weights are always normalized even if
the raw fit parameters move during minimization, so the grouped amplitudes stay
on a physically interpretable simplex.

Evaluate Model and Components
-----------------------------

.. code-block:: python

   import numpy as np

   t = np.linspace(0.0, 6.0, 200)
   y = model.function(
       t,
       A_1=20.0,
       Lambda=0.4,
       A_2=8.0,
       frequency=2.0,
       phase=0.0,
       A_bg=0.6,
   )

   # Useful for plotting stacked contributions
   additive_curves = model.evaluate_components(
       t,
       additive_only=True,
       A_1=20.0,
       Lambda=0.4,
       A_2=8.0,
       frequency=2.0,
       phase=0.0,
       A_bg=0.6,
   )

For multiplicative models, pass the shared chain amplitude only once:

.. code-block:: python

   damped_cosine = CompositeModel(
       component_names=["Exponential", "Oscillatory"],
       operators=["*"],
   )

   y = damped_cosine.function(
       t,
       A_1=20.0,
       Lambda=0.4,
       frequency=2.0,
       phase=0.0,
   )

In symbolic previews, downstream multiplicative factors suppress the redundant
``1*`` amplitude term, so the displayed formula stays readable.

Use with FitEngine
------------------

.. code-block:: python

   from asymmetry.core.fitting import FitEngine, Parameter, ParameterSet

   params = ParameterSet([
       Parameter("A_1", value=20.0, min=0.0),
       Parameter("Lambda", value=0.4, min=0.0),
       Parameter("A_2", value=8.0, min=0.0),
       Parameter("frequency", value=2.0, min=0.0),
       Parameter("phase", value=0.0),
       Parameter("A_bg", value=0.0),
   ])

   result = FitEngine().fit(dataset, model.function, params)
   print(result.success)

If the model contains a multiplicative chain, include only that chain's shared
amplitude parameter in the fit table.

Available Components
--------------------

The following components are registered in ``COMPONENTS`` and can be used by
name in ``CompositeModel``:

.. list-table::
   :header-rows: 1
   :widths: 22 40 38

   * - Key
     - Formula
     - Parameters
   * - ``Exponential``
     - :math:`A e^{-\Lambda t}`
     - ``A``, ``Lambda``
   * - ``Gaussian``
     - :math:`A e^{-(\sigma t)^2}`
     - ``A``, ``sigma``
   * - ``Oscillatory``
     - :math:`A \cos(2\pi f t + \phi)`
     - ``A``, ``frequency`` (MHz), ``phase``
   * - ``OscillatoryField``
     - :math:`A \cos(2\pi \gamma_\mu B t + \phi)`
     - ``A``, ``field`` (Gauss), ``phase``. The precession frequency is
       computed internally using :math:`\gamma_\mu = 13.554\,\text{MHz/kG}`.
       Use this component when the fit parameter should be the applied field
       rather than the precession frequency directly.
   * - ``StretchedExponential``
     - :math:`A e^{-(|\Lambda| t)^\beta}`
     - ``A``, ``Lambda``, ``beta``
   * - ``StaticGKT_ZF``
     - :math:`A \left[\tfrac{1}{3} + \tfrac{2}{3}(1-\Delta^2 t^2)e^{-\Delta^2 t^2/2}\right]`
     - ``A``, ``Delta``
   * - ``LongitudinalFieldKT``
     - Hayano LF-KT :math:`G_z(t;\Delta,B_L)` — see :ref:`fit-lf-kubo-toyabe`
     - ``A``, ``Delta``, ``B_L`` (Gauss)
   * - ``MuF``
     - Analytical single mu-F polarization :math:`D_z(t)`
     - ``A``, ``r_muF`` (Å)
   * - ``FmuF_Linear``
     - Analytical collinear F-mu-F polarization
     - ``A``, ``r_muF`` (Å)
   * - ``FmuF_General``
     - Numerical powder-averaged F-mu-F polarization
     - ``A``, ``r1`` (Å), ``r2`` (Å), ``theta`` (°)
   * - ``Constant``
     - :math:`A_{\mathrm{bg}}`
     - ``A_bg``

Runnable Example
----------------

See ``examples/composite_models.py`` for a complete executable script.
