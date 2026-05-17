Composite Models
================

Composite models let you build time-domain fit functions from reusable
components and arithmetic operators.

Building a Composite Function
-----------------------------

.. code-block:: python

   from asymmetry.core.fitting import CompositeModel

   model = CompositeModel(
       component_names=["Exponential", "Oscillatory", "Constant"],
       operators=["+", "+"],
   )
   print(model.formula_string())

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
     - Hayano LF-KT :math:`G_z(t;\Delta,B_L)` — see :ref:`lf-kubo-toyabe`
     - ``A``, ``Delta``, ``B_L`` (Gauss)
   * - ``MuF``
     - Analytical single mu-F polarization :math:`D_z(t)`
     - ``A``, ``r_muF`` (Å)
   * - ``FmuF_Linear``
     - Analytical collinear F-mu-F polarization
     - ``A``, ``r_muF`` (Å)
   * - ``FmuF_General``
     - Numerical powder-averaged F-mu-F polarization
     - ``A``, ``r1`` (Å), ``r2`` (Å), ``theta`` (rad)
   * - ``Constant``
     - :math:`A_{\mathrm{bg}}`
     - ``A_bg``

Runnable Example
----------------

See ``examples/composite_models.py`` for a complete executable script.
