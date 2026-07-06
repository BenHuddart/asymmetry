Composite models
================

.. image:: /_generated/screenshots/composite_models_builder.png
   :alt: Fit Function Builder dialog with an Oscillatory + Exponential + Constant expression
   :width: 100%

*The Build Fit Function dialog searches a component library on the left*
*(type to match names, aliases, parameters, or descriptions) and assembles*
*the model as structured rows on the right, one per component, joined by*
*editable operators. Selecting two or more '+'-joined rows and pressing*
*"Group as fractions" binds them into a shared-amplitude fraction group,*
*shown as an accented container. The preview line at the bottom shows the*
*compiled formula with mangled parameter names (A_1, A_2, …) for each*
*component (Blundell et al.,* Muon Spectroscopy: An Introduction *(Oxford University Press, Oxford, 2022), Ch. 6.4).*

A realistic μSR asymmetry almost never has the algebraic form of a single
depolarisation function. It is typically the product of a relaxation
envelope with an oscillatory signal, or the sum of contributions from
distinct muon populations (e.g. magnetic and paramagnetic fractions near a
transition, multiple stopping sites in a molecular magnet), with a small
constant offset added in. Composite models are how Asymmetry
expresses these combinations: a free-form arithmetic expression over
registered baseline-free components, parsed into a compiled callable that
the fit engine drives like any other model. The two patterns the builder
is designed for are *multiplicative* combinations, where one physical
effect modulates another (a Gaussian envelope multiplying a transverse-field
(TF) precession signal in the vortex state; an exponential damping multiplying a Larmor
oscillation), and *additive* combinations, where independent populations
of muons contribute separately. Fraction groups, documented below, let
several additive components share one overall amplitude budget — the
natural representation for two muonium states in a semiconductor whose
fractions must sum to one, or for the magnetic and paramagnetic
fractions of a sample passing through a transition.

For real examples that build composites step-by-step, see
:doc:`/workflows/temperature_scan_magnetism` and
:doc:`/workflows/superconductor_penetration_depth`.

.. image:: /_generated/screenshots/composite_fractions_dialog.png
   :alt: Fit Function Builder with a fraction-group expression
   :width: 100%

Building a composite function
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

Fraction groups
---------------

Composite models can also share one overall amplitude across several additive
components while fitting the fractional weight of each term inside that group.
In Python, write the grouped sum as ``(...){frac}``:

.. code-block:: python

  fraction_model = CompositeModel.from_expression(
     "( Exponential + Gaussian + Constant ){frac} + Constant"
  )

A group of :math:`n` additive terms has :math:`n - 1` **free** fraction
parameters, named after their components — ``f_Exponential``, ``f_Gaussian``,
... (duplicates are suffixed ``_2``, ``_3``, ...). Each free fraction is a
weight in :math:`[0, 1]`. The **last** term of the group has *no* fitted
parameter: its weight is the remainder, :math:`1 - \sum_i f_i`, clamped to
:math:`[0, 1]`. There is no sum-normalisation — the free fractions are the
weights directly, so together with the derived remainder they partition the
group's amplitude.

In the GUI fit-function builder, you do not need to type ``{frac}``. Instead,
select two or more additive rows and press **Group as fractions**; the grouped
terms then sit inside a single accented container, and the typeset preview
colours each fraction group so it is clear which terms share an amplitude
budget. In the fit-panel parameter table the free fractions are ordinary
editable rows; the group's remainder appears as a muted, read-only row that
updates automatically as you edit the others.

Parameter naming rules
----------------------

Composite models generate unique parameter names automatically:

* Additive terms get their own amplitude parameters: ``A_1``, ``A_2``, ...
* Components joined by ``*`` or ``/`` share a single amplitude for that
    multiplicative chain. For example, ``Exponential * Gaussian`` uses only
    ``A_1``.
* Fraction groups share one amplitude across the whole grouped sum (``A_1``)
  and add one free fraction per term named after its component —
  ``f_Exponential``, ``f_Gaussian``, ... (duplicates suffixed ``_2``, ``_3``,
  ...) — for all but the last term. The last term carries no parameter; its
  weight is the derived remainder :math:`1 - \sum_i f_i`.
* Repeated symbols are indexed: ``Lambda_1``, ``Lambda_2``
* Unique symbols remain unindexed: ``frequency``
* Constant background uses ``A_bg``

This keeps the parameterisation closer to the usual physics notation for
products such as an exponentially damped oscillation, where the envelope and
oscillation share one overall asymmetry.

For fraction groups the free fractions *are* the weights (each clamped to
:math:`[0, 1]`); the final term takes whatever remains, :math:`1 - \sum_i f_i`
(floored at zero if the free fractions over-subscribe). There is no
sum-normalisation, so the fitted free values are directly interpretable as the
physical fractional weights.

Always read ``.param_names`` before building a ``ParameterSet``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The numbering is **collision-driven**, not blanket per-component, so the only
reliable way to know the parameter names is to ask the compiled model. A symbol
that appears in more than one component is suffixed with that component's
1-based position; a symbol that is unique across the whole expression stays
bare. So in ``Oscillatory * Exponential + Constant`` the amplitude ``A``
collides (shared across the ``*`` chain → ``A_1``) and ``Constant`` contributes
``A_bg``, but ``frequency``, ``phase``, and ``Lambda`` are each unique and keep
their plain names:

.. code-block:: python

   from asymmetry.core.fitting import CompositeModel

   model = CompositeModel.from_expression("Oscillatory * Exponential + Constant")
   print(model.param_names)
   # ['A_1', 'frequency', 'phase', 'Lambda', 'A_bg']
   # note: 'Lambda' (unique) is NOT 'Lambda_1'; 'Constant' contributes 'A_bg'

The 1-based position is counted across **every** component in the expression,
not per repeated symbol — so when a symbol first collides at a later component,
its suffix is that component's index, which can *skip* numbers. Add a second
``Oscillatory`` and the precession symbols of the two oscillatory terms become
``_1`` and ``_3`` (their component positions), never ``_1`` and ``_2``:

.. code-block:: python

   model = CompositeModel.from_expression(
       "Oscillatory * Exponential + Oscillatory + Constant"
   )
   print(model.param_names)
   # ['A_1', 'frequency_1', 'phase_1', 'Lambda', 'A_3',
   #  'frequency_3', 'phase_3', 'A_bg']
   # the second Oscillatory is component #3 → frequency_3 / phase_3, NOT _2

``model.param_names`` and ``model.to_model_definition().param_names`` return the
same list. Build your :class:`~asymmetry.core.fitting.ParameterSet` from that
list rather than from guessed names — guessing ``Lambda_1`` or ``frequency_2``
here would silently fail to bind.

A ``ParameterSet`` is keyed **by parameter name**, not by position. Look a
parameter up with its name; integer indexing raises ``KeyError`` (it is treated
as a missing name, not a sequence index). Its own name list is the ``.names``
attribute (a plain ``list``, not a method):

.. code-block:: python

   from asymmetry.core.fitting import Parameter, ParameterSet

   ps = ParameterSet([Parameter("A_1", value=0.2), Parameter("Lambda", value=0.4)])
   ps.names          # ['A_1', 'Lambda']  (attribute, not ps.names())
   ps["A_1"]         # Parameter(name='A_1', value=0.2, ...)
   ps[0]             # KeyError: 0  — there is no positional access

So the safe recipe is always: compile the model, read ``model.param_names``,
then build the ``ParameterSet`` entries under those exact names.

.. _components-vs-models:

Expression (``COMPONENTS``) names are not the standalone (``MODELS``) names
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. important::

   **"GaussianRelaxation is not a component" is a naming mismatch, not a missing
   feature.** ``GaussianRelaxation`` is a key of the standalone ``MODELS``
   registry; inside a composite expression the same physics is the component
   ``Gaussian``. Likewise ``ExponentialRelaxation`` → ``Exponential`` and
   ``LFKuboToyabe`` → ``LongitudinalFieldKT``. If a name you expect is "absent"
   from ``COMPONENTS``, check the table below before concluding the model does
   not exist.

Asymmetry keeps **two** registries, and they use different names for the same
physics:

* ``COMPONENTS`` — the building blocks you combine **inside a composite
  expression** (``CompositeModel.from_expression(...)`` and the GUI builder).
  Their natural parameters are baseline-free (``A``, ``Lambda``, ``sigma``, …)
  and the parser then *mangles* them into the composite names (``A_1`` / ``A_bg``
  / ``Lambda_2`` / …) described above.
* ``MODELS`` — the **standalone** models you hand straight to
  ``FitEngine().fit(...)`` (see :doc:`fitting`). Their parameters carry an
  explicit amplitude and baseline (``A0``, …, ``baseline``).

Most names are shared between the two registries, but three of the everyday
relaxation forms are spelled differently — exactly the ones testers most often
report as "missing":

.. list-table:: ``MODELS`` (standalone) ↔ ``COMPONENTS`` (expression) name map
   :header-rows: 1
   :widths: 30 18 26 26

   * - Standalone ``MODELS`` key
     - Its parameters
     - Expression ``COMPONENTS`` name
     - Its parameters
   * - ``ExponentialRelaxation`` ⚠
     - ``A0``, ``Lambda``, ``baseline``
     - ``Exponential`` ``+ Constant``
     - ``A``, ``Lambda`` (+ ``A_bg``)
   * - ``GaussianRelaxation`` ⚠
     - ``A0``, ``sigma``, ``baseline``
     - ``Gaussian`` ``+ Constant``
     - ``A``, ``sigma`` (+ ``A_bg``)
   * - ``LFKuboToyabe`` ⚠
     - ``A0``, ``Delta``, ``B_L``, ``baseline``
     - ``LongitudinalFieldKT`` ``+ Constant``
     - ``A``, ``Delta``, ``B_L`` (+ ``A_bg``)
   * - ``StretchedExponential``
     - ``A0``, ``Lambda``, ``beta``, ``baseline``
     - ``StretchedExponential``
     - ``A``, ``Lambda``, ``beta``
   * - ``StaticGKT_ZF``
     - ``A0``, ``Delta``, ``baseline``
     - ``StaticGKT_ZF``
     - ``A``, ``Delta``
   * - ``Abragam``
     - ``A0``, ``Delta``, ``nu``, ``baseline``
     - ``Abragam``
     - ``A``, ``Delta``, ``nu``
   * - ``Keren``
     - ``A0``, ``Delta``, ``nu``, ``B_L``, ``baseline``
     - ``Keren``
     - ``A``, ``Delta``, ``nu``, ``B_L``
   * - ``DynamicGaussianKT``
     - ``A0``, ``Delta``, ``nu``, ``B_L``, ``baseline``
     - ``DynamicGaussianKT``
     - ``A``, ``Delta``, ``nu``, ``B_L``
   * - ``DynamicLorentzianKT``
     - ``A0``, ``a_L``, ``nu``, ``B_L``, ``baseline``
     - ``DynamicLorentzianKT``
     - ``A``, ``a_L``, ``nu``, ``B_L``
   * - ``Oscillatory``
     - ``A0``, ``frequency``, ``phase``, ``Lambda``, ``baseline``
     - ``Oscillatory``
     - ``A``, ``frequency``, ``phase``

The rows marked ⚠ are the three where the **name itself differs**; the rest
share a name but still differ in their parameters (standalone models add an
explicit ``A0`` amplitude and ``baseline``; the matching ``Constant`` component
supplies the ``A_bg`` background in an expression). The standalone ``Oscillatory``
also carries its own ``Lambda`` damping and ``baseline`` that the bare
``Oscillatory`` component does not.

Mixing the two vocabularies is a common source of "unknown parameter" errors.
Decide which API you are using (composite expression vs standalone ``MODELS``
entry) and take the names from that one.

Worked example — building ``GaussianRelaxation`` inside an expression
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A standalone ``GaussianRelaxation`` is :math:`A_0\,e^{-(\sigma t)^2} +
\mathrm{baseline}`. The expression-domain equivalent is the ``Gaussian``
component plus a ``Constant`` background — note how the amplitude is mangled to
``A_1`` and the baseline becomes ``A_bg``:

.. code-block:: python

   from asymmetry.core.fitting import CompositeModel

   model = CompositeModel.from_expression("Gaussian + Constant")
   print(model.param_names)
   # ['A_1', 'sigma', 'A_bg']
   # A_1 ≡ standalone A0, sigma ≡ sigma, A_bg ≡ standalone baseline

The authoritative component list is the live registry — the
`Available Components`_ table below is a commonly-used subset. Print the full set
(the output below is current as of this writing)::

   from asymmetry.core.fitting import COMPONENTS, MODELS
   sorted(COMPONENTS)   # names usable inside CompositeModel expressions
   # ['Abragam', 'Bessel', 'Constant', 'ConstantBackground', 'DipolarPairField',
   #  'DipolarSpinJ', 'DynamicFmuF', 'DynamicGaussianKT', 'DynamicLorentzianKT',
   #  'ElectronDipole', 'Exponential', 'FmuF_General', 'FmuF_Linear',
   #  'FmuF_Triangle', 'Gaussian', 'GaussianBroadenedKT', 'GaussianPeak', 'Keren',
   #  'LinearBackground', 'LongitudinalFieldKT', 'LorentzianPeak', 'MuF',
   #  'MuoniumHighTF', 'MuoniumHighTFAniso', 'MuoniumLFRelax', 'MuoniumLowTF',
   #  'MuoniumTF', 'MuoniumZF', 'Oscillatory', 'OscillatoryField', 'ProtonDipole',
   #  'RischKehr', 'StaticGKT_ZF', 'StretchedExponential']

   sorted(MODELS)       # standalone model keys (different names)
   # ['Abragam', 'DynamicGaussianKT', 'DynamicLorentzianKT',
   #  'ExponentialRelaxation', 'GaussianRelaxation', 'Keren', 'LFKuboToyabe',
   #  'Oscillatory', 'StaticGKT_ZF', 'StretchedExponential']

Evaluate model and components
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

Available components
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
     - Hayano longitudinal-field Kubo–Toyabe (LF-KT) :math:`G_z(t;\Delta,B_L)` — see :ref:`fit-lf-kubo-toyabe`
     - ``A``, ``Delta``, ``B_L`` (Gauss)
   * - ``MuF``
     - Analytical single mu-F polarisation :math:`D_z(t)`
     - ``A``, ``r_muF`` (Å)
   * - ``FmuF_Linear``
     - Analytical collinear F-mu-F polarisation
     - ``A``, ``r_muF`` (Å)
   * - ``FmuF_General``
     - Numerical powder-averaged F-mu-F polarisation
     - ``A``, ``r1`` (Å), ``r2`` (Å), ``theta`` (°)
   * - ``Constant``
     - :math:`A_{\mathrm{bg}}`
     - ``A_bg``

Runnable example
----------------

See ``examples/composite_models.py`` for a complete executable script.
