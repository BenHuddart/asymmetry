.. _exponential-relaxation:

Exponential Relaxation
======================

The **Exponential** component describes muon-spin polarisation that decays
exponentially with time,

.. math::

   A(t) = A\,e^{-\lambda t},

and is the workhorse depolarisation function for any system in which the muon
experiences a rapidly fluctuating local field. It is the Redfield motional-
narrowing limit of slower-relaxation forms: when the fluctuation rate
:math:`\nu` of the local field is large compared with its static second moment
:math:`\Delta`, the depolarisation collapses from a Kubo–Toyabe shape onto a
pure exponential with rate :math:`\lambda \simeq 2\Delta^2/\nu`. In practice
this covers paramagnetic spin fluctuations, electronic relaxation in metals
and semiconductors at temperatures where coherent dynamics are absent, and the
high-temperature regime of essentially every diffusive system once the
correlation time has dropped below the muon precession scale.

The same functional form arises in two physically distinct contexts that this
chapter distinguishes explicitly: as the *time-domain* ``Exponential``
component used inside a composite :math:`A(t)` expression, and as the
*standalone* ``ExponentialRelaxation`` model that adds an explicit ``baseline``
parameter. The first is what the GUI builder inserts. The second appears in
the ``MODELS`` registry for direct use through the Python API.

Time-domain component
---------------------

.. math::

   A(t) = A\,e^{-\lambda t}

Parameters as registered in
``PARAM_INFO_REGISTRY``:

==========  ==========  ==========  ===================================================
Name        Symbol      Unit        Description
==========  ==========  ==========  ===================================================
``A``       :math:`A`   %           Component asymmetry amplitude.
``Lambda``  :math:`\lambda` μs⁻¹    Exponential relaxation rate.
==========  ==========  ==========  ===================================================

Both parameters are constrained non-negative by default.

Sensible initial values depend on the host. For a paramagnetic salt at room
temperature, :math:`\lambda \sim 0.1{-}1\;\mu s^{-1}` is typical; for fast
electronic dynamics in metals near a magnetic transition, values of a few
:math:`\mu s^{-1}` are not unusual. Above roughly :math:`\lambda \sim
10\;\mu s^{-1}` the relaxation occurs almost entirely inside the instrumental
deadtime window and ``Lambda`` becomes ill-conditioned; in that regime the
signal is better described as missing initial asymmetry than as a fast
exponential.

Combining with other components is the usual route to a fittable :math:`A(t)`:
add a ``Constant`` term to capture detector imbalance and any time-independent
background, multiply by an oscillatory term for damped TF precession, or use
``Exponential`` as the dynamic envelope on a static-field component such as
``MuF`` or ``LongitudinalFieldKT``. For example,

.. code-block:: text

   FmuF_Linear * Exponential + Constant

is the standard fluoride-host expression and

.. code-block:: text

   LongitudinalFieldKT * Exponential + Constant

is appropriate when a static field distribution is masked by an additional
dynamic relaxation channel.

Standalone model
----------------

The ``MODELS`` registry exposes
``ExponentialRelaxation`` with an explicit baseline:

.. math::

   A(t) = A_0\,e^{-\lambda t} + A_{\mathrm{bg}}.

The standalone variant is convenient for scripted single-component fits where
you do not want the overhead of constructing a ``CompositeModel``. Parameters
are ``A0`` (initial asymmetry, %), ``Lambda`` (μs⁻¹) and ``baseline`` (%).

.. code-block:: python

   from asymmetry.core.fitting.engine import FitEngine
   from asymmetry.core.fitting.models import MODELS
   from asymmetry.core.fitting.parameters import Parameter, ParameterSet

   model = MODELS["ExponentialRelaxation"]
   params = ParameterSet([
       Parameter("A0", value=25.0, min=0.0),
       Parameter("Lambda", value=0.5, min=0.0),
       Parameter("baseline", value=0.0),
   ])
   result = FitEngine().fit(dataset, model.function, params)

For anything more elaborate than a single channel — multiple amplitudes,
shared baselines across detector groups, multiplicative envelopes — use the
composite-model path documented in :doc:`composite_models` instead.

Numerical notes
---------------

The implementation clips the exponent at :math:`-700` to prevent overflow at
unphysically large ``Lambda``; this is invisible at any value a fit will
realistically converge to but matters if you evaluate the model on a grid that
extends well beyond the data range.

When :math:`\lambda` is being recovered alongside a slow Gaussian or
Kubo–Toyabe channel, expect correlation between the two rates: short-time data
constrain :math:`A` and the combined initial slope, while the late-time tail
disambiguates the two relaxation mechanisms. Restricting the fit range or
adding LF-decoupling data of the same sample is usually the cleanest way to
break that degeneracy.

Physics references
------------------

- A. Yaouanc and P. Dalmas de Réotier, *Muon Spin Rotation, Relaxation, and
  Resonance: Applications to Condensed Matter*, Oxford University Press (2011).
- S. J. Blundell, R. De Renzi, T. Lancaster, and F. L. Pratt (eds.),
  *Muon Spectroscopy: An Introduction*, Oxford University Press (2021),
  Ch. 4–5.
