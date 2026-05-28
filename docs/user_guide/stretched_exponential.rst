.. _stretched-exponential:

Stretched Exponential
=====================

The **StretchedExponential** component describes muon-spin polarisation that
decays with a Kohlrausch–Williams–Watts time dependence,

.. math::

   A(t) = A\,e^{-(|\lambda| t)^{\beta}},

interpolating continuously between a simple exponential (:math:`\beta = 1`)
and a Gaussian (:math:`\beta = 2`). It is the standard phenomenological
relaxation function for systems in which the muon ensemble samples a
*distribution* of relaxation rates rather than a single :math:`\lambda` —
spin glasses near and below the freezing temperature, dilute and
concentrated magnetic alloys with broad RKKY-coupling distributions, frustrated
magnets with quenched disorder, and any host whose static or dynamic field
distribution is heavy-tailed enough that the second moment alone does not
characterise it.

The stretching exponent :math:`\beta` is the physically informative
parameter. A value near 1 indicates a narrow distribution of dynamic rates
and is consistent with weak inhomogeneity in an otherwise exponential
regime; a value close to :math:`0.5` is the Walstedt–Walker form expected
for a frozen, broadly distributed RKKY-like field [1]; values closer to 2
indicate a static, near-Gaussian field distribution and are usually better
fitted with a dedicated Kubo–Toyabe form (see :doc:`static_gkt_zf` and
:doc:`lf_kubo_toyabe`) that exposes the field-distribution width directly.
A fitted :math:`\beta` that drifts systematically with temperature through a
magnetic transition is one of the canonical μSR signatures of glassy
freezing: :math:`\beta \approx 1` above :math:`T_f` collapsing toward
:math:`\beta \approx 1/3` at and below the transition was the empirical
result that first established the technique in spin-glass studies [2].

Time-domain component
---------------------

.. math::

   A(t) = A\,e^{-(|\lambda| t)^{\beta}}

Parameters as registered in
``PARAM_INFO_REGISTRY``:

==========  ==============  =========  ===================================================
Name        Symbol          Unit       Description
==========  ==============  =========  ===================================================
``A``       :math:`A`       %          Component asymmetry amplitude.
``Lambda``  :math:`\lambda` μs⁻¹       Relaxation rate scale.
``beta``    :math:`\beta`   --         Stretching exponent.
==========  ==============  =========  ===================================================

All three parameters are constrained non-negative by default. The absolute
value around :math:`\lambda` in the model expression makes the fit insensitive
to the sign of small trial values during minimisation, but
:math:`\beta \le 0` is unphysical and should be constrained out explicitly
(``min=0.1`` or similar). In practice it is also wise to cap
:math:`\beta \le 2` since values above that have no standard physical
interpretation within this functional form.

Sensible initial values: :math:`\lambda` of order the inverse of the visible
relaxation time scale, :math:`\beta \approx 1` if you have no prior
information. Once the fit settles, the deviation of :math:`\beta` from unity
is the diagnostic — *not* the value of :math:`\lambda` in isolation, which
absorbs both the rate scale and the shape distortion when :math:`\beta \neq
1`.

The most common composite is

.. code-block:: text

   StretchedExponential + Constant

for a single relaxing channel. When two physically distinct channels are
present — for example a fast paramagnetic component and a slow nuclear
background — fit them as separate components with their own rates rather
than absorbing the second into a small :math:`\beta`, which trades
interpretability for one fewer free parameter.

Standalone model
----------------

The ``MODELS`` registry exposes
``StretchedExponential`` with an explicit baseline:

.. math::

   A(t) = A_0\,e^{-(|\lambda| t)^{\beta}} + A_{\mathrm{bg}}.

.. code-block:: python

   from asymmetry.core.fitting.engine import FitEngine
   from asymmetry.core.fitting.models import MODELS
   from asymmetry.core.fitting.parameters import Parameter, ParameterSet

   model = MODELS["StretchedExponential"]
   params = ParameterSet([
       Parameter("A0", value=25.0, min=0.0),
       Parameter("Lambda", value=0.5, min=0.0),
       Parameter("beta", value=1.0, min=0.1, max=2.0),
       Parameter("baseline", value=0.0),
   ])
   result = FitEngine().fit(dataset, model.function, params)

Numerical notes
---------------

The exponent is clipped at :math:`-700` to prevent overflow. The fit is
typically well behaved for :math:`0.3 \lesssim \beta \lesssim 1.8`; near
:math:`\beta \to 0` the function collapses onto a step that is degenerate
with a missing-asymmetry artefact, and near :math:`\beta \to 2` it becomes
indistinguishable from a Gaussian relaxation across any reasonable noise
level.

:math:`\lambda` and :math:`\beta` are strongly correlated: changing
:math:`\beta` rescales the effective rate, so the marginal uncertainty on
:math:`\lambda` reported by the Hessian usually understates the true
uncertainty. Where this matters, either fix :math:`\beta` at a physically
motivated value and report :math:`\lambda` at that fixed shape, or quote
the joint :math:`(\lambda, \beta)` covariance.

Physics references
------------------

1. R. E. Walstedt and L. R. Walker, *Phys. Rev. B* **9**, 4857 (1974).
2. Y. J. Uemura, T. Yamazaki, D. R. Harshman, M. Senba, and E. J. Ansaldo,
   *Phys. Rev. B* **31**, 546 (1985) — the foundational μSR study of
   stretched-exponential relaxation in spin glasses.
3. I. A. Campbell, A. Amato, F. N. Gygax, D. Herlach, A. Schenck, R. Cywinski,
   and S. H. Kilcoyne, *Phys. Rev. Lett.* **72**, 1291 (1994) — temperature
   dependence of :math:`\beta` across the glass transition.
