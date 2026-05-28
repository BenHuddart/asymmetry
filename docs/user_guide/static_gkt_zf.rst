.. _static-gkt-zf:

Static Gaussian Kubo–Toyabe (Zero Field)
========================================

The static Gaussian Kubo–Toyabe function describes the muon-spin
depolarisation produced by a static, isotropic, Gaussian-distributed local
field at the muon site, measured in zero applied field. It is the
foundational zero-field response for any host in which a *static* (or, on
the muon time scale, effectively static) distribution of nuclear or
electronic moments dominates the local field, and it is the diagnostic
signature for distinguishing a frozen field distribution from a dynamic one:
the characteristic recovery to a :math:`1/3` plateau at late times can only
arise when the field is static over many muon precession periods. If your
zero-field data show a deep Gaussian minimum followed by a recovery toward
roughly :math:`A_0/3`, this is the right model; if they decay monotonically
through the plateau value, the underlying field distribution is dynamic and
a stretched exponential (:doc:`stretched_exponential`) or a dynamic-KT model
is more appropriate.

The :math:`1/3` tail has a direct geometric origin. In three dimensions the
randomly oriented local field projects onto the muon spin with mean square
:math:`\langle\cos^2\theta\rangle = 1/3`, so one third of the ensemble has
its spin parallel to the local field and is not depolarised, while the
remaining two thirds precess and dephase into the Gaussian dip at
:math:`t \sim \sqrt{2}/\Delta`. The same width parameter

.. math::

   \Delta \;=\; \gamma_\mu\sqrt{\langle B^2 \rangle}

sets both the position of the dip and the timescale of the early-time
Gaussian behaviour. For metals and other diamagnetic hosts with only
nuclear moments, :math:`\Delta` is typically in the range
:math:`0.1{-}0.5\;\mu s^{-1}`; values appreciably above this almost always
imply an electronic contribution to the static-field distribution.

Time-domain component
---------------------

.. math::

   A(t) = A\,\left[\tfrac{1}{3} + \tfrac{2}{3}(1-\Delta^2 t^2)\,
                   e^{-\Delta^2 t^2/2}\right]

Parameters as registered in
``PARAM_INFO_REGISTRY``:

=========  ==============  =========  ===================================================
Name       Symbol          Unit       Description
=========  ==============  =========  ===================================================
``A``      :math:`A`       %          Component asymmetry amplitude.
``Delta``  :math:`\Delta`  μs⁻¹       Static Gaussian field-distribution width.
=========  ==============  =========  ===================================================

Both parameters are constrained non-negative by default. The Kubo–Toyabe
shape is sufficiently rigid that a sensible initial value of :math:`\Delta`
can be read straight off the data: the time of the minimum is
:math:`t_{\min} = \sqrt{2}/\Delta`, so :math:`\Delta \approx
\sqrt{2}/t_{\min}` is almost always a good seed.

Limits worth keeping in mind. At short times,

.. math::

   G_{\mathrm{KT}}^{\mathrm{ZF}}(t)
   \;=\; 1 - \tfrac{1}{2}\Delta^2 t^2 + \mathcal{O}(\Delta^4 t^4),

so when the data window does not extend past the dip the function is
indistinguishable from a Gaussian with :math:`\sigma = \Delta/\sqrt{2}`
(see :doc:`gaussian_relaxation`). At very large times the function flattens
onto :math:`A/3`, so a constant background that is itself near :math:`A/3`
can be hard to disentangle from the tail unless the dip region carries
enough statistics to pin :math:`\Delta`.

The most useful composites are

.. code-block:: text

   StaticGKT_ZF + Constant

for a pure nuclear-dipole background and

.. code-block:: text

   StaticGKT_ZF * Exponential + Constant

when an additional dynamic relaxation channel modulates the static-field
recovery — for example a slowly fluctuating electronic moment superimposed
on the nuclear dipole background.

The decisive experiment for confirming that the field distribution is
genuinely static is a small longitudinal decoupling field. Under
:math:`B_L \gtrsim 5{-}10\,\Delta/\gamma_\mu`, a static distribution
decouples and the polarisation recovers toward unity; a dynamic
distribution does not. The full LF-KT form that interpolates between
:math:`B_L = 0` and the high-field limit is documented separately in
:doc:`lf_kubo_toyabe`.

Standalone model
----------------

The ``MODELS`` registry exposes
``StaticGKT_ZF`` with an explicit baseline:

.. math::

   A(t) = A_0\,\left[\tfrac{1}{3} + \tfrac{2}{3}(1-\Delta^2 t^2)\,
                     e^{-\Delta^2 t^2/2}\right] + A_{\mathrm{bg}}.

.. code-block:: python

   from asymmetry.core.fitting.engine import FitEngine
   from asymmetry.core.fitting.models import MODELS
   from asymmetry.core.fitting.parameters import Parameter, ParameterSet

   model = MODELS["StaticGKT_ZF"]
   params = ParameterSet([
       Parameter("A0", value=25.0, min=0.0),
       Parameter("Delta", value=0.3, min=0.0),
       Parameter("baseline", value=0.0),
   ])
   result = FitEngine().fit(dataset, model.function, params)

Numerical notes
---------------

The exponent is clipped at :math:`-700`; this is purely defensive and never
affects a physically reasonable fit. The function itself is fully analytic
— there are no integrals, no quadrature settings to tune, and evaluation
cost is negligible.

What this model does *not* describe is field-distribution dynamics. If
:math:`\Delta` extracted from a Kubo–Toyabe fit changes with temperature
across an apparent transition, the natural explanations are (a) the local
moment is itself temperature dependent, or (b) the system is leaving the
fully static regime and a dynamic Kubo–Toyabe treatment is required.
Asymmetry currently implements only the static and LF static cases; a full
dynamic KT model is on the roadmap.

Physics references
------------------

- R. Kubo and T. Toyabe, in *Magnetic Resonance and Relaxation*,
  R. Blinc (ed.), North-Holland (1967), p. 810.
- R. S. Hayano, Y. J. Uemura, J. Imazato, N. Nishida, T. Yamazaki, and
  R. Kubo, *Phys. Rev. B* **20**, 850 (1979).
- S. J. Blundell et al., *Muon Spectroscopy: An Introduction*, Ch. 5.
