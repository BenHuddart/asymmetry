.. _gaussian-relaxation:

Gaussian Relaxation
===================

The **Gaussian** component describes muon-spin polarisation that decays
as a Gaussian in time,

.. math::

   A(t) = A\,e^{-(\sigma t)^2}.

It is the natural relaxation envelope when the muon ensemble experiences a
*static* Gaussian distribution of local fields and the experimental time
window is short compared with :math:`1/\sigma`. Under those conditions
the full static Gaussian Kubo–Toyabe function (see :doc:`static_gkt_zf`)
reduces to a Gaussian on its early-time branch: expanding

.. math::

   G_{\mathrm{KT}}^{\mathrm{ZF}}(t)
   = \tfrac{1}{3} + \tfrac{2}{3}(1-\Delta^2 t^2)\,e^{-\Delta^2 t^2/2}

for :math:`\Delta t \ll 1` gives :math:`1 - \tfrac{1}{2}\Delta^2 t^2 +
\mathcal{O}(\Delta^4 t^4)`, which matches :math:`e^{-(\sigma t)^2}` to leading
order with :math:`\sigma = \Delta/\sqrt{2}`. The Gaussian component therefore
applies wherever the underlying physics is a nuclear dipolar field
distribution (or a frozen-disorder analogue) but the data either do not reach,
or are not expected to recover, the characteristic :math:`1/3` tail. The
corresponding rate is set by the second moment of the field distribution at
the muon site,

.. math::

   \sigma \;=\; \gamma_\mu\sqrt{\langle B^2 \rangle}.

The same form also captures the Gaussian damping envelope of a Gaussian
TF-precession signal, where :math:`\sigma` is again related to the width of
the field distribution sampled by the muon.

This chapter documents both the time-domain ``Gaussian`` component used in
composite expressions and the standalone ``GaussianRelaxation`` model exposed
through the Python API.

Time-domain component
---------------------

.. math::

   A(t) = A\,e^{-(\sigma t)^2}

Parameters as registered in
``PARAM_INFO_REGISTRY``:

=========  ==========  =========  ===================================================
Name       Symbol      Unit       Description
=========  ==========  =========  ===================================================
``A``      :math:`A`   %          Component asymmetry amplitude.
``sigma``  :math:`\sigma` μs⁻¹    Gaussian relaxation rate.
=========  ==========  =========  ===================================================

Both parameters are constrained non-negative by default.

For polycrystalline samples with only nuclear moments, typical values lie in
the range :math:`\sigma \sim 0.1{-}0.5\;\mu s^{-1}` (most metallic hydrogen-
free hosts) and up to about :math:`1\;\mu s^{-1}` when light, high-moment
nuclei such as :sup:`1`\ H or :sup:`19`\ F are dense at the muon site. Values
substantially larger than this almost always signal that a coupled F–μ–F (or
similar) entangled state is being mis-fitted as a Gaussian envelope; in that
case use the dedicated components in :doc:`muon_fluorine`.

If the data window includes times :math:`t \gtrsim 1/\sigma` and the signal
does not flatten onto a :math:`1/3` baseline as the static-field hypothesis
would predict, the relaxation is *not* purely Gaussian — switch to
``StretchedExponential`` (with :math:`\beta < 2`) or to the full
``StaticGKT_ZF`` and assess the fit quality at late times.

The ``Gaussian`` component is most often used inside expressions of the form

.. code-block:: text

   Gaussian + Constant

for a static nuclear background, or

.. code-block:: text

   Oscillatory * Gaussian + Constant

for a Gaussian-damped TF precession signal.

Standalone model
----------------

The ``MODELS`` registry exposes
``GaussianRelaxation`` with an explicit baseline:

.. math::

   A(t) = A_0\,e^{-(\sigma t)^2} + A_{\mathrm{bg}}.

.. code-block:: python

   from asymmetry.core.fitting.engine import FitEngine
   from asymmetry.core.fitting.models import MODELS
   from asymmetry.core.fitting.parameters import Parameter, ParameterSet

   model = MODELS["GaussianRelaxation"]
   params = ParameterSet([
       Parameter("A0", value=25.0, min=0.0),
       Parameter("sigma", value=0.3, min=0.0),
       Parameter("baseline", value=0.0),
   ])
   result = FitEngine().fit(dataset, model.function, params)

Numerical notes
---------------

The exponent is clipped at :math:`-700` to prevent overflow at unphysically
large :math:`\sigma`. The Gaussian shape has its strongest sensitivity to
:math:`\sigma` near :math:`\sigma t \sim 1`; data points well inside the
plateau (:math:`\sigma t \ll 1`) constrain only :math:`A`, while points well
beyond :math:`\sigma t \sim 3` add little because the signal has already
decayed. When :math:`\sigma` is poorly determined, the most effective
remedies are usually to extend the fit range to longer times or — preferably —
to repeat the measurement under a small longitudinal decoupling field and
fit the full LF-KT form simultaneously (see :doc:`lf_kubo_toyabe`).

Physics references
------------------

- R. S. Hayano, Y. J. Uemura, J. Imazato, N. Nishida, T. Yamazaki, and
  R. Kubo, *Phys. Rev. B* **20**, 850 (1979).
- A. Abragam, *Principles of Nuclear Magnetism*, Oxford University Press
  (1961), Ch. IV — the Gaussian short-time limit as a special case of
  motional-narrowing theory.
- S. J. Blundell et al., *Muon Spectroscopy: An Introduction*, Ch. 5.
