.. _fit-functions:

Fit Functions
=============

This chapter documents every fit function (component) available in the
fit-function builder. The pages mirror the submenus of the builder's
component picker exactly, so the section a function is documented under is
the submenu it lives under in the GUI:

.. toctree::
   :maxdepth: 2

   relaxation
   oscillation
   kubo_toyabe
   muonium
   nuclear_dipolar
   background
   frequency_domain

Components are *building blocks*: each evaluates a normalised polarisation or
relaxation shape scaled by its amplitude, and a fittable model is assembled by
combining components with ``+``, ``-``, ``*``, ``/`` and fraction groups in
the builder (see :doc:`../composite_models`). The two canonical patterns are
an additive background,

.. code-block:: text

   StaticGKT_ZF + Constant

and a multiplicative relaxation envelope on an oscillating or entangled-state
signal,

.. code-block:: text

   FmuF_Linear * Exponential + Constant

A small number of single-channel models (with an explicit ``baseline``
parameter) are also exposed through the Python ``MODELS`` registry for
scripted fits; these are noted on the relevant component sections.

Conventions
-----------

Time is measured in μs, frequencies in MHz, fields in Gauss, distances in Å,
and relaxation rates in μs⁻¹; phases are in radians. The muon gyromagnetic
ratio is :math:`\gamma_\mu/2\pi = 135.539` MHz/T. Component notation is kept
consistent with Blundell, De Renzi, Lancaster and Pratt, *Muon Spectroscopy:
An Introduction* (Oxford University Press, 2022); each function's section
cites the original literature for its functional form.

The builder only offers components belonging to the analysis domain being
fitted: time-domain representations see the time-domain categories
(:doc:`relaxation`, :doc:`oscillation`, :doc:`kubo_toyabe`, :doc:`muonium`,
:doc:`nuclear_dipolar`, :doc:`background`), while Fourier spectra are fitted
with the :doc:`frequency_domain` components.

.. _fit-function-doc-policy:

Documentation policy
--------------------

Every fit component must be documented in the page of this chapter that
corresponds to its category in the component picker, in the same pedagogical
style: when the model is physically relevant, the mathematical form (rendered
with LaTeX), parameter table, practical fitting guidance, and APS-style
references to the original literature. This placement is enforced by
``tests/test_fit_function_docs.py`` — adding a new component without
documenting it in the matching page fails the test suite.

Migrating from WiMDA
--------------------

Asymmetry's time-domain catalogue covers everything WiMDA's fitting menu
offers (built-in oscillation/relaxation grid plus the muonium and dipolar
user-function libraries; see ``docs/porting/wimda-fit-function-parity/``).
A few WiMDA conveniences are deliberately *not* separate components because
parameter constraints already express them:

**Scaled frequency rotation** (``otScaledFRotation``) — a cosine at
``frequency × scale``. Use ``Oscillatory`` and tie the frequency with an
**affine tie** (:ref:`affine-ties`), e.g. for a component locked to 1.2× the
first component's frequency::

   from asymmetry.core.fitting import AffineTie, Parameter

   Parameter("frequency_2", value=..., tie=AffineTie(main="frequency_1", scale=1.2))

or use a link group when the ratio is exactly 1.

**Frequency-normalised stretched exponential** (``rtFstr``) — a stretched
exponential whose rate scales with the component's precession frequency
(``Lambda = 2π·c·frequency``). This couples *two* fitted parameters
multiplicatively, so it is **not** an affine tie; it needs the general
expression constraint (``Parameter.expr``), which is reserved but not yet
evaluated by the engine. Until then, fit ``Lambda`` directly, or fix
``frequency`` and use an :ref:`affine tie <affine-ties>`
``AffineTie(main="frequency", scale=2*pi*c)`` for a known ``c``.

**Gaussian variants** (``rtGau2``, ``rtSig2``) — reparameterisations of
``Gaussian`` :math:`e^{-(\sigma t)^2}`: WiMDA's ``Gau2``
:math:`e^{-(\sigma' t)^2/2}` corresponds to :math:`\sigma = \sigma'/\sqrt{2}`
(also the mapping to the textbook's :math:`e^{-\Delta^2 t^2/2}` convention),
and ``Sig2`` fits :math:`s_2 = \sigma^2` directly.

**RIKEN BeCu pressure cell** (``BeCu ZF``) — exactly the composite::

   StaticGKT_ZF + Exponential

with the amplitude split between the two terms. The companion empirical
``BeCu LF 110G`` calibration curve (a polynomial λ(T) for one cell at one
field) is instrument calibration data rather than a fit function and was not
ported.
