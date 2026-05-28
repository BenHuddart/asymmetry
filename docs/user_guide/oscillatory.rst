.. _oscillatory:

Oscillatory Components
======================

A coherent muon-spin precession signal appears in the data whenever the muon
ensemble experiences a well-defined local field — either an applied
transverse field in the normal state, or a spontaneous internal field set up
by magnetic order. Asymmetry exposes two undamped oscillatory components for
this regime: ``Oscillatory``, which parameterises the precession by the
frequency directly, and ``OscillatoryField``, which parameterises it by the
field strength so that the frequency is computed from
:math:`f = \gamma_\mu B / 2\pi` with the muon gyromagnetic ratio
:math:`\gamma_\mu / 2\pi = 135.539\,\mathrm{MHz/T}`.

Both components are pure cosines. Damping is intentionally separated:
multiply by a relaxation envelope to build a physical lineshape. The two
canonical patterns are

.. code-block:: text

   Oscillatory * Exponential + Constant

for a Lorentzian-broadened line (motional narrowing, dynamic disorder) and

.. code-block:: text

   Oscillatory * Gaussian + Constant

for a Gaussian-broadened line (static field distribution, the high-field
limit of TF Kubo–Toyabe). A single ``Oscillatory + Constant`` will fit only
a perfectly coherent signal and will absorb the inevitable lineshape into
spurious phase and frequency residuals.

When a precession signal contains contributions from inequivalent muon sites
or from a distribution of internal fields — incommensurate magnetic order,
mixed phases at a phase transition, or a poorly aligned TF — a sum of two or
three ``Oscillatory`` components is generally preferable to a single
component with a broadened envelope, because the residual will tell you
which physical interpretation is correct. If the field distribution is
genuinely continuous, the time-domain Fourier transform documented in
:doc:`fourier_analysis` is the better first look.

``Oscillatory``: frequency parameterisation
-------------------------------------------

.. math::

   A(t) = A\,\cos(2\pi f t + \phi).

Parameters as registered in
``PARAM_INFO_REGISTRY``:

=============  ============  =====  ===================================================
Name           Symbol        Unit   Description
=============  ============  =====  ===================================================
``A``          :math:`A`     %      Component asymmetry amplitude.
``frequency``  :math:`f`     MHz    Precession frequency.
``phase``      :math:`\phi`  rad    Phase offset.
=============  ============  =====  ===================================================

The frequency is bounded non-negative; the phase is unrestricted.

For applied transverse fields, useful starting frequencies follow from
:math:`f\,[\mathrm{MHz}] \simeq 0.01355\,B\,[\mathrm{G}]` (equivalently
:math:`B\,[\mathrm{G}] \simeq 73.8\,f\,[\mathrm{MHz}]`). In zero-field
measurements on ordered magnets the internal field can range from a few tens
of Gauss in weakly ordered systems (:math:`f \sim 1\;\mathrm{MHz}`) up to
several Tesla in concentrated rare-earth compounds
(:math:`f \sim 10^2\;\mathrm{MHz}`); choose ``Bunch`` and the fit time window
accordingly so the precession is sampled at well above Nyquist.

The phase is determined by the geometric relationship between the muon spin
at implantation and the detector pair. In a well-tuned spectrometer the
phase of the *first* component should sit close to 0 rad; large fitted
phases on subsequent components, or a single component fitted to a
multi-frequency signal, indicate either an instrumental phase offset that
should be calibrated out or that the chosen model is wrong.

``OscillatoryField``: field parameterisation
--------------------------------------------

.. math::

   A(t) = A\,\cos\!\left(2\pi\,\frac{\gamma_\mu}{2\pi}\,B\,t + \phi\right)
   = A\,\cos(\gamma_\mu B\,t + \phi).

Parameters:

=========  ============  =====  ===================================================
Name       Symbol        Unit   Description
=========  ============  =====  ===================================================
``A``      :math:`A`     %      Component asymmetry amplitude.
``field``  :math:`B`     G      Local magnetic field at the muon site.
``phase``  :math:`\phi`  rad    Phase offset.
=========  ============  =====  ===================================================

Use the field parameterisation when the physically interesting quantity is
the field strength itself — for example when extracting the temperature
dependence of a sublattice magnetisation in an ordered magnet, or when
comparing internal fields across runs in a parameter-trending workflow (see
:doc:`parameter_trending`). The two parameterisations are otherwise
mathematically equivalent.

Standalone model
----------------

The ``MODELS`` registry exposes
``Oscillatory`` with both a damping rate and an explicit baseline,
recovering the conventional damped-cosine form in a single component:

.. math::

   A(t) = A_0\,\cos(2\pi f t + \phi)\,e^{-\lambda t} + A_{\mathrm{bg}}.

Parameters: ``A0`` (%), ``frequency`` (MHz), ``phase`` (rad), ``Lambda``
(:math:`\mu s^{-1}`), ``baseline`` (%). This is convenient for scripted
single-frequency fits where you do not want to construct a composite.

.. code-block:: python

   from asymmetry.core.fitting.engine import FitEngine
   from asymmetry.core.fitting.models import MODELS
   from asymmetry.core.fitting.parameters import Parameter, ParameterSet

   model = MODELS["Oscillatory"]
   params = ParameterSet([
       Parameter("A0", value=20.0, min=0.0),
       Parameter("frequency", value=2.0, min=0.0),
       Parameter("phase", value=0.0),
       Parameter("Lambda", value=0.2, min=0.0),
       Parameter("baseline", value=0.0),
   ])

Numerical notes
---------------

Frequencies that approach the inverse of the binned time step are
under-sampled; the fitter will frequently lock onto an alias of the true
frequency and report an excellent reduced :math:`\chi^2` for a physically
wrong answer. The safeguard is to evaluate the Fourier spectrum first
(:doc:`fourier_analysis`) and seed ``frequency`` from a peak in the power
spectrum.

The phase and frequency are weakly correlated over a short fit window. A
fit that runs out of data before completing two or three full periods will
produce a phase whose uncertainty is comparable to :math:`\pi`; either
extend the time range, fix the phase from an instrument calibration, or
move to ``OscillatoryField`` and fix the field if known.

Physics references
------------------

- S. J. Blundell, *Contemp. Phys.* **40**, 175 (1999) — primer on TF and
  ZF precession signals.
- A. Yaouanc and P. Dalmas de Réotier, *Muon Spin Rotation, Relaxation, and
  Resonance*, Ch. 4 (TF measurements) and Ch. 6 (spontaneous precession in
  ordered magnets).
