.. _fit-oscillation:

Oscillation
===========

A coherent muon-spin precession signal appears whenever the muon ensemble
experiences a well-defined local field — an applied transverse field, or a
spontaneous internal field set up by magnetic order. The components here are
*undamped*: damping is intentionally separated, so a physical line shape is
built by multiplying with a relaxation envelope,

.. code-block:: text

   Oscillatory * Exponential + Constant

for a Lorentzian-broadened line (dynamic disorder, dilute static moments) or

.. code-block:: text

   Oscillatory * Gaussian + Constant

for a Gaussian-broadened line (dense static field distribution). A bare
``Oscillatory + Constant`` fits only a perfectly coherent signal and will
absorb the inevitable line shape into spurious phase and frequency residuals.
When a signal contains several inequivalent muon sites, a sum of two or three
``Oscillatory`` components is generally preferable to one component with a
broadened envelope; if the field distribution is genuinely continuous, look
at the Fourier spectrum first (:doc:`../fourier_analysis`), and for an
*incommensurate* distribution use ``Bessel``.

.. _fit-oscillatory:

Oscillatory
-----------

.. math::

   A(t) = A\,\cos(2\pi f t + \phi)

Coherent precession parameterised by frequency. In zero field on an ordered
magnet the spontaneous frequency :math:`f = \gamma_\mu B_{\mathrm{int}}/2\pi`
acts as an order parameter and is the natural quantity to trend versus
temperature (:doc:`../parameter_trending`); in transverse field the frequency
calibrates the local field, :math:`f\,[\mathrm{MHz}] \simeq
0.01355\,B\,[\mathrm{G}]`.

=============  ============  =====  ==========================================
Name           Symbol        Unit   Description
=============  ============  =====  ==========================================
``A``          :math:`A`     %      Component asymmetry amplitude.
``frequency``  :math:`f`     MHz    Precession frequency.
``phase``      :math:`\phi`  rad    Phase offset.
=============  ============  =====  ==========================================

The frequency is bounded non-negative; the phase is unrestricted. In a
well-tuned spectrometer the phase of the first component should sit close to
0; large fitted phases usually indicate an instrumental phase offset that
should be calibrated out, or a wrong model. Frequencies approaching the
inverse binned time step alias: seed ``frequency`` from a Fourier peak
(:doc:`../fourier_analysis`) rather than letting the minimiser search. A fit
window covering fewer than two or three periods leaves the phase uncertain at
the :math:`\sim\pi` level — extend the range or fix the phase.

A standalone damped-cosine ``Oscillatory`` model (with ``Lambda`` and
``baseline``) is available in the ``MODELS`` registry.

**References**

- S. J. Blundell, Contemp. Phys. **40**, 175 (1999).

.. _fit-oscillatory-field:

OscillatoryField
----------------

.. math::

   A(t) = A\,\cos(\gamma_\mu B\,t + \phi)

The same precession parameterised by the local field :math:`B` (Gauss), with
:math:`f = \gamma_\mu B/2\pi`. Use when the physically interesting quantity
is the field itself — extracting the temperature dependence of a sublattice
magnetisation, or comparing internal fields across runs in a
parameter-trending workflow. For a transverse-field muonium experiment,
model the central diamagnetic Mu⁺ line with this component and add
``MuoniumTF`` for the Mu⁰ satellites. Mathematically equivalent to
``Oscillatory``.

=========  ============  =====  ==============================================
Name       Symbol        Unit   Description
=========  ============  =====  ==============================================
``A``      :math:`A`     %      Component asymmetry amplitude.
``field``  :math:`B`     G      Local magnetic field at the muon site.
``phase``  :math:`\phi`  rad    Phase offset.
=========  ============  =====  ==============================================

.. _fit-bessel:

Bessel
------

.. math::

   A(t) = A\,J_0(2\pi f t + \phi)

The polarisation of an **incommensurate** magnet, such as a spin-density-wave
state. When the ordering wavevector is incommensurate with the lattice, the
implanted muons uniformly sample the phase of the modulation and hence the
Overhauser distribution of local fields,
:math:`p(B) = \pi^{-1}(B_1^2 - B^2)^{-1/2}` for :math:`|B| < B_1`; the
resulting polarisation is the zeroth-order Bessel function with
:math:`f = \gamma_\mu B_1/2\pi` set by the field-distribution edge. At late
times

.. math::

   J_0(x) \simeq \sqrt{\tfrac{2}{\pi x}}\,\cos(x - \tfrac{\pi}{4}),

a damped cosine with a characteristic :math:`-45^\circ` phase — so a
free-phase ``Oscillatory`` fit that insists on a phase near
:math:`-45^\circ` is the classic sign that this component is needed.
Compose with a relaxation envelope for additional damping; for commensurate
order use ``Oscillatory`` or ``OscillatoryField``.

=============  ============  =====  ==========================================
Name           Symbol        Unit   Description
=============  ============  =====  ==========================================
``A``          :math:`A`     %      Component asymmetry amplitude.
``frequency``  :math:`f`     MHz    Field-distribution edge, γ\ :sub:`μ`\ B₁/2π.
``phase``      :math:`\phi`  rad    Phase offset.
=============  ============  =====  ==========================================

**References**

- L. P. Le *et al.*, Phys. Rev. B **48**, 7284 (1993).

.. _fit-vortex-lattice:

VortexLattice / VortexLatticePowder
-----------------------------------

.. math::

   A(t) = A\,\mathrm{Re}\!\left[e^{i(2\pi\gamma_\mu B t + \phi)}\,
          R_{VL}(t;\lambda,B_{c2})\right]

Transverse-field precession in the **mixed state of a type-II superconductor**.
Below :math:`T_c` the muon samples the inhomogeneous field of the flux-line
lattice, whose distribution :math:`p(B)` is strongly **non-Gaussian** — a sharp
low-field cutoff at the saddle point, a most-probable field below the mean, and
a long tail to high field near the vortex cores (a positively skewed line). The
relaxation :math:`R(t)=\langle e^{i 2\pi\gamma_\mu(B-\bar B)t}\rangle` is the
characteristic function of the *modified-London* field distribution of an ideal
triangular lattice. Fitting the lineshape directly — rather than a single
``Gaussian`` proxy, whose returned rate depends on the fit window and binning —
gives a window-independent penetration depth :math:`\lambda` and upper critical
field :math:`B_{c2}`.

``VortexLatticePowder`` applies the :math:`3^{1/4}\lambda_{ab}` polycrystalline
average and returns the ab-plane depth :math:`\lambda_{ab}`. The line's second
moment is calibrated to the Brandt result (see :doc:`../sc_penetration_depth`),
so the depth read from this lineshape matches the field-domain
``SC_Brandt_VortexLattice`` trend models. Multiply by a ``Gaussian`` for the
nuclear dipolar background and add ``Oscillatory + Constant`` for the
(weakly relaxing) sample-holder signal:

.. code-block:: text

   VortexLatticePowder * Gaussian + Oscillatory + Constant

==============  ===================  =====  =====================================
Name            Symbol               Unit   Description
==============  ===================  =====  =====================================
``A``           :math:`A`            %      Component asymmetry amplitude.
``field``       :math:`B`            G      Applied transverse field (usually fixed).
``phase``       :math:`\phi`         rad    Phase offset.
``lambda_ab``   :math:`\lambda`      nm     Penetration depth (ab-plane for powder).
``Bc2``         :math:`B_{c2}`       T      Upper critical field (core-size cutoff).
==============  ===================  =====  =====================================

``field`` starts fixed at the applied value. :math:`B_{c2}` is weakly
constrained by a single low-field run (where :math:`b=B/B_{c2}\to 0`); fix it
from the literature or fit the field dependence to pin it. ``lambda_ab`` is
strongly correlated with the nuclear ``Gaussian`` rate, so constrain the latter
from a normal-state (above :math:`T_c`) measurement at the same field.

**References**

- E. H. Brandt, Phys. Rev. B **68**, 054506 (2003).
- J. E. Sonier, J. H. Brewer, and R. F. Kiefl, Rev. Mod. Phys. **72**, 769 (2000).
- F. L. Pratt *et al.*, Phys. Rev. B **79**, 052508 (2009).
