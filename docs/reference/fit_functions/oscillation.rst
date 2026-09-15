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
*incommensurate* distribution use ``Bessel``. ``Bessel`` is the bare
precessing line; for a **powder** of the same incommensurate, single-site
structure, where a non-precessing ⅓ tail is also present, use
``OverhauserPowder`` instead. When the field distribution of a single-q
structure has two non-zero cut-offs rather than running to zero, choose by
how the local field moves: ``HelicalPowder`` when the field vector rotates on
an ellipse (helical or cycloidal order), and the
``OverhauserPowderCutoff``/``OverhauserPowderCentre`` pair when the field
keeps one direction and only its size is modulated. ``HelicalCrystal`` is the
single-crystal form of ``HelicalPowder``.

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
order use ``Oscillatory`` or ``OscillatoryField``. For a **powder** sample,
where the ⅓ non-precessing tail is also visible, use ``OverhauserPowder``
instead.

=============  ============  =====  ==========================================
Name           Symbol        Unit   Description
=============  ============  =====  ==========================================
``A``          :math:`A`     %      Component asymmetry amplitude.
``frequency``  :math:`f`     MHz    Field-distribution edge, γ\ :sub:`μ`\ B₁/2π.
``phase``      :math:`\phi`  rad    Phase offset.
=============  ============  =====  ==========================================

**References**

- L. P. Le *et al.*, Phys. Rev. B **48**, 7284 (1993).

.. _fit-overhauser-powder:

OverhauserPowder
----------------

.. math::

   A(t) = A\left[\tfrac{1}{3}\,e^{-\lambda_L t}
          + \tfrac{2}{3}\,J_0(2\pi f t)\,e^{-\lambda_T t}\right]

The polarisation of a **powder** (polycrystalline) sample of the same
incommensurate, single-site structure as ``Bessel`` — a spin-density wave with
one muon site, sampled over every crystallite orientation. Each crystallite
still sees the Overhauser field distribution
:math:`p(B) = \pi^{-1}(B_{\max}^2 - B^2)^{-1/2}` for :math:`|B| < B_{\max}`,
with :math:`f = \gamma_\mu B_{\max}/2\pi` the distribution edge and the
order parameter to trend versus temperature, but the powder average splits
the polarisation into two exact fractions: a non-precessing :math:`\tfrac{1}{3}`
of the muon spins lie along the local field at their site and relax at
:math:`\lambda_L`, while the precessing :math:`\tfrac{2}{3}` follow the
:math:`J_0` line shape of ``Bessel`` and relax at :math:`\lambda_T`.

=================  =================  =======  ==========================================
Name               Symbol             Unit     Description
=================  =================  =======  ==========================================
``A``              :math:`A`          %        Component asymmetry amplitude.
``frequency``      :math:`f`          MHz      Field-distribution edge, γ\ :sub:`μ`\ B\ :sub:`max`\ /2π.
``lambda_T``       :math:`\lambda_T`  µs⁻¹     Relaxation of the precessing ⅔ fraction.
``lambda_L``       :math:`\lambda_L`  µs⁻¹     Relaxation of the non-precessing ⅓ fraction.
=================  =================  =======  ==========================================

``frequency``, ``lambda_T`` and ``lambda_L`` are bounded non-negative. A
Bessel-like line does not on its own prove incommensurate order — a
distribution of fields from a commensurate structure with several sites, or
from disorder, can mimic it — so corroborate with the ordering wavevector
from diffraction where possible.

**References**

- A. T. Savici *et al.*, Phys. Rev. B **66**, 014524 (2002).
- L. P. Le *et al.*, Phys. Rev. B **48**, 7284 (1993).
- P. Dalmas de Réotier, A. Yaouanc, and A. Maisuradze, arXiv:1410.2767 (2014).

.. _fit-overhauser-powder-two-cutoff:

OverhauserPowderCutoff and OverhauserPowderCentre
--------------------------------------------------

.. math::

   A(t) = A\left[\tfrac{1}{3}\,e^{-\lambda_L t}
          + \tfrac{2}{3}\,J_0(2\pi\Delta f\,t)\,
            \cos(2\pi f_{\mathrm{av}} t + \phi)\,e^{-\lambda_T t}\right]

The polarisation of a **powder** sample in which the local field at the muon
keeps one direction while its size is modulated sinusoidally between *two*
non-zero cut-offs, :math:`B = B_{\mathrm{av}} + \Delta B\cos(\mathbf{q}\cdot\mathbf{r})`
— a shifted Overhauser distribution,
:math:`p(B) = \pi^{-1}[(B - B_{\min})(B_{\max} - B)]^{-1/2}`, rather than one
running down to zero as in ``OverhauserPowder``. As with
``OverhauserPowder`` the polarisation splits into a non-precessing
:math:`\tfrac{1}{3}` tail relaxing at :math:`\lambda_L` and a precessing
:math:`\tfrac{2}{3}` fraction relaxing at :math:`\lambda_T`; :math:`\phi` is
a phase offset.

The two entries fit the same lineshape in two parametrisations, sharing one
implementation:

- **OverhauserPowderCutoff** — ``frequency`` is the upper cut-off
  :math:`f_{\max}` and ``ratio`` is :math:`r = B_{\min}/B_{\max}`. Share
  ``ratio`` across a series, since it is a structural constant of the
  magnetic structure, and trend ``frequency`` as the order parameter.
- **OverhauserPowderCentre** — ``frequency`` is the centre :math:`f_{\mathrm{av}}`
  and ``delta_frequency`` is the half-width :math:`\Delta f`, the form of
  Amato *et al.* (2014). Use it to fit the two edges as independent numbers,
  or to compare directly against literature written in :math:`(f_{\mathrm{av}},
  \Delta f)`.

The two parameter sets convert into each other directly:

.. math::

   f_{\mathrm{av}} = f_{\max}\,\frac{1+r}{2}, \qquad
   \Delta f = f_{\max}\,\frac{1-r}{2}, \qquad
   f_{\min} = r\,f_{\max}

.. math::

   f_{\max} = f_{\mathrm{av}} + \Delta f, \qquad
   f_{\min} = f_{\mathrm{av}} - \Delta f, \qquad
   r = \frac{f_{\mathrm{av}} - \Delta f}{f_{\mathrm{av}} + \Delta f}

=====================  =========================  =======  ==========================================
Name                    Symbol                     Unit     Description
=====================  =========================  =======  ==========================================
``A``                   :math:`A`                  %        Component asymmetry amplitude.
``frequency``           :math:`f_{\max}`           MHz      Upper cut-off frequency (OverhauserPowderCutoff).
``ratio``               :math:`r`                  —        :math:`B_{\min}/B_{\max}` (OverhauserPowderCutoff).
``phase``               :math:`\phi`               rad      Phase offset.
``lambda_T``            :math:`\lambda_T`          µs⁻¹     Relaxation of the precessing ⅔ fraction.
``lambda_L``            :math:`\lambda_L`          µs⁻¹     Relaxation of the non-precessing ⅓ fraction.
=====================  =========================  =======  ==========================================

=====================  =========================  =======  ==========================================
Name                    Symbol                     Unit     Description
=====================  =========================  =======  ==========================================
``A``                   :math:`A`                  %        Component asymmetry amplitude.
``frequency``           :math:`f_{\mathrm{av}}`    MHz      Centre frequency (OverhauserPowderCentre).
``delta_frequency``     :math:`\Delta f`           MHz      Half-width between the two cut-offs.
``phase``               :math:`\phi`               rad      Phase offset.
``lambda_T``            :math:`\lambda_T`          µs⁻¹     Relaxation of the precessing ⅔ fraction.
``lambda_L``            :math:`\lambda_L`          µs⁻¹     Relaxation of the non-precessing ⅓ fraction.
=====================  =========================  =======  ==========================================

In the Fit Wizard, ``ratio`` and :math:`\phi` are bounded to :math:`[0, 1]`
and :math:`[-\pi, \pi]` respectively; in a manual fit only the lower bound of
0 is set on ``ratio``, ``delta_frequency``, ``lambda_T``, ``lambda_L`` and
``frequency``. ``delta_frequency`` exceeding :math:`f_{\mathrm{av}}` is not
clamped: it is telling you :math:`B_{\min}` has reached zero, and you should
switch to ``OverhauserPowder`` rather than read a negative :math:`f_{\min}`.
In the cut-off form :math:`r > 1` is not the same line: it makes
``frequency`` the *lower* cut-off and :math:`r f` the upper one, so set an
upper bound of 1 on ``ratio`` in the fit table (the Fit Wizard does), or
read a fit that crosses it with the two edges swapped.

The closed form above is exact for that fixed-direction field. For a helical
or cycloidal structure, where the field *vector* rotates on an ellipse, the
distribution is instead
:math:`p(B) = (2/\pi)\,B\,[(B^2 - B_{\min}^2)(B_{\max}^2 - B^2)]^{-1/2}`,
which carries an extra factor of :math:`2B/\sqrt{(B+B_{\min})(B_{\max}+B)}`,
and the closed form is Amato *et al.*'s narrow-distribution approximation to
it: exact as :math:`r \to 1` (a pure cosine at :math:`f_{\max}`) and worse as
:math:`r` falls, with a maximum deviation of the precessing part of about
0.03 at :math:`r = 0.8`, 0.07 at :math:`r = 0.6`, 0.12 at :math:`r = 0.4` and
0.31 at :math:`r = 0` (in units of the precessing amplitude). Use
``HelicalPowder`` for that geometry. At :math:`r = 0` the closed form does
**not** reduce to ``OverhauserPowder``'s :math:`J_0(2\pi f_{\max} t)`: a
fixed-direction field swinging between zero and :math:`B_{\max}` is not a
spin-density wave, whose field reverses sign. A fit that drives ``ratio`` (or
:math:`f_{\min}`) to zero is worth comparing with ``OverhauserPowder`` and
with ``HelicalPowder``, which reduces to it there. As with ``Bessel``, a
Bessel-like line does not on its own prove incommensurate or single-*q*
order — several commensurate sites, or disorder, can mimic it — so
corroborate with the ordering wavevector from diffraction where possible.

The Fit Parameters panel's **Create Composite Parameter**
(:doc:`../parameter_trending`) recovers the other parametrisation's
quantities as derived, uncertainty-propagated trends. From the centre form:

- ``frequency + delta_frequency`` for :math:`f_{\max}`
- ``frequency - delta_frequency`` for :math:`f_{\min}`
- ``(frequency - delta_frequency)/(frequency + delta_frequency)`` for :math:`r`

From the cut-off form:

- ``frequency*(1 + ratio)/2`` for :math:`f_{\mathrm{av}}`
- ``frequency*(1 - ratio)/2`` for :math:`\Delta f`
- ``frequency*ratio`` for :math:`f_{\min}`

In a composite model, a name shared by several components carries the index
shown in the fit table (e.g. ``frequency_1``).

The Fit Wizard offers ``OverhauserPowder`` and ``OverhauserPowderCutoff``,
each with an optional extra ``Exponential``, in its **Precession** family for
the single-run and Global Fit Wizards. ``OverhauserPowderCentre`` is not
offered as a wizard template, because it is numerically identical to the
cut-off form.

**References**

- A. Amato *et al.*, Phys. Rev. B **89**, 184425 (2014).
- P. Dalmas de Réotier *et al.*, Phys. Rev. B **93**, 144419 (2016).
- P. Dalmas de Réotier, A. Yaouanc, and A. Maisuradze, arXiv:1410.2767 (2014).

.. _fit-helical-powder:

HelicalPowder
-------------

.. math::

   A(t) = A\left[\tfrac{1}{3}\,e^{-\lambda_L t}
          + \tfrac{2}{3}\,e^{-\lambda_T t}\int_{B_{\min}}^{B_{\max}} p(B)\,
            \cos(\gamma_\mu B t + \phi)\,dB\right],
   \qquad
   p(B) = \frac{2}{\pi}\,\frac{B}{\sqrt{(B^2 - B_{\min}^2)(B_{\max}^2 - B^2)}}

The polarisation of a **powder** sample of a helical, cycloidal or other
single-*q* magnetic structure in which the local field at the muon site
rotates on an ellipse centred on zero, with semi-axes :math:`B_{\max}` and
:math:`B_{\min}`. As the helix phase runs uniformly over the sites, the field
magnitude follows :math:`p(B)`, which diverges at both cut-offs. ``frequency``
is the upper cut-off :math:`f = \gamma_\mu B_{\max}/2\pi`, the order parameter
to trend versus temperature, and ``ratio`` is :math:`r = B_{\min}/B_{\max}`, a
structural constant to share across a series. The powder average splits the
polarisation into a non-precessing :math:`\tfrac{1}{3}` tail relaxing at
:math:`\lambda_L` and a precessing :math:`\tfrac{2}{3}` fraction relaxing at
:math:`\lambda_T`.

The transform of :math:`p(B)` has no closed form, so it is evaluated
numerically to :math:`10^{-12}` of the line amplitude. :math:`p(B)` is the
shifted Overhauser density of ``OverhauserPowderCutoff`` times the smooth
factor :math:`h(B) = 2B/\sqrt{(B+B_{\min})(B+B_{\max})}`, and expanding
:math:`h` in Chebyshev polynomials turns the integral into a series of Bessel
functions,

.. math::

   \int p(B)\cos(\gamma_\mu B t + \phi)\,dB
   = \sum_{n \ge 0} c_n\,J_n(2\pi\Delta f\,t)\,
     \cos\!\left(2\pi f_{\mathrm{av}} t + \phi + \tfrac{n\pi}{2}\right),

with :math:`f_{\mathrm{av}} = f(1+r)/2`, :math:`\Delta f = f(1-r)/2`, and
coefficients :math:`c_n` that depend only on :math:`r`. The first term,
:math:`c_0 = 1`, is the ``OverhauserPowderCutoff`` line; the rest are the
correction. The series needs more terms as :math:`r` falls, so a fit near
:math:`r = 0` evaluates more slowly; at :math:`r = 0` the line is exactly
``OverhauserPowder``'s :math:`J_0(2\pi f t)`, and at :math:`r = 1` a single
cosine at :math:`f`.

=============  =================  =======  ==========================================
Name           Symbol             Unit     Description
=============  =================  =======  ==========================================
``A``          :math:`A`          %        Component asymmetry amplitude.
``frequency``  :math:`f`          MHz      Upper cut-off, γ\ :sub:`μ`\ B\ :sub:`max`\ /2π.
``ratio``      :math:`r`          —        :math:`B_{\min}/B_{\max}`.
``phase``      :math:`\phi`       rad      Phase offset; starts fixed at 0.
``lambda_T``   :math:`\lambda_T`  µs⁻¹     Relaxation of the precessing ⅔ fraction.
``lambda_L``   :math:`\lambda_L`  µs⁻¹     Relaxation of the non-precessing ⅓ fraction.
=============  =================  =======  ==========================================

``phase`` starts fixed at 0. In zero field every muon starts precessing from
the direction of its initial polarisation, and every field :math:`\mathbf{B}`
on the ellipse is paired with :math:`-\mathbf{B}`, so the true phase is zero.
A fit that improves when the phase is freed is usually absorbing a time-zero
offset, which shifts the phase in proportion to frequency, or a missing
component, rather than describing the lineshape. ``ratio`` above 1 describes
the same ellipse with its axes relabelled, so ``frequency`` then names the
lower cut-off; bound ``ratio`` to :math:`[0, 1]` to keep the usual meaning.
Compose with an additional relaxation component when the line is broader than
the ideal distribution.

**References**

- A. Amato *et al.*, Phys. Rev. B **89**, 184425 (2014).
- A. Yaouanc and P. Dalmas de Réotier, *Muon Spin Rotation, Relaxation, and
  Resonance* (Oxford University Press, Oxford, 2011).

.. _fit-helical-crystal:

HelicalCrystal
--------------

.. math::

   A(t) = A\left[W_0\,e^{-\lambda_L t}
          + e^{-\lambda_T t}\int_{B_{\min}}^{B_{\max}} p(B)\,
            \bigl[1 - b_z^2(B)\bigr]\cos(\gamma_\mu B t + \phi)\,dB\right]

The same elliptical field distribution as ``HelicalPowder`` in a **single
crystal**, where the non-precessing weight of each field is fixed by its
direction relative to the initial muon polarisation :math:`\hat{P}_0`
rather than averaged over orientations. With :math:`a` and :math:`c` the
components of :math:`\hat{P}_0` along the :math:`B_{\max}` and
:math:`B_{\min}` axes of the ellipse,

.. math::

   a = \sin\theta_h\cos\phi_h, \qquad c = \sin\theta_h\sin\phi_h,

the squared component of the field direction along :math:`\hat{P}_0` depends
on the field magnitude only,

.. math::

   b_z^2(B) = c^2 + (a^2 - c^2)\,\frac{1 - B_{\min}^2/B^2}{1 - r^2},
   \qquad
   W_0 = \int p(B)\,b_z^2(B)\,dB = \frac{a^2 + c^2 r}{1 + r}.

:math:`\theta_h` is the angle between :math:`\hat{P}_0` and the normal to the
plane the field rotates in, and :math:`\phi_h` the angle, within that plane,
between the projection of :math:`\hat{P}_0` and the :math:`B_{\max}` axis.
Fields parallel to the polarisation do not precess, so the orientation
reshapes the line as well as setting :math:`W_0`: with :math:`\hat{P}_0`
along the :math:`B_{\max}` axis (:math:`\theta_h = 90^\circ`,
:math:`\phi_h = 0`) the upper cut-off drops out of the oscillation, and along
the :math:`B_{\min}` axis the lower one does. The magic orientation
:math:`\theta_h = 54.7356^\circ`, :math:`\phi_h = 45^\circ` gives
:math:`a^2 = c^2 = \tfrac{1}{3}` and reproduces ``HelicalPowder`` exactly. The
line is evaluated by the same series as ``HelicalPowder``.

=============  =================  =======  ==========================================
Name           Symbol             Unit     Description
=============  =================  =======  ==========================================
``A``          :math:`A`          %        Component asymmetry amplitude.
``frequency``  :math:`f`          MHz      Upper cut-off, γ\ :sub:`μ`\ B\ :sub:`max`\ /2π.
``ratio``      :math:`r`          —        :math:`B_{\min}/B_{\max}`.
``theta_h``    :math:`\theta_h`   °        Angle between the polarisation and the helix-plane normal.
``phi_h``      :math:`\phi_h`     °        In-plane angle of the polarisation from the B\ :sub:`max` axis.
``phase``      :math:`\phi`       rad      Phase offset; starts fixed at 0.
``lambda_T``   :math:`\lambda_T`  µs⁻¹     Relaxation of the precessing fraction.
``lambda_L``   :math:`\lambda_L`  µs⁻¹     Relaxation of the non-precessing fraction.
=============  =================  =======  ==========================================

Take ``theta_h`` and ``phi_h`` from the known magnetic structure and the
crystal mounting and hold them fixed where possible; freeing them alongside
``A`` and ``ratio`` leaves strong correlations. Only :math:`a^2` and
:math:`c^2` enter, so both angles are defined on :math:`[0^\circ, 90^\circ]`.
Symmetry-inequivalent muon sites, or magnetic domains, generally see
differently oriented ellipses: model them with one ``HelicalCrystal`` per
inequivalent set, sharing ``frequency`` and ``ratio`` where the structure
requires it. The weights assume the detector axis lies along
:math:`\hat{P}_0`; with ``ratio`` above 1 the roles of the two axes, and so of
``phi_h``, are exchanged.

**References**

- A. Amato *et al.*, Phys. Rev. B **89**, 184425 (2014).
- A. Yaouanc and P. Dalmas de Réotier, *Muon Spin Rotation, Relaxation, and
  Resonance* (Oxford University Press, Oxford, 2011).

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
