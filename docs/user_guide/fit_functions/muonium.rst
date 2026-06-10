.. _fit-muonium:

Muonium
=======

Muonium (Mu = μ⁺e⁻) forms when the implanted muon captures an electron, and
its spin dynamics are governed by the hyperfine coupling :math:`A_{\mathrm{hf}}`
(4463 MHz for vacuum muonium; reduced in semiconductors and molecules). In a
field :math:`B` the four energy levels follow the Breit–Rabi diagram with
reduced field

.. math::

   x = \frac{B}{B_0}, \qquad
   B_0 = \frac{A_{\mathrm{hf}}}{\gamma_e + \gamma_\mu}
   \;(\approx 1585\ \mathrm{G\ for\ vacuum\ Mu}),

and the observable transverse-field transitions :math:`\nu_{ij}` carry
amplitudes :math:`(1 \pm \delta)/4` with :math:`\delta = x/\sqrt{1+x^2}`.
All components on this page compute their frequencies from the exact
Breit–Rabi levels, use the positive-frequency (same-phase) convention, and
exclude the central diamagnetic Mu⁺ line — model that separately with
:ref:`fit-oscillatory-field`. Compose with a relaxation envelope for damping.

.. _fit-muonium-tf:

MuoniumTF
---------

.. math::

   A(t) = \frac{A}{4}\sum_{ij}(1\pm\delta)\cos(2\pi\nu_{ij} t + \phi),
   \qquad \nu_{ij} = E_i - E_j

The general transverse-field muonium component: all four hyperfine
transitions (:math:`\nu_{12}`, :math:`\nu_{23}`, :math:`\nu_{14}`,
:math:`\nu_{34}`) with their Breit–Rabi amplitudes, parameterised by field
:math:`B` and hyperfine coupling :math:`A_{\mathrm{hf}}`. In the
shallow-donor (small :math:`A_{\mathrm{hf}}`) limit it reduces to two
satellites straddling the diamagnetic line with separation
:math:`A_{\mathrm{hf}}`, so the hyperfine constant can be read off directly.

==========  =======================  =====  ====================================
Name        Symbol                   Unit   Description
==========  =======================  =====  ====================================
``A``       :math:`A`                %      Muonium asymmetry amplitude.
``field``   :math:`B`                G      Applied transverse field.
``A_hf``    :math:`A_{\mathrm{hf}}`  MHz    Hyperfine coupling constant.
``phase``   :math:`\phi`             rad    Phase offset.
==========  =======================  =====  ====================================

For genuinely shallow-donor satellites it is often more robust to fit three
independent ``Oscillatory`` lines with linked frequencies; this component
targets genuine muonium where the relative transition weights matter.

**References**

- B. D. Patterson, Rev. Mod. Phys. **60**, 69 (1988).

.. _fit-muonium-low-tf:

MuoniumLowTF
------------

.. math::

   A(t) = \frac{A}{4}\left[(1+\delta)\cos(2\pi\nu_{12} t + \phi)
   + (1-\delta)\cos(2\pi\nu_{23} t + \phi)\right]

The low-field approximation: only the two intratriplet transitions
:math:`\nu_{12}` and :math:`\nu_{23}`, which are the lines observable at low
transverse field (:math:`x \ll 1`), where the other two transitions sit near
:math:`A_{\mathrm{hf}}` and are beyond the spectrometer bandwidth. Use when
only the low-frequency pair is resolved; otherwise prefer ``MuoniumTF``.
Parameters as for ``MuoniumTF``.

**References**

- B. D. Patterson, Rev. Mod. Phys. **60**, 69 (1988).

.. _fit-muonium-zf:

MuoniumZF
---------

.. math::

   A(t) = \frac{A}{6}\sum_k a_k\cos(2\pi f_k t + \phi),
   \qquad f_1 = A_{\mathrm{hf}} - D,\quad
   f_2 = A_{\mathrm{hf}} + \tfrac{D}{2},\quad
   f_3 = \tfrac{3D}{2}

Zero-field muonium with an **axially anisotropic** hyperfine interaction:
three lines set by the isotropic coupling :math:`A_{\mathrm{hf}}` and the
axial component :math:`D`, with weights :math:`(1, 2, 2)/6` and an optional
Lorentzian cutoff ``f_cut`` suppressing lines beyond the spectrometer
bandwidth. There is no applied field, so no diamagnetic line. Relevant for
anisotropic muonium centres (e.g. bond-centred Mu in semiconductors)
measured in zero field.

==========  =========================  =====  ==================================
Name        Symbol                     Unit   Description
==========  =========================  =====  ==================================
``A``       :math:`A`                  %      Muonium asymmetry amplitude.
``A_hf``    :math:`A_{\mathrm{hf}}`    MHz    Isotropic hyperfine coupling.
``D_mu``    :math:`D`                  MHz    Axial hyperfine anisotropy.
``f_cut``   :math:`f_{\mathrm{cut}}`   MHz    Lorentzian cutoff (0 = off).
``phase``   :math:`\phi`               rad    Phase offset.
==========  =========================  =====  ==================================

**References**

- B. D. Patterson, Rev. Mod. Phys. **60**, 69 (1988).

.. _fit-muonium-high-tf:

MuoniumHighTF
-------------

.. math::

   A(t) = \frac{A}{2}\left[\cos(2\pi\nu_{12} t + \phi)
   + \cos(2\pi\nu_{34} t + \phi)\right],
   \qquad \nu_{12} + \nu_{34} = A_{\mathrm{hf}}

The high transverse-field muonium pair. Above :math:`B_0` only the two
muon-spin-flip transitions survive (the :math:`1-\delta` amplitudes vanish as
:math:`\delta \to 1`), and their frequencies sum to the hyperfine constant —
so fitting the pair measures :math:`A_{\mathrm{hf}}` directly even when
neither line is individually assigned. The equal :math:`1/2` weights are the
high-field limit; at lower fields, where the amplitudes differ, use
``MuoniumTF``. Parameters as for ``MuoniumTF``.

**References**

- B. D. Patterson, Rev. Mod. Phys. **60**, 69 (1988).

.. _fit-muonium-high-tf-aniso:

MuoniumHighTFAniso
------------------

.. math::

   A(t) = \frac{A}{2}\left\langle
   \cos\!\left[2\pi\nu_{12}(\theta)\,t+\phi\right]
   + \cos\!\left[2\pi\nu_{34}(\theta)\,t+\phi\right]
   \right\rangle_{\cos\theta},
   \qquad \nu_{12}(\theta) + \nu_{34}(\theta) \simeq
   A_{\mathrm{hf}} + \tfrac{D}{2}\left(3\cos^2\theta - 1\right)

The high-TF pair with an axially anisotropic hyperfine interaction, powder
averaged: writing the hyperfine tensor as an isotropic part
:math:`A_{\mathrm{hf}}` plus an axial (traceless) part :math:`D`, the two
muon-spin-flip frequencies are obtained for each crystallite orientation by
**exact diagonalization of the 4-level Hamiltonian**
:math:`H = \gamma_e B S_z^e - \gamma_\mu B S_z^\mu + S^e\!\cdot\!A(\theta)\!\cdot\!S^\mu`,
batched over a 32-node Gauss–Legendre :math:`\cos\theta` grid. Both lines
co-shift so that each orientation's pair sum tracks the secular effective
coupling :math:`A_{\mathrm{eff}}(\theta) = A_{\mathrm{hf}} +
\tfrac{D}{2}(3\cos^2\theta - 1)`, producing the characteristic asymmetric
(Pake-like) powder broadening. Use for bond-centred muonium in semiconductors
or muoniated radicals in powders; :math:`D = 0` reduces exactly to
``MuoniumHighTF``, and for single crystals fit the orientation-dependent
lines directly. (WiMDA's ``AnisMuoniumPairRot`` instead splits its two signed
line frequencies by a symmetric :math:`\pm d/2`, which is only approximate —
fitted :math:`D` values are not directly comparable.)

==========  =======================  =====  ====================================
Name        Symbol                   Unit   Description
==========  =======================  =====  ====================================
``A``       :math:`A`                %      Muonium asymmetry amplitude.
``field``   :math:`B`                G      Applied transverse field.
``A_hf``    :math:`A_{\mathrm{hf}}`  MHz    Isotropic hyperfine coupling.
``D_mu``    :math:`D`                MHz    Axial hyperfine anisotropy.
``phase``   :math:`\phi`             rad    Phase offset.
==========  =======================  =====  ====================================

**References**

- B. D. Patterson, Rev. Mod. Phys. **60**, 69 (1988).
- E. Roduner and H. Fischer, Chem. Phys. **54**, 261 (1981).

.. _fit-muonium-lf-relax:

MuoniumLFRelax
--------------

.. math::

   A(t) = A\,e^{-\lambda t}, \qquad
   \lambda = \frac{(1-\delta)\,\delta_{ex}^2\,\tau_c}
   {1 + (2\pi\nu_{12}\tau_c)^2}, \qquad
   \delta = \frac{x}{\sqrt{1+x^2}}

Longitudinal-field spin-lattice (T₁) relaxation of muonium by a fluctuating
coupling — nuclear hyperfine fields modulated by muonium hopping, or electron
spin exchange with carriers — sampled at the intratriplet :math:`\nu_{12}`
transition in the BPP/Redfield form used throughout the muonium
quantum-diffusion literature. The :math:`(1-\delta)` prefactor quenches the
relaxation as the muon repolarises in high field; together with the growing
:math:`\nu_{12}` this produces the LF quenching curves from which hop rates
are extracted. Measuring :math:`\lambda` versus :math:`B` and locating the
T₁ minimum (:math:`2\pi\nu_{12}\tau_c \approx 1`) determines both
:math:`\delta_{ex}` and :math:`\tau_c`.

============  =======================  =====  ==================================
Name          Symbol                   Unit   Description
============  =======================  =====  ==================================
``A``         :math:`A`                %      Relaxing amplitude.
``delta_ex``  :math:`\delta_{ex}`      MHz    Fluctuating-coupling amplitude.
``tau_c``     :math:`\tau_c`           μs     Correlation time.
``B_L``       :math:`B_L`              G      Applied longitudinal field.
``A_hf``      :math:`A_{\mathrm{hf}}`  MHz    Hyperfine coupling (normally fixed).
============  =======================  =====  ==================================

:math:`\nu_{12}` is computed from the exact Breit–Rabi levels —
intentionally *not* WiMDA's approximate expression (see
``docs/porting/wimda-fit-function-parity/``), so fitted
:math:`\delta_{ex}`/:math:`\tau_c` are not directly comparable with WiMDA's.
``A_hf`` defaults to vacuum muonium and should normally stay fixed. This is a
relaxation envelope: multiply an oscillating component, or use it standalone
for the repolarised muonium fraction.

**References**

- R. F. Kiefl *et al.*, Phys. Rev. Lett. **62**, 792 (1989).
- R. Kadono *et al.*, Phys. Rev. Lett. **64**, 665 (1990).
