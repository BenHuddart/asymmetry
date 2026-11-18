.. _fit-muonium:

Muonium
=======

Muonium (Mu = μ⁺e⁻) forms when the implanted muon captures an electron, and
its spin dynamics are governed by the hyperfine coupling :math:`A_{\mathrm{hf}}`
(4463.302765 MHz for vacuum muonium; reduced in semiconductors and molecules).
The vacuum starting value is the measurement of W. Liu et al., Phys. Rev. Lett.
**82**, 711 (1999), `doi:10.1103/PhysRevLett.82.711
<https://doi.org/10.1103/PhysRevLett.82.711>`_. In a
field :math:`B` the four energy levels follow the Breit–Rabi diagram with
reduced field

.. math::

   x = \frac{B}{B_0}, \qquad
   B_0 = \frac{A_{\mathrm{hf}}}{\gamma_e + \gamma_\mu}
   \;(\approx 1585\ \mathrm{G\ for\ vacuum\ Mu}),

and the observable transverse-field transitions :math:`\nu_{ij}` carry
amplitudes :math:`(1 \pm \delta)/4` with :math:`\delta = x/\sqrt{1+x^2}`.
The isotropic TF components use exact Breit–Rabi frequencies with a common
phase on positive-frequency cosines. Their ideal initial-spin interpretation
is checked at zero phase; a free common phase is a signal-fitting convention.
The axial ZF and anisotropic high-field definitions are specified separately
below. Model a central diamagnetic Mu⁺ line with :ref:`fit-oscillatory-field`
and compose with a relaxation envelope for additional damping.

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

At zero phase the retained pair starts at :math:`A/2`, rather than :math:`A`.
The omitted high-frequency transitions account for the remaining initial
polarisation in the ideal isotropic Hamiltonian. Their experimental
visibility depends on the instrument response.

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

This is the defined three-line component, with :math:`A(0)=5A/6` when
``phase = 0`` and ``f_cut = 0``. For an unpolarised electron and an initially
polarised muon, the ideal static powder Hamiltonian with principal couplings
:math:`(A_{\mathrm{hf}}-D,A_{\mathrm{hf}}-D,A_{\mathrm{hf}}+2D)` gives the
full polarisation by adding a further constant :math:`A/6`. That sample term
is separate from an unrelated sample-holder background. At degeneracies a
retained line can also be constant; for example, :math:`D=0` leaves the
:math:`f_3=0` contribution in the three-line component.

The cutoff replaces each bare weight by
:math:`a_k/[1+(f_k/f_{\mathrm{cut}})^2]`. It is an empirical bandwidth model,
not a calculation of an instrument's time response. The full-polarisation
identity above applies only without this filtering and with zero phase.
The ``D_mu`` convention differs from :ref:`fit-muonium-high-tf-aniso`:
the same tensor requires :math:`D_{\mathrm{highTF}}=2D_{\mathrm{ZF}}`.

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
**exact diagonalisation of the 4-level Hamiltonian**
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

Here the tensor principal values are
:math:`(A_{\mathrm{hf}}-D/2,A_{\mathrm{hf}}-D/2,A_{\mathrm{hf}}+D)`.
The frequencies are exact for that tensor, but retaining two lines with equal
weights is a high-field approximation to the observable; weak transitions
and unequal intensities are omitted. A fixed orientation quadrature does
not guarantee uniform accuracy for arbitrarily large :math:`|D t|`.

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

.. image:: /_generated/screenshots/muonium_lf_component_info.png
   :alt: Muonium LF component help showing the rate envelope and angular-rate units
   :width: 100%

.. math::

   A(t) = A\,e^{-\lambda t}, \qquad
   \lambda = \frac{\delta_{ex}^2\,\tau_c}
   {1 + (2\pi\nu_{12}\tau_c)^2}

An intratriplet approximation to longitudinal-field spin-lattice (T₁)
relaxation from fluctuating nuclear hyperfine fields, as in muonium self
diffusion. With :math:`\nu=1/\tau_c` and
:math:`\omega_{12}=2\pi\nu_{12}`, the rate is Eq. (22) of Kadono and Ito
(2025), attributed there to Kadono *et al.* (1990). It retains only the
dominant low-frequency transition; it is not a general electron spin-exchange
model or a full calculation of all transitions.

The measured polarisation in Eq. (19) of Kadono and Ito has a separate
field-dependent repolarisation amplitude :math:`g_z` multiplying this decay.
This component supplies the decay envelope scaled by ``A``; it does not
calculate :math:`g_z` and has no additional :math:`(1-\delta)` rate factor.
:math:`\delta_{ex}` is an angular-rate amplitude in μs⁻¹, without an extra
:math:`2\pi`; :math:`\nu_{12}` is a cyclic frequency in MHz.

============  =======================  =====  ==================================
Name          Symbol                   Unit   Description
============  =======================  =====  ==================================
``A``         :math:`A`                %      Relaxing amplitude.
``delta_ex``  :math:`\delta_{ex}`      μs⁻¹   Fluctuating angular-rate amplitude.
``tau_c``     :math:`\tau_c`           μs     Correlation time.
``B_L``       :math:`B_L`              G      Applied longitudinal field.
``A_hf``      :math:`A_{\mathrm{hf}}`  MHz    Hyperfine coupling (normally fixed).
============  =======================  =====  ==================================

:math:`\nu_{12}` is computed from the exact isotropic Breit–Rabi levels.
``A_hf`` defaults to vacuum muonium and normally stays fixed. Estimating a
microscopic hop rate requires the nuclear-field fluctuation assumptions and
enough field-dependent data to constrain both amplitude and correlation time.

**References**

- R. F. Kiefl *et al.*, Phys. Rev. Lett. **62**, 792 (1989).
- R. Kadono *et al.*, Phys. Rev. Lett. **64**, 665 (1990).
- R. Kadono and T. U. Ito, J. Phys. Soc. Jpn. **94**, 064601 (2025),
  Eqs. (19) and (22), https://doi.org/10.7566/JPSJ.94.064601.
