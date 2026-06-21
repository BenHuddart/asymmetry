.. _fit-nuclear-dipolar:

Nuclear dipolar
===============

.. image:: /_generated/screenshots/muon_fluorine_pbf2.png
   :alt: Main window with a PbF₂ F-μ-F dataset and FmuF_Linear+Constant model selected
   :width: 100%

*Synthetic PbF₂ ZF dataset with r\ :sub:`μF` = 1.17 Å, captured over 20 μs so
the F–μ–F beat envelope is fully resolved. PbF₂ is a particularly clean
F–μ–F host: the heavy Pb nuclei carry no significant nuclear moment, so the
analytical FmuF_Linear component captures the full polarisation.*

When the implanted muon stops close to one or a few nuclei with substantial
moments, the muon and nuclear spins evolve as an entangled few-spin system
under the magnetic dipole–dipole interaction, producing characteristic slow
beats in the zero-field (or weak-LF) asymmetry. These beats are the textbook
signature of a well-defined stopping site, and their envelope encodes the
site geometry directly through the muon–nucleus distances. All components on
this page derive from the dipolar Hamiltonian

.. math::

   H_{ij} = \omega_{ij}\left[\mathbf{S}_i\cdot\mathbf{S}_j
   - 3\left(\mathbf{S}_i\cdot\hat{\mathbf{r}}_{ij}\right)
   \left(\mathbf{S}_j\cdot\hat{\mathbf{r}}_{ij}\right)\right],
   \qquad
   \omega_{ij} = \frac{\mu_0}{4\pi}\,\gamma_i\gamma_j\hbar\,r_{ij}^{-3},

with distances entered in Å. They are polarisation *building blocks*, not
full asymmetry models: the expected workflow is to multiply by a relaxation
envelope and add a background, e.g. ``FmuF_Linear * Exponential + Constant``.
Distinguishing an F–μ–F beat from simple two-frequency precession is usually
fastest in the frequency domain — F–μ–F gives a three-line (collinear) or
multi-line (general) pattern with characteristic spacing, while precession
gives one line per site (:doc:`../fourier_analysis`).

Choosing a model: use ``MuF``/``ProtonDipole``/``ElectronDipole`` when one
spin-½ partner dominates (or ``DipolarPairField`` to fit the dipolar field
directly); ``FmuF_Linear`` for the classic symmetric linear centre;
``FmuF_General`` for bent or asymmetric two-fluorine geometries;
``FmuF_Triangle`` when a third fluorine matters; ``DynamicFmuF`` when the
F–μ–F signal is washed out by muon hopping; and ``DipolarSpinJ`` for a
single quadrupolar (:math:`J > 1/2`) partner such as Cu or Nb.

.. _fit-muf:

MuF
---

.. math::

   D_z(t) = \frac{1}{6}\left[1 + 2\cos\left(\frac{\omega_d t}{2}\right)
   + \cos(\omega_d t) + 2\cos\left(\frac{3\omega_d t}{2}\right)\right]

The entangled two-spin μ–F pair: a muon strongly coupled to a single dominant
:sup:`19`\ F nucleus (:math:`I = 1/2`, 100 % abundant, no quadrupole),
giving the characteristic three-frequency pattern. This is Case I of the
molecular-magnet analysis of Lancaster *et al.* — the relevant model when a
symmetric site between two fluorines is chemically disfavoured, as in
CuF₂(H₂O)₂(pyz).

=========  ===================  =====  ========================================
Name       Symbol               Unit   Description
=========  ===================  =====  ========================================
``A``      :math:`A`            %      Component asymmetry amplitude.
``r_muF``  :math:`r_{\mu F}`    Å      Muon–fluorine distance.
=========  ===================  =====  ========================================

Not intended for cases where two fluorines contribute comparably, or where an
extra nearby nucleus (e.g. a proton) materially affects the spectrum.

**References**

- T. Lancaster *et al.*, Phys. Rev. Lett. **99**, 267601 (2007).

.. _fit-proton-dipole:

ProtonDipole
------------

The same two-spin form as ``MuF`` with the proton gyromagnetic ratio and an
optional transverse damping :math:`\lambda_T` applied to the oscillating
:math:`5/6` part only (the non-oscillating :math:`1/6` term arises from field
components parallel to the muon spin, which do not dephase):

.. math::

   A(t) = \frac{A}{6}\left[1 + e^{-\lambda_T t}\left(
   2\cos\tfrac{\omega_d t}{2} + \cos\omega_d t
   + 2\cos\tfrac{3\omega_d t}{2}\right)\right]

Use for stopping sites adjacent to a single dominant proton — hydroxyl
groups, hydrides, water of crystallisation. The fitted :math:`r_{\mu H}` is
the muon–proton distance through the :math:`r^{-3}` coupling; :math:`\lambda_T`
absorbs weaker couplings to more distant nuclei. Proton moments are roughly
ten times weaker than :sup:`19`\ F at the same distance, so resolvable
oscillations require a close, well-defined μ–H pair.

============  ==================  =====  =======================================
Name          Symbol              Unit   Description
============  ==================  =====  =======================================
``A``         :math:`A`           %      Component asymmetry amplitude.
``r_muH``     :math:`r_{\mu H}`   Å      Muon–proton distance.
``lambda_T``  :math:`\lambda_T`   μs⁻¹   Transverse damping of the oscillation.
============  ==================  =====  =======================================

**References**

- P. F. Meier, Hyperfine Interact. **18**, 427 (1984).

.. _fit-electron-dipole:

ElectronDipole
--------------

As ``ProtonDipole`` with the electron gyromagnetic ratio: a muon coupled by
the dipolar interaction to a single **localised** electronic moment at
distance :math:`r_{\mu e}`, static on the muon time scale — a dilute
paramagnetic defect or rare-earth ion adjacent to the muon site. Frequencies
are about three orders of magnitude higher than the nuclear pairs at the same
distance, so :math:`r_{\mu e}` of several Å still gives MHz-scale
oscillations. Not appropriate for muonium (contact hyperfine dominates — use
the :doc:`muonium` components) or for dense magnets (use ``Oscillatory`` or
``Bessel`` with an internal field). Parameters: ``A`` (%), ``r_mue`` (Å),
``lambda_T`` (μs⁻¹).

**References**

- P. F. Meier, Hyperfine Interact. **18**, 427 (1984).

.. _fit-dipolar-pair-field:

DipolarPairField
----------------

The same spin-½ pair polarisation parameterised by the **dipolar field** at
the muon, :math:`\omega_d = \gamma_\mu B_{\mathrm{dip}}`, for when it is
preferable to fit the field directly rather than assume a nucleus and
distance — e.g. when the coupled nucleus is unknown, or when comparing with
dipolar-field calculations of candidate sites. A fitted
:math:`B_{\mathrm{dip}}` converts to a distance through
:math:`B_{\mathrm{dip}} = \mu_0\hbar\gamma_j/(4\pi r^3)` once the partner
nucleus is identified. Parameters: ``A`` (%), ``B_dip`` (G), ``lambda_T``
(μs⁻¹).

**References**

- P. F. Meier, Hyperfine Interact. **18**, 427 (1984).

.. _fit-dipolar-spin-j:

DipolarSpinJ
------------

Zero-field polycrystalline precession of a muon coupled to **one nucleus of
spin** :math:`J > 1/2` with both dipolar and quadrupolar interactions. The
implanted μ⁺ produces an electric field gradient that quadrupole-splits the
neighbouring nucleus, so the two-spin spectrum depends on the quadrupolar
splitting :math:`f_{\mathrm{quad}}` (sign-sensitive) as well as the dipolar
coupling :math:`f_{\mathrm{dip}}`. The component implements the closed-form
eigen-solution of Celio and Meier, averaged as
:math:`(P_z + 2P_x)/3` for a polycrystal. Typical applications are
μ⁺–⁶³Cu (:math:`J = 3/2`) and μ⁺–⁹³Nb (:math:`J = 9/2`) pairs in metals.

============  ==========================  =====  ===============================
Name          Symbol                      Unit   Description
============  ==========================  =====  ===============================
``A``         :math:`A`                   %      Component asymmetry amplitude.
``f_dip``     :math:`f_{\mathrm{dip}}`    MHz    Dipolar coupling frequency.
``f_quad``    :math:`f_{\mathrm{quad}}`   MHz    Quadrupolar splitting.
``J_spin``    :math:`J`                   —      Nuclear spin (hold fixed).
============  ==========================  =====  ===============================

:math:`J` is fixed by default (the model is piecewise-constant in it); set
it to the known nuclear spin. For :math:`J = 1/2` the quadrupole is inactive
and the function reduces exactly to the spin-½ pair. For more than one
strongly coupled nucleus use the F–μ–F family or a dedicated multi-spin
model. Note that the implementation uses the **signed** block mixing angle,
verified against exact diagonalisation; WiMDA's ``Dip gen ZF PCR`` drops the
sign and is wrong for every :math:`J > 1/2`, so fitted parameters are not
comparable with WiMDA for those spins.

**References**

- M. Celio and P. F. Meier, Hyperfine Interact. **18**, 435 (1984).
- O. Hartmann, Phys. Rev. Lett. **39**, 832 (1977).

.. _fit-fmuf-linear:

FmuF_Linear
-----------

.. math::

   G_{F\mu F}(t)=\frac{1}{6}\left[3 + \cos(\sqrt{3}\,\omega_d t)
   + \left(1-\frac{1}{\sqrt{3}}\right)
   \cos\left(\frac{3-\sqrt{3}}{2}\,\omega_d t\right)
   + \left(1+\frac{1}{\sqrt{3}}\right)
   \cos\left(\frac{3+\sqrt{3}}{2}\,\omega_d t\right)\right]

The classic collinear three-spin F–μ–F centre of ionic fluorides: the muon
pulls two fluorines together into a hydrogen-bond-like linear configuration
and sits midway between them. This closed form (which neglects the weak F–F
coupling) is the correct starting point for LiF, NaF, CaF₂, BaF₂ and similar
hosts; Brewer *et al.* extracted typical μ–F distances of about 1.17 Å
(F–F separation ≈ 2.34–2.38 Å). Parameters: ``A`` (%), ``r_muF`` (Å). Do not
use for inequivalent fluorines or bent geometries — use ``FmuF_General``.

**References**

- J. H. Brewer *et al.*, Phys. Rev. B **33**, 7813 (1986).

.. _fit-fmuf-general:

FmuF_General
------------

For a bent or asymmetric two-fluorine geometry there is no compact closed
form. The polarisation is computed numerically: the full three-spin dipolar
Hamiltonian (including the F–F coupling) is diagonalised for each powder
orientation and

.. math::

   D_z(t) = \frac{1}{N}\sum_{m,n}
   \left|\langle m|\sigma_z^{\mu}|n\rangle\right|^2
   \cos\left[(\omega_m-\omega_n)t\right]

is averaged over orientations (Gauss–Legendre × uniform Euler-angle
quadrature); the geometry-dependent eigenspectrum is cached. This is the
model for distorted two-fluorine stopping states such as the
[Cu(NO₃)(pyz)₂]PF₆ site of Lancaster *et al.*
(:math:`r_1 = 0.106(3)` nm, :math:`r_2 = 0.156(3)` nm,
:math:`\theta = 143(1)^\circ`).

=========  ===============  =====  ============================================
Name       Symbol           Unit   Description
=========  ===============  =====  ============================================
``A``      :math:`A`        %      Component asymmetry amplitude.
``r1``     :math:`r_1`      Å      First muon–fluorine distance.
``r2``     :math:`r_2`      Å      Second muon–fluorine distance.
``theta``  :math:`\theta`   °      F–μ–F bond angle.
=========  ===============  =====  ============================================

Assumes exactly three coupled spins; it does not cover configurations
requiring an additional nucleus, such as the proton-coupled HF₂⁻ state (and
for a third *fluorine*, use ``FmuF_Triangle``).

**References**

- T. Lancaster *et al.*, Phys. Rev. Lett. **99**, 267601 (2007).
- J. H. Brewer *et al.*, Phys. Rev. B **33**, 7813 (1986).

.. _fit-fmuf-triangle:

FmuF_Triangle
-------------

A collinear F–μ–F pair (both fluorines at :math:`r_{\mu F}`) plus a **third
fluorine** at distance :math:`r_3` and angle :math:`\phi_3` to the F–μ–F
axis, solved exactly in the 16-dimensional four-spin space with *all* μ–F
and F–F dipolar couplings and a full powder average. Use when second-neighbour
fluorines visibly modify the F–μ–F beat pattern, as established for ionic
fluorides by second-neighbour analyses. As :math:`r_3 \to \infty` it
approaches the collinear limit of ``FmuF_General`` (i.e. ``FmuF_Linear`` plus
the F–F coupling).

=========  ==================  =====  ==========================================
Name       Symbol              Unit   Description
=========  ==================  =====  ==========================================
``A``      :math:`A`           %      Component asymmetry amplitude.
``r_muF``  :math:`r_{\mu F}`   Å      Muon–fluorine distance of the linear pair.
``r3``     :math:`r_3`         Å      Distance to the third fluorine.
``phi3``   :math:`\phi_3`      °      Angle of the third fluorine to the axis.
=========  ==================  =====  ==========================================

Unlike WiMDA's ``F-u-F-F`` function this includes the F–F couplings and a
proper powder average, so fitted distances are not directly comparable with
WiMDA results. Evaluation is cached per geometry; fits are slower than the
analytic F–μ–F forms.

**References**

- J. H. Brewer *et al.*, Phys. Rev. B **33**, 7813 (1986).
- J. M. Wilkinson and S. J. Blundell, Phys. Rev. Lett. **125**, 087201 (2020).

.. _fit-dynamic-fmuf:

DynamicFmuF
-----------

The collinear F–μ–F polarisation dynamicised by the strong-collision model at
fluctuation rate :math:`\nu`:

.. math::

   G^{\mathrm{d}}(t) = G_{F\mu F}(t)\,e^{-\nu t}
   + \nu\int_0^t G^{\mathrm{d}}(t-t')\,G_{F\mu F}(t')\,e^{-\nu t'}\,dt'.

Use when an F–μ–F signal that is clear at low temperature progressively damps
and loses its oscillations on warming because the muon hops away from the
site (or the coupling fluctuates). :math:`\nu = 0` recovers ``FmuF_Linear``
exactly; large :math:`\nu` gives motional narrowing toward
:math:`\exp(-2\omega_d^2 t/\nu)` via an Abragam-form interpolation that
keeps the model smooth in :math:`\nu` across the solver crossover (seam
below ~1 % at physical distances). Fitting a temperature series with shared
:math:`r_{\mu F}` and free :math:`\nu` yields the hop rate and hence an
activation energy for muon diffusion in the fluoride. Assumes the
equal-distance collinear geometry of ``FmuF_Linear``; the solver and caching
follow :ref:`fit-dynamic-gaussian-kt`.

=========  ==================  =====  ==========================================
Name       Symbol              Unit   Description
=========  ==================  =====  ==========================================
``A``      :math:`A`           %      Component asymmetry amplitude.
``r_muF``  :math:`r_{\mu F}`   Å      Muon–fluorine distance.
``nu``     :math:`\nu`         MHz    Fluctuation (hop) rate.
=========  ==================  =====  ==========================================

**References**

- J. H. Brewer *et al.*, Phys. Rev. B **33**, 7813 (1986).
- R. S. Hayano, Y. J. Uemura, J. Imazato, N. Nishida, T. Yamazaki, and
  R. Kubo, Phys. Rev. B **20**, 850 (1979).
