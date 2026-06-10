.. _wimda-fit-functions:

WiMDA Fit-Function Parity
=========================

This page documents the time-domain fit components added to close the gap with
WiMDA's fitting menu (built-in oscillation/relaxation grid plus the muonium and
dipolar user-function libraries), and gives migration recipes for the WiMDA
functions that were deliberately *not* ported because Asymmetry's parameter
constraints already express them. Notation follows Blundell, De Renzi,
Lancaster & Pratt, *Muon Spectroscopy: An Introduction* (OUP, 2022) — cited as
*MS-Intro* below. The porting study lives in
``docs/porting/wimda-fit-function-parity/``.

The component picker groups all time-domain components into submenus:
**Relaxation**, **Oscillation**, **Kubo-Toyabe**, **Muonium**,
**Nuclear dipolar**, and **Background**.

Relaxation
----------

RischKehr
^^^^^^^^^

.. math::

   A(t) = A\, e^{\Gamma t}\,\mathrm{erfc}\!\left(\sqrt{\Gamma t}\right)

Relaxation by a spin carrier diffusing in **one dimension** (e.g. a polaron on
a polymer chain): the 1D random walk keeps returning the carrier to the muon,
producing a :math:`(\pi\Gamma t)^{-1/2}` long-time tail instead of an
exponential [Risch & Kehr, *Phys. Rev. B* **46**, 5246 (1992)]. A
stretched-exponential fit drifting toward :math:`\beta \approx 1/2` at early
times is the usual hint to try this form. :math:`\Gamma` (μs⁻¹) is
constrained non-negative; WiMDA's mirrored branch for negative rates is not
ported. Evaluated via the scaled complementary error function
(``erfcx``), so there is no asymptotic-branch switch at large
:math:`\Gamma t` (WiMDA switches forms at :math:`\Gamma t = 20`).

Oscillation
-----------

Bessel
^^^^^^

.. math::

   A(t) = A\,J_0(2\pi f t + \phi)

The polarisation of an **incommensurate** (spin-density-wave) magnet, where
the muon ensemble samples the Overhauser distribution of fields between
:math:`-B_1` and :math:`+B_1` (MS-Intro eqn 6.47, with
:math:`f = \gamma_\mu B_1/2\pi`). At late times it resembles a damped cosine
with a :math:`-45^\circ` phase shift (MS-Intro eqn 6.48) — a free-phase
``Oscillatory`` fit that insists on a phase near :math:`-45^\circ` is the
classic sign that this component is needed. WiMDA once shipped (and later
withdrew) a Bessel oscillation; it is included here because SDW data cannot
otherwise be fitted.

Kubo-Toyabe
-----------

GaussianBroadenedKT
^^^^^^^^^^^^^^^^^^^

.. math::

   A(t) = A \int \! d\Delta'\, p(\Delta')\, G_{\mathrm{KT}}(t; \Delta', B_L),
   \qquad p = \mathcal{N}\!\left(\Delta, (w_\Delta \Delta)^2\right)

The static (longitudinal-field) Gaussian Kubo-Toyabe averaged over a Gaussian
**distribution of widths** :math:`\Delta` — for disordered hosts where the KT
dip is too sharp and the 1/3-tail recovery too pronounced (the
"Gaussian-broadened Gaussian" of Noakes & Kalvius, *Phys. Rev. B* **56**, 2352
(1997); WiMDA's ``Gau broad KT``). ``w_rel`` is the fractional standard
deviation of the distribution; WiMDA's ``rel width`` equals
:math:`w_\Delta\sqrt{2}`. ``w_rel = 0`` reduces exactly to
:doc:`LongitudinalFieldKT <lf_kubo_toyabe>`. Beware: broadening and dynamics
both fill in the dip — vary temperature or field before preferring one over
:doc:`DynamicGaussianKT <dynamic_relaxation>`.

Muonium
-------

MuoniumHighTF
^^^^^^^^^^^^^

.. math::

   A(t) = \frac{A}{2}\left[\cos(2\pi\nu_{12} t + \phi)
        + \cos(2\pi\nu_{34} t + \phi)\right],
   \qquad \nu_{12} + \nu_{34} = A_{\mathrm{hf}}

The high transverse-field muonium pair: above
:math:`B_0 = A_{\mathrm{hf}}/(\gamma_e+\gamma_\mu)` (≈ 1585 G for vacuum Mu)
only the two muon-spin-flip transitions survive, with frequencies summing to
the hyperfine constant (MS-Intro §4.4, eqn 4.65) — fitting the pair measures
:math:`A_{\mathrm{hf}}` directly. Frequencies come from the exact Breit-Rabi
levels; the equal :math:`1/2` weights are the high-field limit (use
``MuoniumTF`` at lower fields).

MuoniumHighTFAniso
^^^^^^^^^^^^^^^^^^

The same pair with an **axially anisotropic** hyperfine interaction, powder
averaged: each crystallite shifts the lines by :math:`\pm d/2` with
:math:`d = \tfrac{D}{2}(3\cos^2\theta - 1)`, in the isotropic + traceless
decomposition of the hyperfine tensor (MS-Intro eqn 4.68). Use for
bond-centred muonium or muoniated radicals in powders; :math:`D = 0` reduces
to ``MuoniumHighTF``. The angular average uses a 32-node Gauss-Legendre grid
(WiMDA: 15-point midpoint rule).

MuoniumLFRelax
^^^^^^^^^^^^^^

.. math::

   A(t) = A\,e^{-\lambda t}, \qquad
   \lambda = \frac{(1-\delta)\,\delta_{ex}^2\,\tau_c}
   {1 + (2\pi\nu_{12}\tau_c)^2}, \qquad
   \delta = \frac{x}{\sqrt{1+x^2}}

Longitudinal-field T₁ relaxation of muonium by a fluctuating coupling
(nuclear-hyperfine modulation by hopping, or spin exchange) sampled at the
intratriplet :math:`\nu_{12}` transition — the BPP/Redfield form used in the
muonium quantum-diffusion literature [Kiefl *et al.*, *Phys. Rev. Lett.*
**62**, 792 (1989); Kadono *et al.*, *Phys. Rev. Lett.* **64**, 665 (1990)].
Locating the T₁ minimum versus :math:`B_L` (at
:math:`2\pi\nu_{12}\tau_c \approx 1`) determines both :math:`\delta_{ex}` and
:math:`\tau_c`. :math:`\nu_{12}` uses the exact Breit-Rabi levels —
**deliberately not** WiMDA's approximate expression (see the porting study);
fitted :math:`\delta_{ex}`/:math:`\tau_c` values are therefore not directly
comparable with WiMDA's. ``A_hf`` defaults to vacuum muonium (4463 MHz) and
should normally stay fixed.

Nuclear dipolar
---------------

The spin-1/2 pair family shares the polycrystalline two-spin form (MS-Intro
eqn 4.80; Meier, *Hyperfine Interact.* **18**, 427 (1984)):

.. math::

   A(t) = \frac{A}{6}\left[1 + e^{-\lambda_T t}\left(
   2\cos\tfrac{\omega_d t}{2} + \cos\omega_d t +
   2\cos\tfrac{3\omega_d t}{2}\right)\right]

with the transverse damping :math:`\lambda_T` applied **only to the
oscillating 5/6 part** (the 1/6 term comes from local-field components
parallel to the muon spin, which do not dephase).

==================  ==============================================================
Component           :math:`\omega_d` parameterisation
==================  ==============================================================
DipolarPairField    :math:`\omega_d = \gamma_\mu B_{\mathrm{dip}}` (fit the field)
ProtonDipole        :math:`\hbar\omega_d = \mu_0\hbar^2\gamma_\mu\gamma_p/4\pi r^3`
ElectronDipole      :math:`\hbar\omega_d = \mu_0\hbar^2\gamma_\mu\gamma_e/4\pi r^3`
``MuF`` (existing)  fluorine at distance :math:`r` (see :doc:`muon_fluorine`)
==================  ==============================================================

``ProtonDipole``/``ElectronDipole`` compute :math:`\omega_d` from CODATA
gyromagnetic ratios (MS-Intro eqn 4.76); WiMDA's proton variant uses an
empirical constant its own source marks as approximate, so fitted distances
differ slightly.

DipolarSpinJ
^^^^^^^^^^^^

Zero-field polycrystalline precession of a muon coupled to **one nucleus of
spin J** with dipolar coupling ``f_dip`` and quadrupolar splitting ``f_quad``
(sign-sensitive) — the closed-form eigen-solution of Celio & Meier,
*Hyperfine Interact.* **18**, 435 (1984) (cf. the quadrupole Hamiltonian of
MS-Intro eqn 4.87). Use for μ⁺-⁶³Cu (J = 3/2), μ⁺-⁹³Nb (J = 9/2) and similar
pairs; hold ``J`` fixed at the known nuclear spin. ``J = 1/2`` reduces
exactly to the spin-1/2 pair.

DynamicFmuF
^^^^^^^^^^^

The collinear F-μ-F polarisation (MS-Intro eqn 4.81) dynamicized by the
strong-collision model (MS-Intro eqn 5.30) at fluctuation rate :math:`\nu` —
for F-μ-F signals that damp on warming as the muon starts hopping.
:math:`\nu = 0` recovers ``FmuF_Linear`` exactly; large :math:`\nu` gives
motional narrowing :math:`\exp(-2\omega_d^2 t/\nu)`. A temperature series
fitted with shared ``r_muF`` and free :math:`\nu` yields the hop rate and its
activation energy. Unlike WiMDA there is no user-visible ``tmax`` grid
parameter — the integration horizon comes from the data range.

FmuF_Triangle
^^^^^^^^^^^^^

A collinear F-μ-F pair (both fluorines at ``r_muF``) plus a **third
fluorine** at distance ``r3`` and angle ``phi3`` to the F-μ-F axis, solved
exactly in the 16-dimensional four-spin space with *all* μ-F and F-F dipolar
couplings and a full powder average (cf. the second-neighbour analysis of
Wilkinson & Blundell, *Phys. Rev. Lett.* **125**, 087201 (2020)). This
deliberately supersedes WiMDA's ``F-u-F-F``: WiMDA neglects the F-F couplings,
uses a single-crystal :math:`(P_z + 2P_x)/3` proxy instead of a powder
average, and its geometry parameterisation is internally inconsistent — fitted
distances are **not** directly comparable. Evaluation is cached per geometry;
expect slower fits than the analytic F-μ-F forms.

Migration recipes for unported WiMDA functions
----------------------------------------------

These WiMDA conveniences are equivalent to existing components plus parameter
constraints, so they were not ported as separate components:

**Scaled frequency rotation** (``otScaledFRotation``) — a cosine at
``frequency x scale``. Use ``Oscillatory`` and tie the frequency with an
expression constraint, e.g. for a component locked to 1.2x the first
component's frequency::

   frequency_2 = 1.2 * frequency_1      (expr constraint on frequency_2)

or use a link group when the ratio is exactly 1.

**Frequency-normalised stretched exponential** (``rtFstr``) — a stretched
exponential whose rate scales with the component's precession frequency. Use
``Oscillatory * StretchedExponential`` and constrain::

   Lambda = 2 * pi * c * frequency      (expr constraint on Lambda; fit c)

**Gaussian variants** (``rtGau2``, ``rtSig2``) — reparameterisations of
``Gaussian`` :math:`e^{-(\sigma t)^2}`: WiMDA's ``Gau2``
:math:`e^{-(\sigma' t)^2/2}` corresponds to :math:`\sigma = \sigma'/\sqrt{2}`
(this is also the mapping to the textbook's :math:`e^{-\Delta^2 t^2/2}`,
i.e. :math:`\sigma = \Delta/\sqrt{2}`), and ``Sig2`` fits
:math:`s_2 = \sigma^2` directly.

**RIKEN BeCu pressure cell** (``BeCu ZF``) — exactly the composite::

   StaticGKT_ZF + Exponential

with the amplitude split between the two terms. The companion empirical
``BeCu LF 110G`` calibration curve (a polynomial λ(T) for one cell at one
field) is instrument calibration data rather than a fit function and was not
ported.
