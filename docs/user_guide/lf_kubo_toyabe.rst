.. _lf-kubo-toyabe:

Longitudinal-Field Kubo-Toyabe Depolarization
==============================================

.. image:: /_generated/screenshots/lf_kt_series_plot.png
   :alt: Overlay of five Ag LF Kubo–Toyabe runs spanning the decoupling regime
   :width: 100%

*Synthetic Ag polycrystal LF series with Δ ≈ 0.39 μs⁻¹ and B_L = 0, 5, 10,*
*25, 50 G, spanning the textbook decoupling units γ_μB_L/Δ ∈ {0, 1, 2, 5,*
*10} (cf. Fig 5.6 of Blundell et al.). The 0 G run shows the characteristic*
*1/3 tail; as B_L grows the muon spins decouple from the nuclear dipolar*
*field and the polarisation recovers toward unity (Hayano et al. PRB 20,*
*850, 1979).*

The **LongitudinalFieldKT** component implements the static Gaussian Kubo–Toyabe depolarization
function in the presence of a longitudinal magnetic field, following Hayano et al. (1979).

For an end-to-end walk-through, see
:doc:`workflows/lf_decoupling_dynamics`.

Overview
--------

The longitudinal-field Kubo–Toyabe (LF-KT) depolarisation function
describes the time-dependent polarisation of muons precessing in a
Gaussian-distributed static magnetic field with an applied longitudinal
decoupling field. It is the workhorse model for any magnetically
disordered host where the local field is static (or effectively static)
on the muon time scale — magnetic glasses, frustrated magnets, frozen
spin systems, dilute nuclear-dipole hosts — and the experiment is
designed to extract the width :math:`\Delta` of the local-field
distribution by sweeping :math:`B_L` through the decoupling crossover.
The zero-field limit recovers the static Gaussian KT function (see
:doc:`static_gkt_zf`) and is the right model for ZF spectra that exhibit
the characteristic :math:`1/3` tail; the high-field limit recovers a
slowly decaying envelope set by the Gaussian damping. In between, the
recovery of the polarisation toward unity as :math:`\gamma_\mu B_L`
exceeds :math:`\Delta` is the unambiguous diagnostic of a static field
distribution and is the experimental signature that distinguishes a
frozen system from one whose field is dynamic on the muon time scale.

Formalism (Hayano et al. 1979)
------------------------------

The implementation follows the static Gaussian-field treatment from
Hayano et al. (Phys. Rev. B 20, 850 (1979)).

Zero-field Gaussian KT:

.. math::

   G_{\mathrm{KT}}^{\mathrm{ZF}}(t)
   = \frac{1}{3}
   + \frac{2}{3}\left(1-\Delta^2 t^2\right)\exp\left(-\frac{1}{2}\Delta^2 t^2\right)

Longitudinal-field Gaussian KT:

.. math::

   G_{\mathrm{KT}}^{\mathrm{LF}}(t)
   = 1
   - \frac{2\Delta^2}{\omega_0^2}\left[1 - \exp\left(-\frac{1}{2}\Delta^2 t^2\right)\cos(\omega_0 t)\right]
   + \frac{2\Delta^4}{\omega_0^3}\int_0^t \exp\left(-\frac{1}{2}\Delta^2\tau^2\right)\sin(\omega_0\tau)\,d\tau

with

.. math::

   \omega_0 = \gamma_\mu B_L

and a static isotropic Gaussian local-field distribution:

.. math::

   \rho(\mathbf{B})
   = \frac{\gamma_\mu^3}{\left(2\pi\Delta^2\right)^{3/2}}
     \exp\left(-\frac{\gamma_\mu^2(B_x^2 + B_y^2 + B_z^2)}{2\Delta^2}\right)

The width parameter satisfies:

.. math::

   \Delta = \gamma_\mu\sqrt{\langle B^2 \rangle}

and the decoupling parameter is

.. math::

   b = \frac{\omega_0}{\Delta} = \frac{\gamma_\mu B_L}{\Delta}

Definitions:

- :math:`\Delta`: static Gaussian relaxation rate (μs⁻¹)
- :math:`B_L`: applied longitudinal field (Gauss)
- :math:`\omega_0`: longitudinal Larmor frequency (rad/μs)
- :math:`\gamma_\mu`: muon gyromagnetic ratio

The asymmetry at time *t* is given by:

.. math::

   A(t) = A \cdot G_z(t) + \text{baseline}

Using LongitudinalFieldKT in the Fit Builder
--------------------------------------------

The ``LongitudinalFieldKT`` component appears under the **General**
category in the **Edit Function...** dialog of the fit panel. The most
common composites are ``LongitudinalFieldKT + Constant`` for a pure
LF-KT decoupling fit on top of a detector-imbalance background,
``LongitudinalFieldKT + Exponential + Constant`` when an additional
relaxing channel is present, and ``(LongitudinalFieldKT * Exponential)
+ Constant`` when a slow dynamic relaxation modulates the static-field
recovery. When a dataset's metadata carries a known applied field, the
fit panel initialises ``B_L`` from that value automatically.

Parameters
----------

The LongitudinalFieldKT component has three fitted parameters plus an optional baseline:

**A** (Amplitude)
   Initial muon asymmetry at *t* = 0.
   
   - Default: 25%
   - Unit: Dimensionless (%)
   - Constraint: A ≥ 0

**Delta** (Field-distribution width Δ)
   Width of the Gaussian static magnetic field distribution.
   
   - Default: 0.5 μs⁻¹
   - Unit: μs⁻¹
   - Constraint: Δ ≥ 0
   - Physical meaning: Characterizes the width of frozen magnetic moment distribution

**B_L** (Longitudinal field)
   Applied longitudinal magnetic field for decoupling.
   
   - Default in fitting: dataset run field ``B (G)`` when available; otherwise 0.0 G
   - Unit: Gauss (G)
   - Constraint: B_L ≥ 0
   - Physical meaning: Controls transition from zero-field (B_L = 0) to high-field decoupling regime

**baseline** (optional)
   Constant additive baseline. Typically handled separately via **Constant** component.
   
   - Default: 0.0 (%)
   - Unit: Dimensionless (%)

Behavior and Limits
-------------------

**Zero-field limit (B_L → 0)**
   When B_L is very small (:math:`|\omega_0| < 10^{-10}`), the function automatically switches to the
   zero-field Kubo–Toyabe formula for numerical stability:

   .. math::

      G_z(t) \approx \frac{1}{3} + \frac{2}{3}\left(1 - (\Delta t)^2\right)e^{-(\Delta t)^2/2}

**High-field limit (B_L → large)**
   As B_L increases, the muon precession becomes faster, and the envelope of oscillation
   is set by the Gaussian damping. This represents progressive decoupling from the field distribution.

**Monotonic decay**
   For B_L = 0, the function exhibits monotonically decreasing asymmetry (except for small
   oscillations on the time scale of 1/Δ).

Example Fits
------------

**Example 1: Zero-field KT in polycrystalline magnet**

.. code-block:: python

   # Fit data in zero longitudinal field
   model = CompositeModel(["LongitudinalFieldKT", "Constant"], ["+"])
   
   # Set initial guesses
   params = ParameterSet([
       Parameter("A", value=0.25, min=0.0, max=1.0),     # ~25% asymmetry
       Parameter("Delta", value=0.5, min=0.0, max=5.0),  # Field-distribution width
       Parameter("B_L", value=0.0, fixed=True),          # Keep at zero
       Parameter("A_bg", value=0.0, min=-0.1, max=0.1),  # Background
   ])

**Example 2: Field-decoupling crossover**

.. code-block:: python

   # Fit data with varying longitudinal field
   # LF-KT should show transition from depolarization to coherent oscillation
   
   model = CompositeModel(["LongitudinalFieldKT", "Constant"], ["+"])
   params = ParameterSet([
       Parameter("A", value=0.25, min=0.0, max=1.0),
       Parameter("Delta", value=0.5, min=0.0, max=5.0),
          Parameter("B_L", value=500.0, min=0.0, max=5000.0),  # Gauss
       Parameter("A_bg", value=0.0, min=-0.1, max=0.1),
   ])

**Example 3: Composite model with damping**

.. code-block:: python

   # LF-KT depolarization multiplied by exponential relaxation
   model = CompositeModel(
       ["LongitudinalFieldKT", "Exponential"],
       ["*"]
   )
   # This captures depolarization combined with sample relaxation

Known Limitations
-----------------

Numerical Integration
   The LF-KT formula includes a numerical integral term for the oscillatory part:
   
   .. math::
   
      I(t) = \int_0^t e^{-\Delta^2\tau^2/2}\sin(\omega_0\tau)\,d\tau

   This integral can be challenging to compute numerically when:
   
   - Very small longitudinal fields (:math:`|\omega_0| < 10^{-8}`): Consider fixing B_L = 0 and using pure zero-field KT
   - Very large field distributions (Δ > 10): Integration may require tuning
   - Noisy data with few time points: Parameter recovery (especially Δ) may be degraded
   
   The implementation uses ``scipy.integrate.quad`` with adaptive quadrature. For edge cases,
   consider:
   
   1. **Fixing B_L** if it is poorly constrained
   2. **Starting near expected values** to improve convergence
   3. **Using a simpler model** (pure zero-field KT or Exponential) as a first pass

Physics References
-------------------

- **Hayano et al.** (1979): R. S. Hayano, Y. J. Uemura, J. Imazato, N. Nishida, T. Yamazaki,
  and R. Kubo, "Zero and low field muon spin relaxation and depolarization rates in a
  random static Gaussian field," *Physical Review B*, **20**, 850–859.

- **Hillier & Cywinski** (1997): A. D. Hillier and R. Cywinski, "The study of µSR in
  randomly oriented powders and polycrystals," *Applied Magnetic Resonance*, **13**, 95–104.
  (Discussion of KT depolarization in powdered samples)

- **Kadono** (1996): Y. Kadono, "Muon spin rotation/relaxation in magnetic materials,"
  *Journal of the Physical Society of Japan*, **65**, 765–776.
  (Overview of depolarization functions including LF-KT)

Practical Notes on Fitting
--------------------------

For a zero-field-only experiment, fix ``B_L = 0`` and use the much
cheaper :doc:`static_gkt_zf` form directly; the LF-KT integral is only
needed when :math:`B_L` is genuinely a fitted or per-run-varied
quantity. When :math:`\Delta` is poorly constrained — a common
situation for noisy data or for fit windows that do not extend past the
dip — either fix :math:`B_L` from the known applied field and let only
:math:`\Delta` and the amplitude float, or fit a global LF series and
share :math:`\Delta` across runs (:doc:`global_fit_wizard`), which uses
the decoupling crossover itself to pin the width. Where a static-field
recovery is masked by additional slow dynamics, the right compose is
``(LongitudinalFieldKT * Exponential) + Constant`` rather than an
ad-hoc enlargement of :math:`\Delta`, which trades a physical width for
a phenomenological one and obscures the decoupling diagnostic.
