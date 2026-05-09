.. _lf-kubo-toyabe:

Longitudinal-Field Kubo-Toyabe Depolarization
==============================================

The **LongitudinalFieldKT** component implements the static Gaussian Kubo–Toyabe depolarization
function in the presence of a longitudinal magnetic field, following Hayano et al. (1979).

Overview
--------

The longitudinal-field Kubo–Toyabe (LF-KT) depolarization function describes the time-dependent
polarization of muons precessing in a Gaussian-distributed static magnetic field with an applied
longitudinal decoupling field. This is particularly useful for:

- Studying static magnetic field distributions (e.g., in magnetically disordered systems)
- Testing field-decoupling regimes and crossovers
- Measuring the width of local field distributions (Δ parameter)
- Analyzing relaxation in zero-field or weak longitudinal-field geometries

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

Using LongitudinalFieldKT in Build Fit
---------------------------------------

1. **Open Build Fit Dialog**: Click **Edit Function...** in the Fit panel.

2. **Search or select LongitudinalFieldKT**: In the component browser, look under the
   **General** category for **LongitudinalFieldKT** (or type the name in the expression builder).

3. **Combine with other components**: You can combine LF-KT with other depolarization functions
   using operators:

   - ``LongitudinalFieldKT + Constant`` — LF-KT with additive background
   - ``LongitudinalFieldKT + Exponential`` — LF-KT plus exponential relaxation
   - ``(LongitudinalFieldKT * Exponential) + Constant`` — Multiplicative damping

4. **Click Accept** to use the model in your fit.

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
   When B_L is very small (|ω₀| < 10⁻¹⁰), the function automatically switches to the
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
   
   - Very small longitudinal fields (|ω₀| < 10⁻⁸): Consider fixing B_L = 0 and using pure zero-field KT
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

Tips for Fitting
----------------

1. **Start with zero field** (B_L fixed = 0) if your experiment is designed for zero-field geometry.

2. **Monitor fit quality**: If reduced χ² is poor, consider:
   - Collecting more data at early times (where signal is strongest)
   - Simplifying the model (e.g., remove high-order components)
   - Checking for instrumental artifacts or phase shifts

3. **Parameter uncertainty**: Use the fit engine's covariance matrix to assess uncertainties.
   Large errors on Δ may indicate the field distribution width is poorly constrained
   (common in noisy or short-duration datasets).

4. **Combining with other physics**: You can build more complex models by composing LF-KT
   with relaxation functions:

   .. code-block:: python

      # Gaussian-damped KT with weak exponential relaxation
      model = "(LongitudinalFieldKT * Gaussian) + Constant"
      
      # This captures static-field depolarization masked by additional
      # dynamic relaxation or sample inhomogeneity.
