Superconducting penetration depth models
========================================

.. image:: /_generated/screenshots/parameter_trending_mgb2.png
   :alt: σ(T) two-gap fit on synthetic MgB₂ data
   :width: 100%

*Synthetic MgB₂ σ(T) data (Tc = 36 K) fitted with the two-gap*
*``SC_TwoGap_SS`` superconductor model. The σ(T) curve maps onto the*
*London penetration depth via ρ_s(T) ∝ 1/λ²(T) and the relation*
*σ ∝ 1/λ², making μSR one of the few bulk probes of the superfluid*
*density in unconventional superconductors (Sonier et al., Rev. Mod. Phys.*
*72, 769 (2000)).*

For an end-to-end walk-through that exercises every model on this
page, see :doc:`/workflows/superconductor_penetration_depth`.

This page documents the superconducting sigma(T) models used for
transverse-field (TF) μSR vortex-state analysis and interleaves the core theory with the API for each
model family.

Scope and terminology
---------------------

In this context, ``lambda`` denotes the London penetration depth
:math:`\lambda_L(T)`, not the time-domain relaxation-rate symbol ``Lambda``
used in asymmetry models for :math:`A(t)`.

The superconducting workflow is:

1. Fit each TF-muSR run in the time domain to extract a Gaussian rate
   :math:`\sigma(T_i)`.
2. Fit the resulting :math:`\sigma(T)` trend with superconducting gap models.
3. Interpret the fitted model in terms of :math:`\rho_s(T)` and optionally
   convert to :math:`\lambda_L(T)`.

Core equations
--------------

In the London limit for a triangular vortex lattice,

.. math::

   \sigma_{sc}(T) \propto \lambda_L^{-2}(T) \propto \rho_s(T),

with normalised superfluid density

.. math::

   \rho_s(T) = \left[\frac{\lambda(0)}{\lambda(T)}\right]^2.

The code returns ``sigma(T)`` through either

.. math::

   \sigma(T) = \sigma_0\,\rho_s(T) + \sigma_{bg}

or the quadrature convention

.. math::

   \sigma^2(T) = \sigma_{sc}^2\,\rho_s^2(T) + \sigma_{nm}^2.

The quadrature form is motivated when superconducting and non-superconducting
linewidth channels are treated as statistically independent Gaussian
contributions to the field-distribution second moment [3].

The gap is factorised as

.. math::

   \Delta(T,\mathbf{k}) = \Delta_0\,\delta(T/T_c)\,g(\mathbf{k}),

with the normalised superfluid density :math:`\rho_s(T)` obtained from the
semiclassical response integral and the reduced-gap temperature law
:math:`\delta(T/T_c)`. The derivation-level detail of both — the response
tensor and the per-symmetry gap-amplitude constants used by each model — is
collapsed below; the practical starting point for most users is the
model-selection table that follows.

Model selection at a glance
---------------------------

.. list-table:: Physical Appropriateness Guide
   :header-rows: 1

   * - Model
     - Node Structure
     - Typical Low-T Trend
     - Use When
   * - ``SC_SWave``
     - Fully gapped
     - Activated/exponential
     - Baseline nodeless BCS scenario
   * - ``SC_DWave``
     - Line nodes
     - Approximately linear (clean)
     - Cuprate-like nodal behaviour is expected
   * - ``SC_AnisotropicS_Cos4``
     - Nodeless for :math:`|a|<1`
     - Weaker than nodal d-wave
     - Fourfold anisotropy without committed nodal symmetry
   * - ``SC_SPlusG``
     - Strong anisotropy, can be near-nodal
     - Intermediate between simple s and d
     - Anisotropic singlet systems not captured by pure d-wave
   * - ``SC_NonmonotonicD``
     - d-wave with shifted angular maximum
     - Can show sub-linear curvature
     - Electron-doped cuprate phenomenology
   * - ``SC_TwoGap_SS``
     - Two nodeless gaps
     - Curvature from two energy scales
     - MgB2-style multiband superconductivity [2]
   * - ``SC_TwoGap_SD``
     - Mixed nodal and nodeless
     - Interpolates between s and d signatures
     - One band appears nodal and another appears nodeless

.. dropdown:: Mathematical detail: the normalised superfluid-density integral

   The semiclassical response tensor form used in the literature is [1]

   .. math::

      R_{ij}(T) = \frac{e^2}{4\pi^3\hbar c}
      \int_{FS} dS_k\,\frac{v_{Fi}v_{Fj}}{|v_F|}
      \left[
      1 + 2\int_{\Delta(\mathbf{k},T)}^{\infty}
      \frac{\partial f}{\partial E}
      \frac{E\,dE}{\sqrt{E^2-\Delta^2(\mathbf{k},T)}}
      \right].

   With :math:`\rho_{ii}(T)=R_{ii}(T)/R_{ii}(0)`, the practical fitting form is

   .. math::

      \rho_s(T)=1+2\left\langle
      \int_{\Delta(T,\mathbf{k})}^{\infty}
      \frac{\partial f}{\partial E}
      \frac{E\,dE}{\sqrt{E^2-\Delta^2(T,\mathbf{k})}}
      \right\rangle_{FS}.

   The implementation evaluates the energy and angular integrals with
   Gauss-Legendre quadrature and enforces stable limits
   :math:`\rho_s(0)=1`, :math:`\rho_s(T\ge T_c)=0`.

.. dropdown:: Mathematical detail: gap-amplitude approximations per symmetry

   The implementation does not apply one temperature-amplitude law to every
   symmetry.

   For isotropic s-wave-derived channels, the reduced gap uses the
   Carrington-Manzano interpolation [2]

   .. math::

       \delta_{CM}(t)=\tanh\left(1.82[1.018(1/t-1)]^{0.51}\right),\quad t=T/T_c,

   while models with tabulated symmetry-dependent weak-coupling shape factors
   use the generalised Gross-style form discussed in the review literature [1]

   .. math::

       \delta_{gen}(t)=\tanh\left[
       \frac{\pi}{\Delta_0(0)/(k_B T_c)}\sqrt{a\left(\frac{1}{t}-1\right)}
       \right],\quad t=T/T_c.

   This reduced form is algebraically equivalent to the review expression once
   energies are written in units of :math:`k_B T_c`.

   The current code paths are:

   - Isotropic s-wave, alpha model, and s-wave channels in two-gap models:
     use :math:`\Delta_0(0)/(k_B T_c)=1.764`, following Carrington-Manzano.
   - d-wave and nonmonotonic d-wave channels:
     default :math:`\Delta_0(0)/(k_B T_c)=2.14`, generalised form with
     :math:`a=4/3`.
   - extended s-wave channel with :math:`g(\phi)=\cos(2\phi)` or
     :math:`|\cos(2\phi)|`:
     default :math:`\Delta_0(0)/(k_B T_c)=2.14`, generalised form with
     :math:`a=4/3`.
   - s+g channel:
     default :math:`\Delta_0(0)/(k_B T_c)=2.77`, generalised form with
     :math:`a=2`.
   - Anisotropic s-wave and p-wave examples:
     model-dependent gap ratio; use the generalised form only when the user
     supplies a positive ``shape_factor_a``. Otherwise these channels fall back
     to the Carrington-Manzano interpolation.

   In the GUI, ``shape_factor_a`` may be left at ``0`` to indicate that no
   symmetry-specific weak-coupling value is being supplied. A positive fixed
   value uses the generalised amplitude law directly, while a positive free
   value allows the fit to determine :math:`a`.

Superconducting gap models
--------------------------

Isotropic s-wave
^^^^^^^^^^^^^^^^

.. math::

   g(\phi)=1,\qquad \Delta_0(0)/(k_B T_c)\approx 1.764.

Use when low-T behaviour is exponentially activated and no nodal signatures are
required [1].

.. autofunction:: asymmetry.core.fitting.sc.models.rho_s_wave
   :no-index:

.. autofunction:: asymmetry.core.fitting.sc.models.sc_s_wave
   :no-index:

d_{x^2-y^2} d-wave
^^^^^^^^^^^^^^^^^^

.. math::

   g(\phi)=\cos(2\phi),\qquad \Delta_0(0)/(k_B T_c)\approx 2.14.

Use when line nodes are physically expected (for example cuprate-like systems)
and low-T data are inconsistent with activated behaviour [1].

.. autofunction:: asymmetry.core.fitting.sc.models.rho_d_wave
   :no-index:

.. autofunction:: asymmetry.core.fitting.sc.models.sc_d_wave
   :no-index:

Anisotropic s-wave (cos4)
^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

   g(\phi)=1+a\cos(4\phi).

Use when an s-wave baseline is too rigid but a strict nodal d-wave model is
not yet justified. For :math:`|a|<1` the model remains nodeless. If
``shape_factor_a`` is left at ``0``, the model uses the Carrington-Manzano
temperature dependence; if ``shape_factor_a > 0``, it switches to the
generalised weak-coupling reduced-gap law with that value.

.. autofunction:: asymmetry.core.fitting.sc.models.rho_anisotropic_s_cos4
   :no-index:

.. autofunction:: asymmetry.core.fitting.sc.models.sc_anisotropic_s_cos4
   :no-index:

s+g anisotropic singlet
^^^^^^^^^^^^^^^^^^^^^^^

.. math::

   g(\theta,\phi)=\frac{1-\sin^4\theta\cos(4\phi)}{2}.

Use when strong anisotropy is indicated but pure
:math:`d_{x^2-y^2}` does not capture the observed shape [1].

.. autofunction:: asymmetry.core.fitting.sc.models.rho_s_plus_g
   :no-index:

.. autofunction:: asymmetry.core.fitting.sc.models.sc_s_plus_g
   :no-index:

Nonmonotonic d-wave
^^^^^^^^^^^^^^^^^^^

.. math::

   g(\phi)=\beta\cos(2\phi)+(1-\beta)\cos(6\phi).

Use when monotonic d-wave cannot reproduce curvature and an electron-doped
cuprate-like nonmonotonic form is physically plausible [1].

.. autofunction:: asymmetry.core.fitting.sc.models.rho_nonmonotonic_d
   :no-index:

.. autofunction:: asymmetry.core.fitting.sc.models.sc_nonmonotonic_d
   :no-index:

Extended s-wave
^^^^^^^^^^^^^^^

The extended-s channel is implemented from the :math:`\cos(2\phi)` basis
(:math:`\cos(2\phi)` or :math:`|\cos(2\phi)|`) and uses the generalised
weak-coupling reduced-gap form with :math:`a=4/3`, consistent with the same
angular basis used for the d-wave tabulation in Ref. [1].

.. autofunction:: asymmetry.core.fitting.sc.models.rho_extended_s
   :no-index:

.. autofunction:: asymmetry.core.fitting.sc.models.sc_extended_s
   :no-index:

p-wave examples
^^^^^^^^^^^^^^^

These are phenomenological alternatives for anisotropic or unconventional
scenarios when domain-specific theory suggests them. If ``shape_factor_a`` is
left at ``0``, the model uses the Carrington-Manzano temperature dependence; if
``shape_factor_a > 0``, it switches to the generalised weak-coupling
reduced-gap law with that value.

.. autofunction:: asymmetry.core.fitting.sc.models.rho_p_wave_axial
   :no-index:

.. autofunction:: asymmetry.core.fitting.sc.models.sc_p_wave_axial
   :no-index:

.. autofunction:: asymmetry.core.fitting.sc.models.rho_p_wave_polar_3d
   :no-index:

MgB2-style two-gap weighted sums
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The standard multiband weighted-sum form is [2]

.. math::

   \rho_{tot}(T)=w\rho_1(T)+(1-w)\rho_2(T),\qquad 0\le w\le 1.

For ``SC_TwoGap_SS`` both channels are isotropic s-wave. For ``SC_TwoGap_SD``
one channel is isotropic s-wave and one is d-wave.

.. autofunction:: asymmetry.core.fitting.sc.models.sc_two_gap_ss
   :no-index:

.. autofunction:: asymmetry.core.fitting.sc.models.sc_two_gap_sd
   :no-index:

Alpha-model and quadrature conventions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``SC_AlphaModel`` rescales the weak-coupling s-wave ratio,
:math:`\Delta_0/(k_B T_c)=\alpha_{sc}\times 1.764`.

``SC_SWave_Q`` and ``SC_DWave_Q`` use

.. math::

   \sigma(T)=\sqrt{(\sigma_{sc}\rho_s(T))^2+\sigma_{nm}^2}.

Use quadrature models when the non-superconducting linewidth contribution is
best represented as an independent Gaussian channel that combines at the
second-moment level rather than by direct addition.

.. autofunction:: asymmetry.core.fitting.sc.models.sc_alpha_model
   :no-index:

.. autofunction:: asymmetry.core.fitting.sc.models.sc_s_wave_q
   :no-index:

.. autofunction:: asymmetry.core.fitting.sc.models.sc_d_wave_q
   :no-index:

.. autofunction:: asymmetry.core.fitting.sc.models.sc_s_plus_g_q
   :no-index:

Field-dependent vortex-lattice line width (Brandt)
--------------------------------------------------

The models above fit the line width as a function of **temperature** at fixed
field. The complementary measurement sweeps the **applied field** at fixed
(low) temperature: in a type-II superconductor the vortex-lattice
field-distribution second moment depends on field through Brandt's
Ginzburg-Landau result [5], and fitting it yields the absolute penetration
depth :math:`\lambda` and the upper critical field :math:`B_{c2}` directly.

For an ideal triangular flux-line lattice (:math:`\kappa \gg 1`) the Gaussian
muon depolarisation rate is

.. math::

   \sigma(B_0) = \sigma_0(\lambda)\,\frac{(1-b)\,[1 + 1.21\,(1-\sqrt{b})^3]}
   {1 + 1.21},\qquad b = \frac{B_0}{B_{c2}},

where :math:`\sigma_0(\lambda)=\gamma_\mu\,C_B\,\Phi_0/\lambda^2` is the
field-independent London limit (the :math:`b\to 0` maximum, the same scale used
by :func:`asymmetry.core.fitting.sc.constants.lambda_nm_to_sigma_us`). The
field factor is normalised so that :math:`\sigma(b\to0)=\sigma_0(\lambda)` and
:math:`\sigma(b\ge1)=0`. Equivalently, in the commonly cited single-crystal
form, :math:`\sigma\,[\mu s^{-1}] \approx 4.85\times10^{4}\,(1-b)[1+1.21
(1-\sqrt{b})^3]\,\lambda^{-2}\,[\mathrm{nm}^{-2}]` [6].

The field width is what the literature often reports as
:math:`B_{\mathrm{rms}} = \sigma/\gamma_\mu`; fitting in :math:`\sigma`
(:math:`\mu s^{-1}`) keeps the workflow identical to the time-domain Gaussian
rate. An optional field-independent nuclear/background channel ``sigma_bg``
adds in quadrature, :math:`\sigma=\sqrt{\sigma_{VL}^2+\sigma_{bg}^2}` (Pratt
*et al.* Eq. (2) [6]).

These are **field-scope** components (they appear for field-trend fits, not
temperature trends). ``x`` is the applied field in **gauss**; ``Bc2`` is a
parameter in **tesla**; ``lambda_ab`` is in **nm**.

``SC_Brandt_VortexLattice``
   Single-crystal Brandt :math:`\sigma(B_0)` for a type-II superconductor.

``SC_Brandt_VortexLattice_Powder``
   Polycrystalline variant applying the :math:`3^{1/4}` ab-plane powder
   average (so the line width is the single-crystal value divided by
   :math:`\sqrt{3}`, Pratt Eq. (3) [6]). Use this for powder samples — a
   single-crystal fit of powder data under-estimates :math:`\lambda_{ab}` by a
   factor :math:`3^{1/4}=1.316`.

.. autofunction:: asymmetry.core.fitting.sc.models.brandt_field_width_sigma
   :no-index:

.. autofunction:: asymmetry.core.fitting.sc.models.brandt_field_factor
   :no-index:

Worked example: extract lambda from sigma(B0)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fit a field sweep (line widths already extracted per run in the time domain) of
a powder type-II superconductor, recovering :math:`\lambda_{ab}` in nm:

.. code-block:: python

   import numpy as np
   from asymmetry.core.fitting import Parameter, ParameterSet, ParameterCompositeModel
   from asymmetry.core.fitting.parameter_models import fit_parameter_model

   # Applied transverse field in gauss; sigma in us^-1 (one point per run).
   B0 = np.array([100.0, 200.0, 800.0, 1600.0, 3200.0, 4000.0, 6000.0])
   model = ParameterCompositeModel(["SC_Brandt_VortexLattice_Powder"])

   params = ParameterSet([
       Parameter("lambda_ab", value=200.0, min=0.0),  # nm
       Parameter("Bc2", value=20.0, min=0.0),          # tesla
       Parameter("sigma_bg", value=0.0, min=0.0, fixed=True),
   ])

   sigma = model.function(B0, lambda_ab=195.0, Bc2=25.0, sigma_bg=0.0)
   sigma_err = np.full_like(B0, 0.02)
   result = fit_parameter_model(B0, sigma, sigma_err, model, params)
   # result.parameters -> lambda_ab ~ 195 nm; B_rms = sigma / gamma_mu.

Shared parameter semantics
--------------------------

``sigma_0``
   Additive superconducting scale in :math:`\mu s^{-1}`.

``sigma_bg``
   Additive background term in :math:`\mu s^{-1}`.

``sigma_sc``, ``sigma_nm``
   Superconducting and normal/nuclear terms in the quadrature convention.

``Tc``
   Critical temperature in K.

``lambda_ab``
   Magnetic (ab-plane) penetration depth in nm, fitted directly by the Brandt
   field-dependence models.

``Bc2``
   Upper critical field in tesla, setting the reduced field :math:`b=B_0/B_{c2}`
   in the Brandt vortex-lattice line width.

``gap_ratio``, ``gap_ratio_*``
   Dimensionless :math:`\Delta_0/(k_B T_c)` values.

``weight``
   Band weight constrained to :math:`0\le w\le 1`.

``a_anis``, ``beta_nm``
   Angular anisotropy controls for anisotropic-s and nonmonotonic-d models.

``shape_factor_a``
   Optional weak-coupling shape-factor parameter for anisotropic-s and p-wave
   channels. In the GUI, ``0`` means "not supplied" and keeps the
   Carrington-Manzano fallback; any positive fixed or free value enables the
   generalised reduced-gap law.

``alpha_sc``
   Scaling factor multiplying the weak-coupling s-wave ratio.

``signed_gap``
   Extended-s convention control (signed versus magnitude form).

Info helpers for trend components
---------------------------------

Each model component shown in parameter-trend fitting (for example
``SC_SWave``, ``SC_DWave``, ``SC_SPlusG``, ``SC_TwoGap_SS``) has a concise
physics summary in
:mod:`asymmetry.core.fitting.parameter_models.PARAMETER_MODEL_COMPONENTS` via
the ``description`` field. Parameter-level helper text is provided by
:func:`asymmetry.core.fitting.get_param_info`.

Assumptions and limitations
---------------------------

The current implementation is intended for practical experimental fitting and
follows standard approximations:

- London-limit treatment.
- Clean-limit-like thermal kernel without explicit impurity self-energy.
- Isotropic angular weighting by default.
- No explicit field-dependent nonlinear Meissner corrections.

Treat fitted parameters accordingly when comparing strongly disordered,
strongly anisotropic, or very low-field datasets.

Working with lambda(T) and lambda^{-2}(T)
------------------------------------------

The same :math:`\rho_s(T)` can be mapped to penetration-depth observables:

.. math::

   \lambda^{-2}(T)=\rho_s(T)\lambda^{-2}(0),\qquad
   \lambda(T)=\frac{\lambda(0)}{\sqrt{\rho_s(T)}}.

.. autofunction:: asymmetry.core.fitting.sc.models.rho_to_lambda_inv_sq
   :no-index:

.. autofunction:: asymmetry.core.fitting.sc.models.rho_to_lambda
   :no-index:

.. autofunction:: asymmetry.core.fitting.sc.models.lambda_inv_sq_from_model
   :no-index:

.. autofunction:: asymmetry.core.fitting.sc.models.lambda_from_model
   :no-index:

Example: Fit sigma(T) with an s-wave model
------------------------------------------

.. code-block:: python

   import numpy as np
   from asymmetry.core.fitting import Parameter, ParameterSet, ParameterCompositeModel
   from asymmetry.core.fitting.parameter_models import fit_parameter_model

   T = np.linspace(1.0, 25.0, 20)
   model = ParameterCompositeModel(["SC_SWave"])

   params = ParameterSet([
       Parameter("sigma_0", value=1.0, min=0.0),
       Parameter("Tc", value=24.0, min=0.0),
       Parameter("gap_ratio", value=1.764, min=0.0),
       Parameter("sigma_bg", value=0.0, min=0.0),
   ])

   y = model.function(T, sigma_0=1.2, Tc=24.0, gap_ratio=1.9, sigma_bg=0.03)
   yerr = np.full_like(T, 0.02)
   result = fit_parameter_model(T, y, yerr, model, params)

References
----------

[1] R. Prozorov and R. W. Giannetta, Supercond. Sci. Technol. **19**, R41 (2006).

[2] A. Carrington and F. Manzano, Physica C **385**, 205 (2003).

[3] J. E. Sonier, J. H. Brewer, and R. F. Kiefl, Rev. Mod. Phys. **72**, 769 (2000).

[4] R. Prozorov, M. A. Tanatar, R. T. Gordon, C. Martin, H. Kim, V. G. Kogan,
N. Ni, M. E. Tillman, S. L. Bud'ko, and P. C. Canfield, Physica C **469**, 582
(2009).

[5] E. H. Brandt, Phys. Rev. B **37**, 2349 (1988); Phys. Rev. B **68**, 054506 (2003).

[6] F. L. Pratt, P. J. Baker, S. J. Blundell, T. Lancaster, H. J. Lewtas,
P. Adamson, M. J. Pitcher, D. R. Parker, and S. J. Clarke, Phys. Rev. B **79**,
052508 (2009).
