Superconducting Penetration Depth Models
========================================

This page documents the superconducting sigma(T) models used for TF-muSR
vortex-state analysis and interleaves the core theory with the API for each
model family.

Scope and Terminology
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

Core Equations
--------------

In the London limit for a triangular vortex lattice,

.. math::

   \sigma_{sc}(T) \propto \lambda_L^{-2}(T) \propto \rho_s(T),

with normalized superfluid density

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

The gap is factorized as

.. math::

   \Delta(T,\mathbf{k}) = \Delta_0\,\delta(T/T_c)\,g(\mathbf{k}).

Normalized Superfluid-Density Integral
--------------------------------------

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

Gap-Amplitude Approximations
----------------------------

The code uses the Carrington-Manzano approximation [2]

.. math::

   \delta_{BCS}(t)=\tanh\left(1.82[1.018(1/t-1)]^{0.51}\right),\quad t=T/T_c,

while a common generalized form discussed in the review literature is [1]

.. math::

   \Delta_0(T)=\Delta_0(0)
   \tanh\left[
   \frac{\pi T_c}{\Delta_0(0)}\sqrt{a\left(\frac{T_c}{T}-1\right)}
   \right].

Representative weak-coupling values from Ref. [1]:

.. list-table:: Symmetry-Dependent Reference Parameters
   :header-rows: 1

   * - Symmetry
     - :math:`g(\mathbf{k})`
     - :math:`\Delta_0(0)/(k_B T_c)`
     - :math:`a`
   * - Isotropic s-wave
     - :math:`1`
     - 1.76
     - 1.0
   * - :math:`d_{x^2-y^2}`
     - :math:`\cos(2\phi)`
     - 2.14
     - :math:`4/3`
   * - s+g
     - :math:`(1-\sin^4\theta\cos 4\phi)/2`
     - 2.77
     - 2.0
   * - Nonmonotonic d-wave
     - :math:`1.43\cos 2\phi + 0.43\cos 6\phi`
     - 1.19
     - 0.38

Model Selection At A Glance
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
     - Cuprate-like nodal behavior is expected
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

Superconducting Gap Models
--------------------------

Isotropic s-wave
^^^^^^^^^^^^^^^^

.. math::

   g(\phi)=1,\qquad \Delta_0(0)/(k_B T_c)\approx 1.764.

Use when low-T behavior is exponentially activated and no nodal signatures are
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
and low-T data are inconsistent with activated behavior [1].

.. autofunction:: asymmetry.core.fitting.sc.models.rho_d_wave
   :no-index:

.. autofunction:: asymmetry.core.fitting.sc.models.sc_d_wave
   :no-index:

Anisotropic s-wave (cos4)
^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

   g(\phi)=1+a\cos(4\phi).

Use when an s-wave baseline is too rigid but a strict nodal d-wave model is
not yet justified. For :math:`|a|<1` the model remains nodeless.

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

Extended s-wave and p-wave examples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These are phenomenological alternatives for anisotropic or unconventional
scenarios when domain-specific theory suggests them.

.. autofunction:: asymmetry.core.fitting.sc.models.rho_extended_s
   :no-index:

.. autofunction:: asymmetry.core.fitting.sc.models.sc_extended_s
   :no-index:

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

Shared Parameter Semantics
--------------------------

``sigma_0``
   Additive superconducting scale in :math:`\mu s^{-1}`.

``sigma_bg``
   Additive background term in :math:`\mu s^{-1}`.

``sigma_sc``, ``sigma_nm``
   Superconducting and normal/nuclear terms in the quadrature convention.

``Tc``
   Critical temperature in K.

``gap_ratio``, ``gap_ratio_*``
   Dimensionless :math:`\Delta_0/(k_B T_c)` values.

``weight``
   Band weight constrained to :math:`0\le w\le 1`.

``a_anis``, ``beta_nm``
   Angular anisotropy controls for anisotropic-s and nonmonotonic-d models.

``alpha_sc``
   Scaling factor multiplying the weak-coupling s-wave ratio.

``signed_gap``
   Extended-s convention control (signed versus magnitude form).

Info Helpers For Trend Components
---------------------------------

Each model component shown in parameter-trend fitting (for example
``SC_SWave``, ``SC_DWave``, ``SC_SPlusG``, ``SC_TwoGap_SS``) has a concise
physics summary in
:mod:`asymmetry.core.fitting.parameter_models.PARAMETER_MODEL_COMPONENTS` via
the ``description`` field. Parameter-level helper text is provided by
:func:`asymmetry.core.fitting.get_param_info`.

Assumptions and Limitations
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

[1] R. Prozorov and R. W. Giannetta, Supercond. Sci. Technol. 19, R41 (2006).

[2] A. Carrington and F. Manzano, Physica C 385, 205 (2003).

[3] J. E. Sonier, J. H. Brewer, and R. F. Kiefl, Rev. Mod. Phys. 72, 769 (2000).

[4] R. Prozorov, M. A. Tanatar, R. T. Gordon et al., Physica C 469, 582 (2009).
