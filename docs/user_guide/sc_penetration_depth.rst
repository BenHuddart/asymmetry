Superconducting Penetration Depth Models
========================================

This page documents the superconducting sigma(T) models used for TF-muSR
vortex-state analysis.

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

Context: sigma(T), lambda(T), and superfluid density
----------------------------------------------------

In transverse-field muSR on a vortex lattice, the Gaussian relaxation rate
``sigma`` tracks the second moment of the field distribution. In the London
limit for a triangular lattice, the superconducting contribution follows

.. math::

   \sigma_{sc}(T) \propto \lambda_L^{-2}(T) \propto \rho_s(T),

where :math:`\lambda_L` is the London penetration depth and
:math:`\rho_s(T) = \lambda^2(0)/\lambda^2(T)` is normalized superfluid density.

The module in :mod:`asymmetry.core.fitting.sc` computes :math:`\rho_s(T)` from
gap-symmetry models and returns ``sigma(T)`` through either

.. math::

   \sigma(T) = \sigma_0\,\rho_s(T) + \sigma_{bg}

or the quadrature convention

.. math::

   \sigma^2(T) = \sigma_{sc}^2\,\rho_s^2(T) + \sigma_{nm}^2.

Conventions Implemented in Code
-------------------------------

- Canonical internal quantity: normalized superfluid density

   .. math::

       \rho_s(T) = \frac{\lambda^2(0)}{\lambda^2(T)}
       = \frac{\lambda^{-2}(T)}{\lambda^{-2}(0)}.

- Gap factorization:

   .. math::

       \Delta(T,\mathbf{k}) = \Delta_0\,\delta_{BCS}(T/T_c)\,g(\mathbf{k}).

- Angular variable for quasi-2D models: :math:`\phi \in [0, 2\pi)`.
- 3D optional models average over :math:`(\theta,\phi)` on a sphere.
- Observables use :math:`|g(\mathbf{k})|` by default, because the quasiparticle
   excitation energy depends on :math:`|\Delta|`. Signed conventions are
   available where relevant.

Thermal Kernel
--------------

The implemented kernel is

.. math::

   \rho_s(T) = 1 + 2\left\langle
   \int_{\Delta(T,\mathbf{k})}^{\infty}
   \frac{\partial f}{\partial E}
   \frac{E\,dE}{\sqrt{E^2-\Delta^2(T,\mathbf{k})}}
   \right\rangle_{FS},

with

.. math::

   \Delta(T,\mathbf{k}) = \Delta_0\,\delta_{BCS}(T/T_c)\,g(\mathbf{k}).

The reduced BCS gap :math:`\delta_{BCS}` uses the Carrington-Manzano analytic
approximation, accurate for fitting workflows and stable near both
:math:`T=0` and :math:`T=T_c`.

Numerical method
^^^^^^^^^^^^^^^^

- Energy integral and angular averages are evaluated with Gauss-Legendre
   quadrature.
- The implementation explicitly handles limiting cases:

   - :math:`T \to 0`: :math:`\rho_s \to 1`
   - :math:`T \ge T_c`: :math:`\rho_s = 0`
   - nodal points where :math:`\Delta(T,\mathbf{k})=0`

- Exponential arguments are clipped for numerical stability.

BCS Gap Temperature Dependence
------------------------------

The reduced gap helper follows the Carrington-Manzano interpolation:

.. math::

    \delta_{BCS}(t)
    = \tanh\!\left(1.82\,[1.018(1/t-1)]^{0.51}\right),\quad t=T/T_c.

The code accepts two equivalent gap-magnitude conventions:

- Dimensionless ratio :math:`\Delta_0/(k_B T_c)` (``gap_ratio``).
- Gap magnitude in meV (``gap_mev``), converted internally to
   :math:`\Delta_0/(k_B T_c)`.

This makes literature comparisons straightforward when papers quote either
ratio values or meV values.

Gap Symmetry Models
-------------------

The following representative models are available as parameter-trend
components (``x_key='temperature'``):

- ``SC_SWave``: isotropic s-wave, :math:`g(\phi)=1`
- ``SC_DWave``: nodal d-wave, :math:`g(\phi)=\cos(2\phi)`
- ``SC_AnisotropicS_Cos4``: :math:`g(\phi)=1+a\cos(4\phi)`
- ``SC_NonmonotonicD``: :math:`g(\phi)=\beta\cos(2\phi)+(1-\beta)\cos(6\phi)`
- ``SC_PWaveAxial``: 2D axial p-wave, :math:`g(\phi)=\cos(\phi)`
- ``SC_ExtendedS``: extended s-wave using :math:`\cos(2\phi)`
   (signed or absolute-value convention)
- ``SC_AlphaModel``: alpha-model scaling of weak-coupling s-wave
- ``SC_TwoGap_SS``: weighted two-gap isotropic model
- ``SC_TwoGap_SD``: weighted mixed-symmetry model
- ``SC_SWave_Q`` / ``SC_DWave_Q``: quadrature sigma conventions

Model-by-model physics interpretation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``SC_SWave``
   :math:`g(\phi)=1`, fully gapped. Low-temperature behavior is exponentially
   activated.

``SC_DWave``
   :math:`g(\phi)=\cos(2\phi)`, line nodes. Low-temperature behavior is
   stronger and approximately power-law/linear in clean limits.

``SC_AnisotropicS_Cos4``
   :math:`g(\phi)=1+a\cos(4\phi)`.
   Nodeless for :math:`|a|<1`; accidental nodes may appear for
   :math:`|a|\ge 1`.

``SC_NonmonotonicD``
   :math:`g(\phi)=\beta\cos(2\phi)+(1-\beta)\cos(6\phi)`.
   Useful when simple monotonic d-wave does not capture angular anisotropy.

``SC_PWaveAxial``
   2D p-wave example :math:`g(\phi)=\cos(\phi)`.

``SC_ExtendedS``
   Based on :math:`\cos(2\phi)`, with signed or absolute-value convention.

``SC_AlphaModel``
   Isotropic model that scales weak-coupling s-wave by ``alpha_sc``:
   :math:`\Delta_0/(k_B T_c)=\alpha_{sc}\times 1.764`.

``SC_TwoGap_SS`` and ``SC_TwoGap_SD``
   Weighted sums:

   .. math::

      \rho_s(T)=w\,\rho_1(T)+(1-w)\,\rho_2(T),\quad 0\le w\le 1.

   ``SC_TwoGap_SS`` uses two isotropic gaps; ``SC_TwoGap_SD`` mixes isotropic
   and d-wave bands.

``SC_SWave_Q`` / ``SC_DWave_Q``
   Quadrature convention for combining superconducting and nuclear
   contributions.

Node behavior notes
^^^^^^^^^^^^^^^^^^^

- ``SC_SWave`` is fully gapped.
- ``SC_DWave`` has line nodes and stronger low-T variation.
- ``SC_AnisotropicS_Cos4`` is nodeless for :math:`|a|<1`, and can develop
  accidental nodes for :math:`|a|\ge 1`.
- Mixed and two-gap models interpolate between constituent behaviors through
  weight ``w``.

Parameter Semantics
-------------------

``sigma_0``
   Superconducting scale factor at :math:`T=0` in additive models.

``sigma_bg``
   Additive non-superconducting background term in :math:`\mu s^{-1}`.

``sigma_sc``, ``sigma_nm``
   Superconducting and nuclear terms in quadrature models.

``Tc``
   Critical temperature in K.

``gap_ratio``, ``gap_ratio_*``
   Dimensionless :math:`\Delta_0/(k_B T_c)` values.

``weight``
   Band weight with physical bounds :math:`0\le w\le 1`.

``a_anis``, ``beta_nm``
   Angular anisotropy parameters for anisotropic-s and nonmonotonic-d forms.

``alpha_sc``
   Alpha-model scaling relative to weak-coupling s-wave.

Assumptions and Limitations
---------------------------

The current implementation is intended for practical experimental fitting and
follows standard approximations:

- London-limit treatment (vortex-lattice field-distribution details are not
  explicitly re-fit).
- Clean-limit-like thermal kernel without explicit impurity self-energy.
- Isotropic Fermi-velocity weighting in angular averaging.
- No explicit field-dependent nonlinear Meissner corrections.

Interpret fitted parameters in this approximation framework, especially when
comparing to strongly disordered, strongly anisotropic, or very low-field data.

If your system requires impurity-driven crossovers (for example dirty d-wave
behavior), treat this as a baseline model and extend the component set.

Practical Fitting Guidance
--------------------------

1. Start with ``SC_SWave`` and ``SC_DWave`` as baseline hypotheses.
2. Add ``SC_AnisotropicS_Cos4`` when low-T deviations are present but a full
   nodal model is not required.
3. Use ``SC_TwoGap_SS`` for known multiband materials (for example MgB2-like
   behavior).
4. Use ``SC_TwoGap_SD`` if one band appears nodal and another nodeless.
5. Apply bounds to keep physically meaningful parameters, for example
   ``0 <= weight <= 1`` and positive gap ratios.
6. Compare competing models with reduced :math:`\chi^2`, AIC/BIC-like criteria,
   and residual structure, not just visual overlap.
7. Validate limiting behavior of the best-fit model:

   - :math:`\rho_s(0) \approx 1`
   - :math:`\rho_s(T_c) \approx 0`
   - expected low-T trend (activated vs nodal power-law)

Working with lambda(T) and lambda^{-2}(T)
------------------------------------------

The trend components fit :math:`\sigma(T)` directly, but the same
:math:`\rho_s(T)` can be mapped to penetration-depth observables:

.. math::

   \lambda^{-2}(T) = \rho_s(T)\,\lambda^{-2}(0),
   \qquad
   \lambda(T) = \frac{\lambda(0)}{\sqrt{\rho_s(T)}}.

Helper functions in :mod:`asymmetry.core.fitting.sc.models` provide these
conversions for post-fit interpretation.

Example: fit sigma(T) with an s-wave model
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

   # Replace y/yerr with sigma(T) extracted from per-run TF-muSR time-domain fits.
   y = model.function(T, sigma_0=1.2, Tc=24.0, gap_ratio=1.9, sigma_bg=0.03)
   yerr = np.full_like(T, 0.02)

   result = fit_parameter_model(T, y, yerr, model, params)
   print(result.success, result.reduced_chi_squared)

Brandt conversion helpers
-------------------------

Use conversion helpers for approximate absolute penetration-depth estimates:

.. code-block:: python

   from asymmetry.core.fitting.sc.constants import sigma_to_lambda_nm

   lambda_nm = sigma_to_lambda_nm(0.8)

The conversion uses the common Brandt proportionality constant for a
triangular vortex lattice in the London limit. Treat absolute values as model-
dependent estimates unless your field regime and vortex-lattice conditions are
well controlled.

References
----------

- R. Prozorov and R. W. Giannetta, Supercond. Sci. Technol. 19, R41 (2006).
- A. Carrington and F. Manzano, Physica C 385, 205 (2003).
- J. E. Sonier, J. H. Brewer, and R. F. Kiefl, Rev. Mod. Phys. 72, 769 (2000).
- R. Prozorov et al., Phys. Rev. B 78, 224506 (2008) and related 122/1111 studies.
