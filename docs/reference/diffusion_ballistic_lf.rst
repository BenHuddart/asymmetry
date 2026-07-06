.. _diffusion-ballistic-lf:

Field-Dependent Transport Models (DiffusionLF / BallisticLF)
============================================================

When muon-spin relaxation is dominated by motion of the muon (or its host
spin environment) rather than by a static field distribution, the diagnostic
observable is the *field* dependence of the relaxation rate rather than the
detailed shape of any single :math:`A(t)`. The standard workflow is
two-stage: fit each longitudinal-field (LF) decoupled run to an empirical relaxation form to
extract :math:`\lambda(B_{LF})`, then fit the resulting
:math:`\lambda` versus :math:`B_{LF}` curve with a transport model whose
parameters carry direct physical meaning — a diffusion rate, a hopping
rate, the dimensionality of the motion. Asymmetry implements two families
of such models, both consumed by the parameter-trending framework documented
in :doc:`parameter_trending`:

- **DiffusionLF_1D / 2D / 3D** for incoherent random-walk transport, in
  one, two, or three spatial dimensions.
- **BallisticLF_1D / 2D / 3D** for coherent ballistic transport at the
  same set of dimensionalities.

Both families share a common construction: an autocorrelation function
:math:`S(t)` of the fluctuating local field is built from the appropriate
transport propagator, its one-sided cosine transform gives the spectral
density,

.. math::

   J(\omega) \;=\; 2\int_0^{\infty} S(t)\,\cos(\omega t)\,dt,

and the muon relaxation rate is read out at the longitudinal Larmor
frequency,

.. math::

   \lambda(B_{LF}) \;=\; \frac{A^2}{4}\,J(\omega_e),
   \qquad \omega_e = \gamma_e B_{LF}.

The amplitude :math:`A` is the coupling strength of the muon to the
fluctuating field, in :math:`\mathrm{MHz}` (numerically equal to
:math:`\mu s^{-1}` here). The Larmor frequency is taken from the electron
gyromagnetic ratio :math:`\gamma_e` because the field that the muon sees in
these systems is typically set by an unpaired electron spin on a host atom.

Choosing between the two families is a physics question, not a fitting
question: are you describing a random walk (diffusion) or coherent motion
along well-defined channels (ballistic)? In a strongly disordered or
strongly scattered system the random-walk picture is correct and the
``DiffusionLF_*`` models apply; in a clean, low-dimensional conductor with
long mean free path the coherent-transport picture is correct and the
``BallisticLF_*`` models apply. Dimensionality is the other axis of choice
and is usually fixed by the crystal structure of the host: one-dimensional
chains, two-dimensional layers, or three-dimensional bulk.

Diffusive transport (DiffusionLF_*)
------------------------------------

The autocorrelation for incoherent diffusion in :math:`n` accessible
dimensions, following Pratt [2], is

.. math::

   S_{nD}(t) \;=\; \bigl[e^{-2 D_{nD} t}\,I_0(2 D_{nD} t)\bigr]^{n}
                 \,\bigl[e^{-2 D_\perp t}\,I_0(2 D_\perp t)\bigr]^{3-n},
   \quad n \in \{1,2,3\},

with :math:`I_0` the modified Bessel function of the first kind.
:math:`D_{nD}` is the in-plane (or in-chain) diffusion rate;
:math:`D_\perp` allows for a slower perpendicular diffusion channel and may
be fixed to zero in genuinely low-dimensional cases. Substituting into the
spectral-density expression above gives the field-dependent rate
:math:`\lambda_{nD}^{\mathrm{diff}}(B_{LF})`.

The full parameter-trending expression that the documentation in [1]
employs for a quasi-2D conductor is a sum of four physically distinct
contributions,

.. math::

   \lambda(B_{LF})
   \;=\; \lambda_{nD}^{\mathrm{diff}}(B_{LF})
       + \lambda_{0D}(B_{LF})
       + \lambda_{BG}
       + \lambda_{LCR}(B_{LF}),

a dynamic transport term, a local (0D) dynamic term (the ``Redfield``
component in :doc:`parameter_trending`), a field-independent background
(``Lambda_bg``), and an optional level-crossing-resonance Gaussian
(``GaussianLCR``). Asymmetry exposes each of these as a separate basis
function in the parameter-model builder so the user can include only the
terms motivated by their data.

Parameters
~~~~~~~~~~

==============  ==================  =========  ===================================================
Name            Symbol              Unit       Description
==============  ==================  =========  ===================================================
``A``           :math:`A`           MHz        Coupling amplitude (numerically equal to μs⁻¹).
``D_nD``        :math:`D_{nD}`      μs⁻¹       In-plane / in-chain diffusion rate.
``D_perp``      :math:`D_\perp`     μs⁻¹       Perpendicular diffusion rate.
``B_LF``        :math:`B_{LF}`      G          Longitudinal field (independent variable).
==============  ==================  =========  ===================================================

The dimensionality is encoded in the choice of component (1D, 2D, or 3D)
rather than as a fit parameter. ``D_perp`` may be fixed at zero for cleanly
low-dimensional systems; freeing it lets the fit accommodate weak
inter-chain or inter-layer leakage.

Ballistic transport (BallisticLF_*)
-----------------------------------

For coherent propagation, the autocorrelation in :math:`n` dimensions
follows the Bessel-function form derived in *Muon Spectroscopy: An
Introduction* [3],

.. math::

   S_{nD}^{\mathrm{ball}}(t)
   \;=\; \bigl[J_0(2 D_{\mathrm{hop}}\,t)\bigr]^{2n},
   \quad n \in \{1,2,3\},

with :math:`J_0` the zeroth-order Bessel function of the first kind and
:math:`D_{\mathrm{hop}}` the ballistic hopping rate. The spectral-density
construction is identical to the diffusive case, and the field-dependent
rate is again

.. math::

   \lambda_{nD}^{\mathrm{ball}}(B_{LF})
   \;=\; \frac{A^2}{4}\,J(\omega_e),
   \quad \omega_e = \gamma_e B_{LF}.

The one-dimensional ballistic case has a useful low-frequency
approximation,

.. math::

   J(\omega) \;\approx\; \frac{0.318}{D_{\mathrm{hop}}}
                         \,\ln\!\left(\frac{16 D_{\mathrm{hop}}}{\omega}\right),

so that :math:`\lambda` plotted against :math:`\log B_{LF}` is expected to
be close to linear over the corresponding field window. Curvature relative
to that prediction is usually the cleanest indicator that motion is not
strictly one-dimensional or not strictly ballistic.

Parameters
~~~~~~~~~~

============  ========================  =========  ===================================================
Name          Symbol                    Unit       Description
============  ========================  =========  ===================================================
``A``         :math:`A`                 MHz        Coupling amplitude.
``D_hop``     :math:`D_{\mathrm{hop}}`  μs⁻¹       Ballistic hopping rate.
``B_LF``      :math:`B_{LF}`            G          Longitudinal field (independent variable).
============  ========================  =========  ===================================================

Choosing dimensionality
-----------------------

The dimensionality choice should be made from the crystallography and any
prior transport measurements on the same compound, not from comparing fit
qualities across the three options.

- ``*LF_1D`` for transport along chains (e.g. organic conductor stacks) or
  other strongly one-dimensional pathways.
- ``*LF_2D`` for layered systems with in-plane mobility (e.g. cuprates,
  intercalates, transition-metal dichalcogenides).
- ``*LF_3D`` when the transport is bulk-like and isotropic on the muon
  time scale.

A fit that selects ``BallisticLF_1D`` over ``DiffusionLF_2D`` purely on
:math:`\chi^2` is rarely conclusive on its own: the field dependence over
any realistic experimental window is similar enough between dimensionalities
that the choice usually rests on the physics. Where the data genuinely
support discrimination, the model-selection workflow in the
parameter-trending fits (AIC / BIC) will report it; treat that as
confirmation rather than as a primary tool.

Using these components
----------------------

These components live in the parameter-model registry
(``PARAMETER_MODEL_COMPONENTS``)
and are applied to a :math:`\lambda` versus :math:`B_{LF}` series extracted
from a previous round of time-domain fits. The end-to-end workflow is
documented in :doc:`parameter_trending`; the call signature for direct
evaluation is

.. code-block:: python

   import numpy as np
   from asymmetry.core.fitting.parameter_models import PARAMETER_MODEL_COMPONENTS

   b_lf = np.linspace(10.0, 3000.0, 60)  # Gauss

   diff = PARAMETER_MODEL_COMPONENTS["DiffusionLF_2D"]
   lam_diff = diff.function(b_lf, A=0.8, D_nD=2.0, D_perp=0.0)

   ball = PARAMETER_MODEL_COMPONENTS["BallisticLF_1D"]
   lam_ball = ball.function(b_lf, A=0.8, D_hop=3.0)

In a composite parameter model, combine the transport term with the
field-independent background and (if needed) the level-crossing-resonance
term:

.. code-block:: python

   from asymmetry.core.fitting.parameter_models import ParameterCompositeModel

   model = ParameterCompositeModel(
       ["DiffusionLF_2D", "Lambda_bg"], operators=["+"]
   )

The GUI parameter-model builder filters the available components by the
selected x-axis so that field-only components are not offered for
temperature trends, and vice versa.

References
----------

[1] Phys. Rev. B **106**, L060401 (2022) — the four-term decomposition
    used in the quasi-2D dynamic-relaxation analysis the diffusive
    components were built to match.

[2] F. L. Pratt, J. Phys.: Conf. Ser. **2462**, 012038 (2023) —
    autocorrelation forms for one-, two-, and three-dimensional diffusion.

[3] S. J. Blundell, R. De Renzi, T. Lancaster, and F. L. Pratt,
    *Muon Spectroscopy: An Introduction* (Oxford University Press, Oxford,
    2022) — derivation of the ballistic Bessel-function autocorrelation.
