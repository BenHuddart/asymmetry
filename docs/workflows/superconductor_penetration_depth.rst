Superconductor penetration depth from σ(T)
==========================================

This chapter is a worked example of the canonical superconductor μSR
workflow: extract the temperature dependence of the magnetic
penetration depth :math:`\lambda_L(T)` from transverse-field (TF) μSR data
in the vortex state, then fit a gap model to identify the pairing
symmetry. The synthetic data corresponds to MgB₂
(:math:`T_c \approx 36\;\mathrm{K}`, two-gap s+s structure with
:math:`\Delta_1 / k_B T_c \approx 1.1` and
:math:`\Delta_2 / k_B T_c \approx 2.3`) — the textbook example used
in Blundell *et al.* 2022 Ch 9.5 and discussed in detail in
Amato & Morenzoni 2024 Ch 6. The same workflow applies to other
type-II superconductors: high-:math:`T_c` cuprates (where d-wave gap
fitting requires ``SC_TwoGap_SD``), iron pnictides, and recently-
discovered correlated superconductors. The screenshots below are
generated from the synthetic dataset shipped with the documentation
and are intended to show what each stage of the analysis produces
rather than as exercises for the reader.

Physical motivation
-------------------

In the vortex state of a type-II superconductor (between
:math:`H_{c1}` and :math:`H_{c2}`), the magnetic field penetrates the
sample as a triangular lattice of flux tubes. A muon stops at a
random position relative to this lattice and sees a local field drawn
from the lattice's internal field distribution :math:`P(B)`. The
distribution is asymmetric: a sharp peak at the saddle-point field
between three vortices, with a long tail toward the high-field cores
(see Brandt, Phys. Rev. B **37**, 2349 (1988)). The second moment of :math:`P(B)`
sets the Gaussian damping rate :math:`\sigma` of the time-domain TF
asymmetry,

.. math::

   \sigma^2 = \gamma_\mu^2 \langle \Delta B^2 \rangle.

For a triangular vortex lattice in the London limit,

.. math::

   \sigma = 0.0609\,\frac{\Phi_0}{\lambda_L^2}\,\gamma_\mu,

so :math:`\sigma \to \lambda_L` follows directly from a constant of
nature and a measured number. The temperature dependence of
:math:`\lambda_L(T)` — and equivalently the superfluid density
:math:`\rho_s(T) \propto 1/\lambda^2(T)` — reveals the
superconducting gap structure: an s-wave gap freezes
:math:`\rho_s(T)` exponentially as :math:`T \to 0`; a d-wave gap
gives a linear-in-:math:`T` decrease at low :math:`T`; a two-gap
(s+s) model shows a kink or shoulder marking the smaller gap's
freezing scale.

The data
--------

The example uses 28 TF runs from 1.5 K to 35 K at :math:`B_{TF}
\approx 200\;\mathrm{mT}` (well above :math:`H_{c1}`). The synthetic
generator (``make_mgb2_sigma_t``) returns the :math:`\sigma(T)` curve
directly, saving the per-run TF fit step for this walk-through. In
practice each TF run would be fitted individually to extract
:math:`\sigma(T)` point by point — see :doc:`/reference/fitting`
and :doc:`/reference/grouped_time_domain_fitting`.

Step 1 — Load and inspect the TF series
---------------------------------------

.. image:: /_generated/screenshots/fourier_tf.png
   :alt: YBCO vortex-lattice TF Fourier spectrum (shown as a
       representative line shape)
   :width: 100%

The screenshot shows the **Frequency** domain of the central
workspace on a representative TF μSR dataset in the vortex state
(YBCO is used in place of MgB₂ for the FFT image because its line
shape is qualitatively similar but more visually distinctive). The
asymmetric line shape is the canonical signature of a triangular
vortex lattice: a sharp low-field peak at the saddle-point van Hove
singularity, with a long tail extending toward the higher-field
vortex cores. The width of the tail scales with
:math:`1/\lambda_L^2`; the temperature dependence of that width is
what the analysis extracts.

Above :math:`T_c` the line shape collapses to a narrow Gaussian set
by the nuclear-dipolar contribution alone — a useful cross-check
that the measurement is in the right physical regime.

Step 2 — Fit each TF run for σ(T)
---------------------------------

For each temperature, the per-run workflow is to open the **Fit**
dock on a single dataset, build the composite
``Oscillatory + Gaussian + Constant`` (or equivalently
``Oscillatory * Gaussian + Constant`` for a Gaussian envelope on a
single precession line) via the function builder (see
:doc:`/reference/composite_models`), fit, and record the Gaussian
envelope's :math:`\sigma` parameter — the second moment of
:math:`P(B)` — along with its uncertainty. The ``Oscillatory``
component carries the average field's Larmor frequency.

For pedagogical purposes, the synthetic generator skips ahead and
gives :math:`\sigma(T)` directly:

.. code-block:: python

   from docs.screenshots.data.archetypes import make_mgb2_sigma_t
   payload = make_mgb2_sigma_t()
   T = payload["T_K"]
   sigma = payload["sigma"]
   sigma_err = payload["sigma_err"]

Step 3 — Trend σ(T) in the parameter-trending panel
---------------------------------------------------

.. image:: /_generated/screenshots/parameter_trending_mgb2.png
   :alt: MgB₂ σ(T) with two-gap SC_TwoGap_SS curve overlaid
   :width: 100%

The screenshot shows the resulting :math:`\sigma(T)` curve with the
two-gap ``SC_TwoGap_SS`` model overlaid at the literature MgB₂
decomposition (gap ratios 1.1 and 2.3, weight 0.55,
:math:`T_c = 36\;\mathrm{K}`). The characteristic shape is plain:
the curve flattens at low :math:`T` (the large gap freezes out
quasiparticles), shows a kink near :math:`T/T_c \approx 0.3{-}0.5`
that marks the smaller gap's freezing scale, and drops to zero as
:math:`T \to T_c`. The kink is the visual signature of multiband
superconductivity that a single-gap model cannot reproduce.

Step 4 — Fit a gap model
------------------------

Asymmetry's parametric-model registry includes the standard family of
superconducting gap models:

- ``SC_SWave`` — single isotropic s-wave gap (parameters
  :math:`\sigma_0`, :math:`T_c`, :math:`\Delta_s / k_B T_c`).
- ``SC_TwoGap_SS`` — two s-wave gaps; the MgB₂ canonical model
  (:math:`\sigma_0`, :math:`T_c`, two gap ratios, weight :math:`w`).
- ``SC_TwoGap_SD`` — s + d hybrid (high-:math:`T_c` cuprates).
- ``SC_TwoGap_DD`` — two d-wave (exotic; ignore unless specific
  evidence demands it).

The two-gap model is the right choice for MgB₂. Fitting it to the
synthetic :math:`\sigma(T)` requires bounds on :math:`T_c` (the
data set ends at :math:`T = 35\;\mathrm{K} = T_c - 1`, so an
unconstrained :math:`T_c` may drift):

.. code-block:: python

   from asymmetry.core.fitting.sc.models import sc_two_gap_ss
   from scipy.optimize import curve_fit
   import numpy as np

   def model(T, s0, Tc, r1, r2, w, s_bg):
       return sc_two_gap_ss(
           T, sigma_0=s0, Tc=Tc,
           gap_ratio_1=r1, gap_ratio_2=r2,
           weight=w, sigma_bg=s_bg,
       )

   p0 = [1.25, 36.0, 1.1, 2.3, 0.55, 0.03]
   bounds = (
       [0.1, 10.0, 0.1, 0.1, 0.0, 0.0],
       [5.0, 60.0, 10.0, 10.0, 1.0, 0.5],
   )
   popt, pcov = curve_fit(
       model, T, sigma, sigma=sigma_err,
       p0=p0, bounds=bounds,
   )

The recovered parameters from the synthetic dataset are close to the
inputs, though the two-gap fit is well known to be degenerate
between the small-gap fraction and the background. Without data
above :math:`T_c` the fitted :math:`T_c` saturates at the upper bin
of the data range; in real experiments at least a few points above
:math:`T_c` should be included so the high-:math:`T` tail anchors
both :math:`T_c` and :math:`\sigma_{bg}`. The literature values for
MgB₂ are :math:`\sigma_0 \approx 1.25\;\mu\mathrm{s}^{-1}`,
:math:`T_c \approx 36\;\mathrm{K}`,
:math:`\Delta_1 / k_B T_c \approx 1.1`,
:math:`\Delta_2 / k_B T_c \approx 2.3`, weight
:math:`w \approx 0.55`, and
:math:`\sigma_{bg} \approx 0.03\;\mu\mathrm{s}^{-1}`.

Step 5 — Convert σ to λ_L
-------------------------

.. image:: /_generated/screenshots/mgb2_lambda_t.png
   :alt: MgB₂ penetration depth λ_L(T) derived from σ(T)
   :width: 100%

The screenshot shows the same data after inverting σ → λ via
Asymmetry's
:func:`asymmetry.core.fitting.sc.constants.sigma_to_lambda_nm` (which
implements the Brandt triangular-lattice formula above), with the
background :math:`\sigma_{bg}` subtracted before the inversion so
that :math:`\lambda_L` reflects the superconducting contribution
alone:

.. code-block:: python

   from asymmetry.core.fitting.sc.constants import sigma_to_lambda_nm
   sigma_sc = sigma - 0.03   # subtract sigma_bg first
   lambda_nm = sigma_to_lambda_nm(sigma_sc)

The curve rises smoothly from a low-:math:`T` plateau (where the
superfluid density is fully developed) and diverges as
:math:`T \to T_c`. The absolute value of :math:`\lambda_L(0)`
depends on the convention used by the σ↔λ helper; Asymmetry's
implementation uses :math:`\gamma_\mu` in rad/(s·T), so for the
MgB₂ synthetic data it returns
:math:`\lambda_L(0) \approx 290\;\mathrm{nm}`. Comparing to the
muSR literature for MgB₂ (Niedermayer *et al.*, Phys. Rev. B **65**, 094512
(2002), which reports :math:`\lambda_L(0) \approx 110{-}130\;\mathrm{nm}`
depending on sample purity and orientation) requires accounting
for the factor-of-:math:`\sqrt{2\pi}` ambiguity between the
"rad/s" and "Hz" conventions for :math:`\sigma`. Either way, the
*shape* of :math:`\lambda_L(T)` — flat at low :math:`T`, divergent
near :math:`T_c`, with the two-gap kink visible — is convention-
independent and is what determines the gap structure.

Interpretation
--------------

What the analysis tells you about the material:

- **Single-gap fit succeeds, two-gap doesn't help.** The material is
  well-described by a single isotropic gap. The
  :math:`\Delta / k_B T_c` value tells you whether it's weak-coupling
  (BCS limit, 1.764) or strong-coupling (above ≈ 1.9–2.5).
- **Two-gap fit is needed.** The material has multiband
  superconductivity (e.g. MgB₂). The two gap ratios give the
  relative coupling strengths; the weight gives how the superfluid
  density partitions between bands.
- **d-wave fit is needed.** The material has nodes in the gap (e.g.
  cuprates). At low :math:`T`, :math:`\rho_s(T)` is linear in
  :math:`T`, not exponential.
- **The fit fails to converge.** Possible causes: insufficient
  low-:math:`T` points to constrain :math:`\Delta_s`; sample
  inhomogeneity (try a two-component composite); wrong gap structure
  (try another ``SC_*`` model).

Common pitfalls
---------------

- **Treating σ(T) as the only observable.** σ → λ is only exact in
  the London limit (:math:`\lambda \gg \xi`). For materials where
  :math:`\xi` approaches :math:`\lambda`, the Brandt expression has
  corrections; see Sonier, Rev. Mod. Phys. **72**, 769 (2000) Appendix B.

- **Ignoring the nuclear-dipolar background** :math:`\sigma_{bg}`.
  Even above :math:`T_c` the muon sees a non-zero :math:`\sigma`
  from nuclear dipoles. Always fit it as a parameter and subtract it
  before the σ → λ conversion.

- **Field too low or too high.** Below :math:`H_{c1}` the field does
  not penetrate as a vortex lattice — that's the Meissner state.
  Above :math:`H_{c2}` superconductivity is suppressed. Stay
  comfortably in the mixed state.

- **Data range too narrow.** Without points above :math:`T_c`, the
  fit cannot independently determine :math:`T_c` and
  :math:`\sigma_{bg}`. Without enough points below
  :math:`T/T_c \approx 0.3`, the small gap (in a two-gap model) is
  poorly constrained.

- **Wrong gap model for the universality class.** Don't fit d-wave
  to MgB₂ or s+s to YBCO. Cross-check against other techniques
  (specific heat, scanning tunnelling).

Forward pointer to the roadmap
------------------------------

This workflow exercises four planned improvements:

- `maxent-spectrum
  <../../porting/candidates/maxent-spectrum/>`_ would sharpen the
  vortex-lattice FFT line shape at low :math:`T` where data windows
  are short.
- `dynamic-kubo-toyabe
  <../../porting/candidates/dynamic-kubo-toyabe/>`_ is needed if the
  sample shows fluctuating background magnetism that competes with
  the vortex-lattice signal.
- `theory-library-expansion
  <../../porting/candidates/theory-library-expansion/>`_ would add a
  time-domain Brandt vortex :math:`P(B)` model so raw TF asymmetry
  can be fitted directly rather than via σ.
- `minos-error-analysis
  <../../porting/candidates/minos-error-analysis/>`_ delivers
  asymmetric uncertainties on the gap parameters that publication
  reviewers will increasingly expect.

Further reading
---------------

- S. J. Blundell, R. De Renzi, T. Lancaster, and F. L. Pratt, *Muon
  Spectroscopy: An Introduction* (Oxford University Press, Oxford, 2022),
  Ch. 9 (superconductors), especially Ch. 9.5 (penetration depth and gap
  structure), Fig. 9.5 (MgB₂ σ(T)).
- A. Amato and E. Morenzoni, *Introduction to Muon Spin Spectroscopy:
  Applications to Solid State and Material Sciences*, Lecture Notes in Physics
  Vol. 961 (Springer, Cham, 2024), Ch. 6 — particularly the sections on the
  vortex-state field distribution, multi-band superconductivity, and
  unconventional pairing symmetry.
- J. E. Sonier, J. H. Brewer, and R. F. Kiefl, Rev. Mod. Phys. **72**, 769
  (2000) — the canonical review of TF μSR in superconductors.
- Niedermayer *et al.*, Phys. Rev. B **65**, 094512 (2002) for the
  MgB₂ penetration-depth μSR measurement.

Cross-references
----------------

- :doc:`/reference/loading_data`
- :doc:`/reference/composite_models` — building the
  ``Oscillatory + Gaussian + Constant`` time-domain composite.
- :doc:`/reference/grouped_time_domain_fitting` — when per-detector
  second-moment analysis is needed.
- :doc:`/reference/fourier_analysis` — for inspecting the
  :math:`P(B)` line shape directly.
- :doc:`/reference/parameter_trending` — for the
  σ → λ → gap-model pipeline.
- :doc:`/reference/sc_penetration_depth` — the parametric-model
  reference page.
