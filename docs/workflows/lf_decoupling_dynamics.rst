LF decoupling and static-vs-dynamic field distributions
=======================================================

This chapter is a worked example of a longitudinal-field decoupling
series on a nonmagnetic host (Ag polycrystal — the standard
calibration sample at every μSR facility). The analysis measures the
Gaussian width :math:`\Delta` of the static local-field distribution
and verifies that the distribution is genuinely static (not dynamic)
by checking that the polarisation recovers fully at large LF. The
screenshots below are taken from the GUI driving the synthetic
dataset shipped with the documentation; they are intended to show
what each stage of the analysis looks like in practice. The
synthetic data corresponds to the textbook example in Blundell
*et al.* 2022 Fig. 5.6 (after Hayano *et al.*, Phys. Rev. B **20**, 850
(1979)); Amato & Morenzoni 2024 Ch 4 covers the same formalism with
additional emphasis on dynamic-regime crossovers.

Physical motivation
-------------------

In a paramagnetic or non-magnetic host, the muon spin samples a
distribution of local fields :math:`B_\mu` that arise mainly from
randomly-oriented nuclear dipoles. If the distribution is isotropic
Gaussian with standard deviation :math:`\Delta / \gamma_\mu` along
each Cartesian axis, the zero-field polarisation function is the
**static Gaussian Kubo–Toyabe** form

.. math::

   P_z^{\mathrm{KT}}(t) = \frac{1}{3} + \frac{2}{3}
   (1 - \Delta^2 t^2) \exp\!\left(-\frac{\Delta^2 t^2}{2}\right).

The "1/3 tail" — the long-time recovery to one-third of the initial
polarisation — is the fingerprint of an isotropic static
distribution. Two-thirds of the muons see a perpendicular field
component and dephase; one-third see a parallel component and don't.
Applying a longitudinal field :math:`B_L` quenches the perpendicular
dephasing once :math:`\gamma_\mu B_L \gg \Delta`. Hayano *et al.*
computed the full LF-KT polarisation function; in practice the
decoupling progression is read off at five field values spanning
:math:`\gamma_\mu B_L / \Delta \in \{0, 1, 2, 5, 10\}`.

If the field distribution is **dynamic** — the muon's local field
fluctuates on a timescale :math:`\tau_f \sim \nu^{-1}` — the static KT
formula no longer applies. The signature is that the 1/3 tail itself
decays. In strong applied LF the dynamic system shows residual
relaxation that the static formula cannot reproduce:
:math:`\lambda_{\min}(B_L \to \infty) \to 2 \Delta^2 / \nu > 0`.

The data
--------

The example uses five Ag polycrystal runs at :math:`T = 20\;\mathrm{K}`
with applied longitudinal field
:math:`B_L \in \{0, 5, 10, 25, 50\}\;\mathrm{G}`. With Ag's known
:math:`\Delta \approx 0.39\;\mu\mathrm{s}^{-1}`, this maps to
:math:`\gamma_\mu B_L / \Delta \approx \{0, 1.0, 2.0, 5.0, 9.6\}` —
the textbook decoupling progression. The global-fit screenshot
further down uses a four-field subset
(:math:`B_L \in \{0, 15, 50, 100\}\;\mathrm{G}`) to span an even
wider decoupling range.

Step 1 — Load and group as a series
-----------------------------------

.. image:: /_generated/screenshots/lf_kt_series_plot.png
   :alt: Ag LF Kubo–Toyabe field-decoupling series overlay
   :width: 100%

The screenshot shows the five runs loaded into a data group named
"LF decoupling — Ag", with **Overlay** enabled so the central plot
draws all five runs on the same axes. The canonical decoupling
progression is plain to read off:

- :math:`B_L = 0\;\mathrm{G}`: clear KT dip near
  :math:`t \approx \sqrt{3}/\Delta \approx 4.4\;\mu\mathrm{s}`
  followed by recovery to :math:`A_0/3 \approx 8\,\%` (the initial
  asymmetry is 24 %).
- :math:`B_L = 5\;\mathrm{G}` (:math:`\gamma_\mu B/\Delta \approx 1`):
  dip fills in, tail rises.
- :math:`B_L = 10\;\mathrm{G}` (:math:`\gamma_\mu B/\Delta \approx 2`):
  nearly flat above ~1/3.
- :math:`B_L = 25, 50\;\mathrm{G}`
  (:math:`\gamma_\mu B/\Delta \approx 5, 10`): completely decoupled,
  asymmetry slowly relaxing only from residual paramagnetism.

If the :math:`B_L = 50\;\mathrm{G}` run still showed clear relaxation,
the system would be dynamic — see the dynamic-regime section below.

Step 2 — Choose the model and parameter classification
------------------------------------------------------

.. image:: /_generated/screenshots/global_fit_lfkt.png
   :alt: Global fit setup for the Ag LF-KT decoupling series
   :width: 100%

The model for every run is ``LongitudinalFieldKT + Constant``. The
screenshot shows the global-fit tab populated with the four-field
subset and the parameter table set up for the joint fit:

- :math:`A_1` — initial asymmetry (Global, shared across runs).
- :math:`\Delta` — Gaussian width (Global; this is the quantity the
  fit measures).
- :math:`B_L` — applied LF in gauss (Local per run, fixed to the
  experimentally-set value).
- :math:`A_{bg}` — constant background (Global).

The **Type** column drives the parameter classification: ``Global``
parameters share one value across every run, ``Local`` parameters
take an independent value per run, and ``File`` parameters are fixed
per run to the value stored in the run metadata.

Step 3 — Run the global fit
---------------------------

.. image:: /_generated/screenshots/lf_kt_global_results.png
   :alt: Converged global fit on the Ag LF-KT decoupling series
   :width: 100%

After ``Run Global Fit`` finishes, the parameter table reports the
fitted global values. For the four-field synthetic series shown here
the engine recovers :math:`\Delta = 0.3902\;\mu\mathrm{s}^{-1}`
(input 0.39, Hessian uncertainty around :math:`4 \times 10^{-4}`,
about 0.1 %), :math:`A_1 = 23.91\,\%`, and
:math:`A_{bg} = 0.31\,\%`. The log panel shows the per-run reduced
:math:`\chi^2` averaging to 1.02, confirming a clean joint
convergence. The central plot overlays the LF-KT fit curve on the
selected run (here :math:`B_L = 0\;\mathrm{G}`); the curve passes
through the dip and the 1/3 recovery cleanly.

Computing :math:`\gamma_\mu B_L / \Delta` for each run from the
fitted :math:`\Delta` gives a monotonic progression
:math:`\{0, 5.2, 17.4, 34.7\}` across the four-field subset (using
:math:`\gamma_\mu/2\pi = 13.55\;\mathrm{kHz/G}`), confirming that
the highest field is deep in the decoupled regime.

Step 4 — Diagnose static vs dynamic
-----------------------------------

The decoupling diagnostic is read off the high-field run. At
:math:`\gamma_\mu B_L / \Delta \gtrsim 5` the LF-KT model predicts
:math:`P_z(t \to \infty) \approx 1`. If the
:math:`B_L = 100\;\mathrm{G}` run's late-time asymmetry plateaus near
:math:`A_0 + A_{bg}` within the fit uncertainty, the distribution is
static and the measured :math:`\Delta` is the final answer. If the
data continue to relax linearly or exponentially at late times, the
distribution is dynamic and a dynamic-KT analysis is required (see
below).

For the synthetic Ag dataset the high-field run flattens cleanly —
the global fit's per-run :math:`\chi^2_r` is near unity for every
field, including the most-decoupled one, which is the quantitative
form of the same statement. Ag at :math:`T = 20\;\mathrm{K}` is in
the static regime (nuclear-dipolar fluctuations are negligible at
this temperature), and the analysis stops here.

Forward pointer: dynamic-regime workflow
----------------------------------------

If the data show dynamic relaxation in the decoupled limit, the
correct model is the **strong-collision dynamic Kubo–Toyabe** of
Hayano *et al.* (1979). It introduces a third parameter :math:`\nu`
(fluctuation rate, in :math:`\mu\mathrm{s}^{-1}`) and reduces to the
static form as :math:`\nu \to 0`.

**Asymmetry ships dynamic KT.** The strong-collision dynamic
Kubo–Toyabe is available as two models: ``DynamicGaussianKT`` (dense
nuclear-dipolar fields) and ``DynamicLorentzianKT`` (dilute or
randomly-diluted moments). Both are registered in **both** registries —
the ``COMPONENTS`` expression registry (so they combine in composite
expressions, e.g. ``DynamicGaussianKT + Constant``) and the ``MODELS``
standalone registry (``MODELS["DynamicGaussianKT"]``) — so you can fit
the dynamic regime directly rather than approximating it.

The standard tactic for a dynamic LF/ZF series is to **fix the static
width** :math:`\Delta` from a low-temperature (static) reference and
**float only** the fluctuation rate :math:`\nu`. The worked example
:doc:`dynamic_kt_copper` walks through this end-to-end on a copper run
(muon diffusion); the reference pages :ref:`fit-dynamic-gaussian-kt`
and :ref:`fit-dynamic-lorentzian-kt` give the parameter lists and
formulas.

Interpretation
--------------

What the analysis tells you:

- :math:`\Delta` is the standard deviation of the local-field
  distribution divided by :math:`\gamma_\mu`. For nuclear-dipolar
  fields it sets the effective spread of dipolar couplings — useful
  for confirming the muon stopping site against a calculated dipolar
  tensor (e.g. via MuFinder or µ-LFC).
- **Static vs dynamic**: a clean LF decoupling rules out motional
  narrowing within the muon's observation window. If dynamics are
  present, :math:`\tau_f` has been measured indirectly.
- **Hayano "calibration" of** :math:`\gamma_\mu`: running the same
  analysis on a well-characterised host (Ag, Cu, KBr) is a standard
  cross-check for instrument calibration.

Common pitfalls
---------------

- **Insufficient LF coverage.** Omitting the high-field point
  (:math:`\gamma_\mu B_L / \Delta \gtrsim 5`) means static and dynamic
  cannot be distinguished. Always include at least one far-decoupled
  point.

- **Fixed-bound** :math:`B_L`. If the applied field is off by 5 %
  (common for low-field instruments), the static fit accommodates the
  miscalibration by inflating :math:`\Delta`. Cross-check the applied
  field against a known calibration sample.

- **Forgetting background.** The asymmetry doesn't return to
  :math:`A_0` because of muons stopping in the sample holder.
  Subtract a constant background or fit it.

- **Misreading the "1/3 tail".** It's :math:`A_0/3 + A_{bg}`, not
  one third of unity. Easy mistake when the background is large.

Further reading
---------------

- S. J. Blundell, R. De Renzi, T. Lancaster, and F. L. Pratt, *Muon
  Spectroscopy: An Introduction* (Oxford University Press, Oxford, 2022),
  Ch. 5.2 — static Gaussian KT (eq. 5.13), LF-KT (eq. 5.27), dynamic KT
  (Ch. 5.3, the strong-collision derivation).
- A. Amato and E. Morenzoni, *Introduction to Muon Spin Spectroscopy:
  Applications to Solid State and Material Sciences*, Lecture Notes in Physics
  Vol. 961 (Springer, Cham, 2024), Ch. 4 — covers the same formalism plus the
  relationship between :math:`\Delta` and the calculated dipolar tensor.
- R. S. Hayano *et al.*, Phys. Rev. B **20**, 850 (1979) — the original static
  and dynamic KT formulas; still the canonical reference.
- A. Yaouanc and P. Dalmas de Réotier, *Muon Spin Rotation, Relaxation, and
  Resonance: Applications to Condensed Matter* (Oxford University Press,
  Oxford, 2011), Ch. 6 — the most detailed mathematical treatment.

Cross-references
----------------

- :doc:`/reference/fit_functions/kubo_toyabe` — the LF-KT reference page.
- :doc:`dynamic_kt_copper` — the dynamic Kubo–Toyabe worked example
  (fitting the fluctuation rate :math:`\nu`).
- :doc:`/reference/global_fit_wizard` — the Global Fit Wizard.
- :doc:`/reference/composite_models` — for composite envelopes.
- :doc:`/reference/fit_wizard` — for model recommendation.
