Temperature scan through a magnetic transition
==============================================

This chapter is a worked example showing how Asymmetry handles a
zero-field (ZF) μSR temperature scan through a magnetic ordering
transition. The synthetic data corresponds to the textbook EuO example
(:math:`T_c \approx 69` K; see Blundell *et al.* 2022 Fig 6.6 and the
discussion of cubic ferromagnets in Amato & Morenzoni 2024 Ch 5). The
same workflow applies, with minor adaptations, to antiferromagnets
(where the muon precesses in the staggered local field) and to canted
or molecular magnets. The screenshots below are taken directly from
the GUI driving the synthetic dataset shipped with the documentation —
they are intended to show what each stage of the analysis looks like
in practice rather than as exercises for the reader.

Physical motivation
-------------------

A muon stopped in an ordered magnetic phase sees a static local
field :math:`B_\mu` set by the dipolar (and, for metals, RKKY)
contributions of the surrounding ions. Its spin precesses at the
Larmor frequency

.. math::

   \nu_\mu = \frac{\gamma_\mu}{2\pi}\, B_\mu

where :math:`\gamma_\mu / 2\pi \approx 13.55\;\mathrm{kHz/G}`. As
temperature rises toward :math:`T_c` the sublattice magnetisation
falls — and so does :math:`B_\mu` — until the precession washes out
completely. For a mean-field magnet the order parameter follows a
power law near :math:`T_c`,

.. math::

   M(T) = M_0 \left(1 - \frac{T}{T_c}\right)^{\beta},

with :math:`\beta` depending on the model (:math:`\beta = 1/2` in
Landau mean field, :math:`\beta \approx 0.32{-}0.37` in the
3D Heisenberg model — see Blundell Ch 6.2 for the full taxonomy of
critical exponents). Measuring :math:`\beta` from μSR is one of the
most direct ways to test which universality class a given material
belongs to.

The data
--------

The example uses six ZF runs at temperatures
:math:`T = 30, 50, 65, 69, 73, 90\;\mathrm{K}`. The synthetic
generator (``docs/screenshots/data/archetypes.py:make_euo_tf_tscan``)
produces asymmetry traces that:

- **Below** :math:`T_c`: oscillate at
  :math:`\nu = 28\,(1 - T/T_c)^{0.40}\;\mathrm{MHz}` with a Lorentzian-
  peaked damping rate :math:`\lambda(T)` at :math:`T_c`.
- **Above** :math:`T_c`: relax as a single paramagnetic exponential
  with damping that peaks just above :math:`T_c` due to critical
  fluctuations and tails away at high :math:`T`.

Step 1 — Load and inspect
-------------------------

.. image:: /_generated/screenshots/main_window.png
   :alt: EuO ZF temperature scan loaded in the main window
   :width: 100%

The screenshot shows Asymmetry with the six runs loaded. The data
browser on the left lists one row per temperature; the central plot
shows the currently-selected run (here :math:`T = 65\;\mathrm{K}`,
just inside the ordered phase, where the spontaneous-field precession
is at its slowest and the critical damping is largest, so the
time-domain signal damps away within ~1 μs). Clicking through the
rows shows how the character of the signal changes across the
transition: a clear, slowly-damped precession at :math:`T = 30\;\mathrm{K}`
well below :math:`T_c`; precession damped within a microsecond by
critical fluctuations at :math:`T = 65` and :math:`69\;\mathrm{K}`;
and pure exponential decay with no oscillation at :math:`T = 73` and
:math:`90\;\mathrm{K}` above :math:`T_c`. This visual inspection is
the first half of model selection — at this stage it is already clear
that an oscillatory model is needed below :math:`T_c` and an
exponential one above.

.. image:: /_generated/screenshots/logbook_view.png
   :alt: Data browser sorted by temperature
   :width: 100%

The data browser doubles as a run logbook (clicking the *T*\ (K)
column header sorts the scan from coldest to hottest), so the scan
reads top-to-bottom for the rest of the workflow.

Step 2 — Group and bunch identically
------------------------------------

The **Grouping** dialog (opened from the toolbar) verifies that the
forward/backward detector assignments and the :math:`\alpha` value are
consistent across runs. For synthetic data the defaults are correct;
on real data the same dialog is opened on the lowest-:math:`T` run,
the detector mapping is checked against the instrument geometry, the
:math:`\alpha` value is recorded (typically 1.0–1.4 for an ideal F–B
pair), and the resulting grouping is applied to all selected runs via
**Apply to selection**. The bunch factor is set on the toolbar; for
the EuO time window (0–6 μs) a factor of ×4 keeps the Nyquist limit
comfortably above the ~22 MHz Larmor at low :math:`T` while still
reducing per-bin noise.

Step 3 — Choose a fit model per regime
--------------------------------------

.. image:: /_generated/screenshots/fit_wizard_result.png
   :alt: Fit Wizard result page — answer card and decision trail (Ag dataset shown as a reference)
   :width: 100%

For the :math:`T = 65\;\mathrm{K}` run (mid-:math:`T_c`) the
**Fit Wizard** (toolbar → ``Fit`` → ``Fit Wizard...``) fingerprints
the data, fits a curated set of around a dozen candidate composite
models, and recommends one with a confidence grade. The screenshot
shows the wizard's result page on a static-field reference dataset
(Ag): an answer card with the recommended model, its confidence, and
the data-with-fit overlay, above a decision trail whose steps expand to
the underlying candidate rankings. For the EuO mid-:math:`T_c`
run the top candidates are ``Oscillatory + Exponential + Constant``
and ``StaticGKT × Exponential``, the former being the natural choice
inside the ordered phase. For runs well below :math:`T_c`,
``Oscillatory + Constant`` (a single damped cosine) is sufficient;
above :math:`T_c`, ``Exponential + Constant``. The runs nearest
:math:`T_c` may benefit from a stretched exponential or a
two-component model with a fraction group — see
:doc:`/reference/composite_models`.

Step 4 — Fit each run
---------------------

.. image:: /_generated/screenshots/euo_fit_oscillatory.png
   :alt: Converged single-run fit on the EuO T=30 K dataset
   :width: 100%

The screenshot shows the fit panel after running
``Oscillatory * Exponential + Constant`` on the :math:`T = 30\;\mathrm{K}`
run. The parameter table on the right reports the converged values
with their Hessian uncertainties: an amplitude
:math:`A_1 = 22.00\,\%`, the precession frequency
:math:`\nu = 22.29\;\mathrm{MHz}` (the expected
:math:`\nu_0\,(1 - 30/69)^{0.40} = 22.29\;\mathrm{MHz}`), a phase
consistent with zero, a damping rate
:math:`\lambda = 0.10\;\mu\mathrm{s}^{-1}`, and a small constant
background :math:`A_{bg} = 0.39\,\%`. The reduced
:math:`\chi^2 = 0.98` confirms the model captures the data within the
quoted uncertainties. The central plot overlays the fit curve on the
data so the agreement can be checked visually.

The same fit is repeated for the other below-:math:`T_c` runs. Above
:math:`T_c` the model becomes ``Exponential + Constant`` (the
oscillatory term carries no signal). Locking the background parameter
to the value established on the lowest-:math:`T` run is a useful
trick once that value is well constrained — backgrounds are
typically temperature-independent over a single run series, and
letting the fit re-discover the background at every temperature
contaminates the trend in :math:`\lambda`.

Step 5 — Trend the order parameter
----------------------------------

Opening the **Fit Parameters** dock (toolbar → ``Params``) and
selecting all the runs populates a sortable trend table; selecting a
parameter from the y-axis dropdown plots that parameter against
temperature. The expected shape for the EuO data is
:math:`\nu(T)` starting at :math:`\sim 22\;\mathrm{MHz}` at
:math:`T = 30\;\mathrm{K}` (the lowest temperature in the scan) and
falling to zero at :math:`T_c` with a downward concavity
(:math:`\beta < 1`), and :math:`\lambda(T)` showing a peak at
:math:`T_c` — the hallmark of critical slowing-down of the spin
fluctuations.

Step 6 — Fit the order parameter to a power law
-----------------------------------------------

.. image:: /_generated/screenshots/temperature_trend_fit.png
   :alt: Fit Parameters trending panel showing EuO ν(T) with the fitted OrderParameter (Landau) curve
   :width: 100%

In the trending panel, click **Model Fit** on the ``f (MHz)`` row and
fit the built-in ``OrderParameter`` (Landau power-law) model

.. math::

   \nu(T) = y_0 \left[1 - (T/T_c)^{\alpha}\right]^{\beta},
   \quad T < T_c,

which reduces to the Landau form
:math:`\nu_0 (1 - T/T_c)^{\beta}` when :math:`\alpha = 1`. The model
vanishes identically at and above :math:`T_c`, so the paramagnetic
runs (:math:`\nu = 0`) constrain :math:`T_c` directly.

For the synthetic EuO data the panel recovers
:math:`y_0 \approx 27.5\;\mathrm{MHz}`,
:math:`T_c \approx 69.0\;\mathrm{K}`, :math:`\beta \approx 0.39`, and
:math:`\alpha \approx 1.0` — matching the input parameters and within
the range expected for an isotropic 3D ferromagnet. The screenshot
shows the six-point :math:`\nu(T)` trend (per-run ZF fit results) with
the three points at and above :math:`T_c` — 69, 73 and 90 K — sitting
on the :math:`\nu = 0` axis, and the fitted ``OrderParameter`` curve
overlaid (the **Model Fit\*** button flags the active fit).

The same fit can be reproduced outside the GUI by exporting the trend
(**Export TSV**) and fitting with ``scipy.optimize.curve_fit``:

.. code-block:: python

   import numpy as np
   from scipy.optimize import curve_fit

   T = np.array([30.0, 50.0, 65.0, 69.0, 73.0, 90.0])
   nu = np.array([22.19, 16.78, 9.11, 0.0, 0.0, 0.0])   # MHz from the fits
   nu_err = np.full_like(T, 0.4)

   def landau(T, nu0, Tc, beta):
       arg = np.clip(1 - T / Tc, 1e-9, None)
       return nu0 * arg ** beta

   popt, _ = curve_fit(landau, T, nu, sigma=nu_err, p0=[28.0, 69.0, 0.4])
   nu0, Tc, beta = popt
   print(f"ν0 = {nu0:.2f} MHz, Tc = {Tc:.2f} K, β = {beta:.3f}")

Interpretation
--------------

The analysis exposes four physical quantities:

- :math:`T_c` is the magnetic ordering temperature.
- :math:`\beta` classifies the universality class —
  :math:`\beta \approx 0.5` for mean-field / Landau,
  :math:`\beta \approx 0.37` for 3D Heisenberg,
  :math:`\beta \approx 0.33` for 3D Ising, :math:`\beta = 0.125` for
  2D Ising. EuO sits close to the 3D Heisenberg universality class.
- :math:`\nu_0` is the muon-site local field at :math:`T = 0`.
  Combined with a calculated dipolar tensor (e.g. from MuFinder or
  μ-LFC) it pins down which crystallographic site the muon occupies.
- The :math:`\lambda` peak at :math:`T_c` is the critical-fluctuation
  signature; its width in temperature is set by the correlation
  length divergence.

A more accurate :math:`\beta` would need more temperature points near
:math:`T_c` (within :math:`\pm 2\;\mathrm{K}`), a careful cross-check
that all runs use the same grouping, and asymmetric error analysis
on the per-run fits.

Common pitfalls
---------------

- **One composite model across all temperatures.** The qualitative
  change of regime at :math:`T_c` means a single model — say always
  ``Oscillatory + Constant`` — will silently underweight the
  paramagnetic runs. Use different models per regime, or a stretched-
  exponential composite that interpolates.

- **Forgetting to lock the background.** Backgrounds drift slowly
  with detector efficiency; letting the fit float them at every run
  contaminates the trend in :math:`\lambda` with the background
  trend.

- **Over-bunching.** ×16 bunching loses Nyquist for a ~22 MHz signal
  — the fit then reports an unphysically high :math:`\lambda` to
  compensate.

- **Ignoring critical slowing.** The damping peak at :math:`T_c`
  isn't an artefact; it's the physical signature of the transition.
  A :math:`\lambda(T)` that shows no peak indicates the critical
  region is finer than the temperature spacing and more runs are
  needed.

Further reading
---------------

- S. J. Blundell, R. De Renzi, T. Lancaster, and F. L. Pratt, *Muon
  Spectroscopy: An Introduction* (Oxford University Press, Oxford, 2022),
  Ch. 6.1–6.2 (magnetism, Landau theory, critical exponents); the EuO data
  are summarised in Fig. 6.6.
- A. Amato and E. Morenzoni, *Introduction to Muon Spin Spectroscopy:
  Applications to Solid State and Material Sciences*, Lecture Notes in Physics
  Vol. 961 (Springer, Cham, 2024), Ch. 5 (μSR in ordered magnets, with
  expanded discussion of antiferromagnets, frustrated systems, and
  unconventional order parameters).
- T. Lancaster *et al.*, Phys. Rev. B **75**, 094421 (2007) — a real-data
  example on the molecular magnet Cu(pyz)₂(ClO₄)₂ (three precession
  frequencies near :math:`T_c`).

Cross-references
----------------

- :doc:`/reference/loading_data` — load formats.
- :doc:`/reference/detector_grouping` — group definitions.
- :doc:`/reference/fit_wizard` — the model-recommendation tool.
- :doc:`/reference/parameter_trending` — the trend panel.
- :doc:`/reference/composite_models` — combining oscillatory and
  relaxation envelopes.
