Spin dynamics in a Dirac spin liquid: a cross-temperature global fit
====================================================================

This chapter reproduces the field-scan analysis of Wu *et al.* on the
Dirac U(1) quantum-spin-liquid candidate YbZn₂GaO₅ [1]_, working
entirely on synthetic data. The experiment probes the muon-spin
relaxation rate :math:`\lambda` as a function of longitudinal field at
eight temperatures spanning the quantum-to-classical crossover; the
field dependence :math:`\lambda(B)` at each temperature resolves into
four physical contributions — two-dimensional spin diffusion, a
zero-dimensional Redfield term, a flat background, and a
level-crossing-resonance peak — some of whose parameters are shared
across temperature and some of which vary with it. Extracting them is a
textbook use of the cross-group global fit: batch-fit each temperature's
field scan, trend :math:`\lambda` against field, then fit all eight
:math:`\lambda(B)` curves jointly with a common set of shared parameters.

The data here are **synthetic data generated from the published
parameter values** (Table I and Figs. 3–4 of Ref. [1]_); no
experimental data are used or distributed. They are produced by the
:mod:`asymmetry.examples.ybzn2gao5` module, whose docstring records the
unit conventions and the truth used. One honest caveat: the per-run
fluctuation rate :math:`\nu` is deliberately retuned to a smaller scale
than the paper's Fig. 4 reading so that the sharp :math:`m = 7` cutoff
falls inside the accessible field window and the synthetic global fit
stays cleanly identifiable — see the module docstring for the full
rationale. The eight shared (Table I) parameters are used exactly.

The screenshots below are taken from the GUI driving the synthetic
dataset; they show what each stage of the analysis looks like in
practice rather than serving as exercises for the reader.

Physical motivation
-------------------

A quantum spin liquid evades magnetic order down to zero temperature,
its spins remaining dynamically correlated. In a Dirac U(1) spin liquid
the low-energy excitations are gapless spinons with a linear (Dirac)
dispersion, and the resulting spin dynamics leave a characteristic
fingerprint in the field and temperature dependence of the muon
relaxation rate. Following Ref. [1]_, the longitudinal-field relaxation
rate is written as the sum of four terms,

.. math::

   \lambda(B) = \lambda_{2\mathrm{D}}(B) + \lambda_{0\mathrm{D}}(B)
   + \lambda_{\mathrm{BG}} + \lambda_{\mathrm{LCR}}(B),

where each term names a physical channel:

- :math:`\lambda_{2\mathrm{D}}(B) = (A^2/4)\,J_{2\mathrm{D}}(D_{2\mathrm{D}}, \omega_e)`
  is **two-dimensional spin diffusion**, with :math:`A` the coupling
  amplitude and :math:`D_{2\mathrm{D}}` the in-plane diffusion rate. Its
  logarithmic field dependence is the signature of correlated,
  low-dimensional dynamics.
- :math:`\lambda_{0\mathrm{D}}(B) = (D^2/4)(2/\nu)\,[1 + (\omega_\mu/\nu)^m]^{-1}`
  is a **zero-dimensional Redfield term** with fluctuation rate
  :math:`\nu` and a sharp spectral-density cutoff of exponent :math:`m`.
  A cutoff much steeper than the Lorentzian :math:`m = 2` — here
  :math:`m \approx 7` — is direct evidence of non-trivial, non-diffusive
  local dynamics.
- :math:`\lambda_{\mathrm{BG}}` is a **flat, temperature-independent
  background** from nuclear-dipolar and instrumental contributions.
- :math:`\lambda_{\mathrm{LCR}}(B) = f\,G(B; B_0, B_{\mathrm{wid}})` is a
  **level-crossing resonance** — a Gaussian peak centred at the field
  :math:`B_0` where a muon–electron level crossing enhances relaxation,
  of width :math:`B_{\mathrm{wid}}` and amplitude :math:`f`.

Three of the parameters vary with temperature and are therefore
**local** to each scan: the diffusion rate :math:`D_{2\mathrm{D}}`, the
fluctuation rate :math:`\nu`, and the resonance amplitude :math:`f`. The
remaining six — the amplitudes :math:`A` and :math:`D`, the background
:math:`\lambda_{\mathrm{BG}}`, the cutoff exponent :math:`m`, and the
resonance centre and width :math:`B_0` and :math:`B_{\mathrm{wid}}` — are
genuinely shared physics and are held **global** across all eight
temperatures. Sharing them pools the eight scans and pins the shared
parameters far more tightly than any single scan could.

The data
--------

The example uses 160 runs — eight temperatures
(:math:`T = 0.05, 0.2, 0.4, 1.6, 3.2, 6, 9, 12\;\mathrm{K}`) at 20
log-spaced longitudinal fields each, spanning
:math:`10\;\mathrm{G}\;(0.001\;\mathrm{T})` to
:math:`45\,000\;\mathrm{G}\;(4.5\;\mathrm{T})`. The exchange scale
:math:`J \approx 3.2\;\mathrm{K}` separates a low-temperature quantum
plateau (where :math:`D_{2\mathrm{D}}` is roughly temperature-independent
and the resonance amplitude :math:`f` is zero) from the classical regime
above :math:`J` (where :math:`D_{2\mathrm{D}}` rises steeply and the
level-crossing peak switches on). Each run's muon asymmetry is a simple
exponential relaxation on a flat background,
:math:`a(t) = a_0 \exp(-\lambda t) + a_{\mathrm{BG}}` with
:math:`a_0 \approx 21\,\%` and :math:`a_{\mathrm{BG}} \approx 4\,\%`,
matching the paper's low-temperature :math:`\beta = 1` description; the
per-run rate is :math:`\lambda(B, T)` evaluated from the model above. The
event budget is sized so that a single-run exponential fit returns a
:math:`\lambda` error of a few percent.

Because the whole analysis runs on the repository's own component
functions, the synthetic truth and the fit share one implementation of
the physics — the field-scan fit does not use a separate reference
formula.

Step 1 — Generate the dataset
-----------------------------

Generate the runs from the command line:

.. code-block:: bash

   python -m asymmetry.examples.ybzn2gao5 --out ./ybzn2gao5_runs

This writes 160 NeXus V1 ``.nxs`` files — one per (temperature, field) —
into the chosen directory, with run numbers in a fictitious 90xxx range
and the generic sample name "YbZn2GaO5 (synthetic)". Each run's title
records its temperature and field so they reload with full provenance.
The generation is deterministic: the default seed produces byte-identical
files on every run, so the numbers quoted below are reproducible. Passing
``--fields`` changes the number of field points per temperature and
``--seed`` selects a different Poisson realisation.

Step 2 — Load and group by temperature
--------------------------------------

.. image:: /_generated/screenshots/ybzn2gao5_runs_loaded.png
   :alt: The 160 synthetic YbZn2GaO5 runs loaded and grouped by temperature
   :width: 100%

Load all 160 ``.nxs`` files (**File ▸ Open**, or drag the directory onto
the data browser). The runs carry temperature and field metadata, so the
data browser sorts them into the eight-temperature, twenty-field grid.
Group the runs by temperature into eight data groups — one per
temperature — so that each group holds a single :math:`\lambda(B)` field
scan. The group name conventionally carries the temperature, e.g.
"T = 3.2 K", which becomes the panel label later in the analysis.

For real data the detector grouping and :math:`\alpha` value would be
checked on one run and applied to the whole series first (see
:doc:`calibration_grouping_emu`); for the synthetic data the defaults are
correct.

Step 3 — Batch-fit each temperature
-----------------------------------

.. image:: /_generated/screenshots/ybzn2gao5_batch_fit.png
   :alt: Batch exponential relaxation fits across the field scan at one temperature
   :width: 100%

The level-1 model for every run is ``Exponential + Constant``: an
exponential relaxation :math:`a_0 \exp(-\lambda t)` plus the flat
background :math:`a_{\mathrm{BG}}`. Fit each temperature group as a batch
so every field point in the scan is fitted with the same model and shared
seeding (see :doc:`superconductor_penetration_depth` for the batch-fit
mechanics on a field/temperature series).

Check the fits before trending. Each per-run :math:`\lambda` should carry
an uncertainty of a few percent, and the per-run reduced
:math:`\chi^2` should sit near unity — the batch machinery reports both.
A run whose :math:`\lambda` error blows up (typically the most-decoupled,
counts-depleted high-field point) is a sign to check the fit window
rather than to trust the point.

Step 4 — Trend λ against field
------------------------------

.. image:: /_generated/screenshots/ybzn2gao5_trend_lambda.png
   :alt: Lambda versus longitudinal field on a logarithmic field axis, eight temperature series
   :width: 100%

Open the **Fit Parameters** dock (toolbar → ``Params``) and select
:math:`\lambda` from the y-axis dropdown. Because the runs carry field
metadata, choose **field** as the x-axis; the fields span three decades,
so set the field axis to logarithmic. Selecting all eight temperature
series overlays the eight :math:`\lambda(B)` curves on one plot. The
qualitative story of Fig. 3 is already visible: a smooth
diffusion-dominated fall-off at low temperature, and — from
:math:`T = 3.2\;\mathrm{K}` upward — a resonance bump near
:math:`2.7\;\mathrm{T}` riding on top of it.

Step 5 — Set up the global fit
------------------------------

.. image:: /_generated/screenshots/ybzn2gao5_setup_dialog.png
   :alt: The New global parameter fit setup dialog with the eight temperature series selected
   :width: 100%

With two or more series selected, the trend panel's **Model Fit** button
relabels to **Global fit (8 groups)…**; clicking it (or **Analysis ▸ New
global parameter fit…**) opens the setup dialog. Configure the joint fit:

- **Parameter:** :math:`\lambda` — the observable being fitted across
  groups.
- **Series:** all eight temperature series, ticked.
- **X axis:** **field** — the abscissa of each :math:`\lambda(B)` curve.
- **Group variable:** **temperature** — the coordinate that distinguishes
  the eight groups and against which the local parameters will be
  plotted. The per-group values default to each group's median
  temperature; leave them as inferred.

Click **Continue** to move to the fit dialog.

Step 6 — Classify the parameters and fit
----------------------------------------

.. image:: /_generated/screenshots/ybzn2gao5_roles_dialog.png
   :alt: The cross-group fit dialog with the four-component model and parameter roles assigned
   :width: 100%

In the fit dialog, enter the four-component composite model
string

.. code-block:: text

   DiffusionLF_2D + Redfield + Lambda_bg + GaussianLCR

and assign each parameter a **role** in the **Type** column. ``Global``
parameters share one value across all eight temperatures, ``Local``
parameters take an independent value per temperature, and ``Fixed``
parameters are held at a set value:

.. list-table:: Parameter roles for the YbZn₂GaO₅ global fit
   :header-rows: 1
   :widths: 20 20 60

   * - Parameter
     - Role
     - Meaning
   * - :math:`D_{2\mathrm{D}}`
     - Local
     - In-plane 2D spin-diffusion rate (varies with :math:`T`)
   * - :math:`\nu`
     - Local
     - 0D Redfield fluctuation rate (varies with :math:`T`)
   * - :math:`f`
     - Local
     - Level-crossing-resonance amplitude (zero below :math:`J`)
   * - :math:`A`
     - Global
     - 2D-diffusion coupling amplitude (shared)
   * - :math:`D`
     - Global
     - 0D Redfield amplitude (shared)
   * - :math:`\lambda_{\mathrm{BG}}`
     - Global
     - Flat background rate (shared)
   * - :math:`m`
     - Global
     - Redfield spectral-density cutoff exponent (shared)
   * - :math:`B_0`
     - Global
     - Level-crossing-resonance centre (shared)
   * - :math:`B_{\mathrm{wid}}`
     - Global
     - Level-crossing-resonance width (shared)
   * - :math:`D_\perp`
     - Fixed 0
     - Out-of-plane diffusion, held at zero (2D limit)

If you would rather not assign the roles by hand, the **Suggest roles…**
button searches global-versus-local partitions with an information
criterion (AICc by default, with AIC and BIC alternatives) and applies
the recommended classification, showing a per-parameter rationale you can
adjust before fitting. On this dataset it recovers the split above.

Set the **Errors** selector to **Column** so the fit uses the per-point
:math:`\lambda` uncertainties from the batch fits. Run the fit. The
Filon fast path keeps the joint fit to a handful of seconds.

Step 7 — Read the results
-------------------------

.. image:: /_generated/screenshots/ybzn2gao5_results_window.png
   :alt: The global parameter fit results window, eight per-temperature panels with stacked components
   :width: 100%

The results window opens on a grid of eight panels, one per temperature,
each overlaying the fitted :math:`\lambda(B)` total on that group's data
with the component contributions stacked beneath it — the reproduction of
Fig. 3. The level-crossing-resonance peak appears only in the
:math:`T \geq 3.2\;\mathrm{K}` panels, exactly where its amplitude
:math:`f` becomes non-zero; below :math:`J` the curve is pure diffusion
plus Redfield plus background. The quality bar above the global table
reports the joint reduced :math:`\chi^2`, and each panel carries its own
per-group :math:`\chi^2_r` chip.

The global table lists the six shared parameters with their
uncertainties — the reproduction of Table I. For the default seed the
recovered globals sit within uncertainty of the published truth
(values in the paper's units — fields in tesla, not gauss — and varying
slightly from seed to seed):

.. list-table:: Published (Table I) versus recovered globals
   :header-rows: 1
   :widths: 30 25 25 20

   * - Parameter
     - Published
     - Recovered
     - Unit
   * - :math:`A`
     - 63
     - 64.8(13)
     - MHz
   * - :math:`D`
     - 18.4
     - 18.40(2)
     - MHz
   * - :math:`\lambda_{\mathrm{BG}}`
     - 0.067
     - 0.0655(15)
     - :math:`\mu\mathrm{s}^{-1}`
   * - :math:`m`
     - 7
     - 7.05(5)
     - —
   * - :math:`B_0`
     - 2.7
     - 2.698(42)
     - T
   * - :math:`B_{\mathrm{wid}}`
     - 1.3
     - 1.321(17)
     - T
   * - :math:`\chi^2_r`
     - —
     - 0.86
     - —

Every shared parameter is recovered, and the joint reduced
:math:`\chi^2_r \approx 0.86` confirms a clean simultaneous fit of all
eight scans. The tightly-constrained :math:`m = 7.05(5)` is the headline:
a cutoff this far from the Lorentzian :math:`m = 2` is the quantitative
evidence for non-diffusive local dynamics, and it is only pinned this
sharply because the eight scans are fitted jointly with a shared
:math:`m`.

.. image:: /_generated/screenshots/ybzn2gao5_locals_vs_T.png
   :alt: The local parameters D_2D, nu and f plotted against temperature
   :width: 100%

The local-parameter pane plots the per-temperature values against the
group variable — the reproduction of Fig. 4. Selecting
:math:`D_{2\mathrm{D}}` shows the quantum plateau below
:math:`J \approx 3.2\;\mathrm{K}` giving way to a steep classical rise;
:math:`\nu` climbs smoothly with temperature; and :math:`f` is zero below
:math:`J` and switches on above it, tracking the appearance of the
level-crossing peak in the Fig. 3 panels. Each local curve is itself a
trend series and can be fitted with a parametric model in the same panel,
so the analysis recurses naturally (see
:doc:`/reference/parameter_trending`).

Model selection: does the resonance term earn its place?
--------------------------------------------------------

.. image:: /_generated/screenshots/ybzn2gao5_compare.png
   :alt: Comparison of the full model against the model with the level-crossing term removed
   :width: 100%

The four-term model asserts that a genuine level-crossing resonance sits
in the high-temperature scans. To test that assertion, **Duplicate** the
study from the results-window sidebar, **Edit fit…** on the copy to drop
the ``GaussianLCR`` term (leaving
``DiffusionLF_2D + Redfield + Lambda_bg``), **Refit**, and **Compare
with…** the original. The full model wins on AICc: removing the resonance
term worsens the information criterion, so the peak is not an artefact of
over-fitting but a feature the data demand. This is the synthetic echo of
the paper's sharp :math:`m = 7` cutoff evidence — the field-resolved fit
distinguishes a real spectral feature from a smooth background.

Reproduction recipe
-------------------

The whole analysis is reproducible from the following, without the prose:

- **Generate the data:**

  .. code-block:: bash

     python -m asymmetry.examples.ybzn2gao5 --out ./ybzn2gao5_runs

  (160 runs: 8 temperatures × 20 log-spaced fields,
  :math:`10\;\mathrm{G}` to :math:`45\,000\;\mathrm{G}`; default seed.)

- **Grouping rule:** group by temperature into 8 data groups, one
  :math:`\lambda(B)` scan each.
- **Level-1 model:** ``Exponential + Constant``, batch-fit per group.
- **Trend axis:** :math:`\lambda` versus **field**, logarithmic field
  axis, all 8 series selected.
- **Global-fit model string:**

  .. code-block:: text

     DiffusionLF_2D + Redfield + Lambda_bg + GaussianLCR

- **Parameter roles:**

  - **Local:** :math:`D_{2\mathrm{D}}`, :math:`\nu`, :math:`f`
  - **Global:** :math:`A`, :math:`D`, :math:`\lambda_{\mathrm{BG}}`,
    :math:`m`, :math:`B_0`, :math:`B_{\mathrm{wid}}`
  - **Fixed:** :math:`D_\perp = 0`

- **Group variable:** temperature (per-group median).
- **Error mode:** Column (per-point :math:`\lambda` uncertainties).
- **Expected quality:** joint reduced :math:`\chi^2_r \approx 0.85`–1.0,
  every Table I global recovered within uncertainty.

Cross-references
----------------

- :doc:`/reference/parameter_trending` — the trend panel, cross-group
  fitting, group variables, and recursive trending.
- :doc:`superconductor_penetration_depth` — a batch field/temperature
  trend workflow.
- :doc:`lf_decoupling_dynamics` — a longitudinal-field global fit at the
  asymmetry level.
- :doc:`/reference/composite_models` — combining component functions.

.. rubric:: References

.. [1] H. C. H. Wu, F. L. Pratt, B. M. Huddart, D. Chatterjee,
   P. A. Goddard, J. Singleton, D. Prabhakaran, and S. J. Blundell,
   arXiv:2502.00130 (2025).
