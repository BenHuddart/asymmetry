Global fitting across fields: Li⁺ ionic motion in a garnet electrolyte
=====================================================================

This chapter is the flagship worked example of a **global fit** — a single
simultaneous fit of several runs that share some parameters and differ in
others. It takes a real longitudinal-field (LF) decoupling series measured on
EMU at ISIS and, one temperature at a time, fits a triplet of runs recorded at
three applied fields with one shared model, then trends the fitted fluctuation
rate against temperature to recover an activation energy for lithium-ion
motion. The sample is an aluminium-doped lithium garnet,
Li₇La₃Zr₂O₁₂ (LLZ), a candidate solid electrolyte for a lithium-ion battery;
the same runs were published by Amores *et al.*, J. Mater. Chem. A **4**, 1729
(2016), so the analysis here has a paper-grade target to aim at.

Its synthetic sibling, :doc:`lf_decoupling_dynamics`, builds the same story on a
textbook Ag decoupling series and works through the static-versus-dynamic
formalism in detail. Read that page for the theory of Kubo–Toyabe decoupling;
this page is the real-data counterpart and concentrates on *why you need a
global fit at all*, and on the practical mechanics of tying parameters across a
field triplet.

Why a global fit is unavoidable here
-------------------------------------

In a solid electrolyte the muon spin relaxes because it sits in a fluctuating
local field: the nuclear moments of the lithium (and other) ions that diffuse
past it produce a field distribution of width :math:`\Delta` that fluctuates at
a rate :math:`\nu` set by the ionic hop rate. Both quantities are physical
unknowns we want per temperature. The trouble is that a single relaxing curve
constrains them only jointly — a broad, slowly-fluctuating distribution and a
narrower, faster one can produce almost the same early-time decay. Fitting one
field in isolation is therefore degenerate: :math:`\Delta` and :math:`\nu`
trade off against each other along a valley in parameter space, and the fit
reports a tightly-correlated pair rather than two independent numbers.

Applying a longitudinal field breaks the degeneracy. A field along the initial
muon spin direction progressively **decouples** the muon from the nuclear
fields: as :math:`\gamma_\mu B_L` grows the relaxation slows, and *how fast* it
slows with field depends on :math:`\Delta` and :math:`\nu` in different ways. A
run at 0 G, one at 5 G, and one at 10 G thus carry complementary information,
and a single model fitted to all three at once — with :math:`\Delta` and
:math:`\nu` forced to take one shared value across the triplet while the field
:math:`B_L` is held at each run's own set value — pins both parameters. That
simultaneous, parameter-sharing fit is what Asymmetry (and the WiMDA "Multi
Fit" it descends from) calls a global fit.

The runs
--------

The corpus example ships 40 EMU runs collected in April 2015 (experiment
RB1510349): one transverse-field calibration run and thirteen temperatures each
measured at three longitudinal fields.

.. list-table::
   :header-rows: 1
   :widths: 26 18 56

   * - Run(s)
     - Field / mode
     - Role
   * - ``51315``
     - TF 20 G (300 K)
     - Calibration run — its precession amplitude fixes the detector balance
       :math:`\alpha` before any science fit.
   * - ``51341``–``51379``
     - LF 0 / 5 / 10 G
     - 13 temperatures × 3 fields = 39 science runs. Each temperature's triplet
       is the three consecutive runs ``(zf, zf+1, zf+2)`` = (0 G, 5 G, 10 G);
       the zero-field run opens each set (``51341``, ``51344``, … ``51377``).

The temperature setpoints span 160–404 K (measured sample temperatures
157–391 K). All 40 files are HDF4-based ISIS NeXus histograms; the EMU loader
reads each run's applied field and setpoint temperature straight from the file
header, which — as the workflow below relies on — means the per-run field can be
supplied to the fit automatically rather than typed in by hand.

Step 1 — Calibrate α from the TF run
------------------------------------

Every time-domain analysis starts from a balanced asymmetry, so the first job
is to calibrate :math:`\alpha`, the forward/backward detector normalisation, on
the transverse-field run ``51315``. The full mechanics of the grouping profile
and the calibration dialog are covered in :doc:`calibration_grouping_emu`; here
it is a single preparatory step.

.. figure:: /_generated/corpus_screenshots/corpus_llz_calibration.png
   :width: 90%
   :align: center
   :alt: The inline alpha calibration on the Al-LLZ TF 20 G run 51315, showing
      the before (α = 1) and after (fitted α) asymmetry with the fitted value
      α = 0.876 in the α card.

   The inline alpha calibration (the Grouping window's **α (detector
   balance)** card) on the TF 20 G run ``51315``. With the
   **Diamagnetic (TF)** method selected, **Estimate α** finds the :math:`\alpha`
   that makes the transverse-field precession oscillate symmetrically about
   zero; the α card reports
   :math:`\alpha = 0.876` for this silver-free garnet run. The grey
   :math:`\alpha = 1` (before) and blue fitted (after) traces show the
   balancing directly. The precession is clean out to about 20 µs; past ~25 µs
   the forward/backward counts have run down and the raw asymmetry ratio grows
   noisy, which is normal for pulsed EMU data and does not affect the estimate.

With :math:`\alpha` calibrated and the grouping profile applied, every LF run of
the series inherits the same detector balance automatically.

Step 2 — Read the raw field triplet
-----------------------------------

Before fitting anything, load one temperature's triplet and overlay the three
fields. This is where the decoupling signature — and the case for a joint fit —
becomes visible.

.. figure:: /_generated/corpus_screenshots/corpus_llz_lf_triplet.png
   :width: 100%
   :alt: The Al-LLZ 160 K longitudinal-field triplet (runs 51341/51342/51343 at
      0/5/10 G) overlaid over 0–12 µs, showing the zero-field run relaxing
      fastest and the 10 G run least.

   The 160 K triplet — runs ``51341`` (0 G), ``51342`` (5 G) and ``51343``
   (10 G) — loaded into a data group and drawn with **Overlay** enabled. The
   three fields separate clearly: the zero-field trace (blue) relaxes fastest,
   the 5 G trace (orange) more slowly, and the 10 G trace (green) least of all.
   That progressive slowing with field *is* the decoupling, and each field
   samples the :math:`(\Delta, \nu)` pair differently — the raw justification
   for fitting them together. The plot is clipped to the 0–12 µs analysis
   window (see the fit-range note below).

The separation between the three curves is modest — this is *weak* decoupling,
only a few gauss — which is exactly why one field alone cannot resolve
:math:`\Delta` from :math:`\nu` and why the three must be fitted jointly.

Step 3 — The model and the parameter classification
---------------------------------------------------

The model applied to every run in the triplet is the **Keren** relaxation
function plus a flat background, entered as the composite
``Keren + Constant``. Asymmetry displays its assembled form as

.. code-block:: text

   A(t): A_1*exp(-Gamma(t; Delta=Delta, nu=nu, B_L=B_L)) + A_bg

The Keren function is the standard analytic model for LF decoupling by a
fluctuating nuclear-dipolar field: it carries the static field-distribution
width :math:`\Delta`, the fluctuation rate :math:`\nu`, and the applied field
:math:`B_L` directly as parameters, and is accurate in the fast/intermediate
regime that ionic motion occupies. It is the same model Amores *et al.* used on
this dataset. The flat ``Constant`` term is the background the guide asks for —
muons that stop outside the sample and do not relax.

.. dropdown:: The Keren relaxation function

   Keren's analytic generalisation of the Abragam function to a longitudinal
   field gives the muon polarisation as :math:`A(t) = A\,e^{-\Gamma(t)}` with

   .. math::

      \Gamma(t)=\frac{2\Delta^2}{(\omega_0^2+\nu^2)^2}
      \Big[(\omega_0^2+\nu^2)\,\nu t
      +(\omega_0^2-\nu^2)(1-e^{-\nu t}\cos\omega_0 t)
      -2\nu\omega_0\, e^{-\nu t}\sin\omega_0 t\Big],

   where :math:`\omega_0 = \gamma_\mu B_L` is the muon Larmor frequency in the
   applied longitudinal field. It is a strong-collision result valid when
   :math:`\nu \gtrsim \Delta` and reduces to the Abragam function at
   :math:`B_L = 0`. The full derivation and parameter list are on the
   :ref:`Keren reference page <fit-keren>`; where fluctuations are slow
   (:math:`\nu \lesssim \Delta`) the full dynamic Kubo–Toyabe of
   :doc:`lf_decoupling_dynamics` is the better model.

The heart of a global fit is deciding, parameter by parameter, whether a
quantity is *shared* across the runs or *individual* to each. That decision is
made in the **Batch** tab's **Parameter classification** table.

.. figure:: /_generated/corpus_screenshots/corpus_llz_global_setup.png
   :width: 100%
   :alt: The Batch fit tab on the 160 K triplet, showing the Keren + Constant
      model and the parameter-classification table with A_1, Δ, ν and A_bg set
      to Global and B_L set to File, seeded with the guide's starting values
      over a 12 µs fit range.

   The **Batch** tab set up for the joint fit of the 160 K triplet. The
   **Parameter classification** table is the whole story of the fit: each row's
   **Type** decides whether that parameter is shared or per-run. The **Fit
   range** is capped at 12 µs and the guide's seed values are entered in the
   **Seed** column, with all three runs selected so **Run Batch Fit** acts on
   the loaded triplet.

Walk down the classification table row by row — this is the tying picture that
makes the fit a global one:

.. list-table::
   :header-rows: 1
   :widths: 16 12 14 58

   * - Parameter
     - Seed
     - Type
     - Role in the joint fit
   * - ``A_1`` (%)
     - 15
     - **Global**
     - Sample-signal amplitude. The relaxing garnet signal is the same physical
       fraction of the asymmetry at every field, so one shared value serves the
       triplet.
   * - :math:`\Delta` (µs⁻¹)
     - 0.3
     - **Global**
     - Static field-distribution width. This is a property of the sample at this
       temperature, not of the applied field — one value, shared. Sharing it
       across fields is precisely what lifts the :math:`\Delta`–:math:`\nu`
       degeneracy.
   * - :math:`\nu` (MHz)
     - 0.2
     - **Global**
     - Fluctuation (hop) rate — the quantity whose temperature dependence gives
       the activation energy. Also a sample property, hence shared.
   * - ``B_L`` (G)
     - 0
     - **File**
     - Applied longitudinal field. This is the one parameter that *differs*
       between the runs, and it is not fitted at all: **File** fixes it per run
       to the 0 / 5 / 10 G value read from each run's own header. This is why
       the loader reading the field from the file matters.
   * - ``A_bg`` (%)
     - 5
     - **Global**
     - Flat background amplitude. Muons stopping outside the sample give the
       same field-independent background at each field, so it too is shared.

The **Type** column is the general mechanism: a ``Global`` parameter takes one
value across every run in the batch; a ``File`` parameter is fixed per run to
the value stored in that run's metadata; and a per-run free parameter (``Local``
in the general case) would take an independent fitted value per run. Here only
:math:`B_L` varies, and it varies in a known way, so the triplet is described by
just four shared numbers plus three fixed fields — the textbook shape of a
well-posed global fit.

Two practical settings complete the setup. The **Fit range** is limited to
:math:`t \le 12` µs: past about 13 µs the forward/backward counts of pulsed EMU
data have run down and the asymmetry ratio diverges, so restricting the window
both speeds the fit and keeps the noisy tail out of it. The **Seed** values
(:math:`A_1 = 15` %, :math:`A_{bg} = 5` %, :math:`\Delta = 0.3`,
:math:`\nu = 0.2`) are the guide's suggested starting points at 160 K — starting
values, not results. Warm-starting each higher temperature from the previous
fit's converged values ("propagate up in temperature") makes the whole series
converge cleanly.

Step 4 — Run the joint fit
--------------------------

With the triplet selected and the parameters classified, **Run Batch Fit**
performs the simultaneous fit.

.. figure:: /_generated/corpus_screenshots/corpus_llz_global_result.png
   :width: 100%
   :alt: The converged global Keren fit of the 160 K triplet, with the red
      Batch Fit curve overlaid on the zero-field data and the fitted shared
      Δ = 0.358 µs⁻¹, ν = 0.267 MHz reported in the classification table.

   The converged joint fit. The red **Batch Fit** curve traces the zero-field
   run cleanly through the decoupling shoulder, and the classification table now
   carries the fitted *shared* values: :math:`\Delta = 0.358` µs⁻¹ and
   :math:`\nu = 0.267` MHz at 160 K, with :math:`A_1 = 14.1` %. The log reports
   the joint quality, ``average χ²ᵣ = 1.674`` across the three runs — one fit
   describing all three fields at once.

The fitted :math:`\Delta = 0.358` µs⁻¹ and :math:`\nu = 0.267` MHz sit close to
the guide's seeds, as expected for the lowest temperature where the ions are
nearly frozen. Note that the background amplitude fits to a small *negative*
value here (:math:`A_{bg} \approx -3.6` %) rather than the +5 % seed: with a
single flat term standing in for the true baseline the fit is free to place it
below zero, a reminder that the background model is a modelling choice (see
`Assumptions and limitations`_).

Step 5 — Trend ν(T) and extract the activation energy
-----------------------------------------------------

Repeating the joint fit at all thirteen temperatures builds a
:math:`\nu(T)` series — the payoff of the whole workflow. Physically we expect
:math:`\nu` to sit on a low, roughly constant plateau while the lithium ions are
immobile and then rise steeply once thermally-activated hopping switches on. In
this reproduction :math:`\nu` is flat near 0.27 MHz up to about 250 K and then
climbs to roughly 1.1 MHz at the top of the range, matching the paper's
"plateau, then activated rise above ~290 K". (Over the same series
:math:`\Delta` decreases smoothly — the width the muon sees narrows as the ions
mobilise — from 0.358 µs⁻¹ down to about 0.27 µs⁻¹.)

An activation energy comes from an Arrhenius analysis of the rising branch. In
the parameter-trending panel this is done natively by transforming the axes so
that an Arrhenius law becomes a straight line, then fitting a ``Linear`` model
to it.

.. figure:: /_generated/corpus_screenshots/corpus_llz_nu_arrhenius.png
   :width: 100%
   :alt: The ν(T) trend rendered as an Arrhenius plot — log(ν − baseline)
      against 1/T — with a straight Linear fit through the eight activated-branch
      points and five plateau points excluded.

   The :math:`\nu(T)` trend as a native Arrhenius plot. The abscissa is
   transformed to :math:`1/T` (reciprocal) and the ordinate to
   :math:`\log(\nu - 0.274324)` via a **Custom** transform; the section chip
   reads ``1/x · log(y - 0.274324)``. A ``Linear`` model fit (**Model Fit\***)
   runs on the eight activated-branch points (:math:`T \ge 264` K), whose slope
   is :math:`-E_a/k_B`. The five plateau points are excluded from the trend
   (``8/13 members in trend · 5 excluded``): three sit low with large
   propagated error bars and two fall below the baseline and drop out entirely.

The reason the transform subtracts a baseline — rather than using the plain
:math:`\ln` preset — is important, and it is the subtle part of this analysis.
The hop rate does not vanish on the low-temperature plateau; it saturates at a
residual value :math:`c \approx 0.274` MHz. A rate of the form
:math:`\nu(T) = \nu_0\,e^{-E_a/k_BT} + c` is **not** linearised by
:math:`\ln\nu` against :math:`1/T`, because :math:`\ln(\nu_0 e^{-E_a/k_BT}+c)`
is not linear in :math:`1/T`; fitting it returns an activation energy biased low
and, worse, one that depends on where the branch is cut. Subtracting the
plateau first — the ``Custom`` expression ``log(x - 0.274324)`` with the
plateau value read off the low-temperature end — makes :math:`\ln(\nu-c)`
genuinely straight and the slope branch-insensitive. The recipe, and the
general "Arrhenius on a plateau" caution, are documented under
:ref:`trend-axis-transforms` in :doc:`/reference/parameter_trending`.

The slope of the fitted line gives

.. math::

   E_a = 0.222(8)\ \mathrm{eV},

against the paper's µSR value of :math:`0.19(1)` eV from the same data. The two
agree in magnitude and both describe the same physics — a lithium-ion hopping
barrier of about 0.2 eV, seen *locally* by the muon. (For contrast, Amores
*et al.* measured 0.55(3) eV by impedance spectroscopy on the same sample; the
much larger transport barrier is dominated by inter-grain resistance that the
local muon probe does not see — one of the paper's central points.) The residual
~15 % gap between this reproduction and the paper's µSR number is honest and
attributable: the background model is not uniquely pinned, and there is a known
factor-of-:math:`2\pi` ambiguity in whether :math:`\Delta` and :math:`\nu` are
quoted in MHz or in angular µs⁻¹.

.. dropdown:: From the hop rate to a diffusion coefficient

   Amores *et al.* convert the fluctuation rate into a lithium diffusion
   coefficient with a jump-diffusion (Einstein) relation summed over the two
   distinct Li hop paths in the garnet structure (24d → 96h and 96h → 24d):

   .. math::

      D = \sum_i \frac{1}{N_i}\, Z_{v,i}\, s_i^2\, \nu,

   where for hop path :math:`i`, :math:`N_i` is the number of accessible sites,
   :math:`Z_{v,i}` the destination vacancy fraction, and :math:`s_i` the jump
   distance. Using the measured room-temperature :math:`\nu`, the paper reports
   :math:`D = 4.62\times10^{-11}\ \mathrm{cm^2\,s^{-1}}` and a mobile Li-ion
   fraction of 21.7 %. Asymmetry does not compute :math:`D` itself — it stops at
   :math:`\nu(T)` and :math:`E_a` — but the extracted rate is the input this
   relation needs.

Assumptions and limitations
---------------------------

- **The background model is a choice.** The guide asks for "some choice of
  background terms" without prescribing one; a single flat ``Constant`` is the
  simplest, and it can fit slightly negative (as at 160 K above) when it stands
  in for a baseline the data do not tightly constrain. A different background
  (an extra relaxing term, a second Keren component) would shift the fitted
  :math:`\Delta` and :math:`\nu` somewhat, and is part of the residual gap to
  the paper's :math:`E_a`.

- **Δ and ν units carry a 2π ambiguity.** Field-distribution widths and hop
  rates in Keren/dynamic-KT parameterisations are variously quoted in MHz or in
  angular µs⁻¹, which differ by :math:`2\pi`. The seeds are taken as the guide's
  literal values; do not assume a conversion when comparing numbers across
  sources.

- **The Keren caveat at low temperature.** The Keren function is a
  fast-fluctuation result and "does not work for both zero field *and* low
  fluctuation rate", so at the lowest temperatures the zero-field run can be
  excluded, fitting only the 5 / 10 G pair. Here all three fields are kept —
  the joint fits converge with :math:`\chi^2_r \le 1.7` at every temperature —
  but on data where the ZF run pulls the low-temperature fit, dropping it (or
  switching to the full dynamic Kubo–Toyabe, which the guide names as the
  alternative model) is the correct response.

- **The Arrhenius branch and baseline are chosen by eye.** The activated branch
  (:math:`T \ge 264` K) and the plateau baseline (:math:`c = 0.274` MHz) are
  read off the trend, not fitted jointly; the extracted :math:`E_a` is
  reassuringly insensitive to exactly where the branch is cut once the baseline
  is subtracted, but a different baseline would move it. The alternative is to
  skip the linearisation and fit an ``Arrhenius`` model in raw coordinates.

- **Setpoint versus measured temperature.** The trend axis uses the nominal
  setpoints (160–404 K); the instrument-recorded sample temperatures run
  systematically lower (157–391 K). For a higher-precision Arrhenius slope, trend
  against the logged temperatures instead (see :ref:`trend-abscissa-coordinate`).

References
----------

- M. Amores, T. E. Ashton, P. J. Baker, E. J. Cussen, and S. A. Corr, J. Mater.
  Chem. A **4**, 1729 (2016) — the published analysis of this dataset: EMU LF
  0/5/10 G decoupling, a simultaneous Keren fit at each temperature, and the
  Li⁺ activation energy :math:`E_a = 0.19(1)` eV.
- A. Keren, Phys. Rev. B **50**, 10039 (1994) — the analytic longitudinal-field
  relaxation function used here.
- R. S. Hayano, Y. J. Uemura, J. Imazato, N. Nishida, T. Yamazaki, and R. Kubo,
  Phys. Rev. B **20**, 850 (1979) — the dynamic Kubo–Toyabe theory of field
  decoupling that Keren's function generalises.
- S. J. Blundell, R. De Renzi, T. Lancaster, and F. L. Pratt, *Muon
  Spectroscopy: An Introduction* (Oxford University Press, Oxford, 2022), Ch. 5
  — LF decoupling and dynamic relaxation.

Cross-references
----------------

- :doc:`lf_decoupling_dynamics` — the synthetic sibling: the static-versus-
  dynamic Kubo–Toyabe formalism behind LF decoupling, worked on a textbook Ag
  series.
- :doc:`calibration_grouping_emu` — the EMU grouping profile and α calibration
  used in Step 1.
- :ref:`fit-keren` — the Keren fit-function reference page.
- :doc:`/reference/global_fit_wizard` — the Global Fit Wizard for automated
  role selection.
- :doc:`/reference/parameter_trending` — the trending panel, axis transforms,
  and the Arrhenius-on-a-plateau recipe.
- :doc:`/reference/composite_models` — building composite models such as
  ``Keren + Constant``.
