Global fitting across fields: Li⁺ ionic motion in a garnet electrolyte
======================================================================

.. admonition:: Example data

   :ref:`Download the Rigi school data <rigi-school-data>` for the
   EMU garnet electrolyte runs.

   Use several applied fields at each temperature, together with reliable field
   and temperature metadata and a common instrument calibration. The LLZ triplets
   below identify the demonstration. Form your own groups of runs at matched
   temperatures, verify the field units, and decide which parameters can
   physically be shared before fitting them together.

Fit three longitudinal-field (LF) measurements at a time to resolve the
fluctuation rate of the local field, then use its temperature dependence to
estimate an activation energy for lithium-ion motion. The measurements come
from EMU at ISIS and probe an aluminium-doped lithium garnet (Al-LLZ), a
candidate solid electrolyte for lithium-ion batteries. Amores *et al.*
use these measurements to study lithium diffusion within grains [Amores2016]_.

The related demonstration, :doc:`lf_decoupling_dynamics`, uses a magnetic
longitudinal-field series and works through the static-versus-dynamic
interpretation in detail. Read that page for the theory of Kubo–Toyabe decoupling;
this page concentrates on *why you need a global fit at all*, and on the practical mechanics of tying parameters across a
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

The Rigi school data include 40 EMU runs collected in April 2015 (experiment
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
     - TF calibration run to estimate the detector balance :math:`\alpha`.
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
      the calibrated draft preview beside its measured coefficient
      α = 0.858376 in the α card.

   The **Grouping** window on TF run ``51315``. Select **Diamagnetic** and click
   **Estimate calibration** to minimise the weighted asymmetry power in the
   calibration window. The α field shows 0.858376 for this run. **Apply**
   stores the coefficient and its source in the grouping profile.

Use the standard EMU forward and backward groups and the same calibrated
profile for all LF runs. The example keeps the file timing and good-bin
settings, with no deadtime or count background correction.

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

Keren's analytic function describes relaxation in a fluctuating nuclear field.
Its parameters include the field-distribution width :math:`\Delta`, the
fluctuation rate :math:`\nu`, and the applied field :math:`B_L`. In this teaching
model, a fitted constant :math:`A_{bg}` accounts for a flat asymmetry offset
and is free to vary between temperature triplets. Amores *et al.* subtracted
a fixed background and multiplied the Keren polarisation by an additional
relaxation factor whose rate was fixed with temperature for each sample
[Amores2016]_. Here the constant offset is the only term added to Keren.

.. dropdown:: The Keren relaxation function

   Keren's analytic generalisation of the Abragam function to a longitudinal
   field gives the muon polarisation as :math:`A(t) = A\,e^{-\Gamma(t)}` with

   .. math::

      \Gamma(t)=\frac{2\Delta^2}{(\omega_0^2+\nu^2)^2}
      \Big[(\omega_0^2+\nu^2)\,\nu t
      +(\omega_0^2-\nu^2)(1-e^{-\nu t}\cos\omega_0 t)
      -2\nu\omega_0\, e^{-\nu t}\sin\omega_0 t\Big],

   where :math:`\omega_0 = \gamma_\mu B_L` is the muon Larmor frequency in the
   applied longitudinal field. The approximation is valid in the fast-fluctuation
   regime (:math:`\Delta/\nu < 1`) or at early times. The model and parameter
   conventions are on the
   :ref:`Keren reference page <fit-keren>`; where fluctuations are slow
   (:math:`\nu \lesssim \Delta`) the full dynamic Kubo–Toyabe of
   :doc:`lf_decoupling_dynamics` is the better model.

Select the three runs at one temperature in the **Data Browser**. Open the
**Batch** tab in the **Fit** dock, assemble ``Keren + Constant``, and set the
**Type** and **Seed** of each parameter in **Parameter Classification**.

.. figure:: /_generated/corpus_screenshots/corpus_llz_global_setup.png
   :width: 100%
   :alt: The Batch fit tab on the 160 K triplet, showing the Keren + Constant
      model and the parameter-classification table with A_1, Δ, ν and A_bg set
      to Global and B_L set to File, seeded with the guide's starting values
      over a 12 µs fit range.

   The **Batch** tab for the 160 K triplet. In **Parameter Classification**,
   **Global** shares :math:`\Delta`, :math:`\nu`, and the amplitudes across the
   three runs. The 7.5 G shown in **Seed** is the mean of the two nonzero
   fields. **File** holds :math:`B_L` at the three fields from the run headers,
   0 G, 5 G, and 10 G, respectively. The **Fit range** is 0–12 µs.

.. list-table::
   :header-rows: 1
   :widths: 16 10 14 16 44

   * - Parameter
     - Seed
     - Type
     - Min / Max
     - Role in the joint fit
   * - ``A_1`` (%)
     - 15
     - **Global**
     - 0 / 40
     - Common sample asymmetry amplitude within the triplet.
   * - :math:`\Delta` (µs⁻¹)
     - 0.3
     - **Global**
     - 0 / 2
     - One field-distribution width for the three fields at this temperature.
   * - :math:`\nu` (µs⁻¹, displayed in MHz)
     - 0.2
     - **Global**
     - 0 / 8
     - One fluctuation rate for the three fields at this temperature. Its
       temperature dependence gives the activation energy.
   * - ``B_L`` (G)
     - —
     - **File**
     - —
     - The measured longitudinal field from the run header, held at 0 G,
       5 G, or 10 G in the corresponding run.
   * - ``A_bg`` (%)
     - 5
     - **Global**
     - −10 / 20
     - Common flat asymmetry offset within the triplet.

Show **Bounds** to enter the **Min** and **Max** values, set **Fit range** to
0–12 µs, and keep **Bunch** at 1 for the fits. The counts become sparse in the
tail, where the asymmetry ratio is noisy. Each batch uses three fields at a
single temperature so that sharing :math:`\nu` still allows it to vary across
the temperature scan.

Step 4 — Run the joint fit
--------------------------

With the triplet selected and the parameters classified, click **Run series**
to fit the three runs simultaneously. Check the fitted curve and residuals for
each field. Repeat at the remaining temperatures in ascending order, using the
fitted values from one triplet as seeds for the next.

.. figure:: /_generated/corpus_screenshots/corpus_llz_global_result.png
   :width: 100%
   :alt: Batch tab reporting the fit of all three runs at 160 K, with the
      Keren plus Constant fit curve overlaid on the zero-field data.

   The joint fit of the three fields at 160 K. The curve shown is for the 0 G
   run. The fitted values shared across the triplet are
   :math:`\Delta=0.357` µs⁻¹, :math:`\nu=0.266` MHz,
   :math:`A_1=14.0` %, and :math:`A_{bg}=4.02` %. The average reduced
   :math:`\chi^2` shown for the three runs is 1.67.

Step 5 — Trend ν(T) and extract the activation energy
-----------------------------------------------------

As lithium ions become mobile, their nuclear moments change the local field at
an implanted muon more frequently. The fitted :math:`\nu` is close to
0.27 MHz at the five lowest setpoints and rises to about 1.09 MHz at 404 K.
The rising part of the rate is the signal used to estimate a hopping barrier.

Each Batch fit records one series of three fields. In the **Parameters** dock
(**View → Show Fit Parameters**), select that series and open **Table** to
see the shared :math:`\nu` and its fitted standard error under **Global fitting
parameters**. Choose **⋯ → Export TSV…** to save the entry.
In the export's header, read the line ``#   nu (MHz) = ... +/- ...``. Record the
EMU setpoint in kelvin, the rate before ``+/-``, and the standard error after it,
once per triplet. The command-line ``fit-global --groups`` command can also
collect the shared parameters along ``--group-order temperature``.

The **Fit Parameters** figure uses a prepared trend series of those thirteen
summaries. The controls below show how the fit is set up on that series.
Use `Replay from the fitted rates`_ to calculate the result from the values
collected in the Batch fits.

Treat the low-temperature plateau as a residual fluctuation rate :math:`c`
in the teaching law :math:`\nu(T) = c + \nu_0 e^{-E_a/(k_B T)}`. The mean of the
five rates at 160, 180, 200, 220, and 240 K is 0.273517 MHz. This rate offset
belongs to the temperature trend. The fitted asymmetry offset :math:`A_{bg}`
belongs to the time-domain fit and has units of percent.

#. In **Fit Parameters**, choose **𝑇 (K)** on the **x** rail and check
   :math:`\nu` on the **y** rail. On the x rail's **ƒ** menu choose
   ``1/x  (reciprocal)``.
#. On the :math:`\nu` card's **ƒ** menu choose **Custom…** and enter
   ``log(x - 0.273517)``. The expression uses the natural logarithm of the
   numerical rate difference in MHz (:math:`\mathrm{\mu s}^{-1}`).
#. In **Table**, clear **Trend** for the five plateau points. Click **Fit…**
   on the :math:`\nu` card and use the ``Linear`` model with **Errors** set to
   **Column** for the eight setpoints from 264 to 404 K, inclusive. Click the
   **χ²ᵣ** chip to read the slope :math:`m` and its uncertainty in **Fit results**.

.. figure:: /_generated/corpus_screenshots/corpus_llz_nu_arrhenius.png
   :width: 100%
   :alt: The Fit Parameters panel plots ln of the plateau-subtracted
      fluctuation rate against reciprocal setpoint temperature, with a linear
      fit to eight points from 264 to 404 K and five plateau points excluded.

   Arrhenius fit in **Fit Parameters**. The ordinate is
   :math:`\ln[(\nu - 0.273517\,\mathrm{MHz})/(1\,\mathrm{MHz})]` and the
   abscissa is :math:`1/T` in K⁻¹. The line uses the eight setpoints at
   264, 284, 304, 324, 344, 364, 384, and 404 K. Three of the five excluded
   plateau points have :math:`\nu > c` and appear with grey outlines.

Checking the slope
~~~~~~~~~~~~~~~~~~

Subtracting :math:`c` leaves the activated contribution. Its natural
logarithm has slope :math:`m=-E_a/k_B` against :math:`1/T`. You can check the
fit directly from the rate and error columns. For each selected point, take
:math:`x_i=1/T_i`, :math:`y_i=\ln[(\nu_i-c)/(1\,\mathrm{MHz})]`, and
:math:`s_i=\sigma_{\nu_i}/(\nu_i-c)`. Use the **Column** weights
:math:`w_i=1/s_i^2` after flooring :math:`s_i` at half the median selected
error. The weighted slope is

.. math::

   m = \frac{\sum_i w_i (x_i-\bar{x}_w)(y_i-\bar{y}_w)}
            {\sum_i w_i (x_i-\bar{x}_w)^2},

where the bars denote weighted means. This gives :math:`m=-2577(96)` K and
:math:`E_a=-k_Bm=0.222(8)` eV, using
:math:`k_B\simeq 8.61733\times10^{-5}\ \mathrm{eV\,K^{-1}}`. The slope error
holds the plateau fixed.

Multiplying :math:`\nu`, :math:`c`, and :math:`\sigma_\nu` by :math:`2\pi`
adds :math:`\ln(2\pi)` to every :math:`y_i` and preserves each :math:`s_i`.
The refit therefore has the same slope and activation energy. Asymmetry's
Keren fluctuation rate enters the time equation as :math:`\nu t`, with
:math:`\nu` in inverse microseconds. The :math:`2\pi` rescaling here checks the
trend arithmetic after the time-domain fit.

Replay from the fitted rates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Save the thirteen shared rate summaries to ``llz-rates.tsv`` with one row per
setpoint and three tab-separated columns. Use ``T_K``, ``nu_MHz``, and
``sigma_nu_MHz`` as the header. Run this calculation on those columns to check
the energy, conditional fit error, and plateau sensitivity.

.. code-block:: python

   import numpy as np

   from asymmetry.core.utils.constants import BOLTZMANN_CONSTANT_MEV_PER_K

   t, rate, error = np.loadtxt("llz-rates.tsv", delimiter="\t", skiprows=1, unpack=True)
   cold = t < 264
   c = float(f"{rate[cold].mean():.6g}")
   dc = np.linalg.norm(error[cold]) / cold.sum()
   k_b = BOLTZMANN_CONSTANT_MEV_PER_K / 1000  # eV/K

   def fit_line(lower, plateau, scale=1.0):
       selected = t >= lower
       x = 1 / t[selected]
       excess = scale * (rate[selected] - plateau)
       if np.any(excess <= 0):
           raise ValueError("Every selected rate must exceed the plateau")
       y = np.log(excess)
       sigma = scale * error[selected] / excess
       weight = 1 / np.maximum(sigma, np.median(sigma) / 2)**2
       x_mean = np.average(x, weights=weight)
       y_mean = np.average(y, weights=weight)
       spread = np.sum(weight * (x - x_mean)**2)
       slope = np.sum(weight * (x - x_mean) * (y - y_mean)) / spread
       return slope, 1 / np.sqrt(spread), y_mean - slope * x_mean

   m, dm, intercept = fit_line(264, c)
   print(f"{-k_b * m:.4f} ± {k_b * dm:.4f} eV (fixed plateau)")
   np.testing.assert_allclose(
       fit_line(264, c, 2 * np.pi), (m, dm, intercept + np.log(2 * np.pi))
   )
   for lower in (264, 284, 304):
       energies = [-k_b * fit_line(lower, offset)[0] for offset in (c - dc, c, c + dc)]
       print(lower, ", ".join(f"{energy:.4f}" for energy in energies))

Sensitivity to the plateau and temperature interval
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Combining the standard errors of the five separate plateau fits in quadrature
and dividing by five gives 0.00285 MHz for the error in their mean, conditional
on the common calibration and the time-domain model. The table refits the line
at the plateau estimate and at that estimate shifted by this amount in both
directions. Each shift is applied to every rate in the branch together.

.. list-table:: Activation energy in eV for each fit interval and plateau offset
   :header-rows: 1

   * - Setpoint interval (K)
     - :math:`c - 0.00285` MHz
     - :math:`c`
     - :math:`c + 0.00285` MHz
   * - 264–404
     - 0.2180
     - 0.2221
     - 0.2262
   * - 284–404
     - 0.2208
     - 0.2240
     - 0.2273
   * - 304–404
     - 0.2221
     - 0.2247
     - 0.2274

For the eight-point branch, the plateau change shifts the energy by about
0.004 eV in either direction. At the central plateau, dropping the two lowest
branch temperatures increases it by 0.003 eV. These are sensitivity checks
separate from the 0.008 eV conditional fit error.

Amores *et al.* report :math:`0.19(1)` eV from the μSR Arrhenius analysis
(Fig. 7) and :math:`0.55(3)` eV from impedance spectroscopy [Amores2016]_.
The μSR rate probes local lithium motion within grains, while the impedance
measurement also samples transport across grain boundaries. The teaching fit
gives a higher μSR energy than the paper across the tested intervals and
plateau shifts. The sensitivity checks leave the origin of that difference
unresolved.

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

- **The Keren approximation at low temperature.** At 160 K,
  :math:`\Delta/\nu` exceeds one, outside Keren’s fast-fluctuation regime.
  All three fields are retained in this teaching fit. If the zero-field
  curve shows systematic discrepancies, compare a fit using the full dynamic
  Kubo–Toyabe function, which includes the slow-limit polarisation tail.

- **Setpoint versus measured temperature.** The trend axis uses the nominal
  setpoints (160–404 K); the instrument-recorded sample temperatures run
  systematically lower (157–391 K). For a higher-precision Arrhenius slope, trend
  against the logged temperatures instead (see :ref:`trend-abscissa-coordinate`).

References
----------

.. [Amores2016] M\. Amores, T. E. Ashton, P. J. Baker, E. J. Cussen, and
   S. A. Corr, J. Mater. Chem. A **4**, 1729 (2016), section 3.2.2 and
   Figs. 4–7. `DOI <https://doi.org/10.1039/C5TA08107F>`__.
   `Author manuscript <https://strathprints.strath.ac.uk/55263/1/Amores_etal_JOMCA_2015_Fast_microwave_assisted_synthesis_of_Li_stuffed_garnets_and_insights_into_Li_diffusion_from_muon.pdf>`__.

- A\. Keren, Phys. Rev. B **50**, 10039 (1994) — the analytic longitudinal-field
  relaxation function used here.
- R\. S. Hayano, Y. J. Uemura, J. Imazato, N. Nishida, T. Yamazaki, and R. Kubo,
  Phys. Rev. B **20**, 850 (1979) — the dynamic Kubo–Toyabe theory of field
  decoupling that Keren's function generalises.
- S\. J. Blundell, R. De Renzi, T. Lancaster, and F. L. Pratt, *Muon
  Spectroscopy: An Introduction* (Oxford University Press, Oxford, 2022), Ch. 5
  — LF decoupling and dynamic relaxation.

Cross-references
----------------

- :doc:`lf_decoupling_dynamics` — the synthetic sibling: the static-versus-
  dynamic Kubo–Toyabe formalism behind LF decoupling, worked on a generic Gaussian-field
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
