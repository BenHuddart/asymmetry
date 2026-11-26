Temperature scan through a magnetic transition
==============================================

.. admonition:: Example data

   :ref:`Download the Rigi school data <rigi-school-data>` for EuO and
   nickel. The :ref:`molecular antiferromagnet <molecular-antiferromagnet>`
   uses runs from the ISIS DataGateway.

   Use your own zero-field temperature series, with measured sample temperatures
   and the instrument calibration. The EuO, nickel, and molecular-magnet runs below
   identify the demonstrations. Replace their run selections and temperature
   assignments with your own; check each spectrum before choosing a precession
   model or fitting a transition trend.

How does the spontaneous local field change as a magnet approaches its
ordering temperature? In zero field (ZF), the fitted muon precession
frequency tracks that field. Fit the oscillation in each ordered run,
then plot the frequencies against temperature in **Fit Parameters**.
The EuO runs from the Rigi school data are the PSI GPS histograms
``deltat_pta_gps_2923``–``2973``, analysed by Blundell *et al.* in
their study of this localised ferromagnet. Nickel and a molecular
antiferromagnet show how temperature units, usable fit intervals, and
detector balance affect the comparison.

Physical motivation
-------------------

A muon stopped in an ordered magnetic phase sees a static local field
:math:`B_\mu` set by the dipolar and contact (hyperfine) contributions of
the surrounding ions. Its spin precesses at the Larmor frequency

.. math::

   \nu_\mu = \frac{\gamma_\mu}{2\pi}\, B_\mu ,

where :math:`\gamma_\mu / 2\pi = 135.5\;\mathrm{MHz\,T^{-1}}`. As
temperature rises toward :math:`T_C` the sublattice magnetisation falls —
and so does :math:`B_\mu` — until the precession washes out entirely and
the signal reverts to a slow paramagnetic relaxation. The frequency
:math:`\nu(T)` is thus a direct measure of the magnetic order parameter.
Near :math:`T_C` it follows a power law

.. math::

   \nu(T) = \nu_0 \left(1 - \frac{T}{T_C}\right)^{\beta},

with the critical exponent :math:`\beta` depending on the universality
class. A critical-exponent measurement requires frequencies sufficiently
close to the transition on a well-calibrated temperature axis.

The data
--------

The zero-field temperature scan comprises runs 2923–2960. The figure
fits one spectrum at each selected temperature, in the order 2960,
2925, and 2928–2943. It leaves out 2923 and 2924 because of their short
counting times and unstable thermometer readings, and 2926 and 2927
because they repeat the 10 K point. Runs 2944–2959 sample the transition
or the high-temperature phase outside this selected oscillatory branch.
Runs 2961–2973 use a 60 G transverse field and form a separate scan.

Asymmetry reads the PSI ``.bin`` histograms natively. Open the series and
enable **Options → Use temperature from log** to use the sample sensor
for **T (K)**. This recorded sensor agrees with the school logbook for
the selected runs. For run 2960 it reads 1.604 K, with a header setpoint of
1.5 K. The captions round the logged temperature to 1.6 K.

.. _euo-saved-views:

Step 1 — Load and inspect
-------------------------

.. figure:: /_generated/corpus_screenshots/corpus_euo_load_browse.png
   :alt: Saved view 1 of EuO run 2960 showing the early zero-field oscillation at Bunch 4
   :width: 100%

   **Saved views:** **1** shows the first 0.30 μs of run 2960 at 5 ns per
   point. **T (K)** shows the logged sample temperature, 1.604 K (1.6 K
   rounded). The header setpoint is 1.5 K.

Select zero-field run 2960 (``deltat_pta_gps_2960.bin``) in **F-B asymmetry**.
In **Grouping**, keep the file's default Back/Forw detector pair, set
**Binning** to **Fixed** with factor **1**, and use **α (detector balance)**
of 1. In **Deadtime**, select **Off**. In **Background**, choose **Range
average**, open **Edit intervals…**, and enter ``10,85; 10,85``. These
inclusive, zero-based acquisition bins lie before time zero. Click
**Apply** and use the same reduction for both views. Absolute asymmetry
amplitudes require detector balance calibration.

#. Under **Saved views:** click **1** and set **Bunch:** to **4**.
   Start from the full trace. If the x-axis limits are held, click
   **Auto X** first. Beside **X:** enter **0.30** in the right field
   and press **Enter**, then enter **0** in the left field and press
   **Enter**. The native 1.25 ns bins give 5 ns between points, resolving
   the roughly 33 ns period of the 30 MHz spontaneous precession.
#. Click **2** and set **Bunch:** to **4**. Beside **X:** enter **6**
   in the right field and press **Enter**, then enter **1** in the
   left field and press **Enter**. The late trace shows much weaker
   oscillation as the muon ensemble loses phase coherence. Now set
   **Bunch:** to **32**. The 40 ns bins show the slowly varying
   asymmetry and average over more than a precession period.
#. Click **1** again to recover the early oscillation and its 5 ns
   spacing. The x-axis limits frame the plot, while the fit window
   selects the times used for a fit. :ref:`Display bunching
   <display-bunching>` changes the observations available to the next
   time-domain F–B fit. Completed fits retain their original inputs.

For the fitted scan below, use **Bunch:** **1** and set **Background**
back to **None**. Sort the runs by temperature and inspect several traces
as the transition approaches. Their precession slows and becomes more
heavily damped, then disappears above :math:`T_C`.

Step 2 — Group consistently across the scan
-------------------------------------------

The zero-field spontaneous precession is carried by the transverse
Forward/Backward detector pair, and Asymmetry's loader picks that pair by
default for these GPS files, so the **F-B asymmetry** representation shows
the oscillation with no manual grouping. The forward–backward asymmetry
is formed as

.. math::

   G_z(t) = \frac{N_f(t) - \alpha\, N_b(t)}{N_f(t) + \alpha\, N_b(t)} ,

with :math:`\alpha` the detector-balance constant. The frequency example
uses :math:`\alpha = 1` and includes an additive constant to describe the
detector offset. Absolute amplitudes require detector balance calibration.
Use the same forward and backward groups, calibration choices, and timing
across all runs in the trend.

Step 3 — Fit an ordered run
---------------------------

.. figure:: /_generated/corpus_screenshots/corpus_euo_zf_fit.png
   :alt: Converged zero-field oscillation fit on the EuO 1.6 K run
   :width: 100%

   Run 2960 (1.6 K) fitted over 0–6 μs and displayed over 0–0.45 μs to
   resolve the cycles. ``Oscillatory * Exponential + Constant`` gives
   a frequency of about 30.2 MHz and a damping rate of about 3 μs⁻¹.

Open **Analysis → Fit** and select **Single**. For each ordered run,
use a single damped cosine on a constant background,

.. math::

   A(t) = A_1 \cos(2\pi\, f\, t + \varphi)\, e^{-\lambda t} + A_{bg},

entered as the composite ``Oscillatory * Exponential + Constant``. The
frequency at base temperature is about 30.2 MHz, corresponding to a
local field of about 0.22 T.

Seed the frequency near the visible precession line. Fit over 0–6 μs
in ascending temperature order, using each fitted frequency to seed the
next run. The trend here uses 30 MHz initially and starts each subsequent
run 10% below the previous fitted value. Check that each fit resolves a
finite-amplitude oscillation before adding it to the series.

The frequency-domain view gives an independent check that only *one*
precession frequency is present:

.. figure:: /_generated/corpus_screenshots/corpus_euo_fft.png
   :alt: Fourier spectrum of the EuO 1.6 K run showing a single precession line near 30 MHz
   :width: 100%

   The **FFT** of the base-temperature run over the Forward/Back pair,
   with a Lorentzian **Apodisation** matched to the signal's short
   coherence time. A single precession line stands clear at
   :math:`\nu \approx 30\;\mathrm{MHz}`, consistent with the single
   resolved line in Blundell *et al.* Fig. 1(c). The view is framed to
   20–42 MHz.

To see the precession slow before fitting a trend, select the spectra
across the transition and enable **Overlay** and **Waterfall** in the
time plot.

.. figure:: /_generated/corpus_screenshots/corpus_euo_waterfall.png
   :alt: Waterfall of EuO zero-field spectra from 1.6 K to 68.69 K
   :width: 100%

   A waterfall of six zero-field spectra from 1.6 K (bottom) to 68.69 K
   (top). The precession visibly slows as :math:`\nu(T)` falls toward
   :math:`T_C`: the base-temperature trace fits many cycles into the first
   0.6 μs, while the hottest trace barely completes one before it damps
   away. This is the order-parameter collapse read straight off the raw
   time-domain data, before any trend fit.

Step 4 — Trend the order parameter
----------------------------------

Record a fit series with the 18 selected runs. Open
**View → Show Fit Parameters**, select that series, set **x** to **𝑇 (K)**, and
choose ``f (MHz)`` for the plotted parameter. The temperature comes
from the logged sample sensor selected in Step 1.

.. figure:: /_generated/corpus_screenshots/corpus_euo_nu_t_trend.png
   :alt: EuO precession frequency versus logged sample temperature with a fitted curve
   :width: 100%

   The fitted zero-field frequencies fall from 30.19 MHz at 1.604 K to
   5.53 MHz at 68.691 K. Error bars show the symmetric uncertainties of
   the frequency fits.

Step 5 — Fit the order parameter to a power law
-----------------------------------------------

Click **Fit…** on the ``f (MHz)`` card and select ``OrderParameter``.
The function used by the curve is

.. math::

   \nu(T) = \nu_0 \left[1 - (T/T_C)^{\alpha}\right]^{\beta},
   \quad T < T_C ,

with frequency zero at and above :math:`T_C`. Use the whole selected
range, 1.604–68.691 K, and select **Column (propagated errors)**.
Use seeds of 30 MHz, 70 K, 1.5, and 0.4 for :math:`\nu_0`, :math:`T_C`,
:math:`\alpha`, and :math:`\beta`, respectively. The fit bounds keep
:math:`\nu_0 \geq 0`, :math:`\alpha, \beta \geq 0.01`, and :math:`T_C` at or above the
hottest selected temperature. The measured frequency errors span
0.040–0.157 MHz. **Column (propagated errors)** weights each point by
the inverse square of its supplied frequency error. The fit treats the
measured temperatures as fixed coordinates.

The curve gives :math:`\nu_0 = 30.5\;\mathrm{MHz}`,
:math:`T_C = 70.5\;\mathrm{K}`, :math:`\alpha = 1.63`, and
:math:`\beta = 0.48` at the displayed precision. The weighted chi-square
is about 190 for 14 degrees of freedom. Inspect the residuals before
interpreting the curve. At this precision the model misses the observed
frequencies by several measured errors at some temperatures, so 70.5 K
is a model-dependent extrapolation from the ordered branch.

.. dropdown:: Reproduce the trend from the exported series

   With only the EuO series selected in **Fit Parameters**, choose
   **⋯ → Export TSV…**. Save the file as ``euo.tsv``. The
   :download:`EuO example script <../../examples/euo_order_parameter.py>`
   reads the run IDs, logged temperatures, fitted frequencies, and their
   measured errors from that export. It selects the 18 walkthrough runs
   and prints each observation, the range, and the fitted parameters.

   .. code-block:: console

      python euo_order_parameter.py euo.tsv

Interpretation
--------------

The low-temperature frequency and its fall toward the transition follow
the same qualitative trend as the pure EuO data in Fig. 1(d) of
Blundell *et al.* The paper uses the same full-range function in
Fig. 1(d) and (e), with approximate shape exponents :math:`\alpha = 1.5`
and :math:`\beta = 0.4`. Its separate fit to the pure EuO frequencies
in the critical regime gives :math:`\beta = 0.32(1)` and
:math:`T_C = 69.05(1)\;\mathrm{K}` (Fig. 3(a), log–log fit at small
:math:`1 - T/T_C`). The authors note that the exponent is sensitive to
their choice of :math:`T_C` and the stability of the sample temperature.

At base temperature the resolved line gives a local field of about
0.22 T. Blundell *et al.* use the measured frequency together with a
dipolar-field calculation to infer the muon site and discuss the
hyperfine and Lorentz contributions to that field.

Variants
--------

The same load → group → per-run fit → trend workflow carries over to
other magnets. The nickel and molecular antiferromagnet examples span a
range of precession frequencies.

Ferromagnetic nickel — a metallic ferromagnet
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Where does the zero-field precession frequency in nickel fall towards
zero? Below the Curie temperature the muon precesses in nickel's
spontaneous internal field. The ISIS pulse limits the frequencies the time spectrum can
resolve, so the trend uses only the interval where the oscillation is
visible.

.. figure:: /_generated/corpus_screenshots/corpus_ni_zf_precession_fit.png
   :alt: Spontaneous zero-field precession in ferromagnetic nickel at 618 K
   :width: 100%

   EMU run 124232 at 618.15 K (controller label ``T=345``). The
   ``Oscillatory * Exponential + Constant`` fit over 0–4.5 μs gives
   6.13 MHz, corresponding to a local field of about 0.045 T.

The NeXus temperature unit says kelvin, and the run titles repeat the
controller values, such as ``nickel_T=345.0_F=0``. The school run log
puts the controller readings around 358 in the transition region.
Reading them in Celsius gives 618.15 K at 345 °C and 631.15 K at 358 °C,
consistent with Foy *et al.* finding the transition near 630 K.
A 6.13 MHz line at 618 K is also consistent with the softening of the
precession frequency in their Fig. 2. To use the converted temperatures in
**Fit Parameters**, select **𝑇 (K)** in the **x** picker. On that rail,
choose **ƒ → Custom…** and enter ``x + 273.15``. Apply the transform
before fitting so the points and the fit limits both use kelvin.

In **Fit → Single**, use the model from the caption and set **Fit range**
to 0–4.5 μs. Fit each run with two starting frequencies. Start one fit
at the strongest **FFT** magnitude between 2 and 10 MHz, and the other
from the chain in ascending temperature order. The chain starts at
9.6 MHz and then uses 92% of the preceding fitted frequency. Keep
converged fits with an oscillation amplitude greater than three times
its symmetric error and a finite, positive frequency error. Where both
starts give usable fits, keep the one with the lower chi-square.

.. figure:: /_generated/corpus_screenshots/corpus_ni_nu_t_order_parameter.png
   :alt: Nickel precession frequency versus converted sample temperature
   :width: 100%

   The resolved zero-field runs 124227–124236 cover 593.15–629.15 K.
   The frequency falls from 9.42 to 2.80 MHz. The fitted
   ``OrderParameter`` curve fixes :math:`\alpha = 1` to give a single
   power of :math:`T_C - T`.

Select **Column (propagated errors)** for the 593.15–629.15 K fit.
It gives :math:`T_C \approx 631` K and an effective exponent of about 0.39.
The reduced chi-square is about 172. Raise the lower fit limit to
623.15 K to check its range sensitivity. The four remaining runs give
about 0.35 and leave one residual degree of freedom. Even this narrower interval has a reduced chi-square around
15 with the supplied frequency errors. The range sensitivity is much
larger than the conditional errors on the exponents. These fits locate
the transition near 631 K and show why the scan does not discriminate
between the Heisenberg and Ising exponents or establish a universality
class.

.. _molecular-antiferromagnet:

A molecular antiferromagnet — the low-frequency contrast
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At what temperature does the molecular antiferromagnet lose resolved
precession? Inspect the calibrated zero-field traces together and compare
fits with and without an oscillating term. The base-temperature line near
1.55 MHz has a much longer oscillation period than the 30 MHz line in EuO.

.. dropdown:: Download the ISIS runs

   Select a run to open its file in the ISIS DataGateway.
   If you use the automated screenshot capture, rename the downloaded
   files from ``.NXS`` to ``.nxs`` so the scripts can find them.

   .. list-table::
      :header-rows: 1
      :class: isis-run-table

      * - MuSR run
        - Temperature
        - Field and purpose
      * - `17094 <https://data.isis.stfc.ac.uk/browse/instrument/9/facilityCycle/48/investigation/24022691/dataset/24022190/datafile/24642635>`__
        - 1.2 K
        - Zero field
      * - `17095 <https://data.isis.stfc.ac.uk/browse/instrument/9/facilityCycle/48/investigation/24022715/dataset/24022215/datafile/24643149>`__
        - 2 K
        - Zero field
      * - `17096 <https://data.isis.stfc.ac.uk/browse/instrument/9/facilityCycle/48/investigation/24022733/dataset/24022233/datafile/24643562>`__
        - 3 K
        - Zero field
      * - `17097 <https://data.isis.stfc.ac.uk/browse/instrument/9/facilityCycle/48/investigation/24022744/dataset/24022244/datafile/24643935>`__
        - 4 K
        - Zero field
      * - `17098 <https://data.isis.stfc.ac.uk/browse/instrument/9/facilityCycle/48/investigation/24022792/dataset/24022292/datafile/24644645>`__
        - 5 K
        - Zero field
      * - `17099 <https://data.isis.stfc.ac.uk/browse/instrument/9/facilityCycle/48/investigation/24022817/dataset/24022317/datafile/24645166>`__
        - 6 K
        - Zero field
      * - `17100 <https://data.isis.stfc.ac.uk/browse/instrument/9/facilityCycle/48/investigation/24022831/dataset/24022331/datafile/24645534>`__
        - 7 K
        - Zero field
      * - `17101 <https://data.isis.stfc.ac.uk/browse/instrument/9/facilityCycle/48/investigation/24022835/dataset/24022335/datafile/24645677>`__
        - 8 K
        - Zero field
      * - `17102 <https://data.isis.stfc.ac.uk/browse/instrument/9/facilityCycle/48/investigation/24022843/dataset/24022343/datafile/24645850>`__
        - 9 K
        - Zero field
      * - `17103 <https://data.isis.stfc.ac.uk/browse/instrument/9/facilityCycle/48/investigation/24022846/dataset/24022346/datafile/24645895>`__
        - 10 K
        - Zero field
      * - `17104 <https://data.isis.stfc.ac.uk/browse/instrument/9/facilityCycle/48/investigation/24022852/dataset/24022352/datafile/24645999>`__
        - 10 K
        - 20 G transverse field (calibration)

Select run 17104 in **Grouping** and use **Diamagnetic** in the
:doc:`balance calibration panel </reference/detector_grouping>` to estimate
:math:`\alpha = 1.252` from its 20 G transverse-field counts. Apply that
balance to the zero-field runs with the same on-file forward/backward
groups and deadtime correction off. This quick estimate supplies a
central value for :math:`\alpha`. The asymmetry errors propagate counting
errors at that fixed balance.

The NeXus header gives ``sample/magnetic_field_state`` as ``TF`` for
runs 17094, 17099, 17100, and 17104. The recorded
``sample/magnetic_field`` is 0 G in 17094, 17099, and 17100 and 20 G
in 17104, consistent with the school run assignments above.

.. figure:: /_generated/corpus_screenshots/corpus_molafm_zf_fit.png
   :alt: Slow zero-field precession in a molecular antiferromagnet at 1.2 K
   :width: 100%

   The calibrated 1.2 K zero-field run 17094 with a weighted
   ``Oscillatory * Exponential + Constant`` fit over 0.11–8 μs. The
   fitted 1.55 MHz precession has a period of about 0.64 μs. The
   **Transverse-field run** warning follows the file's ``TF`` tag.
   The field recorded for 17094 is 0 G, and the forward/backward
   grouping uses the balance measured on 17104.

At 1.2 K the reduced chi-square is about 21. The normalised residuals
:math:`(A_{\rm data} - A_{\rm fit}) / \sigma_A` have a lag-one
correlation of about 0.91, and the discrepancies are strongest at early
times. Refitting over 0.27–10 μs gives about 1.56 MHz and reduces the
chi-square per degree of freedom to about 15. The 0.003 MHz shift is
larger than the approximately 0.0014 MHz symmetric frequency error from
the first fit. Eight-bin counts-first bunching over 0.11–8 μs leaves
the structured residuals and increases the reduced chi-square to about
131. The frequency scale is stable across these checks, while the
quoted counting-statistical errors give an incomplete description of
the uncertainty for this model.

.. figure:: /_generated/corpus_screenshots/corpus_molafm_zf_overlay.png
   :alt: Calibrated zero-field asymmetry of the 6 K and 7 K molecular-magnet runs
   :width: 100%

   Runs 17099 (6 K) and 17100 (7 K) with the same balance and eight native
   16 ns bins merged at the count level for display. Both headers have
   a ``TF`` tag, and both applied fields are 0 G. The damped oscillation
   at 6 K gives way to a smooth relaxation at 7 K.

In **Fit → Single**, compare ``Oscillatory * Exponential + Constant`` and
``Exponential + Constant`` on the same native bins with **Fit range** set
to 0.11–8 μs. Read their χ² values from the **χ²ᵣ** indicator.
At 6 K the precession fit gives about 0.79 MHz, with weighted chi-square
about 3,566 compared with 7,247 for the relaxation curve. At 7 K
the forced oscillatory fit lands at its 0.3 MHz lower bound and gives
about 78,526 compared with 5,639 for relaxation. The spectra and the
change in relative fit quality indicate a loss of resolved precession
between these runs, bracketing the ordering temperature at 6–7 K.

.. figure:: /_generated/corpus_screenshots/corpus_molafm_nu_t.png
   :alt: Molecular antiferromagnet fitted frequencies versus temperature
   :width: 100%

   The calibrated zero-field frequencies of runs 17094–17099 fall from
   about 1.55 MHz at 1.2 K to 0.79 MHz at 6 K. The ``OrderParameter``
   curve, with :math:`\alpha = 1` and the fitted frequency errors as
   weights, extrapolates to about 6.3 K with a reduced chi-square of
   about 181. The extrapolated temperature depends on the chosen model.

Common pitfalls
---------------

- **One composite model across all temperatures.** The change of regime
  at :math:`T_C` means a single model — always oscillatory, say — will
  silently underweight the paramagnetic runs. Use an oscillatory model
  below :math:`T_C` and a plain relaxation above it.

- **A cold frequency seed.** The single-frequency fit has a spurious
  low-amplitude minimum. Seed the frequency near the expected value and
  warm-start it downward through ascending temperature, so no run collapses
  into the wrong minimum.

- **Reading a critical exponent from the whole curve.** Use data in the
  critical regime, check how the result changes with the chosen window
  and :math:`T_C`, and report the uncertainty of the temperature scale.

- **Trusting a temperature axis blindly.** As the nickel example shows,
  file metadata can be in the wrong units. Confirm what the temperature
  means before quoting a :math:`T_C`.

References
----------

.. rubric:: References

- S\. J. Blundell, T. Lancaster, F. L. Pratt, P. J. Baker, W. Hayes, J.-P.
  Ansermet, and A. Comment, Phys. Rev. B **81**, 092407 (2010).
  `DOI <https://doi.org/10.1103/PhysRevB.81.092407>`__. The EuO
  dataset analysed here; the order parameter is Fig. 1(d) and the
  critical-regime fit (:math:`\beta = 0.32(1)`, :math:`T_C = 69.05(1)` K)
  is Fig. 3.
- M\. L. G. Foy, N. Heiman, W. J. Kossler, and C. E. Stronach, Phys. Rev.
  Lett. **30**, 1064 (1973). The nickel spontaneous-precession
  measurement (:math:`T_C = 630` K, saturation internal field).
- S\. J. Blundell, R. De Renzi, T. Lancaster, and F. L. Pratt, *Muon
  Spectroscopy: An Introduction* (Oxford University Press, Oxford, 2022),
  Ch. 6.1–6.2 (magnetism, Landau theory, and critical exponents).
- A\. Amato and E. Morenzoni, *Introduction to Muon Spin Spectroscopy:
  Applications to Solid State and Material Sciences*, Lecture Notes in
  Physics Vol. 961 (Springer, Cham, 2024), Ch. 5 (μSR in ordered magnets,
  antiferromagnets, and unconventional order parameters).
- T\. Lancaster *et al.*, Phys. Rev. B **75**, 094421 (2007). A real-data
  example on a molecular magnet with several precession frequencies near
  :math:`T_C`.

Cross-references
----------------

- :doc:`/reference/loading_data` — load formats, including PSI ``.bin``.
- :doc:`/reference/detector_grouping` — group definitions and α calibration.
- :doc:`/reference/fit_wizard` — the model-recommendation tool.
- :doc:`/reference/parameter_trending` — the trend panel.
- :doc:`/reference/composite_models` — combining oscillatory and
  relaxation envelopes.
