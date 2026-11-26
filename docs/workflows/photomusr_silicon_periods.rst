Photo-μSR in silicon: carrier recombination from period-mode runs
=================================================================

.. admonition:: Example data

   :ref:`Download the Rigi school data <rigi-school-data>` for the
   two-period HiFi silicon runs. The school guide supplies the injected
   densities and laser delays. For another experiment, use its period labels,
   timing log and density calibration.

In photoexcited μSR, a laser pulse injects electron–hole pairs into silicon.
These excess carriers interact with the bound electron in muonium and relax
the muon spin. At fixed field and temperature, the illuminated relaxation
rate :math:`\lambda` can be calibrated against the excess carrier density
:math:`\Delta n`. Repeating the measurement at different laser-to-muon delays
then tracks carrier recombination.

The school runs
---------------

Runs 103277–103299 were taken on HiFi under ISIS proposal RB1520457.
The recombination data are from an intrinsic silicon wafer at 291 K in a
longitudinal field of 100 G (10 mT). Yokoyama *et al.* report this experiment
in `Phys. Rev. Lett. 119, 226601 <https://doi.org/10.1103/PhysRevLett.119.226601>`__.

- Runs 103277–103286 form the injection calibration. The school run table
  gives :math:`\Delta n` from :math:`8.9 \times 10^{13}` to
  :math:`9.3 \times 10^{12}\;\mathrm{cm^{-3}}` at a fixed, near-coincident
  laser and muon arrival.
- Runs 103287–103298 form the delay scan. At fixed injection, the muons probe
  the remaining carriers :math:`\Delta T = 0.1` to 70 µs after the laser pulse.
- Run 103299 is a transverse-field detector-balance run at 20 G. The figures
  use the file grouping, timing and good bins with detector balance
  :math:`\alpha_{\mathrm{det}} = 1`. Use the transverse-field calibration
  workflow in :doc:`calibration_grouping_emu` to estimate detector balance
  for your own analysis. Its uncertainty is outside the fit errors reported here.

The NeXus field-state tag for run 103277 reads ``TF``, so Asymmetry displays
a transverse-grouping advisory. The school guide and Yokoyama *et al.* give
the 100 G measurements as longitudinal. Keep the file's forward–backward
grouping for these fits.

The calibration densities are supplied by the school guide. In the paper they
come from an optical estimate based on absorption of the 1064 nm light in the
wafer and the carrier density at its centre. Laser pulse energy controls the
injection. The school table gives densities in :math:`\mathrm{cm^{-3}}`, with
no per-run density or timing error estimates.

Select the laser periods
------------------------

The 25 Hz laser illuminates the sample on alternating muon pulses. The HiFi
acquisition stores the illuminated and dark histograms as two periods in each
NeXus file.

Open run 103277 and choose **Grouping → RG Mode**. Select **Green** for the
laser-OFF period and apply the grouping. Select **Red** in the same row and
apply it to inspect the laser-ON period. Check the period assignment against
the experiment log. With these data, period 1 (**Red**) shows the stronger
early-time relaxation and period 2 (**Green**) gives the dark spectrum.
The two fits below use the individually reduced period spectra.

.. figure:: /_generated/corpus_screenshots/corpus_si_grouping_rg.png
   :alt: The Grouping dialog for HiFi run 103277 showing Red, Green, G minus R
         and G plus R in the RG Mode row.
   :width: 100%

   **Grouping** on run 103277. The **RG Mode** row selects which period
   enters the reduction. **G minus R** subtracts the independently reduced
   asymmetries. **G plus R** pools their counts before computing asymmetry.
   See :ref:`selecting-periods` for the reduction conventions.

To inspect both periods in a script, use :func:`~asymmetry.core.io.periods.select_period`.
Both returned datasets retain the file timing and the corresponding period's
good-frame normaliser and dead-time information.

.. code-block:: python

   from asymmetry.core.io import load, select_period

   run = load("HIFI00103277.nxs")
   light_on = select_period(run, "red")
   light_off = select_period(run, "green")

The following figure uses two prepared Data Browser entries, one for each of
those period datasets. With both entries selected, enable **Overlay** in the
Time plot toolbar. The **RG Mode** control chooses the current reduction of
the original file entry.

.. figure:: /_generated/corpus_screenshots/corpus_si_on_off_overlay.png
   :alt: Time-domain asymmetry of run 103277 with the two prepared period
         entries overlaid. The laser-ON trace relaxes more rapidly.
   :width: 100%

   In **Overlay**, the dark trace of run 103277 stays nearly flat, while
   the illuminated trace drops rapidly. The measured observable for this
   walkthrough is the illuminated relaxation rate.

Fit each run’s illuminated relaxation
--------------------------------------

In **Fit**, choose **Exponential** for the **Green** reduction. Set the time
fit range to 0–1 µs, with **Bunch** at 1, and fit
:math:`A(t)=A_1 e^{-\lambda t}` with both parameters free. Record the fitted
amplitude :math:`A_1`. Switch the same run to **Red**, keep the same fit range
and model, and set and fix :math:`A_1` at its dark value. Fit
:math:`\lambda` and record the rate and its symmetric standard error.
Repeat the paired fits for each calibration and delay run.

This uses each dark period to estimate its own run's initial asymmetry and
assumes that illumination leaves that amplitude unchanged. The rate errors
come from the illuminated asymmetry fits, conditional on the fixed dark
amplitudes. The dark amplitude uncertainty and any common error in detector
balance are outside those errors.

.. figure:: /_generated/corpus_screenshots/corpus_si_lambda_fit.png
   :alt: A single-exponential fit of the illuminated period in run 103277
         over the first microsecond, with its amplitude fixed.
   :width: 100%

   Run 103277 gives a dark amplitude of about 15.54% and an illuminated rate
   of :math:`1.270 \pm 0.021\;\mathrm{\mu s^{-1}}`. The file's good bins in
   the nominal 0–1 µs window span approximately 0.108–0.988 µs. The
   illuminated exponential has :math:`\chi^2/\nu \approx 2.68`; the formal
   rate error does not cover departures from the exponential.

The paper also fits the first microsecond and excludes the first 100 ns
because of the muon pulse. A single rate in that window approximates the
carrier density as constant. A lifetime of about 10 µs implies about a 9%
density drop over 1 µs, so the fit yields an effective early-time rate.
The rate can also approach the dark relaxation when the carrier density is low.

Fit and invert the rate calibration
------------------------------------

Use the school injection densities to pair each calibration rate with its
density. The prepared **Fit Parameters** series below uses
:math:`x=\Delta n/\Delta n_0`, where
:math:`\Delta n_0=8.9 \times 10^{13}\;\mathrm{cm^{-3}}` is a fixed
reference density. Fit **PowerLaw** :math:`\lambda=a x^n+c`, with
:math:`c=0` fixed as in Yokoyama *et al.*, Fig. 1(d). The fit uses rates in
:math:`\mathrm{\mu s^{-1}}` and their standard errors on the original
scale. The log axes are for display.

.. figure:: /_generated/corpus_screenshots/corpus_si_lambda_vs_dn.png
   :alt: Illuminated relaxation rate versus normalised injected density on
         log axes, with the weighted zero-offset power law and rate errors.
   :width: 100%

   The calibration gives :math:`a=1.316\;\mathrm{\mu s^{-1}}` and
   :math:`n=0.674`. The paper reports :math:`\beta=1.46(4)\;\mathrm{\mu s^{-1}}`
   and :math:`\alpha=0.68(4)` in its notation. The vertical bars come from
   the fixed-amplitude rate fits. The injected-density errors are unavailable.
   The reduced :math:`\chi^2` is 5.98, indicating scatter beyond these rate
   errors under the power law.

Use the fitted :math:`a`, :math:`n` and :math:`c` to invert each delay rate.
Propagation of its rate standard error :math:`\sigma_\lambda`, with the
calibration held fixed, gives

.. math::

   \Delta n = \Delta n_0 \left(\frac{\lambda-c}{a}\right)^{1/n},
   \qquad
   \sigma_{\Delta n\mid a,n,c}
     = \frac{\Delta n}{n(\lambda-c)}\,\sigma_\lambda.

This inversion requires :math:`\lambda > c`. A positive inferred density can
still fall outside the supported calibration range. Long delays bring the
illuminated rate close to the dark rate, where extrapolating the zero-offset
curve also maps dark relaxation to a positive density.

Fit the carrier lifetime
------------------------

Fit **ExponentialDecay** with :math:`c=0` fixed to the inferred densities
over 0.1–20 µs. The model
:math:`\Delta n(\Delta T)=\Delta n(0)e^{-\Delta T/\tau}` gives the carrier
lifetime :math:`\tau`. At 0.1 µs the inferred density is about 0.9% above
the highest calibration density. This short-delay point is retained in the
main fit. All points from 25 µs onward fall below the calibration range and
are displayed as extrapolations outside the fit. At 70 µs the illuminated rate
is about 0.076 µs⁻¹ and the dark rate is about 0.090 µs⁻¹.

.. figure:: /_generated/corpus_screenshots/corpus_si_tau_decay.png
   :alt: Inferred excess carrier density in units of 10¹³ per cubic centimetre
         versus laser delay in microseconds. The exponential curve fits the
         first six points through 20 microseconds.
   :width: 100%

   The prepared **Fit Parameters** series shows all twelve delay points.
   The bars propagate each illuminated rate error with the calibration
   held fixed. The curve through 20 µs gives :math:`\tau=10.42\;\mathrm{\mu s}`
   and :math:`\Delta n(0)=9.85 \times 10^{13}\;\mathrm{cm^{-3}}`.
   The red fit curve stops at 20 µs, before the labelled late-delay
   extrapolations.

The rate-only lifetime standard error is 0.21 µs. **Fit Parameters** floors
the input errors used for weighting at half their median. The plotted bars
show the input standard errors. The decay fit's reduced :math:`\chi^2` is 7.4,
so the point scatter also exceeds these conditional errors under the
exponential model.

All delay points share the same calibration. To estimate its effect on the
lifetime, shift the prefactor :math:`a` and exponent :math:`n` together by
one standard deviation in both directions along each principal direction of
their fitted covariance, in coordinate order :math:`(a,n)`. Invert all rates
and refit :math:`\tau` after each shift. Half of each paired difference,
combined in quadrature, gives a separate calibration sensitivity of 0.17 µs.
This calculation keeps the offset fixed and treats the nominal injection
densities as given.

Fitting only 3–20 µs gives 9.60 µs, a change of about 0.8 µs when the first point
is omitted. The main estimate is 0.68 µs below the paper's
:math:`11.1(9)\;\mathrm{\mu s}` in Fig. 1(e). The endpoint sensitivity,
short-window approximation, fixed dark amplitude and unavailable injection
errors limit the precision of that comparison.

Replay from the exported rates
--------------------------------

Use **Analysis → Export fit report…** after fitting the illuminated periods. Collect the
value and symmetric error on the ``Lambda = ... +/- ...`` line for each
illuminated run. Save those in ``school-rates.tsv`` with one row per run
103277–103298 and the tab-separated header columns ``run``,
``lambda_per_us`` and ``sigma_lambda_per_us``. If a rate has no reported
standard error, resolve the fit before including it.

The **Fit Parameters** figures use prepared series. Run the
:download:`silicon lifetime example script <silicon_lifetime.py>` to build the
density and delay coordinates from the school run table, fit the calibration,
and reproduce the delay points and lifetime from the exported rates. The script
prints the fitted calibration covariance and both lifetime sensitivity checks.
It uses the installed Asymmetry Python package.

.. code-block:: console

   python silicon_lifetime.py school-rates.tsv

See also
--------

- :doc:`/reference/detector_grouping` for detector grouping and timing.
- :doc:`/reference/parameter_trending` for fitting and viewing parameter trends.
- :doc:`/reference/fit_functions/relaxation` for the exponential model.
- :doc:`/reference/fit_functions/muonium` for muonium dynamics.
