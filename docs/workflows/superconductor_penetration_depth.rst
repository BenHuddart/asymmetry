Superconductor penetration depth from σ(T)
==========================================

.. admonition:: Example data

   :ref:`Download the Rigi school data <rigi-school-data>` for
   BiSCCO and LiFeAs. The :ref:`Re₆Zr example <re6zr-penetration-depth>`
   uses MuSR runs from the ISIS DataGateway.

   Use your own transverse-field temperature series in the mixed state, with
   field and temperature metadata, instrument calibration, and a normal-state
   reference for the nuclear and background contributions. The named material
   runs and numerical results below identify the demonstrations. Establish your
   own component model and non-superconducting linewidth before converting a
   linewidth trend to penetration depth.

How does the width of the local-field distribution change as a
superconductor cools through its transition? In transverse-field (TF)
μSR the spread of precession frequencies damps the oscillation. Fit that
damping in **Fit**, then compare the linewidths at two applied fields in
**Fit Parameters**. The Rigi school example **A high-Tc cuprate** uses a
bismuth-strontium-calcium-copper-oxide (BiSCCO) sample measured on ISIS
MuSR at 400 G and 200 G over 10–125 K.

The same workflow reappears, with material-specific twists, in two further
examples cross-referenced at the end: the iron-pnictide LiFeAs, where
the linewidth is reported as a field width :math:`B_\mathrm{rms}(T)`, and the
noncentrosymmetric superconductor Re₆Zr, where a fitted TF Gaussian
linewidth is shown alongside a zero-field time-reversal-symmetry-breaking
signal.

Physical motivation
-------------------

In the mixed state of a type-II superconductor, magnetic flux
penetrates the sample as vortices. Muons precess faster in stronger local
fields, so their phases spread and the ensemble oscillation damps.
Use the fitted **Gaussian** width :math:`\sigma_G` in μs⁻¹ to
measure that damping. The width includes broadening that survives above
the transition. Keep one normal-state run in each field series to measure
this residual contribution before estimating a penetration depth.

.. _biscco-linewidth:

The BiSCCO data
---------------

The 400 G trend uses all 14 MuSR runs 1276–1289 (10–125 K). The 200 G
trend uses ten runs, 1290, 1292–1294, and 1296–1301 (30–120 K). This is
the subset with comparable fits in the school's reference analysis. Run
1291 (10 K) has a poor reference fit, run 1302 (115 K) uses a different
model, and runs 1295 (80 K) and 1303 (120 K) have no reference fits.

The normal-state references are runs 1276 (125 K, 400 G) and 1290 (120 K,
200 G). Keep them in their respective trends. Use the applied field and
temperature setpoint recorded in each file. The school exercise identifies
the sample as BiSCCO and the MuSR run titles use the label ``HTC``. The
crystallographic phase is unspecified in these records.

Step 1 — Fit a transverse-field run for σ
-----------------------------------------

.. figure:: /_generated/corpus_screenshots/corpus_bscco_tf_fit.png
   :alt: Time-domain fit to the Gaussian-damped BiSCCO precession at 10 K and 400 G
   :width: 100%

   The **Fit** dock (**Single** tab) for run 1277 at 10 K and 400 G. The
   **PARAMETERS** table shows a precession frequency of about 5.24 MHz
   and a Gaussian width of about 1.16 μs⁻¹. The display spans 0.1–2.5 μs
   to show the damping within about 2 μs.

Use the file's MuSR detector grouping. The time-domain fits use
forward/backward asymmetry from file groups 1 and 2 with :math:`\alpha=1`.
In **Grouping**, retain the imported **t0 Bin**, **t_good Offset**, and
**Last Good Bin** settings for each run. Use no extra bunching, dead-time
correction, or background correction.

Open **Analysis → Fit** and choose **Single**. Under **Edit… → Edit as
text**, enter ``Oscillatory * Gaussian + Constant`` and click **OK**.
For run 1277, start the amplitude at 8%, frequency at 5.3 MHz, phase
at zero, ``sigma`` at 1.1 μs⁻¹, and the constant near the asymmetry
median. Leave the parameters free. Enable **Bounds** and set the
minimum of the amplitude and ``sigma`` to zero.

Click **Fit** and read the width and its standard error from the
**χ²ᵣ** indicator. The Gaussian envelope is
:math:`\exp[-(\sigma_G t)^2]`, and the constant absorbs the non-precessing
offset. Inspect how well the curve follows the damping before adding
the width to a series. The quoted results use asymmetry fits, while the
school's WiMDA reference fits use detector-group counts.

Step 2 — Inspect the frequency-domain line shape
------------------------------------------------

The frequency-domain views show how the distribution of precession
frequencies changes between the mixed and normal states. Select the
10 K, 400 G run 1277 and open **FFT**.

.. figure:: /_generated/corpus_screenshots/corpus_bscco_vortex_fft.png
   :alt: FFT showing a broad line near 5.4 MHz for BiSCCO at 10 K and 400 G
   :width: 100%

   The **FFT** in the **Frequency** workspace, with **X Units:**
   ``Frequency (MHz)`` and **Reference:** 400.00 G. The low-temperature
   precession frequencies spread around the nominal 400 G Larmor
   frequency of 5.4 MHz.

.. figure:: /_generated/corpus_screenshots/corpus_bscco_maxent.png
   :alt: BiSCCO maximum-entropy spectra at 400 G showing a broad 10 K line and a narrow 125 K line
   :width: 100%

   The **MaxEnt** reconstruction of runs 1277 and 1276. Normalised to
   unit area, the 10 K line spreads over a wider range of frequencies
   than the 125 K line, and its peak lies slightly below the normal-state
   peak. Keep **Seed phases from data** checked to initialise the four
   MuSR quadrant phases.

Finite observation times and FFT processing affect the FFT line shape.
MaxEnt regularises the reconstructed line shape. Use the time-domain
Gaussian fit for the quantitative linewidth in this example.

Step 3 — Trend the fitted widths with temperature
-------------------------------------------------

Make a data group for each field with the run selections above. Use the
**Batch** tab to fit each group separately. Set **Fit range** to
0–32 μs to cover the available good-bin range after reduction
(approximately 0.09–31.8 μs depending on the run). Keep the oscillation,
Gaussian width, and offset parameters **Local**. Turn on **Bounds** to
set the limits. Start the amplitude at 8% with bounds 0–60%, phase at
zero, and offset near the asymmetry median. At 400 G,
start frequency at 5.3 MHz with bounds 4.8–6.0 MHz and ``sigma`` at
1.15 μs⁻¹. At 200 G, use 2.7 MHz with bounds 2.3–3.2 MHz and
0.90 μs⁻¹. Use 0–3 μs⁻¹ for the width bounds in both series.

Under **Seeding**, choose **Chain from previous run** to initialise
successive fits from the preceding result in temperature order.
Click **Run series** and check convergence and residuals for each run.
The displayed series were fitted with separate per-run seeds. Each width
starts at 90% of the previous fitted width, with a minimum seed of
0.05 μs⁻¹. The other seeds are reset for each run, including the offset
from its median asymmetry. See :ref:`global-fitting` for the series controls.

Open **View → Show Fit Parameters**, select the 400 G series, choose
``𝑇 (K)`` on the **x** rail, and check ``σ (µs⁻¹)`` on the **y** rail.

.. figure:: /_generated/corpus_screenshots/corpus_bscco_sigma_t.png
   :alt: BiSCCO fitted Gaussian linewidths at 400 G, falling from 1.16 at 10 K to 0.064 inverse microseconds at 125 K
   :width: 100%

   The **Fit Parameters** panel plots 14 fitted total Gaussian widths
   :math:`\sigma_G` at 400 G. The width falls from 1.16 μs⁻¹ at 10 K to
   a residual of about 0.06–0.07 μs⁻¹ at 110–125 K, including the
   normal-state reference at 125 K.

Step 4 — Compare the two fields
-------------------------------

With one series selected in **Fit Parameters**, **Shift+click** the
second series chip. Both field scans appear on the same axes and the
legend identifies them. Choose **⋯ → Export TSV…** with both chips
selected. The file retains both series, with ``Series``, ``Run``,
``B (G)``, ``T (K)``, ``sigma (µs⁻¹)``, and ``err_sigma (µs⁻¹)``
columns. For this selection the export contains 14 rows at 400 G and
ten at 200 G. Use **Table** to inspect the active series.

.. figure:: /_generated/corpus_screenshots/corpus_bscco_field_compare.png
   :alt: BiSCCO Gaussian linewidths at 200 and 400 G, with the 400 G points higher at 30, 50, and 70 K
   :width: 100%

   At 30, 50, and 70 K, the fitted Gaussian linewidth is higher at
   400 G than at 200 G. The ordering reverses between 85 and 90 K as the
   widths approach their normal-state values. The plot compares the
   measured linewidths, including the residual above the transition.

In the mixed state the local-field distribution depends on both
penetration depth and vortex arrangement. Vortex disorder and anisotropy
can change the linewidth (Brandt, Phys. Rev. B **37**, 2349 (1988)). The
two-field comparison establishes a
difference in fitted width over part of the temperature range. Locating
a dimensional crossover would require evidence for the vortex structure
and its evolution with field in this specimen.

Estimating a penetration depth
------------------------------

For an estimate from the export, take the ``sigma (µs⁻¹)`` values for
runs 1277 and 1276 in the 400 G series. At 400 G, the fitted :math:`\sigma_G` values are 1.164 μs⁻¹ at
10 K and 0.064 μs⁻¹ at 125 K. If the residual broadening is a
temperature-independent Gaussian contribution statistically independent of
the vortex broadening, their variances add. Subtract in quadrature to
obtain

.. math::

   \sigma_{G,\mathrm{sc}}
   =\sqrt{\sigma_{G,10\,\mathrm{K}}^2-\sigma_{G,125\,\mathrm{K}}^2}
   \approx 1.162\,\mu\mathrm{s}^{-1}.

If an ideal triangular vortex lattice in the London limit describes the
sample, the school guide's rounded London conversion gives
:math:`\lambda_L(10\,\mathrm{K})=\sqrt{75780/\sigma_{G,\mathrm{sc}}}
\approx 255\,\mathrm{nm}`. This estimate assumes the sample dominates the
precessing signal and the fitted Gaussian approximates the second moment
of the asymmetric line. Brandt's relation uses the conversion
:math:`\sigma_2=\sqrt{2}\sigma_G`. The two-field comparison alone leaves the
vortex structure, reduced-field regime, and any additional precessing
components unresolved. Bernhard *et al.* discuss the conditions for
relating this linewidth to penetration depth in separately measured cuprates.

.. dropdown:: Deriving the London-limit relation and the width conventions

   The field distribution of an ideal triangular vortex lattice has a
   peak at the saddle-point field and a high-field tail from the vortex
   cores (Brandt, Phys. Rev. B **37**, 2349 (1988)). For a Gaussian
   approximation, the root-mean-square field width
   :math:`\langle\Delta B^2\rangle^{1/2}` gives
   :math:`\sigma_2=\gamma_\mu\langle\Delta B^2\rangle^{1/2}`.
   The time envelope :math:`\exp(-\sigma_2^2t^2/2)` equals
   :math:`\exp(-\sigma_G^2t^2)` for
   :math:`\sigma_2=\sqrt{2}\sigma_G`.

   For an ideal triangular vortex lattice at large Ginzburg–Landau
   parameter :math:`\kappa=\lambda/\xi`, Brandt's London approximation
   applies in the regime :math:`0.13/\kappa^2 \ll b \ll 1`, where
   :math:`b=\langle B\rangle/B_{c2}` is the mean induction relative to
   the upper critical induction. In this approximation the second-moment
   relaxation rate is

   .. math::

      \sigma = 0.0609\,\gamma_\mu\,\frac{\Phi_0}{\lambda_L^2},

   with :math:`\Phi_0` the flux quantum; the width in this equation is
   :math:`\sigma_2`. Brandt's field variance is
   :math:`\langle\Delta B^2\rangle=0.00371\Phi_0^2/\lambda_L^4`
   (Brandt, Phys. Rev. B **68**,
   `054506 (2003), Eq. (11) <https://arxiv.org/pdf/cond-mat/0304237#page=5>`__).
   The rounded square root gives the coefficient 0.0609. Brandt's numerical
   comparison finds good agreement near the maximum of the field variance
   for :math:`\kappa\gtrsim70`.

   :func:`asymmetry.core.fitting.sc.constants.sigma_to_lambda_nm` takes
   ``width=GaussianWidth.GAUSSIAN_COMPONENT`` for the fitted
   :math:`\sigma_G`, or ``width=GaussianWidth.SECOND_MOMENT`` for
   :math:`\sigma_2`. With the library constants, the coefficient for
   :math:`\sigma_G` is approximately :math:`7.58\times10^4`, consistent with
   the guide's rounded 75780. See :ref:`sc-width-convention` for the API and
   the trend-panel conversion required by the Brandt field models.
   Known Gaussian sources need that explicit conversion at the field-fit request;
   temperature laws retain their input convention. See :ref:`trend-width-inputs`.

Fitting a gap model
-------------------

If the London conversion applies, the corrected superconducting width
:math:`\sigma_\mathrm{sc}(T)` is proportional to the superfluid density
:math:`\rho_s(T)\propto 1/\lambda_L^2(T)`. Fit that width to compare gap
models. See the :ref:`thermal-kernel accuracy <sc-kernel-accuracy>` for
the model calculations. Asymmetry's parametric-model
registry ships the standard family, each documented in the component-info
dialog:

- ``SC_SWave`` — a single isotropic, fully-gapped BCS order parameter;
  :math:`\sigma(T)` saturates exponentially at low :math:`T`.
- ``SC_DWave`` — a nodal d-wave gap; the line nodes give :math:`\sigma(T)`
  a linear low-:math:`T` slope rather than activated saturation.
- ``SC_TwoGap_SS`` — two nodeless gaps for multiband superconductors (the
  MgB₂-type case), with a weight partitioning the superfluid density
  between the two channels.
- ``SC_TwoGap_SD`` — a mixed s + d interpretation for cases where neither
  pure form fits the full temperature range.

The ``_Q`` variants (``SC_SWave_Q``, ``SC_DWave_Q``) combine the
superconducting and normal linewidth contributions in quadrature rather
than linearly, appropriate when the two sources are independent. Fit these
in the trend panel by checking the ``σ (µs⁻¹)`` chip and clicking **Fit…** on
its card, or export the trend (**⋯ → Export TSV…**) and fit outside the GUI with
``scipy.optimize.curve_fit``. In every case, include a few points above
:math:`T_c` so the high-:math:`T` tail anchors both :math:`T_c` and the
nuclear-dipolar background :math:`\sigma_{bg}`, and subtract
:math:`\sigma_{bg}` in quadrature before converting :math:`\sigma \to
\lambda`.

Cross-reference — LiFeAs, the field width B_rms(T)
--------------------------------------------------

The **LiFeAs** example runs the same vortex-lattice workflow on the
"111" iron-arsenide superconductor, but reports the linewidth as a field
width :math:`B_\mathrm{rms}=\sqrt{2}\sigma_{G,\mathrm{VL}}/\gamma_\mu` (mT) rather
than a rate, following Pratt *et al.*, Phys. Rev. B **79**, 052508 (2009).
The 40 mT temperature scan for Sample 1 uses PSI GPS runs 3366–3373.
The superconducting broadening adds in quadrature to a
temperature-independent nuclear width, :math:`\sigma^2 = \sigma_\mathrm{VL}^2
+ \sigma_n^2`, and the powder London limit gives
:math:`B_\mathrm{rms} = \sqrt{0.00371}\,\Phi_0/(3^{1/4}\lambda_{ab})^2`.

.. figure:: /_generated/corpus_screenshots/corpus_lifeas_brms_t.png
   :alt: LiFeAs Sample 1 field width from fits to the 40 mT GPS runs, falling
         towards the 16 K transition with unresolved widths ringed in grey.
   :width: 100%

   Vortex-lattice field width of LiFeAs Sample 1 at 40 mT from Asymmetry
   two-Gaussian fits to the PSI GPS runs. Grey rings mark points excluded
   from the trend where the vortex contribution is unresolved. The dashed
   line gives the published transition temperature of 16 K.
   The original analysis reports :math:`\lambda_{ab} = 195(2)\;\mathrm{nm}`
   for this sample.
   `Pratt et al. <https://doi.org/10.1103/PhysRevB.79.052508>`__.

.. sidebar:: Data handling — pick the right detector pair

   .. figure:: /_generated/corpus_screenshots/corpus_lifeas_pair_select.png
      :alt: LiFeAs base-T run in default Forward/Back pairing versus the Up/Down transverse pair
      :width: 100%

   These are spin-rotated ("TF WED") GPS runs: the muon spin is rotated
   transverse, so the precession appears in the **Up/Down** detector pair.
   The loader's default Forward/Back pairing sees the two detectors in phase
   and the precession *cancels* (top), leaving noise; regrouping onto the
   Up/Down transverse pair recovers the clean 5.44 MHz vortex-damped signal
   (bottom). This is the essential grouping choice for the example, made in
   the **Grouping** dialog.

.. _re6zr-penetration-depth:

Re₆Zr transverse-field linewidth and zero-field relaxation
-----------------------------------------------------------

Does the fitted transverse-field width change through the Re₆Zr
transition? In **Fit → Batch**, fit one asymmetry trace per 40 mT run
(38180–38219) with ``Oscillatory * Gaussian + Constant`` and keep
the parameters **Local**. In **View → Show Fit Parameters**, select
the series, then choose ``𝑇 (K)`` on the **x** rail and
``σ (µs⁻¹)`` on the **y** rail. The Gaussian
component applies
the envelope :math:`\exp[-(\sigma_G t)^2]`, so the fit parameter
``sigma`` gives the linewidth :math:`\sigma_G` directly. The runs come from
the `ISIS dataset for MuSR runs 38176–38275
<https://doi.org/10.5286/ISIS.E.24088625>`__.

.. figure:: /_generated/corpus_screenshots/corpus_trsb_sigma_sc_t.png
   :alt: Fitted Re₆Zr transverse-field Gaussian linewidth σ_G(T) in µs⁻¹ at 40 mT, falling to a finite normal-state width
   :width: 100%

   The **Fit Parameters** panel shows the fitted Gaussian linewidth
   :math:`\sigma_G(T)` in μs⁻¹ at 40 mT. Each point comes from a successful
   fit and its error bar shows the reported fit standard error. The
   width decreases through :math:`T_c=6.75\,\mathrm{K}`
   and retains a finite normal-state value.

Singh *et al.* formed two orthogonal virtual detectors. Their Eq. (1)
uses :math:`\exp[-(\sigma_\mathrm{sc} t)^2/2]` for the precessing sample
signal and models the silver holder as a second, undamped precession.
For the same sample envelope, their
convention gives :math:`\sigma_\mathrm{sc}=\sqrt{2}\sigma_G`. This factor
matches the envelope definitions. The constant offset in the illustrative
fit leaves any precessing background unresolved, and the fitted linewidth
includes the residual broadening seen above :math:`T_c`. Isolating the
vortex contribution requires accounting for both effects. The single
isotropic s-wave gap interpretation comes from the paper's fit in Fig. 2(b).

.. figure:: /_generated/corpus_screenshots/corpus_trsb_sigma_t_step.png
   :alt: Re6Zr zero-field Gaussian Kubo-Toyabe σ(T) showing the small TRSB step below T_c
   :width: 100%

   The separate zero-field measurement uses the Gaussian Kubo–Toyabe
   width :math:`\sigma(T)`, ``Delta`` in Asymmetry. This width enters the
   Kubo–Toyabe function in Eq. (2) of Singh *et al.* The 0.250–0.270 μs⁻¹
   axis resolves the small increase below :math:`T_c = 6.75\,\mathrm{K}`.
   Together with :ref:`decoupling in a 10 mT longitudinal field <re6zr-lf>`,
   this increase is evidence for spontaneous quasistatic fields and broken
   time-reversal symmetry.

Interpretation
--------------

When the corrected linewidth is proportional to the superfluid density,
compare its shape with the candidate gap models.

- Exponential saturation at low temperature. A single isotropic s-wave gap
  (``SC_SWave``); the :math:`\Delta/k_B T_c` value distinguishes
  weak coupling (BCS, 1.764) from stronger coupling.
- A linear decrease with temperature at low temperature. Line nodes in the gap
  (``SC_DWave``); the hallmark of a d-wave order parameter, as in the
  cuprates.
- A shoulder or kink near :math:`T/T_c \approx 0.3\text{–}0.5`.
  Multiband superconductivity with two gaps (``SC_TwoGap_SS``); the smaller
  gap freezes out at its own scale.

Assumptions and limitations
---------------------------

- **The London limit.** The triangular-lattice conversion assumes
  :math:`\lambda \gg \xi` and a field well inside the mixed state (see
  Sonier *et al.*, Rev. Mod. Phys. **72**, 769 (2000), Appendix B).
- **The normal-state width.** Above :math:`T_c` nuclear dipoles and other
  sources such as field inhomogeneity can contribute to the residual width.
  Subtract this width in quadrature if it represents independent Gaussian
  broadening and is also present in the mixed state.
- **Anisotropy and the vortex regime.** To infer a penetration depth from
  the linewidth, choose a vortex model appropriate to the specimen,
  applied field, and field orientation.
- **Width convention.** Convert :math:`\sigma_G` to
  :math:`\sigma_2=\sqrt{2}\sigma_G` when using a second-moment relation.
  Omitting this factor makes :math:`\lambda_L` too long by
  :math:`2^{1/4}`. Geometry and vortex-lattice assumptions must be assessed
  separately; normalised :math:`\rho_s(T)` is unchanged by this width rescaling.
- **Data range.** Without points above :math:`T_c` the fit cannot separate
  :math:`T_c` from :math:`\sigma_{bg}`; without points below
  :math:`T/T_c \approx 0.3` a small gap in a two-gap model is poorly
  constrained.

References
----------

.. rubric:: References

- E\. H. Brandt, Phys. Rev. B **37**, 2349 (1988) — the vortex-lattice field
  distribution and its second moment.
- J\. E. Sonier, J. H. Brewer, and R. F. Kiefl, Rev. Mod. Phys. **72**, 769
  (2000) — the canonical review of TF μSR in superconductors.
- C\. Bernhard, Ch. Niedermayer, U. Binninger, A. Hofer, Ch. Wenger, J. L.
  Tallon, G. V. M. Williams, E. J. Ansaldo, J. I. Budnick, C. E. Stronach,
  D\. R. Noakes, and M. A. Blankson-Mills, Phys. Rev. B **52**,
  `10488 (1995) <https://doi.org/10.1103/PhysRevB.52.10488>`__ —
  the conditions for relating μSR depolarisation and penetration depth in
  polycrystalline cuprates.
- F\. L. Pratt, P. J. Baker, S. J. Blundell, T. Lancaster, H. J. Lewtas, P.
  Adamson, M. J. Pitcher, D. R. Parker, and S. J. Clarke, Phys. Rev. B
  **79**, `052508 (2009) <https://doi.org/10.1103/PhysRevB.79.052508>`__ — LiFeAs :math:`B_\mathrm{rms}(T)` and
  :math:`\lambda_{ab}`.
- R\. P. Singh, A. D. Hillier, B. Mazidian, J. Quintanilla, J. F. Annett, D.
  McK. Paul, G. Balakrishnan, and M. R. Lees, Phys. Rev. Lett. **112**,
  `107002 (2014) <https://doi.org/10.1103/PhysRevLett.112.107002>`__ —
  Re₆Zr Gaussian TF signal and silver background (Eq. (1), Fig. 2), and
  zero-field Kubo–Toyabe relaxation (Eqs. (2)–(3), Fig. 4).
- S\. J. Blundell, R. De Renzi, T. Lancaster, and F. L. Pratt, *Muon
  Spectroscopy: An Introduction* (Oxford University Press, Oxford, 2022),
  Ch. 9 (superconductors), especially Ch. 9.5 (penetration depth and gap
  structure).
- A\. Amato and E. Morenzoni, *Introduction to Muon Spin Spectroscopy:
  Applications to Solid State and Material Sciences*, Lecture Notes in
  Physics Vol. 961 (Springer, Cham, 2024), Ch. 6 — the vortex-state field
  distribution, multiband superconductivity, and unconventional pairing.

Cross-references
----------------

- :doc:`/reference/loading_data` — loading the ISIS ``.nxs`` and PSI
  ``.bin`` example runs.
- :doc:`/reference/detector_grouping` — the transverse (Up/Down) grouping
  choice for spin-rotated runs.
- :doc:`/reference/composite_models` — building the
  ``Oscillatory * Gaussian + Constant`` time-domain composite.
- :doc:`/reference/fourier_analysis` — inspecting the :math:`p(B)` line
  shape by FFT and maximum entropy.
- :doc:`/reference/parameter_trending` — the σ → λ → gap-model pipeline and
  multi-series overlays.
- :doc:`/reference/sc_penetration_depth` — the parametric ``SC_*`` gap-model
  reference page.
