LF decoupling and static-vs-dynamic field distributions
=======================================================

.. admonition:: Example data

   :ref:`Download the Rigi school data <rigi-school-data>` for the
   Ca₃Co₂O₆ plateau scan. The :ref:`Re₆Zr contrast <re6zr-lf>` uses
   MuSR runs from the ISIS DataGateway.

   Use your own longitudinal-field series at a stable temperature, with field
   metadata, instrument calibration, and a zero-field reference where available.
   The Ca₃Co₂O₆ and Re₆Zr runs below identify the demonstrations. Choose the fit
   window and candidate models from your own spectra, and check whether the
   time resolution and field range can distinguish their predictions.

Do the local fields fluctuate on the timescale of the muon measurement?
Static and slowly fluctuating fields can both give a nearly flat
high-field spectrum. In a longitudinal-field (LF) scan, use **Fit**
to measure the relaxation rate as the field changes, then compare the
rate trend with a model in **Fit Parameters**. The time resolution and
the choice of model limit what decoupling can establish.

This chapter works the dynamic case on real data: the Ca₃Co₂O₆ magnetic-plateau
decoupling scan from the Rigi school data (HiFi runs 9023–9051, 15 K,
0–3.8 T), the same measurement published by Baker, Lord, and Prabhakaran,
J\. Phys.: Condens. Matter **23**, 306001 (2011). Its headline is the Redfield
linearisation — :math:`1/\lambda` against :math:`B^2` — built directly in the
parameter-trending panel using the native axis transforms
(:doc:`/reference/parameter_trending`), so the paper's Fig. 2(b) is reproduced
without leaving the GUI. A short contrast section then shows the *static*
counter-example on the superconductor Re₆Zr, where 10 mT of longitudinal field
suppresses the observed relaxation, consistent with static or quasistatic
fields on the experimental timescale.

Physical motivation
-------------------

A muon implanted in a magnetic material samples a distribution of local fields
:math:`B_\mu`. The relaxation of its spin polarisation depends both on the
*width* of that distribution and on its evolution during the usable observation
window. That window depends on the instrument, counting statistics, and signal.

For a **static**, zero-mean, isotropic Gaussian field distribution, define
:math:`\Delta = \gamma_\mu\sqrt{\langle B_x^2\rangle}` in μs⁻¹, using
the angular gyromagnetic ratio, with
:math:`\gamma_\mu/(2\pi)\approx135.5` MHz/T. Time is in μs and field in
tesla in the Redfield equations below. The zero-field polarisation is
the static Gaussian Kubo–Toyabe function

.. math::

   P_z^{\mathrm{KT}}(t) = \frac{1}{3} + \frac{2}{3}
   (1 - \Delta^2 t^2)\exp\!\left(-\frac{\Delta^2 t^2}{2}\right),

whose one-third tail is the angular average
:math:`\langle\cos^2\theta\rangle = 1/3` of the spin projection along the
local field. It does not require a third of the spins to be exactly parallel
to their fields. A longitudinal field suppresses dephasing progressively as
:math:`\gamma_\mu B_L/\Delta` grows. The static prediction approaches unity
as this ratio tends to infinity; there is no sharp finite decoupling threshold.
See :doc:`/reference/fit_functions/kubo_toyabe` and Hayano *et al.* below.

For **fluctuating** fields, one useful limit has an exponential transverse-field
autocorrelation with a single time :math:`\tau`. In the motional-narrowing
regime :math:`\gamma_\mu\Delta_B\tau\ll1`, the polarisation is approximately
:math:`P_z(t)=\exp(-\lambda t)`, with the Redfield rate

.. math::

   \lambda(B_L) = \frac{2\gamma_\mu^2\Delta_B^2\tau}
                       {1+\gamma_\mu^2 B_L^2\tau^2}.

Here :math:`\Delta_B` is a **field** scale in tesla, while the static KT
:math:`\Delta` above is an **angular-frequency** width in μs⁻¹.
For equal finite transverse component variances, the correlation model sets
:math:`\Delta_B^2=\langle B_x^2\rangle=\langle B_y^2\rangle`.
The two width conventions are related by :math:`\Delta=\gamma_\mu\Delta_B`
only when they describe the same component distribution. Baker *et al.* use
:math:`\Delta` for their phenomenological field-width parameter; this page
writes that parameter as :math:`\Delta_B` to distinguish the units.
Their fitted field scale does not establish a static Gaussian distribution.

This form assumes the relevant field amplitude and correlation time are
independent of applied field over the fitted range. Its :math:`1/B_L^2`
falloff applies only for :math:`\gamma_\mu B_L\tau\gg1`. Other correlation
functions, slow dynamics, multiple sites, or field-induced changes need not
follow this law. Separating width from correlation time requires data sensitive
to the crossover; low-field data constrain mainly
:math:`\Delta_B^2\tau`, while the high-field limit constrains mainly
:math:`\Delta_B^2/\tau`.

This workflow fits ``Linear`` to transformed axes. The standalone
``Redfield`` component uses stored :math:`D` and a cyclic-frequency
:math:`\nu`; these are not :math:`\Delta_B` and :math:`1/\tau`. Use the
conversion in :ref:`derived-redfield-convention` when choosing that component.

Ca₃Co₂O₆ is a frustrated Ising-chain magnet: ferromagnetic Ising chains couple
antiferromagnetically on a triangular lattice, and below :math:`T_{N1} = 25`
K the material develops a partial magnetisation plateau at one-third of
saturation over the field range 0.5–3.6 T. Inside that plateau the internal
fields are neither fully frozen nor fully paramagnetic — they *fluctuate slowly*,
and the muon decoupling scan at 15 K is designed to catch them.

The data
--------

The Rigi school archive contains the ISIS HiFi NeXus (HDF4) files for
runs 9023–9051. The 20 G transverse-field calibration at 300 K (run 9023)
provides a reference for :math:`\alpha` and the initial asymmetry. One
calibration run does not determine their full field dependence. The
15 K longitudinal-field scan runs from 9031 (zero field) to 9051 (3.8 T),
stepping the applied field through and beyond the plateau. Ten million decay
positrons were collected per run. The applied field in the 15 K scan is longitudinal throughout,
so this is a decoupling scan, not a transverse-field precession measurement.

Two features of the raw data shape the analysis. First, below about 0.2 T the
relaxation is faster than the ISIS pulse width can resolve and the observed
asymmetry is suppressed, so the zero-field and 0.1 T runs cannot yield a
reliably resolved :math:`\lambda` in this analysis and are excluded from the trend. Second, the initial
asymmetry is only fully recovered by ~0.5 T; the constant background grows with
field as decay positrons spiral in the applied field (from ~7 % at 1 T to ~37 %
at 3.5 T), so every fit carries an additive background term.

Step 1 — Load and overlay the decoupling series
-----------------------------------------------

.. figure:: /_generated/corpus_screenshots/corpus_plateau_lf_overlay.png
   :alt: Four Ca₃Co₂O₆ 15 K LF spectra overlaid at zero field, 0.5, 1.5, and 3.5 T
   :width: 100%

   Raw 15 K longitudinal-field spectra at zero field (9031), 0.5 T (9039),
   1.5 T (9045), and 3.5 T (9050), loaded as the group ``Ca₃Co₂O₆ 15 K LF
   scan`` with **Overlay** enabled. The 3.5 T trace (red) is flat and high
   near 35–40 %: the muon is decoupled and the observed asymmetry has
   recovered. The lower-field traces relax within a few microseconds and sit
   lower, both because their relaxation is faster and because their recovered
   asymmetry is smaller. The vertical spread is therefore a mix of decoupling
   (flattening) and the field-dependent baseline.

Load the four representative runs and tick **Overlay** so all four draw on one
set of axes. The qualitative decoupling picture reads straight off the plot:
the relaxation flattens as the field rises. This is the same information the
static case would give — a flat high-field trace — but here it is only the
*first* half of the story. To tell dynamic from static we must measure how the
rate falls with field, not merely that it falls.

Step 2 — Fit each run with a single exponential
-----------------------------------------------

.. figure:: /_generated/corpus_screenshots/corpus_plateau_exp_fit.png
   :alt: Converged Exponential + Constant fit on the Ca₃Co₂O₆ 1.0 T run 9044
   :width: 100%

   A converged ``Exponential + Constant`` fit on the 1.0 T run (9044),
   displayed bunched ×5 over 0–10 µs. The model line
   ``A_1*exp(-Lambda*t) + A_bg`` gives an initial asymmetry
   :math:`A_1 = 8.72\,\%`, a relaxation rate :math:`\lambda = 1.33\;\mu
   \mathrm{s}^{-1}`, and a background :math:`A_{bg} = 7.23\,\%`. The fit runs
   on the unbinned 0–16 µs data and converges cleanly.

Open **Analysis → Fit** and use **Single** for run 9044. Set
**Fit range** to 0–16 µs and fit ``Exponential + Constant``,
:math:`A(t) = A_1\,\exp(-\lambda t) + A_{bg}`, consistent with the single-exponential limit
of the Redfield picture with an additive background. Seed :math:`\lambda`
high (~9 µs⁻¹) at low field and carry it downward from run to run in field
order: the low-field runs are otherwise prone to walking into a spurious
flat-line minimum. The additive :math:`A_{bg}` absorbs the field-growing
baseline from positron spiralling. Each converged run contributes one
:math:`\lambda(B_L)` point to the trend.

.. note::

   The fit-results panel tags this fit's :math:`\chi^2_r = 1.25` (npar = 3,
   ndof = 991) with the wording *poor*. The verdict is the panel's two-sided
   goodness-of-fit band (:doc:`/reference/parameter_trending`); with nearly a
   thousand degrees of freedom, :math:`\chi^2_r=1.25` is a statistically
   significant excess over unity. Inspect residual structure, model adequacy,
   and the uncertainty assumptions. Late-time noise alone does not establish
   why the fit fails the goodness-of-fit criterion.

Step 3 — Build the λ(B) trend
-----------------------------

.. figure:: /_generated/corpus_screenshots/corpus_plateau_lambda_field.png
   :alt: The relaxation rate lambda against applied field B for the Ca₃Co₂O₆ 15 K scan
   :width: 100%

   The :math:`\lambda(B)` trend in the parameter-trending panel (paper
   Fig. 2(a)), plotted on the panel's native ``B (G)`` axis. The series is
   ``λ(B) — Ca₃Co₂O₆ 15 K``; the y-parameter is ``λ (µs⁻¹)``. The three
   regimes are visible: a rapid drop from ~4.2 µs⁻¹ at 0.2 T (2000 G),
   a slow decrease across the 0.5–3.6 T plateau, and a levelling to ~0.3 µs⁻¹
   above 3.6 T.

Fit each run from 0.2 T upward. Open **View → Show Fit Parameters**,
select the fit series, and plot ``λ (µs⁻¹)`` against ``𝐵 (G)`` to build
the :math:`\lambda(B_L)` curve. Its shape is the physics of the plateau. Below
0.5 T the bulk magnetisation is still rising and :math:`\lambda` falls
steeply; across the 0.5–3.6 T plateau it decreases slowly, the regime where
the Redfield model holds; above 3.6 T the sample saturates into a
field-polarised ferromagnet and :math:`\lambda` is nearly constant. The zero-
field and 0.1 T points are absent by design (Step 2): their relaxation is
unresolvable at the ISIS pulse width, so they carry no physical rate.

A decreasing fitted rate is consistent with field suppression of dynamics,
but it is not unique evidence: static decoupling is also continuous, and a
static KT signal is not generally a single exponential. Compare the full
candidate time-domain predictions rather than interpreting exponential rates
as a universal static-versus-dynamic discriminator.

Step 4 — The Redfield linearisation
-----------------------------------

For the Redfield plot, repeat the time-domain fits on a separate series
with ten runs, 9038–9039 and 9044–9051. This figure uses that selection
throughout, so runs 9040–9043 are outside the series even though their
fields lie on the plateau. In **Fit Parameters**, right-click the
points for 9038 (0.4 T) and 9051 (3.8 T) and choose **Exclude from
trend** for each. This leaves eight points for the trend fit, from 9039
and 9044–9050 (0.5–3.5 T).

On the **x** rail, use **ƒ → x²  (square)**, and on the rate card use
**ƒ → 1/y  (reciprocal)**. Click **Fit…** on that card and choose
``Linear``. Set **Errors** to **Column (stabilised errors)** to raise
small propagated reciprocal-rate errors to half the median included
error when weighting the fit.

.. figure:: /_generated/corpus_screenshots/corpus_plateau_redfield.png
   :alt: Redfield linearisation — 1/lambda against B squared with a linear fit over the plateau
   :width: 100%

   :math:`1/\lambda` in μs against :math:`B^2` in G² in **Fit Parameters**
   (compare Fig. 2(b) of Baker *et al.*). The eight included points follow
   the fitted line. Grey rings mark the 0.4 T point near the intercept
   and the 3.8 T point below the extrapolated line.

Redfield's equation linearises exactly. Inverting it,

.. math::

   \frac{1}{\lambda} = \frac{1}{2\gamma_\mu^2\Delta_B^2\tau}
       + \frac{\tau}{2\Delta_B^2}\,B_L^2 ,

so :math:`1/\lambda` is a straight line in :math:`B_L^2` for constant
:math:`\Delta_B` and :math:`\tau`, with slope :math:`\tau/(2\Delta_B^2)` and
intercept :math:`1/(2\gamma_\mu^2\Delta_B^2\tau)`. No large-field approximation
is needed for this algebra; its interpretation still requires the Redfield
assumptions and field-independent parameters over the chosen interval.

The :doc:`axis transforms </reference/parameter_trending>` set the plotted
and fitted coordinates. The plotted reciprocal-rate errors propagate as
:math:`\sigma_{1/\lambda} = \sigma_\lambda/\lambda^2`.

Reading the slope and intercept back through the algebra gives
:math:`\Delta_B = 41.0\;\mathrm{mT}` and :math:`\tau = 929\;\mathrm{ps}`, against
the paper's :math:`\Delta_B = 40.6(3)\;\mathrm{mT}` and
:math:`\tau = 880(30)\;\mathrm{ps}`. These are conditional model estimates.
The transformed linear fit and a fit
in the original rate coordinates have different finite-error weighting;
reciprocal propagation is a local approximation, particularly for uncertain
rates. With the stabilised weights, the sum of squared weighted
residuals per degree of freedom is 1.83. The paper
reports that its estimates are insensitive to small reductions of its fitted
field interval. The estimates here depend on the eight selected plateau
points and their effective weights.

.. dropdown:: Solving the slope and intercept for :math:`\Delta_B` and :math:`\tau`

   Write the fitted line as :math:`1/\lambda = m\,B_L^2 + b` with slope
   :math:`m = \tau/(2\Delta_B^2)` and intercept
   :math:`b = 1/(2\gamma_\mu^2\Delta_B^2\tau)`. Their ratio removes
   :math:`\Delta_B`:

   .. math::

      \frac{m}{b} = \gamma_\mu^2\tau^2
      \quad\Longrightarrow\quad
      \tau = \frac{1}{\gamma_\mu}\sqrt{\frac{m}{b}},

   and back-substituting recovers the width:

   .. math::

      \Delta_B = \sqrt{\frac{\tau}{2m}}.

   With the transformed fit returning :math:`m = 0.276\;\mu\mathrm{s\,T^{-2}}`
   and :math:`b = 0.441\;\mu\mathrm{s}`, and
   :math:`\gamma_\mu = 2\pi\times 135.5\;\mathrm{MHz\,T^{-1}}`, this gives
   approximately :math:`\tau = 0.93\;\mathrm{ns}` and
   :math:`\Delta_B = 41\;\mathrm{mT}` from the rounded coefficients.

The plotted square uses the native field in gauss, so the abscissa is in
G² and the fitted slope in μs G⁻². Multiply that slope by :math:`10^8`
to use the equations above with field in tesla and slope in μs T⁻².
The intercept is in μs in either convention.

Within the single-timescale Redfield interpretation, the plateau data support
a correlation time just under a nanosecond and an effective field scale near
40 mT. These estimates are conditional on the model, plateau selection, and
background treatment; they do not uniquely identify a microscopic process.

.. _re6zr-lf:

Comparing zero and longitudinal field in Re₆Zr
----------------------------------------------

The Re₆Zr measurements come from the `ISIS dataset for MuSR runs 38176–38275
<https://doi.org/10.5286/ISIS.E.24088625>`__. Select runs 38224 and
38263, choose **F-B asymmetry**, and turn on **Overlay**. Compare
the extent of the relaxation over the same time range in both runs.

.. figure:: /_generated/corpus_screenshots/corpus_trsb_lf_decoupling.png
   :alt: Re₆Zr base-temperature zero-field spectrum overlaid with the 10 mT longitudinal-field spectrum
   :width: 100%

   The static case, for contrast: the superconductor Re₆Zr at base temperature
   (0.3 K), zero field (blue, 38224) overlaid with 10 mT longitudinal field
   (orange, 38263). The zero-field trace relaxes toward its Kubo–Toyabe 1/3
   tail; a mere 10 mT (100 G) of longitudinal field flattens it completely.
   This supports the static or quasistatic interpretation on the observation
   timescale, as stated by Singh *et al.*; the field pair alone cannot rule
   out every dynamic model.

For Re₆Zr, the reported Gaussian KT width near
:math:`\Delta\approx0.26\;\mu\mathrm{s}^{-1}` corresponds to a component
field width :math:`\Delta/\gamma_\mu\approx3.1` G. A 10 mT field (100 G)
is about 33 times that width, consistent with strong static-model decoupling.
A nearly flat spectrum means any remaining relaxation is below the sensitivity
of that measurement; it does not imply an exactly zero rate.

The published interpretation of Re₆Zr is static or quasistatic on the muon
precession timescale. The Ca₃Co₂O₆ field dependence is instead described by a
dynamic Redfield model over its plateau. Neither interpretation follows from
a universal sharp threshold or from a single flat high-field trace.

The static LF-KT decoupling series
----------------------------------

The Ca₃Co₂O₆ scan probes dynamics, and the Re₆Zr example compares two fields.
The textbook static case is illustrated with a synthetic Gaussian-field series
that spans the decoupling crossover.

.. image:: /_generated/screenshots/lf_kt_global_results.png
   :alt: Member Results window from a global generic LF-KT fit, showing shared width and Hessian error
   :width: 100%

The member results window shows the fitted parameters and Hessian errors for
one member of the global fit; its shared width is the same across all members.
The model for every run is ``LongitudinalFieldKT + Constant``, fitted jointly
across the field series in a global fit: the Gaussian width :math:`\Delta` is
shared (Global), the applied field :math:`B_L` is fixed per run (File), and the
initial asymmetry and background are shared. Hayano *et al.* computed the full
LF-KT polarisation function, which recovers to unity once
:math:`\gamma_\mu B_L \gg \Delta`. For the synthetic generic series with input
:math:`\Delta = 0.39\;\mu\mathrm{s}^{-1}` the global fit recovers
:math:`\Delta = 0.3903(4)\;\mu\mathrm{s}^{-1}` with per-run reduced :math:`\chi^2`
near unity. The highest-field run is consistent with the static prediction
:math:`A_0+A_{bg}` within its fit uncertainty. Its static character is known
from the generating model. For acquired data, remaining relaxation could
reflect dynamics, an inadequate static distribution, misalignment, background,
or additional components; it does not select Redfield uniquely.

Interpretation
--------------

What the analysis delivers:

- **Static and dynamic model tests.** The complete field series and its
  time-domain shapes constrain candidate models. Decoupling is consistent with
  static or quasistatic fields; the Redfield field dependence supports a
  particular dynamic interpretation within its domain.
- **Field-width convention.** The KT :math:`\Delta` is an angular-frequency
  component width; the Redfield :math:`\Delta_B` is a field scale. Their
  numerical values cannot be compared without the gyromagnetic conversion
  and matching distribution assumptions.
- **Correlation time.** The fitted :math:`\tau` describes the assumed
  autocorrelation. Assigning it to hopping, a particular magnetic excitation,
  or one stopping site needs independent evidence.

Common pitfalls
---------------

- **Fitting the unresolvable low-field runs.** Below ~0.2 T the fast relaxation
  is beyond the ISIS pulse width and the asymmetry is suppressed, so the single
  exponential is degenerate and :math:`\lambda` collapses to a spurious near-
  zero minimum. Those fits do not resolve a reliable rate for this experiment and are
  excluded from the trend. The limit depends on the pulse profile, timing
  response, usable window, and statistics; it is not a universal ISIS cutoff.

- **Forgetting the field-growing background.** The constant background rises
  from ~7 % at 1 T to ~37 % at 3.5 T as decay positrons spiral in the applied
  field. Fit it (the additive ``Constant`` term) rather than assuming a
  field-independent baseline, or the recovered :math:`\lambda` and
  :math:`A_1` are biased.

- **Reading dynamic where the data are static.** A single flat high-field
  spectrum is consistent with *both* a decoupled static distribution and a
  weakly-relaxing dynamic one. Examine multiple fields and time-domain shapes,
  and report the models and timescales the data can distinguish.

- **Confusing the axis-scale log toggle with a transform.** The Redfield
  linearisation needs the plotted *values* transformed (reciprocal Y, square
  X), not merely the tick spacing changed. The **log** checkbox rescales the
  ticks and leaves the numbers alone; use the **ƒ** transform buttons (the x
  rail's, and each card's own) for the linearisation
  (:doc:`/reference/parameter_trending`).

Further reading
---------------

- P\. J. Baker, J. S. Lord, and D. Prabhakaran, J. Phys.: Condens. Matter
  **23**, 306001 (2011) — the Ca₃Co₂O₆ plateau decoupling measurement worked
  here (`arXiv:1105.2200 <https://arxiv.org/abs/1105.2200>`_, Eq. 4).
- R\. S. Hayano *et al.*, Phys. Rev. B **20**, 850 (1979) — the original static
  and dynamic Kubo–Toyabe polarisation functions, including the LF-KT
  decoupling form; still the canonical reference.
- R\. P. Singh, A. D. Hillier, B. Mazidian, J. Quintanilla, J. F. Annett,
  D\. McK. Paul, G. Balakrishnan, and M. R. Lees, Phys. Rev. Lett. **112**,
  `107002 (2014) <https://doi.org/10.1103/PhysRevLett.112.107002>`_ —
  Re₆Zr LF measurements support static or quasistatic fields.
- S\. J. Blundell, R. De Renzi, T. Lancaster, and F. L. Pratt, *Muon
  Spectroscopy: An Introduction* (Oxford University Press, Oxford, 2022),
  Ch. 5.2–5.3 — static Gaussian KT, the LF-KT decoupling form, and the
  strong-collision dynamic KT.
- A\. Yaouanc and P. Dalmas de Réotier, *Muon Spin Rotation, Relaxation, and
  Resonance: Applications to Condensed Matter* (Oxford University Press,
  Oxford, 2011), Ch. 6 — the most detailed mathematical treatment of dynamic
  relaxation and the Redfield limit.

Cross-references
----------------

- :doc:`/reference/parameter_trending` — the trending panel, axis transforms,
  and the trend model-fit dialog used for the Redfield linearisation.
- :doc:`/reference/fit_functions/kubo_toyabe` — the static and LF-KT
  reference page.
- :doc:`dynamic_kt_copper` — the dynamic Kubo–Toyabe worked example (fitting
  the fluctuation rate :math:`\nu` directly).
- :doc:`/reference/global_fit_wizard` — the Global Fit Wizard used for the
  synthetic LF-KT series.
