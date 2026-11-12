.. _reduction-pipeline:

Count reduction and transformed signals
=======================================

A forward/backward asymmetry compares two groups of acquired positron counts.
The counts must describe corresponding observations before their ratio has a
physical interpretation. Asymmetry therefore prepares counts and their
uncertainty information together, keeping timing, grouping, correction settings,
and the selected interval with the result.

An explicitly selected :ref:`ordinary double-pulse time response
<ordinary-time-response>` acts on the fitted prediction after reduction. It
changes neither acquired counts nor this ratio/error pipeline. Ordinary fits
keep point predictions at retained times rather than integrating source counts
over an arrival-straddling bin.

.. figure:: /_static/reduction-pipeline.svg
   :figclass: reduction-pipeline
   :alt: Acquired counts are selected, detector-corrected, aligned, grouped,
      and background-treated. Separate branches form lifetime-corrected group
      signals for FFT or MaxEnt, and F/B ratios for laboratory fitting,
      two-projection rotation, or scalar filtered display. Native-window scan
      preparation branches into count integrals or explicit time means, period
      operations, condition scans, original-observable fits and derivative display.
      Full-count fits retain their own preparation.

   The observation determines the analysis branch. Solid arrows carry selected
   observations and retained uncertainty information; dashed lines identify
   alternative requests with their own preparation. Corrections act only when
   enabled. Grouped fits and integral observables retain their separate input
   contracts. A filtered scalar display does not feed a fit.
   The :ref:`condition-scan detail <condition-scan-observables>` expands the
   native-window scan branch. On narrow screens the figure can be scrolled horizontally.

Run co-add/subtraction retains original acquisitions behind its raw inspection
view. Each required original detector enters correction and whole-bin alignment
with its own facts before signed weighting and pooling on common observed good
support. Counts use the coefficient and independent variance its square.
The requested group background or shared reference is applied once after
pooling. See :doc:`/reference/run_arithmetic` for exposure and count-model limits.

Branches after count preparation
--------------------------------

The diagram is a map of alternative observables, not a mandatory sequence
through every transform. The same post-decoding scientific operations apply to
ISIS HDF4/HDF5 and PSI BIN/MDU/ROOT when the selected observations and settings
match. Source structure, measured timing, and geometry can guide settings;
facility labels guide defaults, and a file suffix does not determine physics.

* **Lifetime-corrected group signals** multiply prepared group counts and errors
  by :math:`\exp(t/\tau_\mu)`. Grouped FFT and MaxEnt each request uncentred
  signals, then apply their own window-level preparation. FFT uses a
  :math:`1/\sigma` fractional level and optional mean removal; MaxEnt uses an
  inverse-variance starting level and its own phase/amplitude reconstruction.
  Neither finite-window level nor FFT coherent-gain correction establishes an
  absolute polarisation calibration. See :doc:`/reference/fourier_analysis`
  for the selected-window, apodisation, padding, and endpoint limits.
* **F/B ratios** combine prepared F/B counts with α and β. They can be fitted
  in the laboratory frame or transformed as described below. The Fourier
  panel's **F−B asymmetry** source transforms a ratio rather than the default
  lifetime-corrected grouped signals; its correction contract is specified in
  :doc:`/reference/fourier_analysis`.
* **Two measured transverse projections** support pointwise rotation after
  explicit baseline, gain, sense, and phase adjustment. Retained shared
  acquisition/reference/background counting noise propagates through the
  coordinate transformation and packing. Calibration and frame quantities
  remain conditional assumptions. A group label or fitted phase is not proof
  of physical detector geometry. See :ref:`vector-geometry-assumptions` and
  :ref:`vector-rotation-counting-errors` for the authoritative geometry and
  covariance contracts; :doc:`/reference/rotating_frame_projection` also
  describes fitting a supported component.
* **One scalar ratio** supports complex demodulation and FIR low-pass display.
  Same-bin real/imaginary covariance enters first-order magnitude errors, but
  filtered bins remain correlated, low-signal magnitude is biased, and a
  correlated or anisotropic noise pair need not obey an exact Rician law.
  Fits retain the original laboratory observations, rather than treating the
  filtered display as independent data. See :ref:`scalar-rrf-magnitude-errors`.

For pooled G+R vector rotation, Asymmetry requires the Red and Green transverse
baselines to agree; an arithmetic mean is not generally the baseline of a
pooled count ratio. Unequal baselines make that operation unavailable while
Lab G+R, rotating Red, Green, and G−R remain supported. This safeguard does not
restrict decoding or unrelated reductions.

Grouped fits use the full-count model and their own nuisance backgrounds;
active reference-run subtraction is unsupported there, even though it is
supported in grouped Fourier/MaxEnt preparation. Their lifetime-corrected
Gaussian and raw-count Poisson objectives are separate choices. See
:ref:`grouped-count-model` and :doc:`/reference/count_domain_fitting`. Integral observables retain their own
window, balance, and estimator rather than integrating a displayed FFT or
filtered curve. All error statements remain conditional on the documented
calibration and correction assumptions. Correct marginal errors do not supply
a general cross-bin or joint-projection likelihood, or certify overlapping-group
MaxEnt. The operation-specific reference map is in
:ref:`comparison-spectral-conventions`.

From acquired counts to a ratio
-------------------------------

**Decode and select.** NeXus, PSI BIN/MDU, and supported PSI ROOT readers
preserve acquired histograms and resolve file timing and calibration facts.
NeXus physical good times become per-detector, per-period raw-bin intervals
at this boundary, before t0 subtraction; selection does not crop acquired arrays.
Choose the run, period, projection, groups, and exclusions. Selection identifies
observations; it does not add runs or periods together. A file suffix does not
choose pulsed- or continuous-source physics. The format-specific contract is
in :doc:`/reference/loading_data`.

**Prepare individual detectors.** Resolve the effective timing and requested
interval, and apply enabled deadtime correction to individual detector
histograms using the applicable calibration and exposure. The aligned interval
is resolved with the selected channels' common support at the grouping step;
file bounds and explicit windows are different inputs to that resolution.
A selected reference acquisition resolves its own period, timing, detector
preparation, and exposure. **Automatic** scale uses compatible evidenced exposure
ratios, visibly assuming one when exposure is unknown; **Manual** scale uses
the authored positive coefficient. Reference alignment requires compatible
native widths and enough observed support; it does not interpolate or pad counts.
See :doc:`/reference/detector_grouping` for correction assumptions and failures.

Mapped sets resolve these detector preparations against their original acquired
members, including their own coefficients, exposure, file-good bins, and t0.
Only selected members enter timing and correction domains. Missing required
File coefficients withhold correction from the entire requested result.

**Align, sum, and subtract group backgrounds.** Whole-bin shifts align the
selected detectors without interpolation. The maximum selected integer anchor
sets the reference, with acquired-channel ordinal breaking ties; its exact
origin supplies the time stamps. Fractional residuals record the approximation,
not a timing uncertainty. Only common observed support enters the group sums,
including across constituents of a mapped set. Counts and retained variances
pool before one ordinary background operation for that set.
Enabled range, tail-fit, fixed, or reference-run backgrounds then act on those
grouped counts, rather than independently on each raw detector. These are
alternative background modes. Reference subtraction retains both independent
counting variances with fixed scale; accidental levels and physical holder
polarisation are different quantities. See
:doc:`/reference/data_reduction/backgrounds`.

**Bin counts, then form asymmetry.** The good window selects the observations
for analysis bins. Counts and the full retained variance model, including shared
background-estimate components, are combined before the F/B ratio is formed.
Nonuniform targets use the actual left edge and nearest whole-bin membership,
with half-up ties and at least one acquisition bin. Timestamps are actual
contributing-bin midpoints with exact-t0 residuals. Base nonuniform support
retains a partial tail; fixed packing and additional merges trim only incomplete
groups. Averaging already formed ratios is a different operation. For example, with
unity balances, no corrections, and independent Poisson counts, adjacent pairs
(F, B) = (9, 1) and (1, 1) merge to (10, 2), giving asymmetry 2/3 rather than
the mean ratio 0.4. The
``reduce_grouped_asymmetry`` API reports asymmetry
and uncertainty in percent, and time in μs relative to the selected exact
origin. The lower-level :func:`~asymmetry.core.transform.rebin.binned_fb_asymmetry`
returns fractional asymmetry and uncertainty. See
:doc:`/reference/asymmetry_units` for API scales and
:doc:`/reference/data_reduction/binning` for bin edges, partial bins, and errors.

Single/global fit-wizard automatic merges replay the actual histogram
observation through these same numerical stages, preserving base and toolbar
merges and selected output-bin membership. Curves without resolved replay
context and rotated curves retain independent-curve means. Scores and rebuilt
curves refer to the prepared observation; historical saved scores are not refitted
on load. The internal wizard FFT diagnostic still requires uniform input sampling.

Each F/B consumer resolves alpha for its actual selected pair. Display binning,
F/B Fourier input and re-reduction retain that projection's coefficient while
the authored base alpha remains available as the fallback for other pairs.

File-good limits stay on acquired raw bins through timing edits; selected-channel
alignment maps them into the common observed interval. Manual window endpoints
are signed offsets from effective t0, with omitted endpoints retaining mapped
file limits. The resolved window also supplies finite-window calibration
integrals. See :ref:`comparison-file-manual-windows` and
:doc:`/reference/detector_grouping`.

**Fit physical signals.** The asymmetry model can add sample and holder
polarisation components with independent amplitudes and appropriate relaxation
or precession. This stage describes physical muon populations; it does not
subtract another accidental count level. Existing
:doc:`/reference/composite_models` supply these components. Exposure scaling of
an empty-holder run alone does not determine the holder contribution in the
sample acquisition.

Calibration dependencies
-------------------------

Calibration is not a single serial transformation of the counts. File or
chosen timing anchors control alignment; detector coefficients and exposure
control deadtime correction; background estimates or a reference spectrum
control subtraction. The balances α and β enter the asymmetry formula after
count binning. A requested estimate can itself depend on prepared counts.

Fixed and one-shot calibrated values remain stored when grouping changes.
An enabled automatic policy instead uses its current inputs. In particular,
``per_run_estimate`` α uses the current corrected, grouped counts and good
window; it does not repeat a diamagnetic fit. β has fixed and calibrated
policies, without an automatic integral-ratio policy. The owning contracts are
:doc:`/reference/data_reduction/alpha_calibration`,
:doc:`/reference/data_reduction/beta_calibration`, and
:doc:`/reference/grouping_calibration`.

Other observables and observation models
----------------------------------------

.. _condition-scan-observables:

Condition-scan observables
~~~~~~~~~~~~~~~~~~~~~~~~~~

The native-count preparation supplies :doc:`/reference/alc_mode`, including
enabled constant or reference-run background and retained shared counting
responses. Select whole native bins by inclusive time centres, using the good
window when bounds are omitted. Display bunching does not change this support.
The scan detail below separates count integrals from explicit time means.

.. figure:: /_static/condition-scan-pipeline.svg
   :figclass: reduction-pipeline
   :alt: Native prepared counts pool G+R before estimation. Count integrals
      sum counts then form period ratios and G−R; explicit means classify
      native ratios or G−R contrasts before averaging. Scalar results form
      a condition scan. Fits use original observations; a separate derivative
      branch supports display and export only.

   Native-window scan reduction. The two estimator paths share chosen α/β,
   period correction, selected support and conditional uncertainty assumptions.
   On narrow screens the diagram can be scrolled horizontally.

For two-period G+R, each acquired detector's correction and own-t0 whole-bin
alignment precede pooling on common observed support. Group background acts
once on the pooled counts, followed by the selected estimator. G−R prepares
both periods on common native support, then subtracts their ratios. For an
explicit mean, the pointwise contrast is classified before averaging. Retained
shared responses combine before variance contraction, rather than assuming
independent period errors. See :doc:`/reference/data_reduction/period_mapping`.

Original scalar values, mathematical status, conditional errors, correction
outcomes and positional source-sharing facts travel into the condition scan.
Fit selection omits unusable Gaussian observations and checks known covariance
in the actual selected window. Baseline/peak fits propagate the fitted baseline
with fixed regions/model. The derivative branch is display/export only: positive
adjacent steps give midpoint slopes, with no automatic gap rule. Its units are
the original fraction or percent per selected G/K/run unit. Known dependent
input pairs retain values with unavailable errors; marginal derivative errors
do not supply an independent likelihood. Original-observable fitting remains
available under its own selected-input contract.

Run arithmetic has its own compatibility and uncertainty requirements; selecting
runs at the start of the diagram does not authorise adding raw acquisitions
before nonlinear detector corrections. Raw-count likelihood fits and specialised
observation models require their own preparation and statistical assumptions;
they do not all pass through an F/B ratio. In particular, grouped/raw count-fit
inputs reject active reference-run subtraction for either cost. With background
correction disabled or a supported non-reference preflight, the count model
fits full counts and its own accidental-background nuisance. Poisson fitting
requires eligible raw observations, not signed reference differences. The
shared TF calibration follows this count-model branch. It uses Poisson deviance
when deadtime is off or skipped, and weighted corrected counts with retained
measured/supplied variance after an applied deadtime correction. Grouping
background settings retain their availability preflight; the two fitted model
backgrounds are neither subtracted nor replaced by grouping estimates. The
separate count-fit branch in the figure represents this preparation contract,
not a tap from the reference-subtracted counts. See
:doc:`/reference/count_domain_fitting`, :doc:`/reference/negative_muon_analysis`,
:doc:`/reference/run_arithmetic`, and :doc:`/reference/data_reduction/period_mapping`.

Compared with other programs
----------------------------

The operation-by-operation guide in :ref:`reviewed-count-reduction` links the
reviewed grouping, calibration-policy, timing, and loading comparisons. Whole-bin
alignment is closest to WiMDA/musrfit's fixed-bin workflows; Mantid also offers
coordinate-preserving and explicit common-grid operations. Unique membership
and independent disjoint pairs are Asymmetry conventions. Different default
windows are not equivalent requests, and matched count agreement does not
certify physical timing or every correction and uncertainty model.

Automatic timing resolves after effective channel selection and exclusions.
Each required estimated anchor belongs to that acquired channel and period;
run-wide consensus cannot supply a missing detector delay. Acquired file facts
remain separate from effective policy outcomes, including numerically equal
estimates. See :ref:`t0-search`.

Fresh explicit calibration uses TF calibration fit, with beta fixed at one unless
joint calibration is requested for a single pair. The GUI stages the selected
model and conditional errors until Apply; input changes invalidate pending work.
Automatic per-run ratio, manual coefficients, and loaded historical records keep
their own policies. A reduction without a calibration source never starts a TF fit.

Combined condition labels
--------------------------

Run arithmetic retains the first ordered input's field, temperature, and
instrument identity, including display/log overrides. These labels are not
weighted estimates of a physical condition. Missing master values stay unknown;
finite active constituent ranges and signed coefficients remain provenance.
Known detector-role/order conflicts and evidenced implicit period permutations
are checked at the combination request boundary. Unknown correspondence keeps
recorded channel/data-order assumptions. See :doc:`/reference/run_arithmetic`
for the bounded reuse checks, reference comparisons, and repair paths.

.. rubric:: References

* S. J. Blundell, R. De Renzi, T. Lancaster, and F. L. Pratt, *Muon Spectroscopy:
  An Introduction* (Oxford University Press, Oxford, 2022).
* F. L. Pratt, Physica B **289–290**, 710 (2000),
  `WiMDA <https://shadow.nd.rl.ac.uk/wimda/>`_.
* A. Suter and B. M. Wojek, Phys. Procedia **30**, 69 (2012),
  `musrfit manual <https://lmu.pages.psi.ch/musrfit-docu/user-manual.html>`_.
* `Mantid muon reduction tutorial
  <https://docs.mantidproject.org/v6.9.0/tutorials/muon_GUI_course/basics_of_data_reductions.html>`_.

Pulse response belongs to prediction
------------------------------------

A supplied count-fit separation selects a double-delta forward model with equal
incident strengths and muon-survival weighting; it consumes no finite width.
MaxEnt folds the parabolic-plus-pion response into prediction on the resolved
observation clock. Fourier compensation divides amplitude displays by the
response amplitude R and Power by R², scaling existing conditional standard
errors in the same units. Its sampled amplitude-gain limit removes bins from
output support before baseline and area normalisation; removed bins are not
measured zeros. This numerical cutoff depends on the grid and limit, and
amplitude compensation does not invert response phase. These operations
preserve raw counts, file t₀, and good bounds. Explicit Ignore/off is the identity;
active finite-width requests need a positive finite support half-width. Source
labels and ambiguous file width fields do not establish that input. See
:doc:`/reference/count_domain_fitting` and :doc:`/reference/fourier_analysis`.


Fixed supplied TF calibration response
--------------------------------------

TF calibration's supplied fixed double-delta response belongs to the predicted
F/B counts. Equal incident births at the supplied midpoint contribute only at
live ages, with absolute muon survival and each complete age-local polarisation;
flat count backgrounds are added once. This preserves the selected acquired
observations, correction eligibility, and raw-Poisson/applied-deadtime Gaussian
objective. Packed-bin means are evaluated at centres, without rate integration.
Selection compares the same four envelopes at the same supplied response.
Conditional fit errors exclude source/selection uncertainty. Apply transfers
coefficients and a frozen source record while retaining target timing and its
ordinary-fit/Fourier response; see :ref:`alpha-calibration`.
