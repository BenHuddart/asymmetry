.. _reduction-pipeline:

Count-based forward/backward reduction
======================================

A forward/backward asymmetry compares two groups of acquired positron counts.
The counts must describe corresponding observations before their ratio has a
physical interpretation. Asymmetry therefore prepares counts and their
uncertainty information together, keeping timing, grouping, correction settings,
and the selected interval with the result.

.. figure:: /_static/reduction-pipeline.svg
   :figclass: reduction-pipeline
   :alt: Acquired counts pass through selection, detector deadtime correction,
      whole-bin alignment and group summation, group background correction,
      count binning, and forward/backward ratio formation. Calibration and
      reference inputs supply optional dependencies; corrected grouped counts
      also feed integral observables before time-curve binning.

   Standard count-based F/B reduction for one selected run and period. Solid
   arrows carry counts and uncertainty information; dashed arrows supply
   settings, calibration, reference data, or current-count inputs. Corrections
   act only when enabled. The integral branch has its own window and estimator
   and accepts no background or a constant subtraction.
   On narrow screens the figure can be scrolled horizontally.

Run co-add/subtraction retains original acquisitions behind its raw inspection
view. Each required original detector enters correction and whole-bin alignment
with its own facts before signed weighting and pooling on common observed good
support. Counts use the coefficient and independent variance its square.
The requested group background or shared reference is applied once after
pooling. See :doc:`/reference/run_arithmetic` for exposure and count-model limits.

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

The corrected grouped-count stage also supplies :doc:`/reference/alc_mode`
for supported run-level preparation with no background or a constant subtraction.
Reference-run subtraction is unavailable on that branch; its preparation does
not resolve a reference spectrum.
Its count-integral estimator sums a chosen interval before taking a ratio;
its differential estimator averages valid per-bin asymmetries. This branch
therefore precedes the time-curve binning box and still depends on the applicable
balances. It does not imply that every scan is made from a binned asymmetry curve.

For two-period G+R, acquired detector corrections and own-t0 whole-bin
alignment precede count/variance pooling on common observed support. Group
background acts once on that pooled observation, followed by count binning and
one ratio. G−R prepares both sets on that common native support, then subtracts
their independently reduced asymmetries with marginal variances added.
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
observation clock. Fourier compensation uses its amplitude. These operations
preserve raw counts, file t₀, and good bounds. Explicit Ignore/off is the identity;
active finite-width requests need a positive finite support half-width. Source
labels and ambiguous file width fields do not establish that input. See
:doc:`/reference/count_domain_fitting` and :doc:`/reference/fourier_analysis`.
