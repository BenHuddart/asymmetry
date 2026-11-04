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
Reference-background data require their own detector preparation and exposure.
See :doc:`/reference/detector_grouping` for correction assumptions and failures.

**Align, sum, and subtract group backgrounds.** Whole-bin shifts align the
selected detectors without interpolation. The maximum selected integer anchor
sets the reference, with acquired-channel ordinal breaking ties; its exact
origin supplies the time stamps. Fractional residuals record the approximation,
not a timing uncertainty. Only common observed support enters the group sums.
Enabled range, tail-fit, fixed, or reference-run backgrounds then act on those
grouped counts, rather than independently on each raw detector.

**Bin counts, then form asymmetry.** The good window selects the observations
for analysis bins. Counts and the full retained variance model, including shared
background-estimate components, are combined before the F/B ratio is formed.
Averaging already formed ratios is a different operation. For example, with
unity balances, no corrections, and independent Poisson counts, adjacent pairs
(F, B) = (9, 1) and (1, 1) merge to (10, 2), giving asymmetry 2/3 rather than
the mean ratio 0.4. The
``reduce_grouped_asymmetry`` API reports asymmetry
and uncertainty in percent, and time in μs relative to the selected exact
origin. The lower-level :func:`~asymmetry.core.transform.rebin.binned_fb_asymmetry`
returns fractional asymmetry and uncertainty. See
:doc:`/reference/asymmetry_units` for API scales and
:doc:`/reference/data_reduction/binning` for bin edges, partial bins, and errors.

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

For two-period G±R asymmetry, each period is reduced separately on matching bin
edges before the asymmetries are combined, with independent variances added.
Run arithmetic has its own compatibility and uncertainty requirements; selecting
runs at the start of the diagram does not authorise adding raw acquisitions
before nonlinear detector corrections. Raw-count likelihood fits and specialised
observation models require their own preparation and statistical assumptions;
they do not all pass through an F/B ratio. See
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

.. rubric:: References

* S. J. Blundell, R. De Renzi, T. Lancaster, and F. L. Pratt, *Muon Spectroscopy:
  An Introduction* (Oxford University Press, Oxford, 2022).
* F. L. Pratt, Physica B **289–290**, 710 (2000),
  `WiMDA <https://shadow.nd.rl.ac.uk/wimda/>`_.
* A. Suter and B. M. Wojek, Phys. Procedia **30**, 69 (2012),
  `musrfit manual <https://lmu.pages.psi.ch/musrfit-docu/user-manual.html>`_.
* `Mantid muon reduction tutorial
  <https://docs.mantidproject.org/v6.9.0/tutorials/muon_GUI_course/basics_of_data_reductions.html>`_.
