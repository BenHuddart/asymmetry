.. _t0-search:

Time-zero search
================

Time zero establishes the origin of the analysis time axis. File timing is an
instrument calibration; a count feature provides a candidate anchor for review.
Neither agreement with a feature nor a quiet diagnostic certifies physical time
zero. Timing offsets change phase and early-time interpretation.

Selected timing
---------------

**From file** retains each selected channel's usable file calibration.
Integer-only calibration is sufficient, including a legitimate zero. NeXus
integer/exact conflicts retain usable integer anchors and discard only the
conflicting exact declaration. Format-specific unspecified markers remain
separate from explicit zero; see :doc:`/reference/loading_data`.

Legacy LEM ROOT v4/v5 uses numeric file resolution and t0 rather than display
strings or histogram axes. The supported v4 four-channel NPP/PPC banks share
header slots 0–3 while retaining separate count arrays. Zero and fractional
anchors remain usable; negative or nonfinite entries supply no file timing.
Valid file timing remains usable with unknown source metadata, without an
automatic search to compensate for a missed header.

When a required channel has no usable file timing, **From file** searches its
raw counts under an established source strategy. Other selected channels retain
their usable file calibration. The fallback records estimated provenance; it
never relabels the estimate as acquired calibration. **Manual** applies a signed
whole-bin offset to the resolved base. An offset alone cannot supply a missing
base. Switching periods reapplies a Manual offset to that period's own resolved
base; an estimated base from another period is not transferred.

**Auto-detect** requests replacement timing for the actual selected acquired
channels, after exclusions, for each pair and period. Every required channel
must supply its own usable estimate. A failed unused channel does not block a
valid pair. A failed selected channel never receives another detector's anchor
or a median shift. The diagnostic run-wide median is not a detector calibration.

Asymmetry also checks that automatic timing has a known source strategy and
that every required channel resolves. These checks protect the requested
alignment, rather than file admission. An unresolved request shows
**Preview unavailable:**, keeps **Apply** disabled, and preserves the acquired
histograms and editable settings. Repair it by selecting usable channels,
returning to valid file timing, or supplying supported explicit calibration or
an API strategy. Valid file timing remains usable when an advisory search fails.

Source and feature conventions
------------------------------

Known source metadata selects the existing continuous or pulsed strategy.
NeXus ``definition=pulsedTD`` supplies an explicit pulsed-source fact. Missing
or ambiguous source facts remain unknown. A filename does not select a strategy.
The API ``find_t0_for_run(..., pulsed=False)`` explicitly selects the continuous
strategy; ``pulsed=True`` selects pulsed search and takes precedence over metadata.
``resolve_selected_timing`` accepts the same explicit argument when resolving a
selected request. ``T0Policy.strategy`` records a last result for display; it is
not an authored override of unknown source metadata.

* Continuous search takes the maximum-count bin of each acquired histogram,
  choosing the earliest bin on ties. This is a prompt-peak feature, provided
  that the maximum represents that feature.
* Pulsed search walks back from the maximum to the adjacent leading
  half-height crossing, interpolates linearly, and rounds to the nearest bin.
  An early flash can therefore be separate from the chosen leading edge.
  A histogram starting above half-height has no supported leading crossing.

These feature definitions do not identify the physical pulse centre for every
source response. Background, flash, overlap, and detector response can affect
what feature is found. A physical timing calibration requires independent
instrument evidence; changing the estimator requires a separate benchmark.

Search uses raw, unbinned acquired counts. Display binning, Fourier crops, and
good-window choices do not change the feature estimate. The grouping window
reuses a background raw-channel scan across selections of the same acquisition;
acceptance follows the current pair and period. Cached-only periods without raw
histograms are not searched.

Diagnostics and controls
------------------------

.. figure:: /_generated/screenshots/grouping_window_t0_row.png
   :width: 70%
   :align: center
   :alt: Auto-detect timing for a selected pair whose own prompt estimates agree
      with file anchors; an unused empty channel does not block the request.

   **Auto-detect** uses the selected pair's own estimates. Agreement in this
   synthetic example is a feature check, not physical calibration evidence.

The line shows ``File: bin N`` or ``File: none``, followed by ``Detected: …``
while the worker is pending, ``Detected: unavailable`` on failure, or
``Detected: bin M (strategy, spread S)``. The detected bin is the maximum of
selected own estimates. The signed ``Δ`` diagnostic is the median selected
``detected_i − file_i`` shift; the spread is the range of those shifts. They are
different summaries when detector delays differ. A failed selected estimate is
named in the unavailable diagnostic.

A ``⚠`` button exposes advisory messages about file disagreement, detector-shift
spread, and a good window opening on the detected feature. Valid file timing is
not rejected by these messages. The display threshold is the larger of a family
floor (two bins continuous, three bins pulsed) and the median feature width.
Prompt FWHM, pulsed 10–90 % rise width, spread, and thresholds are descriptive
diagnostics, not confidence intervals, localisation limits, or calibration tests.

**Find t0** fills the **Manual** common offset from the cached selected estimate.
It is available after the raw scan completes; **Apply** commits the request.
Use **Auto-detect** to apply each channel's own estimated anchor rather than a
shared manual offset.

Alignment and windows
---------------------

Alignment transports whole acquired bins. Exact file origins retain their
fractional residuals under integer policy shifts; the search does not estimate
new fractional calibration. The selected maximum anchor wins, with acquired
ordinal breaking ties. A pair or projection change resolves the exact origin
from its own selected reference detector, even when the integer anchor is
unchanged. Recorded alignment includes the actual selected channel
origins, effective timing sources, requested search strategy, and residual spread.
Acquired ``t0_source`` remains a separate file fact, including when estimated
and file anchors are numerically equal.

File-good intervals remain attached to acquired bins. **Manual** good-window
endpoints follow effective timing as signed offsets; omitted endpoints retain
mapped file bounds. Only common observed support is reduced. See
:doc:`/reference/detector_grouping`, :doc:`/reference/project_files`, and
:doc:`/explanation/reduction_pipeline`.

For **Background run…**, sample and reference retain independent timing
requests. Changing sample timing cannot overwrite a reference's File, Manual,
or Auto-detect anchors. The same whole-bin mapping and exact-origin residual
records apply to F/B and grouped spectral/MaxEnt inputs. Native bin widths and
required observed support must be compatible at subtraction; raw loading stays
available. See :doc:`backgrounds` for reference replay and compatibility details.

Compared with other programs
----------------------------

WiMDA's **Search for T0** finds a grouped maximum, including for pulsed data.
musrfit 1.12.1 ``musrt0 -g`` writes detector prompt maxima. Unique synthetic
maxima at bins 20 and 31 agree exactly with Asymmetry's continuous feature.
Asymmetry's leading half-height feature for pulsed data is a different choice;
a maximum comparison does not validate physical pulse-centre accuracy.

Mantid 6.16.1 ``MuonPreProcess`` applies a supplied timing table and shifts
spectrum coordinates. That operation tests timing application, not automatic
feature detection. Whole-bin Asymmetry alignment and Mantid's retained shifted
coordinates agree only when selected observations and origin conventions match.
The per-selected-success safeguard above is an Asymmetry request rule; no claim
is made that all three programs enforce it. See
:ref:`comparison-selected-timing` for conditions and limits.

References
----------

* S. J. Blundell, R. De Renzi, T. Lancaster, and F. L. Pratt, *Muon Spectroscopy:
  An Introduction* (Oxford University Press, Oxford, 2022), time-zero calibration.
* A. Suter and B. M. Wojek, Phys. Procedia **30**, 69 (2012).
* `musrfit user manual <https://lmu.web.psi.ch/musrfit/user/html/user-manual.html>`_.
* `Mantid MuonPreProcess <https://docs.mantidproject.org/v6.16.0/algorithms/MuonPreProcess-v1.html>`_.
* `WiMDA <https://shadow.nd.rl.ac.uk/wimda/>`_.
