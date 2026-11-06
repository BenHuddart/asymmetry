Asymmetry in the μSR software landscape
=======================================

This page compares Asymmetry with **WiMDA**, **musrfit**, and **Mantid**.
The count-reduction guide below covers reviewed grouping, calibration-policy,
timing, and loading conventions. Its agreement statements apply to the named
versions, observations, and settings; they do not certify every analysis method
or every supported file.

The intended reader is a μSR practitioner who already uses one of
WiMDA, musrfit, or Mantid and wants to know where Asymmetry fits.

For end-to-end analyses analogous to the typical workflows in
those tools, see :doc:`/workflows/index`.

The reference programs
----------------------

.. list-table::
   :header-rows: 1
   :widths: 18 22 35 25

   * - Tool
     - Authors / facility
     - Language / stack
     - Strength
   * - **WiMDA**
     - Francis Pratt (ISIS)
     - Object Pascal (Delphi)
     - Legacy ISIS standard; rich frequency-domain tools (MaxEnt,
       moments, eigenvalue spectral estimator)
   * - **musrfit**
     - A. Suter, B. Wojek (PSI)
     - C++ / ROOT / Minuit2
     - Hand-editable ``.msr`` workflow files; largest built-in theory
       library (~34 functions); robust Minuit2 fitting
   * - **Mantid**
     - ISIS + ORNL community
     - C++ core / Python GUI / matplotlib
     - Workspace-based pipeline; unique ALC interface; broadest set
       of specialist muonium fit functions
   * - **Asymmetry**
     - Asymmetry contributors
     - Python / PySide6 / matplotlib
     - Modern Python stack; Fit Wizard (AICc-ranked); composite-model
       expression syntax; interactive parameter trending

.. _reviewed-count-reduction:

Reviewed count-reduction conventions
------------------------------------

Start with :doc:`reduction_pipeline` for the standard F/B operation order.
The closest reference depends on the operation, rather than on a single
preferred program. The detailed contracts and versioned comparisons are linked
below; different defaults must be matched before judging numerical agreement.

.. list-table:: Conventions and meaningful differences
   :class: reduction-conventions
   :header-rows: 1
   :widths: 24 36 40

   * - Operation
     - Asymmetry convention / closest comparison
     - Meaningful difference and detail
   * - Membership and F/B pairs
     - Explicit selections, like Mantid detector IDs and musrfit histogram lists
     - Unique members and independent disjoint pairs are Asymmetry choices;
       reuse across projections remains valid. See :ref:`comparison-membership`.
   * - Calibration policies
     - Stored versus explicit estimates, like WiMDA's stored α and Estimate;
       integral α uses the Mantid AlphaCalc convention
     - Enabled automatic policies use current inputs; fixed and one-shot values
       stay fixed. See :ref:`comparison-calibration-policies`.
   * - Alignment and NeXus width
     - Whole-bin alignment and declared-resolution fixed-bin counts, closest
       to WiMDA/musrfit
     - Selected exact origin and residuals are retained; Mantid also preserves
       coordinates and offers common-grid rebinning. See :ref:`comparison-timing`.
   * - PSI intervals
     - Selected file-good intersection; musrfit is the primary PSI comparator
     - Explicit musrfit endpoints and omitted defaults differ; WiMDA uses a
       shared interval. See :ref:`comparison-psi-intervals`.
   * - ROOT calibration and coverage
     - Legacy v4/v5 and modern layouts use common reduction; closest to musrfit
     - Stored bank selection, PPC uncertainty, and the directory-case limit
       have distinct scopes. See :ref:`comparison-root`.

Agreement means agreement for matched observations and uncertainty conventions,
within the stated storage or export precision. Explained convention differences
are identified separately. Unverified paths and unresolved uncertainty
comparisons remain limits; acceptance of a selection alone does not establish
correct covariance or physical calibration. Versioned WiMDA numerical checks
are limited to the background cases explicitly identified below. Other WiMDA
descriptions compare workflows without claiming executable numerical agreement;
the `author's homepage and manual <https://shadow.nd.rl.ac.uk/wimda/>`_
provide general context.

.. _comparison-membership:

Detector membership and exclusions
----------------------------------

Asymmetry groups unique acquired-channel ordinals and applies exclusions without
rewriting raw histograms. Missing preset channels and empty effective groups
make reduction unavailable. Typed overlapping ranges form a set union;
structured duplicate membership is rejected. These ordinal IDs do not certify
physical detector identity across reordered files.

Mantid's `MuonGroupingCounts documentation
<https://docs.mantidproject.org/v6.16.1/algorithms/MuonGroupingCounts-v1.html>`_
specifies detector-ID selection. The `musrfit manual's RUN grouping
<https://lmu.pages.psi.ch/musrfit-docu/user-manual.html#msr-forward-backward>`_
selects histogram numbers. Mantid 6.16.1 sums synthetic channels [10, 20]
and [50, 60] to [60, 80],
retaining supplied errors in quadrature within relative/absolute tolerance
:math:`10^{-12}`. musrfit 1.12.1 (``708f777b``, ROOT 6.40.04), with
corrections off, 0.1 μs bins, and packing one, exports 125000 counts/bin
from constant channels 100000 and 25000; Poisson errors agree within relative
:math:`5\times10^{-6}`, set by six-significant-digit text precision. Both
reject the tested missing member and count a repeated member twice.
Asymmetry's unique
membership rule deliberately rejects that repetition. The tests do not establish
pair independence or equivalence of geometry, timing, or calibration methods.

Asymmetry requires disjoint effective acquired channels within each independent
F/B pair, after exclusions. Native Mantid 6.16.1
``MuonPairingAsymmetry`` rejects tested identical lists but accepts partial
overlap; see its `algorithm documentation
<https://docs.mantidproject.org/v6.16.1/algorithms/MuonPairingAsymmetry-v1.html>`_.
Native musrfit 1.12.1 accepts the tested identical and partially overlapping
pairs. These observations describe request acceptance, not a covariance-aware
error guarantee. Asymmetry retains EMU vector channel reuse across projections
and overlapping auxiliary groups, and never infers identity from equal counts.

WiMDA's **Group Table Edit** forms groups from numbered histogram channels.
A repeated entry contributes again, and a histogram can contribute to both
forward and backward groups. WiMDA can regenerate group membership when the
requested histogram range is unavailable. Asymmetry also checks structured
membership for repeated or missing requested
channels, and requires nonempty disjoint effective channels for independent
F/B analysis. These checks preserve an unweighted authored selection and the
independent-count uncertainty model. Failure concerns the requested reduction:
raw histograms and saved requests remain available for inspection and repair.
See :doc:`/reference/detector_grouping` for the membership contract.

WiMDA can exclude physical-detector contributions before combining them into
mapped histograms. Asymmetry excludes acquired histogram channels without
rewriting the loaded data. When an acquired channel combines several physical
detectors, excluding it removes the whole channel. Neither program can recover
hidden constituent observations from a file that already aggregates them.
See :doc:`/reference/data_reduction/detector_exclusion` for that distinction.

.. _comparison-calibration-policies:

Stored and automatic calibration policies
-----------------------------------------

Asymmetry preserves fixed and one-shot calibrated coefficients during regrouping.
Enabled automatic policies use current inputs: ``per_run_estimate`` α is the
integral ratio of the current corrected grouped counts over the good window.
It is distinct from repeating an optimising α fit. β has fixed and calibrated
policies, without that automatic ratio policy. See
:doc:`/reference/data_reduction/alpha_calibration` and
:doc:`/reference/data_reduction/beta_calibration` for the contracts.

WiMDA keeps α stored during ordinary regrouping; **Estimate** is a separate
action, while enabled background processing is evaluated when groups are
rebuilt. Mantid's `AlphaCalc v1
<https://docs.mantidproject.org/v6.13.1/algorithms/AlphaCalc-v1.html>`_
computes the integral efficiency ratio as a separate operation; it is the
closest convention for Asymmetry's automatic ratio, rather than a claim that
Mantid automatically refits α after every grouping edit. musrfit specifies α/β
through its `RUN/model configuration
<https://lmu.pages.psi.ch/musrfit-docu/user-manual.html#msr-run-block>`_.
These policy distinctions do not establish numerical equivalence between
calibration estimators. The cited Mantid documentation version is distinct
from the 6.16.1 executable used in the numerical comparisons below.

Asymmetry also checks that automatic α has finite positive F/B integrals and
a finite positive ratio. An undefined estimate makes that reduction unavailable
and editable; it does not replace the stored request with unity. Explicit
fixed unity remains valid. This protects the ratio's mathematical inputs,
without claiming that all reference workflows use the same guard.

.. _comparison-timing:

Integer timing alignment
------------------------

WiMDA aligns whole raw bins. Asymmetry also retains integer shifts, but resolves
its exact reference and common observed support from the effective selected
channels after exclusions. The maximum selected anchor defines the reference;
acquired-channel ordinal breaks ties. Fractional residuals and their range are
recorded in μs, without interpolating counts or assigning timing uncertainties.

WiMDA and musrfit use declared resolution for fixed-bin NeXus count analysis.
Asymmetry follows that interpretation, retaining its exact origins and
bin-centre convention. Supplied auxiliary coordinates remain inspectable and
do not override a usable resolution or infer an extra acquisition-start shift.
musrfit 1.12.1 (``708f777b``, ROOT 6.40.04) exports identical arrays for regular,
cumulative-float32, and shortened-terminal synthetic HDF5 coordinates with
unchanged counts and resolution. Mantid 6.16.1's legacy loader retains supplied
coordinate variations in the compared HDF4 inputs while retaining their counts
exactly. These cases do not certify physical t0 or terminal-bin exposure;
:doc:`/reference/loading_data` states the fixed-bin and axis-fallback conventions.

Asymmetry also checks that selected channels have compatible widths before
integer alignment, protecting its whole-bin count transport. A failed request
leaves raw histograms available. When no usable resolution exists, Asymmetry
also checks the selected fallback axis's shape, finiteness, monotonicity, and
uniform spacing to obtain a physical width. Failure then concerns calibrated
loading, because inventing a width would invent physical coordinates.
Asymmetry also checks detector/period timing shapes and redundant integer/exact
t0 agreement: unusable optional timing uses the remaining calibration, while
an exact/integer disagreement is advisory and keeps integer timing for the
affected detectors without discarding coherent exact origins elsewhere. Missing
or unusable file t0 uses the existing search policy, and two-period selection
retains the selected period's own calibration status. These
checks apply to the adopted interpretation, rather than requiring all auxiliary
fields to conform. They do not imply that other programs omit such checks in
every workflow.

The recorded musrfit 1.12.1 (``708f777b``, ROOT 6.40.04) type-0 integer grouping
case uses 0.1 μs bins, ``forward 1 2``, ``t0 2 4``, ``data 1 9``, packing one,
and no corrections. Eight selected counts agree exactly with Asymmetry, with
timestamps agreeing within :math:`10^{-12}` μs. This is a count and coordinate
check, not physical t0 certification. Mantid 6.16.1 keeps per-spectrum shifted
X coordinates; direct grouping requires compatible grids, while explicit
rebinning can redistribute counts onto common edges. See the
`Mantid muon reduction tutorial
<https://docs.mantidproject.org/v6.9.0/tutorials/muon_GUI_course/basics_of_data_reductions.html>`_
and `musrfit manual <https://lmu.pages.psi.ch/musrfit-docu/user-manual.html>`_.
The tutorial version is separate from the recorded executable version.
:doc:`/reference/detector_grouping` gives the selected bins and output mapping.

.. _comparison-file-manual-windows:

File bounds and authored relative windows
-----------------------------------------

Asymmetry keeps acquired detector bounds on raw bins and maps them through
current selected-channel alignment. **From file** preserves the raw
observations under a uniform common t0 change; **Manual** stores each supplied
endpoint as a signed offset from effective t0. An omitted endpoint retains its
mapped file limit. This separates file facts, timing, and an authored interval.

The closest comparator for preserved observations is musrfit's explicit
``data`` channels, with timing specified separately in the
`RUN block <https://lmu.pages.psi.ch/musrfit-docu/user-manual.html#msr-run-block>`_.
For tagged raw bins 11–14, musrfit 1.12.1 (``708f777b``, ROOT 6.40.04), packing
one, and no corrections, unchanged ``data 11 15`` preserves counts and Poisson
errors when integer ``t0`` moves from 10 to 15. Moving the explicitly supplied
interval to ``data 16 20`` instead matches Asymmetry Manual offsets 1 and 4.
Counts and errors agree within the native ASCII precision (relative tolerance
5 × 10⁻⁶). An exact Asymmetry origin of 0.2074 μs differs from musrfit's
integer-centre origin 0.21 μs by a constant 0.0026 μs; changing t0 by five
0.02 μs bins preserves that difference. Fit-time limits and omitted ``data``
defaults are separate operations. Manually matched channels do not establish
an automatic relative-window policy in musrfit's GUI.

Mantid 6.16.1 ``CreateWorkspace`` and
`CropWorkspace <https://docs.mantidproject.org/v6.16.1/algorithms/CropWorkspace-v1.html>`_
provide the closest physical-coordinate comparison. With the same 32 raw
observations, supplied Poisson variances, and 0.02 μs width, translated axes
and explicitly matched crop limits retain identical counts and errors.
Moving both raw crop endpoints by five bins selects 16–19 instead of 11–14.
Centres agree within 10⁻¹⁵ μs. These constructed workspaces test the selected
operation, not NeXus loader defaults or Mantid's GUI policy.

WiMDA's **Use File values** controls timing and a shared interval together.
The interval uses the first histogram's file limits, with a zero-end fallback.
Its editable start is a t0 offset and its end is an absolute last bin.
Asymmetry's independent timing/window selectors and two stored relative
endpoints therefore express a different policy. No WiMDA executable comparison
is claimed; the `WiMDA documentation <https://shadow.nd.rl.ac.uk/wimda/>`_
describes the grouping workflow. No numerical disagreement remains in the
matched native synthetic operations above; automatic-default equivalence and
WiMDA numerical agreement remain unverified.

Asymmetry also checks that explicit manual endpoints are ordered and within
common acquired support. This protects authored selection without inventing
one-bin observations. The failed calculation remains editable and raw data
remain available; it does not reject a facility file. This safeguard does
not assert that all reference programs implement the same validation.

.. _comparison-nexus-good-times:

NeXus physical good-time selection
----------------------------------

The `muon NeXus definition, version 2 revision 8, p. 51
<https://nmi3.eu/nexus_instrument_definitions_for_muon_data-get_file=nexus_instrument_definitions_for_muon_data.pdf.pdf>`_
defines physical good times relative to ``raw_time``. Asymmetry adopts inclusive
raw-centre membership on its nominal fixed-bin grid and retains the resulting
per-detector bounds through reduction and project reopening. Integer attributes retain
precedence for their corresponding endpoints. The definition does not settle
every writer's integer base or sub-bin rounding convention.

In Mantid 6.16.1, a histogram workspace with the same 32 tagged bins, 0.02 μs
width, and 0.2074 μs origin, cropped to raw limits 0.22–0.30 μs, selects the
same four raw observations, centres, and Poisson errors. This comparison uses
``CreateWorkspace`` and
`CropWorkspace <https://docs.mantidproject.org/v6.16.1/algorithms/CropWorkspace-v1.html>`_;
it does not certify every NeXus
loader default. Mantid retains supplied coordinate variation, whereas
Asymmetry's declared-resolution grid is uniform. A physical threshold near a
stored edge can therefore select different observations. Matched selections
must be established before comparing reduction values; legacy first-good
metadata does not by itself establish equal defaults.

musrfit 1.12.1 (revision 708f777b, ROOT 6.40.04), with packing one and explicit
data channels ``11 15``, selects tagged raw bins 11–14. Counts and Poisson
errors agree within the ASCII export's six-significant-digit precision
(relative tolerance 5 × 10⁻⁶). With an explicit integer t0 of 10 channels,
its timestamps agree with Asymmetry's centre origin of 0.21 μs. The compared
single-histogram path uses an integer channel anchor; its fractional ``t0``
request does not reproduce Asymmetry's exact 0.2074 μs calibration. This is a
timing-convention difference, not a count-selection disagreement. Physical
fit-time limits and omitted data defaults are separate controls; no native
HDF4 agreement is claimed for the build with HDF4 support disabled.

WiMDA's **Use File values** uses a shared first-histogram interval and a
zero-end fallback. Its NeXus physical-time conversion uses channel rounding
and t0-related defaults, rather than Asymmetry's per-detector inclusive-centre
intersection. The common choice is to retain whole observations on the raw
acquisition grid; sub-bin thresholds and omitted endpoints need not select the
same bins. No WiMDA executable numerical comparison is claimed. See the
`WiMDA documentation <https://shadow.nd.rl.ac.uk/wimda/>`_ for the workflow and
the `musrfit manual <https://lmu.pages.psi.ch/musrfit-docu/user-manual.html>`_
for channel and fit-time controls.

Asymmetry also checks that the selected file-good intervals have nonempty
common acquired support before reducing independent counts. An unavailable
request retains the acquired histograms and editable grouping. Optional field
shape or unit diagnostics do not impose whole-file rejection when another
usable interpretation remains; see :doc:`/reference/loading_data`.


.. _comparison-root:

PSI legacy ROOT calibration
---------------------------

Asymmetry adopts musrfit's numeric ``TLemRunHeader`` v4/v5 calibration for legacy
LEM ROOT: ns resolution and zero-based per-detector t0, followed by the shared
count reduction. musrfit 1.12.1 (``708f777b``, ROOT 6.40.04) gives identical
selected counts, times, and errors for equivalent synthetic BIN, legacy ROOT,
and modern ``TFolder`` inputs. Asymmetry's F/B asymmetry and Poisson errors
agree within the six-significant-digit ASCII output precision (relative
:math:`5\times10^{-6}`); timing agrees within relative :math:`3\times10^{-8}`
from BIN float32 versus legacy decimal ns storage. Asymmetry reports percent
asymmetry while musrfit reports fractions.

The compared selections are forward raw bins 20–59 and backward raw bins
23–62 with t0 4/7, packing one, alpha=beta=1, zero background, and no deadtime.
Different endpoint requests need not select the same observations. Extra ROOT
container bins remain in the loaded histograms. The generated ``TDirectory``
variant uses ``decayAnaModule``, while this musrfit reader requests
``DecayAnaModule``; Asymmetry reads that representation without claiming a
native numerical comparison for it. WiMDA and Mantid coverage of this legacy
class is unverified. The legacy v5 agreement above uses synthetic files;
the modern acquired examples below extend a different part of ROOT coverage.
Two acquired v4 examples in the musrfit ``708f777b`` example set extend legacy
coverage: all eight decay arrays retain their 66,601 stored bins, numeric
0.1953125 ns width, identities, and available calibration. Header slots 0–3
calibrate both bank 00 (NPP) and bank 20 (PPC); musrfit's ``ROOT-NPP`` and
``ROOT-PPC`` select the respective stored bank. Distinct-anchor checks also
confirm that the timing follows header slots rather than histogram ordinals.
The `musrfit LEM format description
<https://lmu.pages.psi.ch/musrfit-docu/file-formats.html>`_ names these banks.

With channels 1/3, raw bins 3500–3539, packing one, alpha=beta=1, no added
correction, and zero background, summed counts agree exactly. Times, F/B
values, and counting errors agree within relative :math:`5\times10^{-6}`
ASCII precision, allowing absolute :math:`5\times10^{-7}` for F/B values
near zero. Both file t0 3419 and explicit t0 3429 are compared. The PPC check
uses stored observations and the declared square-root-count error convention;
it does not validate pileup physics, likelihood, or corrected-count covariance.
The banks must not be combined as independent observations. The initial pairs
in these two v4 files select bank-00 left/right; this is no general default
geometry certification. Version 5 extra-bank calibration is outside this check.

Serialized v4 regression inputs remain optional external examples. Always-on
normalisation tests cover sparse/reordered identities and unavailable anchors,
but no independently authored serialized-v4 CI fixture is supplied. Loading
checks over 37 external ROOT files establish compatibility and retained data,
not numerical certification of every layout or default reduction. See
:doc:`/reference/loading_data` and the
`musrfit RUN settings <https://lmu.pages.psi.ch/musrfit-docu/user-manual.html#msr-run-block>`_.

.. _comparison-psi-intervals:

PSI good-data intervals
-----------------------

Asymmetry adopts zero-based inclusive raw-bin bounds for file-good intervals
and intersects them after whole-bin alignment over the selected detectors.
BIN upper values at or beyond histogram length are clipped to the final
stored index; interior values are retained. MDU M3/T4/T5 markers are converted
at their hardware scale before clipping. Modern ROOT per-detector bounds use
the same inclusive representation; legacy LEM v4/v5 without good fields defaults
from each detector's t0 through its final stored bin. Container-added bins
remain stored, independently of their counts or the selected interval.

musrfit 1.12.1 (``708f777b``, ROOT 6.40.04) explicit ``data`` intervals at
packing one have an exclusive upper bound. For matched tagged BIN and MDU
M3/T4/T5 selections, summed counts agree exactly, raw-bin identities agree,
and exported Poisson errors agree within relative :math:`5\times10^{-6}`
(six-significant-digit ASCII precision). Timestamp comparisons at that export
precision use the same tolerance. The BIN and ROOT timing comparison above
uses its separate storage-precision tolerance.

Omitted ``data`` does not mean the same default as Asymmetry's file-good
intersection. For the compared 128-bin BIN with t0 4, musrfit's single-channel
fallback selects raw 5..127 whether the supplied last-good field is 60, 127,
or 128. The equivalent 129-bin legacy/modern folder ROOT storage also includes
raw bin 128, a retained terminal zero. Asymmetry's modern ROOT good fields
can exclude that bin; legacy defaults can include it. Neither a terminal zero
nor equal header integers establishes equivalent observations.

For the synthetic MDU marker cases, omitted-data single-channel fallbacks
select raw 23..127 for M3 (t0 7, 0.625 ns bins) and 107..127 for T4/T5
(t0 5, approximately 0.09765625 ns bins). These differ from the converted
file-good intervals 35..80 and 26..60 respectively.

Tested on musrfit 1.12.1 example ROOT files (revision ``708f777b``): modern
LEM with a terminal good bound, and FLAME and HiFi with interior good bounds.
Explicit requests matched to Asymmetry's file-good observations give exact
summed counts and raw-bin identities. Timestamps and positive-count F/B errors
agree within relative :math:`5\times10^{-6}`; asymmetry exports also allow
absolute :math:`5\times10^{-7}` near zero because of ASCII precision.
Omitted native data intervals select different observations in all three
examples. The HiFi initial pair includes an empty channel; its error comparison
uses an additional explicitly selected populated pair. Zero-count uncertainty
conventions are outside that positive-count comparison. For one-sided bins,
Asymmetry retains a unit-fraction uncertainty sentinel, while the native export
supplies count-dependent finite errors. This is selected
numerical coverage of three acquired examples, not agreement for every example
file or every ROOT layout.

WiMDA's **Use File values** uses the first histogram's good interval as the
grouping reference, with zero last-good requesting histogram length. Its
one-based histogram coordinates and shared end differ from Asymmetry's
selected per-detector intersection. No WiMDA executable numerical agreement
is claimed for these PSI boundaries. Mantid 6.16.1 ``LoadPSIMuonBin`` retains
all 128 tagged BIN counts, exposes good-data limits in time units, and does
not crop the raw workspace to those limits. This loading result does not
establish Mantid's default analysis selection. MDU and ROOT interval reductions
in Mantid, WiMDA ROOT, and native musrfit numerical reduction of the generated
lower-case-directory representation remain unverified here.

Asymmetry also checks that the selected channels have common acquired support
and a nonempty intersection for a file-good reduction of independent counts.
This prevents alignment padding from becoming observations. Raw histograms
remain available when the initial pair is unavailable; the grouping request
can be edited. These checks do not certify bin exposure from count values.
See :doc:`/reference/loading_data`, the `musrfit manual
<https://lmu.pages.psi.ch/musrfit-docu/user-manual.html#msr-run-block>`_,
`Mantid LoadPSIMuonBin
<https://docs.mantidproject.org/nightly/algorithms/LoadPSIMuonBin-v1.html>`_,
and the `WiMDA manual <https://shadow.nd.rl.ac.uk/wimda/>`_.

.. _comparison-deadtime-units:

NeXus deadtime units
--------------------

WiMDA converts explicitly declared microsecond and nanosecond deadtimes to
its internal nanosecond scale in its NeXus-v2 path. Asymmetry follows this
physical interpretation, converting once to its canonical μs coefficients at
loading. Thus 20 ns and 0.02 μs give identical corrected counts, including
signed coefficients. WiMDA's legacy v1 path uses a fixed μs convention;
Asymmetry retains that interpretation for v1/HDF4. Missing and other unit
declarations retain Asymmetry's stored-value-as-μs behaviour. This limited
choice does not assert agreement on missing-unit defaults.

The compared **musrfit 1.12.1** (``708f777b``, ROOT 6.40.04) NeXus-v2 path
uses stored coefficient values as μs regardless of the tested ns declaration.
Asymmetry intentionally differs on explicitly declared ns values. With
100000 counts, 0.1 μs bins, 100000 frames, and a 0.02 μs coefficient,
the analytic inverse gives 125000 counts; an equivalent 20 ns declaration
has the same meaning in Asymmetry. The
`musrfit manual <https://lmu.pages.psi.ch/musrfit-docu/user-manual.html>`_
describes its file deadtime correction option; the
`WiMDA manual <https://shadow.nd.rl.ac.uk/wimda/WiMDA%20Manual.doc>`_
provides the correction and calibration context.

Numerical WiMDA agreement and native **Mantid 6.16.1** loader unit equivalence
are not established. The coefficient interpretation does not change the
correction formula or retained measured/supplied counting variance.

.. _comparison-deadtime-availability:

Optional file deadtime calibration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

WiMDA searches for an external calibration when internal coefficients are
absent, then informs the user and turns correction off if none is found.
Asymmetry follows that whole-operation uncorrected result with an explicit
warning, preserving requested File versus actual ``skipped`` and making no
automatic calibration-file search. It applies the same policy to incomplete
required tables, avoiding a mixture of corrected and uncorrected histograms.
A genuinely supplied complete zero table is an available identity.

musrfit 1.12.1 warns and leaves histograms without coefficients uncorrected;
other histograms can still be corrected. Mantid 6.16.1 ``ApplyDeadTimeCorr``
accepts partial tables, its Muon Analysis GUI requires matching table row
counts, and requested legacy-loader table paths can reject short tables.
These are different workflows, rather than a universal missing-table policy.
The musrfit missing, zero, and short-table count exports agree with these
statements; Asymmetry deliberately skips the whole operation for a short table.
Public context is provided by the `WiMDA documentation <https://shadow.nd.rl.ac.uk/wimda/>`_,
`musrfit manual <https://lmu.web.psi.ch/musrfit/user/html/user-manual.html>`_,
and `Mantid correction documentation <https://docs.mantidproject.org/v6.16.0/algorithms/ApplyDeadTimeCorr-v1.html>`_.

Asymmetry also checks non-zero correction exposure and bin width and rejects
saturated or nonfinite inverses over the required F/B observations. Standalone
full-input preparation checks all supplied histograms.
Those calculation failures remain errors; missing optional calibration does
not prevent raw loading or other valid enabled operations.

.. _comparison-deadtime-scope:

Required F/B deadtime observations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

WiMDA traverses requested groups and bins when grouping with deadtime
correction. Mantid 6.16.1's standard ``MuonPreProcess`` corrects before cropping;
its numerical algorithms also permit callers to crop before ``ApplyDeadTimeCorr``.
The compared musrfit 1.12.1 path corrects selected histograms but visits their
raw arrays. These workflows do not share a universal observation domain.

Asymmetry deliberately uses the exact required observations for F/B reduction,
grouped raw-count fits, grouped Fourier inputs, and MaxEnt: the requested
detectors and contributing raw bins, plus active range, tail-fit, and
reference-background inputs. An unused saturated bin or detector cannot block
the calculation, and a missing unused File coefficient cannot suppress it.
A required missing coefficient skips the entire participating sample/reference
operation with a warning; a required pole remains an error. Original detector
ordinals and raw-bin coordinates are preserved. See
:doc:`/reference/detector_grouping` for the inverse and uncertainty convention.
One joint count fit, selected Fourier average, or MaxEnt reconstruction is
one cohort for File fallback. A final exclusion does not remove samples used
by earlier normalisation, background estimation, conditioning, or phase fits.
In particular, MaxEnt interior exclusions follow its weighted-baseline estimate;
FFT preparation retains the input needed by its upstream conditioning.
F/B-asymmetry FFT uses the F/B preparation directly; its group selector is inert.
Standalone full-input preparation still checks every supplied observation.
These preparation choices do not establish identical solver inputs or statistics
across all three reference programs.

Public workflow context: `WiMDA documentation <https://shadow.nd.rl.ac.uk/wimda/>`_,
`musrfit manual <https://lmu.web.psi.ch/musrfit/user/html/user-manual.html>`_, and
`Mantid MuonPreProcess <https://docs.mantidproject.org/v6.16.0/algorithms/MuonPreProcess-v1.html>`_.

Signed deadtime coefficients
----------------------------

Asymmetry preserves finite signed calibration coefficients. A fitted negative
coefficient reduces raw counts under the non-paralysable inverse; it is an
empirical fit parameter, distinct from a detector's physical recovery time.
The published ISIS silver-calibration method and the numerical domain are
explained in :doc:`/reference/detector_grouping`.

A native **Mantid 6.16.1** check of ``ApplyDeadTimeCorr`` v1 used raw counts
[5, 20], 1 µs bins, 10 good frames and coefficients −0.1, 0 and +0.1 µs.
Asymmetry agrees with Mantid and the analytic inverse to relative tolerance
:math:`10^{-12}`. ``CalMuonDeadTime`` v1 recovered noiseless signed synthetic
coefficients −0.02, 0 and +0.02 µs within :math:`2\times10^{-12}` µs; its
`published examples <https://docs.mantidproject.org/v6.13.1/algorithms/CalMuonDeadTime-v1.html>`_
also include negative fitted values. Asymmetry's rational per-detector fit
agrees on these noiseless inputs within :math:`10^{-9}` µs. The inputs are
:math:`N_0=10000` counts/bin, lifetime 2.1969811 µs, one million frames,
0.016 µs bins with edges :math:`0.496+k\times0.016` µs
(:math:`k=0,\ldots,282`), and fit range 0.5–5 µs. Asymmetry weights raw-count
residuals by Poisson standard deviations, while Mantid fits the transformed
regression; this is not a claim of noisy-data estimator equivalence.
Calibration noise and model mismatch can
produce negative fitted coefficients; the sign does not diagnose an individual
calibration's quality.

**WiMDA** preserves signed coefficients in its standard ISIS loading,
manual-input, and correction paths, as do its calibration fit results. No
binary numerical agreement or behaviour of the optional extended models is
established here. The `WiMDA Manual, v1.2 (2005), Sec. 5.9 and Appendix B
<https://shadow.nd.rl.ac.uk/wimda/WiMDA%20Manual.doc>`_ describes the calibration
workflows.

.. _comparison-deadtime-calibration:

Deadtime calibration methods
-----------------------------

A mean response and an estimator are separate choices. Asymmetry's dedicated
Cal/Estimate fits use the rational forward mean corresponding to the inverse
above. They minimise raw-count weighted squared residuals separately for each
detector, with a free non-negative amplitude, signed coefficient, fixed muon
lifetime, and no count background or polarisation term. Weights use the observed
count with a floor of one; supplied variance is not used by this calibration
fit. The published transformed ISIS line is algebraically equivalent to the
mean, but fitting that line is a different objective. Source histograms and
windows also matter; :doc:`/reference/detector_grouping` distinguishes the
Grouping buttons, scripted API, and MaxEnt caller.

**Cal** retains per-detector coefficients. **Estimate** averages those fitted
coefficients arithmetically and broadcasts the mean. **WiMDA** pools early-time
detector observations for its uniform estimate and also offers individual
detector calibration. Its dedicated fitting mean differs from Asymmetry's
rational forward mean. The `WiMDA manual, v1.2, section 5.9 and Appendix B
<https://shadow.nd.rl.ac.uk/wimda/WiMDA%20Manual.doc>`_ gives workflow context;
no executed numerical WiMDA agreement is established here.

For an independent illustration using Asymmetry's rational model, two detectors
with the same 0.02 µs coefficient and equal rates give 0.02 µs both by averaging
fits and by fitting their mean observations. With a 4:1 amplitude ratio,
averaging the independent fits still gives 0.02 µs, while fitting the mean
observations gives about 0.02636 µs. This noiseless illustration uses amplitudes
100000 and 25000 counts/bin at time zero, lifetime 2.1969811 µs, 100000 frames,
0.05 µs bins, and centres 0.525–6.475 µs. It demonstrates noncommuting operations
for a nonlinear response; it is our own pooling experiment, not WiMDA output
or a recommendation to change the estimator.

**Mantid 6.16.1 CalMuonDeadTime** v1 fits a transformed line separately for each
spectrum, using ``goodfrm`` exposure. Its `documented regression
<https://docs.mantidproject.org/v6.13.1/algorithms/CalMuonDeadTime-v1.html>`_
and Asymmetry's rational fit agree on the signed noiseless inputs above.
They do not minimise the same noisy-data objective. A bounded synthetic
sensitivity comparison using matched observations, exposure, and selected bins
shows method and window dependence under noise, a small constant background,
and weak polarisation. Independent Poisson draws here are a chosen sensitivity
model, not certification of physical deadtime statistics, estimator optimality,
or confidence-interval coverage. A teaching silver-run comparison can show
behaviour under matched selection; without known detector truth it cannot
rank accuracy.

**musrfit 1.12.1** (``708f777b``, ROOT 6.40.04) reports the compared
``deadtime-cor estimate`` option as unimplemented; its file correction is a
separate operation. This limited observation does not imply that every musrfit
calibration facility is unimplemented. The `musrfit manual
<https://lmu.pages.psi.ch/musrfit-docu/user-manual.html>`_ describes this option
for pulsed/ISIS NeXus data.

The matched noiseless calibration is agreement under stated conditions;
pooling, objectives, and exported count-error conventions are explained
method differences. Physical continuous-source calibration and corrected ROOT
bank statistics remain unresolved by these comparisons. Generic availability
of Cal does not establish PSI applicability or a new exposure interpretation.

Asymmetry's dedicated calibration errors are conditional Hessian errors from
its chosen objective. Estimate combines independent detector errors in
quadrature and divides by their number; it does not estimate detector scatter.
These errors are distinct from the measured/supplied counting variance retained
during correction below. Coefficient and exposure uncertainty are not propagated
through that correction. A successful count-domain residual-DT₀ fit also need
not supply a complete calibration: skipped File correction or unknown unused
coefficients leaves it usable with a warning and no promotion. Explicit Off
retains its supported zero-base behaviour; see
:doc:`/reference/count_domain_fitting`.

Deadtime count errors
---------------------

The deadtime inverse and its count-error convention are separate choices.
Executed synthetic checks distinguish two reference programs:

* **Mantid 6.16.1**, **ApplyDeadTimeCorr**, retains supplied workspace errors.
  With 1 µs bins, 100 frames, counts [100, 25], and a 0.2 µs coefficient,
  corrected values are [125, 26.315789473684212]; errors [10, 5] remain
  [10, 5]. Supplying [7, 3] instead retains [7, 3]. Signed coefficients,
  unequal detector rates, grouping, and bin merging preserve that behaviour.
* **musrfit 1.12.1**, revision ``708f777b`` with ROOT 6.40.04, exports
  single-histogram errors from corrected counts. With count-rate scaling
  disabled, 100000 counts/bin, 0.1 µs bins, 100000 frames, and a 0.02 µs
  coefficient give 125000 counts and error 353.553. The signed grid
  [−0.02, 0, 0.005, 0.02, 0.05] µs, single and unequal grouped detectors
  (100000 and 25000 counts/bin), and packing factors one and four agree
  with this convention to relative tolerance :math:`5\times10^{-6}`, set
  by the ASCII export precision.

Asymmetry retains measured counting variance, or supplied independent variance,
through deadtime preparation, grouping, background subtraction, and binning.
It agrees with Mantid's supplied-error convention on the recorded synthetic
inputs and differs from the tested musrfit exported-error convention. Calibration
coefficients and frame exposure are fixed in this propagation; the error bars
do not include their uncertainty or physical deadtime-induced correlations.
These comparisons do not assert equality of low-count asymmetry formulas,
background treatment, or fit objectives.
The formulas and limitations are in :doc:`/reference/detector_grouping`.

.. _comparison-reference-timing:

Reference timing and native coordinates
---------------------------------------

Asymmetry resolves sample and selected reference clocks independently, retaining
whole-bin shifts and fractional file-origin residuals in F/B and grouped
Fourier/MaxEnt preparation. Asymmetry also checks compatible selected native bin
widths and required reference support before subtraction. It neither resamples
counts nor restricts raw file loading.

WiMDA 1.326 **FileBG** uses a shared active channel shift. The synthetic NeXus
check uses sample counts 200 per bin with t0 zero, reference counts
``10 + rawbin`` with file t0 zero or two bins, 0.016 μs native width, equal
200-frame exposures, and scale one. WiMDA exports raw group 1 with one detector,
bunch one, deadtime off, and Auto/Manual BG off. After restoring sample timing,
both 1999-row exports agree exactly at count-export precision, beginning
190, 189, 188, 187 counts. Reference loading can change the active timing, so
the matched comparison restores the sample first. Independent Asymmetry timing
instead starts 188, 187, 186, 185 for the two-bin reference, checked against
those tagged observations on bounded 32-bin sample support. The difference is
explained by the alignment convention.

Mantid 6.16.1 generic **Minus** rejects incompatible X arrays; this is explicit
workspace arithmetic, not an equivalent automatic Muon reference GUI workflow.
musrfit's run addition has separate t0 semantics and is distinct from FileBG
subtraction. See :doc:`/reference/data_reduction/backgrounds` for the subtraction
contract and its independent-variance assumptions.

At a glance: feature coverage
-----------------------------

Symbols: ✅ present  ◐ partial / stub  ❌ absent  ★ distinctive
strength.

.. list-table::
   :header-rows: 1
   :widths: 30 14 14 14 14

   * - Category
     - WiMDA
     - musrfit
     - Mantid
     - Asymmetry
   * - Multi-format data ingestion (NeXus, PSI BIN, MUD, ROOT)
     - ✅
     - ✅
     - ✅
     - ✅
   * - Asymmetry calculation (F/B grouping, α)
     - ✅
     - ✅
     - ✅
     - ✅
   * - Automatic phase calibration
     - ❌
     - ❌
     - ★
     - ◐
   * - Rotating Reference Frame
     - ❌
     - ◐
     - ★
     - ✅
   * - Theory function library
     - ◐ ~12
     - ★ ~34
     - ★ ~15 specialist
     - ◐ growing
   * - Static Kubo–Toyabe (ZF / LF)
     - ✅
     - ✅
     - ✅
     - ✅
   * - Dynamic Kubo–Toyabe
     - ✅
     - ✅
     - ★
     - ✅
   * - Composite-model expression syntax
     - ❌
     - ◐
     - ◐
     - ★
   * - MIGRAD / MINOS / HESSE
     - ◐ Hessian only
     - ★ full set
     - ✅ via Mantid Fit
     - ✅ full set
   * - Multi-spectrum / global fit
     - ◐ sequential
     - ✅ shared params
     - ✅
     - ✅
   * - Multi-group time-domain fit
     - ◐
     - ✅
     - ✅
     - ✅
   * - Fourier (FFT + apodisation)
     - ✅
     - ✅
     - ✅
     - ✅
   * - MaxEnt frequency reconstruction
     - ★ Burg pole-scan
     - ◐
     - ★ iterative
     - ✅
   * - Spectral moments analysis
     - ★
     - ❌
     - ❌
     - ✅
   * - Interactive parameter trending
     - ◐ table
     - ◐ ``msr2data`` CLI
     - ◐ table
     - ★ panel
   * - Avoided Level Crossing (ALC) workflow
     - ❌
     - ❌
     - ★
     - ✅
   * - Period arithmetic (pulsed data)
     - ◐
     - ❌
     - ★
     - ✅
   * - Logbook / multi-run manager
     - ✅
     - ◐
     - ✅
     - ★
   * - Synthetic data simulation
     - ★
     - ❌
     - ❌
     - ✅
   * - User-defined functions
     - ◐ DLL
     - ◐ C++ plugin
     - ◐ Mantid plugin
     - ✅ Python plugins
   * - Project files (hand-editable)
     - ❌
     - ★ ``.msr``
     - ◐ ``.mantid``
     - ✅ ``.asymp`` (JSON)
   * - Model-recommendation wizard
     - ❌
     - ❌
     - ❌
     - ★ Fit Wizard

Where Asymmetry leads
---------------------

These are the parts of the workflow where Asymmetry's
implementation is materially richer or more ergonomic than the
alternatives.

**Fit Wizard (AICc-ranked model recommendation).**
  No equivalent in WiMDA, musrfit, or Mantid. The wizard runs a
  curated portfolio of candidate models on the active dataset and
  ranks them by an information-theoretic metric (AICc by default).
  Especially valuable for new users encountering an unfamiliar
  spectrum. See :doc:`/reference/fit_wizard`.

**Composite-model expression syntax.**
  Free-form arithmetic over registered components — ``Exponential *
  Oscillatory + Constant``, with fraction-group syntax ``(...){frac}``
  for shared-amplitude bundles. musrfit's ``FUNCTIONS`` block is the
  closest analogue but is more limited; WiMDA and Mantid require
  building composite functions procedurally. See
  :doc:`/reference/composite_models`.

**Interactive parameter trending.**
  Per-run fit parameters appear as a sortable trend table that
  drives an integrated parametric-model fit panel (e.g.
  ``SC_TwoGap_SS`` for σ(T) → λ(T) in superconductors). musrfit's
  ``msr2data`` is a CLI-only batch tool; Mantid's results table
  does not perform secondary parametric fits in the same window.
  See :doc:`/reference/parameter_trending`.

**Modern PySide6 + matplotlib single-process GUI.**
  musrfit fragments across separate ``musrview``, ``musredit``,
  ``musrWiz``, ``mupp`` processes. Mantid is a heavy install
  (~1 GB). Asymmetry runs in one process with a single ``pip
  install``.

**Schema-versioned JSON project files (``.asymp``).**
  Forward-compatible state serialisation with documented schema
  migrations. ``.mantid`` files are HDF5 binary; ``.msr`` files lack
  a versioned schema. See :doc:`/reference/project_files`.

Where the other tools lead
--------------------------

Areas where Asymmetry currently lags. Each item is tracked as a
candidate in the roadmap (see below).

**Theory function breadth (musrfit / Mantid).**
  musrfit ships ~34 built-in theory functions; Mantid adds ~15
  specialist muon-only functions (``Keren``, ``Meier``,
  ``MuonFInteraction``, four ``*Muonium*`` variants,
  ``MuoniumDecouplingCurve``). Asymmetry's component library is
  smaller and still filling in specialist forms (``Keren``,
  ``Abragam``, ``Bessel``, ``SpinGlass``, ``Meier``). Tracked as
  ``theory-library-expansion``.

**Automatic phase calibration (Mantid).**
  Mantid's ``CalMuonDetectorPhases`` fits *every* detector phase
  automatically from early-time data in one step. Asymmetry
  estimates per-group FFT phases and can fit per-group phases inside
  the MaxEnt reconstruction, but has no equivalent one-click
  full-detector auto-calibration. Tracked as
  ``phase-auto-calibration``.

**musrfit ``.msr`` project import.**
  Asymmetry reads its own ``.asymp`` projects but cannot yet import
  a musrfit ``.msr`` file. Cross-tool interoperability is tracked as
  ``msr-import``.

Shipped since the first landscape survey
----------------------------------------

Much of the original gap list has since landed and is documented in
the reference manual: MINOS asymmetric errors
(:ref:`minos-asymmetric-errors`), dynamic Kubo–Toyabe
(:doc:`/reference/fit_functions/kubo_toyabe`), the MaxEnt
reconstruction and Burg pole-scan (:doc:`/reference/fourier_analysis`,
:doc:`/reference/frequency_finishers`), the rotating-reference-frame
transform (:doc:`/reference/rotating_frame`), spectral moments
(:doc:`/reference/spectral_moments`), synthetic-data simulation
(:doc:`/reference/simulation`), period arithmetic and RF-μSR
resonance (:doc:`/reference/alc_mode`), Python user-function plugins
(:doc:`/reference/user_functions`), and negative-muon analysis
(:doc:`/reference/negative_muon_analysis`).

Automatic FFT phase estimation
------------------------------

Asymmetry's **Auto Phase** estimates one physical phase per detector group
using a nonnegative frequency distribution and the actual sampled FFT response.
It preserves the rotated measured spectrum, includes windowing and known mean
subtraction, and reports conditional uncertainty and processing sensitivity.
Weak, incompatible, or unstable inputs are refused. It is distinct from fitting
an entropy-plus-negativity objective or jointly subtracting a spectral baseline;
see the published-method discussion and citations in
:ref:`the Fourier reference <fourier-phase-estimation>`.

This is an FFT projection, not a maximum-entropy reconstruction or a
full-detector timing calibration. Numerical agreement with another program's
automatic phase estimate depends on its objective, sampling conventions, band,
and preprocessing. A comparison of mathematical formulations alone does not
establish release-specific numerical agreement.

Roadmap — the remaining gaps
----------------------------

The team works through a prioritised list of the gaps that remain.
Priority is set by an explicit ``impact × ease`` score (see
``docs/porting/ROADMAP.md`` in the repository for the full
methodology). The near-term candidates are:

* **Theory library expansion** — continue porting specialist forms
  (Keren, Abragam, Bessel, SpinGlass, Meier, MuoniumDecouplingCurve,
  and the time-domain superconductor vortex-lattice function) from
  musrfit / Mantid.
* **Automatic phase calibration** — a one-step full-detector
  auto-phase for TF datasets, analogous to Mantid's
  ``CalMuonDetectorPhases``.
* **musrfit ``.msr`` import** — read a musrfit project file for
  cross-tool interoperability.

The roadmap is refreshed quarterly. The latest ranked candidate
list lives in ``docs/porting/ROADMAP.md`` in the repository.

References
----------

Software references:

* **WiMDA:** F. L. Pratt, Physica B **289–290**, 710 (2000).
* **musrfit:** A. Suter and B. M. Wojek, Phys. Procedia **30**, 69 (2012).
  Source: https://bitbucket.org/muonspin/musrfit
* **Mantid:** O. Arnold *et al.*, Nucl. Instrum. Methods A **764**, 156
  (2014). Source: https://github.com/mantidproject/mantid

For the underlying physics see S. J. Blundell, R. De Renzi, T. Lancaster,
and F. L. Pratt, *Muon Spectroscopy: An Introduction* (Oxford University
Press, Oxford, 2022).

.. _comparison-selected-timing:

Selected automatic timing
-------------------------

Asymmetry retains the continuous maximum and pulsed leading half-height feature
conventions. WiMDA's **Search for T0** uses a grouped maximum; musrfit 1.12.1
``musrt0 -g`` writes per-detector prompt maxima. With two unique synthetic
prompt features at raw bins 20 and 31, musrt0 and Asymmetry's continuous search
agree exactly (integer-bin tolerance zero). Grouped and detector maxima need
not agree when channels have different delays. Pulsed maximum and leading
half-height features also differ by construction; neither certifies physical
time zero. WiMDA executable/version coverage is unavailable for this comparison.

Mantid 6.16.1 ``MuonPreProcess`` applies supplied time-zero calibration and
retains shifted spectrum coordinates. Explicit-table checks establish timing
application, rather than automatic feature detection. Asymmetry uses whole-bin
transport and carries file fractional residuals; comparisons require matched
raw observations and origin conventions. See the
`MuonPreProcess manual <https://docs.mantidproject.org/v6.16.0/algorithms/MuonPreProcess-v1.html>`_,
`musrfit manual <https://lmu.web.psi.ch/musrfit/user/html/user-manual.html>`_, and
`WiMDA overview <https://shadow.nd.rl.ac.uk/wimda/>`_.

Asymmetry also checks that automatic timing has an established strategy and
that every required selected channel supplies its own usable estimate. This
protects the requested detector alignment without making an unsuccessful
optional search a file-loading failure. Unused failures do not block a valid
selection. Missing-file fallback retains other usable file anchors and marks
its own estimates explicitly; unresolved requests preserve raw inspection and
editable settings. These are Asymmetry safeguards, not a claim about all three
reference programs' failure policies. Their full unknown-source and partial
failure behaviour is outside the executed comparison.

.. _comparison-count-backgrounds:

Count-background estimates
--------------------------

Asymmetry fits one nonnegative exponential-plus-flat Poisson estimating
objective on native bins. The exponential lifetime is fixed, and its shape
is averaged across each bin. Profiling the total expected count and solving
the bounded mixture score makes equivalent origins on identical observations
return the same flat level and likelihood. Output bin merging does not refit
this native-bin estimator. On deadtime-corrected or supplied-variance counts,
the objective is an estimating convention, not a claim of raw Poisson statistics.

WiMDA 1.326 **Auto BG** uses count-error weighting on grouped output points,
with negligible weight for observations at or below four counts. In Mantid
6.16.1, the Muon automatic correction fits ``FlatBackground + ExpDecayMuon``
by default weighted least squares without nonnegative parameter constraints.
A matched Mantid executable check on an independently generated dense exact
exponential-plus-flat mean agrees with Asymmetry. Separate noisy-data checks
with WiMDA and Mantid show different estimates and sensitivity to grouping,
selected support, and supplied zero-bin errors. The WiMDA checks use paired
corrected/uncorrected exports of seeded Poisson draws; Mantid checks use its
public ``Fit`` algorithm with explicitly supplied count errors.
These are selected calculations, not a statistical bias or coverage study or
a full Mantid Muon GUI validation. The `WiMDA homepage
<https://shadow.nd.rl.ac.uk/wimda/>`_ describes the program; the
`Mantid Corrections tab documentation
<https://mantidproject.github.io/docs-versioned/v6.11.0/interfaces/muon/muon_corrections_tab.html>`_
is version 6.11.0 and is distinct from the compared runtime.

In musrfit 1.12.1 (``708f777b``, ROOT 6.40.04), asymmetry preparation uses an
arithmetic region mean. A jointly fitted single-histogram background is a
different operation. See the `musrfit RUN settings
<https://lmu.pages.psi.ch/musrfit-docu/user-manual.html#msr-run-block>`_.
Asymmetry's aligned grouped-region mean can differ from WiMDA's configured
original-detector region when detector timing or corrections differ.

Asymmetry selects the late half of resolved good-data support on native bins,
closest to Mantid's declared selection. For inclusive ``[a,b]`` it starts at
``a + (b-a+1)//2``. WiMDA selects the late half of grouped output points, so
output merging can also change the observations. Asymmetry honours explicit
shared/separate intervals exactly, including values equal to defaults, following
WiMDA/Mantid interval control. musrfit's compared region selection can shorten
explicit intervals and depend on requested packing. Asymmetry applies its
musrfit-inspired period convention only to automatic defaults using native width;
shorter-than-period defaults remain intact. Integer rounding and inclusive
endpoints do not establish exact whole-cycle exposure. Facility labels provide
advice and defaults, without vetoing supported explicit selections.

Asymmetry propagates a shared estimated level and count-estimator covariance
through merged counts. WiMDA's compared background subtraction leaves exported
count errors unchanged; Mantid adds the fitted level error to each bin without
retaining the full shared covariance in the count workspace. musrfit's compared
region-error propagation adds level-estimation variance but does not retain the
full shared/overlap covariance. These differences are deliberate uncertainty
conventions. Supplied variances in Asymmetry enter the fitted estimator's
linearised influence; they do not change its central estimating objective.

Asymmetry also checks sufficient fit support, expected-information rank, and
finite positive uncertainty. These checks prevent a numerically unestimable
level from being reported as usable, even when a solver might report success;
they do not test residual physical polarisation or impose a chi-squared gate.
Its linearised boundary errors do not promise calibrated coverage. An unusable
automatic estimate, including a signed selected tail, leaves the whole requested
result background-unsubtracted with a warning and recorded reason. A genuine
zero-background optimum remains successful. Valid deadtime correction and
supplied variance are retained; loading remains independent of estimability.

Mantid's Muon caller skips per run/group when reduced chi-squared exceeds 10
or the status contains ``Failed to converge``; literal ``success`` is not
required. Asymmetry adds result-wide F/B and contributing-period consistency
without adopting that threshold. WiMDA's automatic path uses its returned level
without an equivalent fit-status skip check. Its negligible low-count weights
also attenuate negative observations. musrfit's region and joint-count operations
do not supply an equivalent automatic-tail failure policy. Finite uncertainty
and information checks in Asymmetry protect downstream covariance even when a
solver might claim success; none establishes physical model adequacy.
See :doc:`/reference/data_reduction/backgrounds` for the model, current window,
and covariance equations.
