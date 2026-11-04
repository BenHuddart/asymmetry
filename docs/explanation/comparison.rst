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
     - Legacy v5 and modern layouts use common reduction; closest to musrfit
     - v4 remains unsupported; acquired-example checks and the directory-case
       limit have distinct scopes. See :ref:`comparison-root`.

Agreement means agreement for matched observations and uncertainty conventions,
within the stated storage or export precision. Explained convention differences
are identified separately. Unverified paths and unresolved uncertainty
comparisons remain limits; acceptance of a selection alone does not establish
correct covariance or physical calibration. WiMDA behaviour below has no
identified executable release version or binary numerical comparison; the
`author's homepage and manual <https://shadow.nd.rl.ac.uk/wimda/>`_ describe
its workflows without supplying a version for these observations.

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

.. _comparison-root:

PSI legacy ROOT calibration
---------------------------

Asymmetry adopts musrfit's numeric ``TLemRunHeader`` v5 calibration for legacy
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
Two legacy v4 examples remain loading failures, outside the supported numeric
v5 decoder. See :doc:`/reference/loading_data` and the
`musrfit RUN settings <https://lmu.pages.psi.ch/musrfit-docu/user-manual.html#msr-run-block>`_.

.. _comparison-psi-intervals:

PSI good-data intervals
-----------------------

Asymmetry adopts zero-based inclusive raw-bin bounds for file-good intervals
and intersects them after whole-bin alignment over the selected detectors.
BIN upper values at or beyond histogram length are clipped to the final
stored index; interior values are retained. MDU M3/T4/T5 markers are converted
at their hardware scale before clipping. Modern ROOT per-detector bounds use
the same inclusive representation; legacy LEM v5 without good fields defaults
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

NeXus physical good-time selection
---------------------------------

The `muon NeXus definition, version 2 revision 8, p. 51
<https://nmi3.eu/nexus_instrument_definitions_for_muon_data-get_file=nexus_instrument_definitions_for_muon_data.pdf.pdf>`_
defines physical good times relative to ``raw_time``. Asymmetry adopts inclusive
raw-centre membership on its nominal fixed-bin grid and retains the resulting
per-detector bounds through reduction and projects. Integer attributes retain
precedence for their corresponding endpoints. The definition does not settle
every writer's integer base or sub-bin rounding convention.

In Mantid 6.16.1, a histogram workspace with the same 32 tagged bins, 0.02 μs
width, and 0.2074 μs origin, cropped to raw limits 0.22–0.30 μs, selects the
same four raw observations, centres, and Poisson errors. This comparison uses
``CreateWorkspace`` and ``CropWorkspace``; it does not certify every NeXus
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
the `musrfit manual <https://lmu.web.psi.ch/musrfit/user/html/user-manual.html>`_
for channel and fit-time controls.

Asymmetry also checks that the selected file-good intervals have nonempty
common acquired support before reducing independent counts. An unavailable
request retains the acquired histograms and editable grouping. Optional field
shape or unit diagnostics do not impose whole-file rejection when another
usable interpretation remains; see :doc:`/reference/loading_data`.
