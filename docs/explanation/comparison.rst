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
cover the background and binning controls explicitly identified below. Other WiMDA
descriptions compare workflows without claiming executable numerical agreement;
the `author's homepage and manual <https://shadow.nd.rl.ac.uk/wimda/>`_
provide general context.

.. _comparison-asymmetry-errors:

Asymmetry values, coordinates, and errors
-----------------------------------------

Asymmetry forms :math:`A=(F-\alpha B)/(\beta F+\alpha B)` from corrected,
packed counts, with alpha on the backward group. Its first-order error uses
retained count variances, including existing shared-background/overlap terms
when counts are merged. See :ref:`asymmetry-counting-errors` for the derivation,
signed-count example, and Gaussian omission policy.
Dataset values and errors are percent; count-to-ratio primitives are fractions.
Neither scale imposes universal estimator bounds.

WiMDA 1.326 ordinary F/B uses the beta-one ratio with a small positive-count
variance adjustment. Its positive-count errors approach the derivative limit
at high counts but are not identical. Its ordinary F/B fitting excludes
nonpositive corrected groups, distinct from plotted fallback points and other
count-fitting workflows. The versioned binning comparisons below check values;
they do not establish identical error bars. The `WiMDA homepage/manual
<https://shadow.nd.rl.ac.uk/wimda/>`_ provides workflow context.

In Mantid's `AsymmetryCalc v1 documentation
<https://docs.mantidproject.org/nightly/algorithms/AsymmetryCalc-v1.html>`_
(reviewed 6.16 development), beta is one and alpha has no error. Its count-based
rule is :math:`\sigma_A^2=(F+\alpha^2B)(1+A^2)/(F+\alpha B)^2`, treating
numerator and denominator as independent. With 6000 forward and 4000 backward
counts and unity alpha, both values are 20 %, but the documented rule gives
1.019803903 percentage points, compared with Asymmetry's 0.979795897.
This is documented arithmetic, not a native Mantid execution comparison.

musrfit 1.12.1 (``708f777b``, ROOT 6.40.04) type 2 fits the raw coordinate
:math:`r=(F-B)/(F+B)` with a transformed model and reciprocal alpha
:math:`\alpha_m=1/\alpha`. Here raw means unbalanced; requested background
subtraction is already applied. Its beta convention agrees after alpha
conversion. For :math:`u=1+r,v=1-r`, converting to Asymmetry's coordinate gives

.. math::

   A(r)=\frac{u-\alpha v}{\beta u+\alpha v},\qquad
   \sigma_A\simeq
   \left|\frac{2\alpha(1+\beta)}{(\beta u+\alpha v)^2}\right|\sigma_r.

For repeated counts :math:`F=60,B=40`, fixed :math:`\alpha=1.25,\beta=0.8`,
zero backgrounds, corrections off, and packing one, native ASCII data/error
columns give :math:`r=0.2,\sigma_r\simeq0.0979796` with :math:`\alpha_m=0.8`.
Converting both gives :math:`A=0.1020408163,\sigma_A=0.1147720100` in fractions,
agreeing with Asymmetry within the six-significant-digit export precision
(relative tolerance :math:`5\times10^{-6}`). The unity dense control also
agrees. These checks use interior points on owned synthetic NeXus inputs,
one native thread, and fixed balances/backgrounds. The fitted theory column
is not a reduction oracle. See the `musrfit type-2 manual
<https://lmu.pages.psi.ch/musrfit-docu/user-manual.html>`_ and
`packing tutorial <https://musruser.psi.ch/Tutorial/Tutorial1.html>`_.

First-order local error agreement after a nonlinear coordinate change does not
establish identical Gaussian objectives for arbitrary residuals or identical
model recovery. musrfit's empty-native-bin variance treatment also differs from
Asymmetry's Gaussian omission policy: musrfit supplies nonzero empty-native-bin
variance, while Asymmetry omits exactly undefined or zero-first-order-variance
histogram ratios. WiMDA omits nonpositive corrected groups; Asymmetry retains
usable signed values and corrected zeros with positive variance. In Mantid,
AsymmetryCalc uses the distinct error approximation above. These are deliberate
method differences, not identical sparse-bin likelihoods. Asymmetry also checks
final ratio usability after requested binning and period composition, reports
omission reasons, and leaves raw/count-fitting access available. Native packing
is not a statistical coverage or wizard-ranking test.

In these reviewed standard F/B workflows, none passes an external alpha/beta
calibration covariance through a full reduced-bin/run covariance to the
subsequent fit. Fitting coefficients within a model, as in musrfit, is a separate
treatment. This statement does not restrict custom analyses in extensible
platforms. Asymmetry's ordinary errors are conditional on the selected
coefficients; see :ref:`conditional-calibration-errors`.

.. _comparison-binning:

Binning and timestamps
----------------------

Asymmetry sums counts and their retained variance before forming F/B asymmetry.
Nonuniform target widths are evaluated at the actual left edge, rounded to the
nearest positive whole number of acquisition bins (half-up), and stamped at
the actual contributing-bin midpoint. The exact 10 μs Variable law, lifetime
Constant error law, and pre-t0 clamp define the target, rather than its realised
integer width. Base nonuniform bins retain inclusive selected support and a
partial tail. Fixed packing and additional recorded-bin merges trim incomplete
groups and retain all complete ones.

The closest width-selection comparison is WiMDA 1.326. With 0.1 μs native
bins, initial width 0.12 μs, and Variable width at 10 μs of 0.25 μs, all eight
Constant error values and eleven shared Variable values agree within 0.0005
percentage points. Actual midpoint times, terminal support, and the literal
10 μs law remain deliberate differences. Half-up is Asymmetry's deterministic
choice, not a verified WiMDA tie result. Positive-count uncertainty conventions
also differ; this comparison does not certify matching corrected/sparse errors.

musrfit 1.12.1 (``708f777b``, ROOT 6.40.04) provides ordinary integer packing,
with matching interior sums and mean native timestamps under matched support
and integer timing. It does not establish the same nonuniform controls.
Mantid histogram-centre conventions offer a coordinate comparison; histogram
Rebunch retains a partial group, while Rebin can redistribute fractional bins.
The Mantid comparison is documented behaviour, with no native binning execution.
See :doc:`/reference/data_reduction/binning` for algorithm references, terminal
qualifications, and uncertainty assumptions.

Ordinary binning honours authored merges without a spectral peak veto, as in
the compared WiMDA 1.326 grouping, musrfit 1.12.1 packing, and documented Mantid
Rebin/Rebunch operations. Actual FFT requests still require a uniform grid.
Asymmetry also checks that an authored count-fit factor and selection leave
usable complete fit bins, before resolving unused corrections. The factor is
never silently reduced to one. Raw inspection and request repair remain
available; identical empty-bin diagnostics in other programs are not established.

Both fit wizards use counts-first automatic preparation for resolved histogram
observations. In the analytic F = [90, 1], B = [10, 9] control, merging gives
65.454545…%, while averaging the separate ratios gives 0%. Curve-only, rotated,
and unresolved historical/custom observations retain independent-curve means.
The native count controls above support the input arithmetic and membership;
they do not validate Asymmetry's wizard ranking or entire search. The wizards
continue fitting asymmetry, separately from TF alpha/beta count calibration.

Point-model fits use output timestamps and conditional point errors. Wide bins
average the signal; their point-model approximation can attenuate oscillations
or shift phase. Mantid's separate histogram-evaluation option is not a general
integration prescription for every count ratio or curve mean.

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

WiMDA 1.326 quick **Diamagnetic** and **General** estimates return alpha
without an uncertainty; Asymmetry follows that error policy. Its Diamagnetic
uses continuous minimisation and internal packing, and its General uses a
two-window decay balance rather than relative-scatter minimisation. Matched
16 ns synthetic beta = 1 controls over raw bins 4–498, background/deadtime off
and bunch one agree within 0.001 absolute alpha: TF Diamagnetic 1.237 versus
1.236610, TF General 1.250 versus 1.249827, and relaxing General 1.250 versus
1.250073 (WiMDA then Asymmetry). These checks do not establish uncertainty
coverage or equivalence on corrected or acquired observations. Asymmetry also
checks rounding-level General contrast and a residual lifetime diagnostic;
their purpose and limits are in :ref:`alpha-calibration`.

Asymmetry also checks that automatic α has finite positive F/B integrals and
a finite positive ratio. An undefined estimate makes that reduction unavailable
and editable; it does not replace the stored request with unity. Explicit
fixed unity remains valid. This protects the ratio's mathematical inputs,
without claiming that all reference workflows use the same guard.

.. _comparison-count-calibration:

Shared TF count calibration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Asymmetry's ``tf_calibration_fit`` core operation uses one exponential/common-phase
F/B count model, fixing beta at one for alpha-only calibration or freeing it
for joint calibration. Both backgrounds and other nuisances belong to one full
covariance. Poisson deviance/dof counts unique free parameters once, and errors
are conventional conditional HESSE standard errors, without residual scaling.
The existing beta ``count_fit`` adapter uses this operation; the independent
``single_histogram`` ratio remains a separate protocol.

Applied File/manual deadtime corrections select the existing weighted count
objective, retaining measured or supplied independent variances. Off/skipped
deadtime uses raw Poisson counts. Backgrounds are fitted model terms; grouping
fixed values, range estimates, and automatic-tail estimates do not subtract or
constrain them. This shares the count model, not musrfit's type-2 asymmetry
objective.

Matched musrfit 1.12.1 type-0 chi-square controls use 16 ns native bins, two
disjoint detectors, 100000 good frames, deadtime 0.02 μs, zero fixed backgrounds,
and packing one/four. After matching its corrected-count variance convention,
fixed/free-beta points agree within 0.0001 absolute and alpha errors within
0.1%. Asymmetry's retained measured-count weights differ: one beta-one control
gives alpha 1.249554 versus native 1.249574, and errors 0.001102 versus 0.001162.
This is a weighting difference, not an uncertainty correction to copy.
Mantid's muon ``ApplyDeadTimeCorr`` preserves supplied errors; WiMDA's ordinary
grouping uses measured counts as its counting-error basis. Those correction
conventions do not establish equivalent native TF calibration objectives or
complete deadtime uncertainty propagation. The comparisons are conditional
weighted fits, not uncertainty coverage or a new detector-response likelihood.

The closest executed counterpart is musrfit 1.12.1 (``708f777b``, ROOT 6.40.04)
global type-0 histogram fitting with ``MAX_LIKELIHOOD``, described in the
`Single Histograms tutorial <https://musruser.psi.ch/Tutorial/Tutorial2.html>`_.
Four owned controls (rounded means and one seeded Poisson realisation each,
fixed/free beta) use identical 16 ns integer arrays, raw bins 4–498,
fixed lifetime, zero backgrounds, phases 0/180 degrees, and matched free
frequency/damping/amplitude parameters. Alpha/beta points agree within 0.0001
absolute after mapping native :math:`N_B/N_F` to reciprocal Asymmetry alpha;
native fractional amplitude maps to Asymmetry percent, MHz to MHz, and degrees
to radians. This is not equivalence to musrfit type-2 asymmetry chi-square,
a coverage study, or validation of acquired detector response.

The installed likelihood route reports a different error scale. With only the
forward normalisation free, the exact Poisson MLE is 19999.9849 and its standard
error is 9.24119 counts/bin. Asymmetry gives about 9.24122; native musrfit gives
about 6.535, approximately :math:`1/\sqrt{2}` as large. The independent control
and full-TF information calculation support Asymmetry's unit-deviance error
criterion (see the `Minuit2 manual
<https://root.cern.ch/root/htmldoc/guides/minuit2/Minuit2.html>`_). No compensating
rescaling is applied. This observation is specific to that installed likelihood
route; it makes no claim about other versions, type-2 chi-square, or all musrfit
uncertainties.

WiMDA 1.326 quick Diamagnetic and Asymmetry's quick method give about 1.237 on
an owned beta-one TF control with generating alpha 1.25; the model fit gives
1.249995, changing mean reduced asymmetry by about −0.53 percentage points.
On a matched original Ag TF example (zero-based raw bins 19–884, 16 ns, full F/B banks,
bunch one, background/deadtime off), native quick alpha is 0.974 and Asymmetry
quick alpha 0.974190, versus TF beta-one 0.966447. The associated mean asymmetry
shift is about +0.39 percentage points. Its deviance/dof is about 1.30, with
phase/model adequacy unresolved and no physical alpha truth asserted. These
results describe different estimators, not a universal accuracy ranking.
WiMDA's separate paired/all-group model-fit errors remain unverified here;
Mantid AlphaCalc is a ratio comparator, not a shared-TF likelihood comparison.

Asymmetry also checks raw-Poisson numerical inputs, disjoint detector banks,
existing count-scale sanity, and free-beta precession/error diagnostics. Their
scope and consequences, and the limits of processed/corrected/pulsed support,
are in :doc:`/reference/data_reduction/alpha_calibration` and
:doc:`/reference/data_reduction/beta_calibration`. Passing these checks does not
certify that an exponential common-phase signal describes a calibration.

The separate ``select_tf_calibration`` script operation compares undamped,
exponential, Gaussian, and stretched envelopes on identical count observations,
using minimum summed objective without a complexity penalty or extra ambiguity
threshold. All candidate states and conditional covariances remain available.
This recommendation is an Asymmetry convenience. The existing single-model
operation and beta adapter retain the exponential model.

musrfit's `MuSRFit interface <https://musruser.psi.ch/?page_id=148>`_ lists
Exponential Cosine, Gaussian Cosine, and Stretch Cosine. The
`musrfit THEORY documentation
<https://lmu.pages.psi.ch/musrfit-docu/user-manual.html#the-theory-block>`_
defines ``simpleGss`` with a Gaussian rate :math:`\sqrt{2}` times Asymmetry's
:math:`\sigma`; ``generExpo`` uses :math:`\exp[-(\Lambda t)^p]` at positive time.
Fresh musrfit 1.12.1 (``708f777b``, ROOT 6.40.04) component controls for all four
shapes agree within 0.0001 absolute alpha/beta with fixed/free beta, zero fixed
backgrounds, phases 0/180 degrees, and the same 16 ns raw bins 4–498. Reciprocal
alpha, fractional amplitude, Gaussian-rate, and degree/radian conventions are
matched. This tests component points, not native automatic selection or an
extension of the likelihood error-scale comparison above.

Mantid's `StretchExpMuon documentation
<https://docs.mantidproject.org/nightly/fitting/fitfunctions/StretchExpMuon.html>`_
defines the corresponding non-oscillatory stretched envelope and describes a
usual exponent range up to two. Asymmetry's positive stretching exponent has no
upper physical cap; its numerical lower search default can be overridden with
positive bounds. No fresh native Mantid stretched fit or WiMDA model-fit
comparison is asserted. The outgoing Diamagnetic/reduced-signal comparison
above concerns a different estimator and does not enter fit-cost ranking.

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

Asymmetry offers **Automatic** and **Manual** reference scaling. Automatic uses
compatible evidenced exposure ratios, or visibly assumes one when exposure is
unknown. Whole-run PSI/ROOT deadtime normalisation one is not measured pulse
exposure. Manual uses its authored positive coefficient regardless of metadata.
WiMDA 1.326's bounded FileBG check uses sample/reference 200/100 good frames,
200/30 counts per native 0.016 μs bin, one detector in group 1, bunch one,
deadtime off, and Auto/Manual BG off. The exported 140 counts agree with the
ratio-two arithmetic within 0.00051 counts (text precision). This does not
establish a physical sample/holder calibration or an equivalent unknown-exposure
label. Mantid 6.16.1 generic `Scale
<https://docs.mantidproject.org/v6.16.1/algorithms/Scale-v1.html>`_ expresses an
explicit coefficient; no corresponding automatic Muon GUI or musrfit FileBG
scale workflow is claimed. Fixed-scale independent variance excludes
estimated-scale and shared-reference cross-result covariance.

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

.. _comparison-reference-errors:

Reference subtraction: deadtime and counting errors
----------------------------------------------------

The following comparison reuses recorded native WiMDA 1.326 (64 bit, Wine
10.0) and Mantid 6.16.1 results on synthetic constant spectra. It is not a new
native execution or a comparison on an acquired sample/holder pair. The sample
has 200/150 counts per native bin in two detectors; the reference has 30/20.
Each group contains one detector. Sample/reference exposures are 200/100 good
frames, native width is 0.016 μs, raw t0 is zero, and the subtraction scale is
two. Enabled Manual deadtime is 1 ns (0.001 μs) per detector. The table shows
the first group with bunch one; WiMDA exports raw group 1 at nonnegative time,
with Auto/Manual BG off and FileBG enabled. Its sample exposure is restored to
200 frames after reference loading, which can change the active frame count.

.. list-table:: First-group count and standard error
   :header-rows: 1
   :widths: 45 28 27

   * - Operation
     - Count
     - Standard error
   * - Asymmetry / explicit Mantid, DT off
     - 140
     - 17.88854382
   * - WiMDA FileBG, DT off
     - 140
     - 14.1421
   * - Asymmetry / explicit Mantid, both sides DT-corrected
     - 152.186836518
     - 17.88854382
   * - WiMDA FileBG, sample DT-corrected, raw reference
     - 153.333
     - 14.1421

For each acquisition, the independently evaluated inverse is
:math:`c(n)=n/[1-n\tau/(F\Delta t)]`, with :math:`\tau` and
:math:`\Delta t` in μs. Asymmetry gives
:math:`c_S(200)-2c_R(30)=152.18683651804673`.
The Mantid sequence explicitly applies `ApplyDeadTimeCorr
<https://docs.mantidproject.org/v6.16.1/algorithms/ApplyDeadTimeCorr-v1.html>`_
to each workspace, then **Scale** and **Minus**. A separately configured
sample-only sequence gives :math:`c_S(200)-2(30)=153.33333333333334`,
explaining the WiMDA FileBG result. These are authored workspace operations,
not an equivalent automatic Muon GUI workflow.

Asymmetry and the configured Mantid arithmetic retain independent input
variance :math:`200+2^2(30)=320` with DT off or on. They do not infer Poisson
variance from corrected means. WiMDA's tested raw-group export retains the
sample-only counting error :math:`\sqrt{200}` after FileBG. These measurement-
variance conventions exclude coefficient/exposure uncertainty and physical
deadtime-induced correlations. The independent-variance rule also excludes
estimated-scale and shared-reference cross-result covariance. Matching this
example does not establish full uncertainty coverage or identical fit objectives.

The Asymmetry/Mantid independent-arithmetic checks use relative tolerance
:math:`10^{-13}`. WiMDA's text-export absolute tolerances are 0.00051 counts
and 0.000051 in the standard error, set by print precision; all 1999 exported
rows satisfy the constant-spectrum comparison. This bounds the claim to the
recorded synthetic raw-group export, not noisy inputs, ROOT-bank statistics,
or every GUI path. See :doc:`/reference/data_reduction/backgrounds` for the
count-reference contract and :ref:`comparison-reference-timing` for the
separate channel-alignment convention.

.. _comparison-physical-backgrounds:

Additive physical signals and amplitude units
---------------------------------------------

The `musrfit tutorial <https://lmu.pages.psi.ch/musrfit-docu/tutorial.html>`_
models sample and holder contributions with separate amplitudes, relaxation,
and fields. These physical signals are distinct from the uncorrelated count
background fitted in a single-histogram model. Asymmetry's existing composite
models likewise express additive physical signals; no external FileBG operation
is implied. A constant holder component is appropriate only for a stationary,
non-relaxing response in the chosen frame.

A reused musrfit 1.12.1 (``708f777b``, ROOT 6.40.04) check fits a generic
``fittype 8`` ASCII curve
:math:`a(t)=0.2\exp(-0.35t)+0.04`, with 81 equally spaced points from
0 to 8 μs, supplied error 0.001, packing one, and rate fixed at 0.35 μs⁻¹.
MIGRAD/HESSE recovers both fraction-valued amplitudes within absolute tolerance
:math:`5\times10^{-6}` (printed parameter precision), starting at 0.15/0.02.
Asymmetry's **Exponential** + **Constant** forward evaluation uses amplitudes
20%/4%; dividing its percent output by 100 gives the same fractional curve
(relative tolerance :math:`10^{-14}`, absolute :math:`10^{-16}`). This is a
model-composition and units illustration. It is not matched raw muon reduction,
a new native fit, a holder-fraction calibration, or a comparison of fit covariance.
See :doc:`/reference/data_reduction/backgrounds` and
:doc:`/reference/asymmetry_units` for modelling advice and units.

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

Calibration persistence
-----------------------

Asymmetry's project records preserve the effective TF model, count objective,
units, correction assumptions, constraints, and full nuisance covariance of an
applied core calibration. Saving and loading do not rerun a reference program,
change the fit, or establish additional numerical agreement. The component,
objective, and conditional-error comparisons above retain their stated versions
and conditions. Model selection is an Asymmetry convenience; stored candidate
covariances do not measure selection uncertainty or full reduced-data covariance.

Workflow calibration entry points
---------------------------------

Explicit workflow/API/CLI TF requests share the documented core operation,
model selection, and canonical alpha/beta record. ``--calibration-method`` also
exposes Diamagnetic, General, and ratio as distinct operations. The no-option
explicit CLI calibration uses TF calibration fit. Survey and
automatic per-run calibration retain the separate integral-ratio policy, which
is not WiMDA's Diamagnetic or General estimate. These adapters add no
reference-program numerical agreement or physical-accuracy claim to the
versioned comparisons above. General count-domain fitting remains a separate
capability.

Fresh explicit calibration routing
----------------------------------

Asymmetry's GUI, CLI ``alpha``/``--alpha-from``, and omitted-option explicit API
calibration use the shared TF count fit with four-envelope selection. Quick
Diamagnetic and Decay balance remain distinct objectives, with stored ``general``
identity retained for Decay balance. Finite windows, weights, frequency, and model
backgrounds can change the fitted detector balance; different estimators need
not agree. The selected envelope and its conditional errors do not establish
acquired physical alpha truth or pulse-model adequacy. Asymmetry also checks
request freshness before displaying or applying GUI results, so changes to the
source, pair, timing, or corrections cannot apply an obsolete result.

Asymmetry also checks that a displayed calibration coefficient is finite and
positive before offering application. A numerical ratio of zero from an empty
forward group cannot be used as a detector calibration; the workflow reports
that diagnostic without changing the ratio formula or its error calculation.

Acquired periods and mapped sets
================================

Asymmetry retains the original acquisitions behind a red/green mapping,
including ignored periods. Independent supplied count variances add when
periods are pooled; ordinary raw Poisson additions remain Poisson. A red-only
mapping supports ordinary histogram preparation and period-1 selection.
Stored data indices remain distinct from evidenced DAE sequence slots and
dwell declarations. See :doc:`/reference/data_reduction/period_mapping`.

WiMDA 1.326 maps acquired periods into red/green count sets and displays
declared dwell slots. Mantid's 6.16 development documentation describes
separate period workspaces in `MuonPreProcess v1
<https://docs.mantidproject.org/nightly/algorithms/MuonPreProcess-v1.html>`_.
musrfit 1.12.1 selects period histogram families explicitly in its
`RUN/grouping manual <https://lmu.pages.psi.ch/musrfit-docu/user-manual.html>`_.
These comparisons establish selection behaviour, not identical uncertainties
or general facility exposure interpretation. No native Mantid period result
is claimed.

With unity calibration and corrections off, equal-total synthetic periods
of 20% and 80% give Red/Green 20%/80%, G−R 60%, and G+R 50% in WiMDA 1.326
under Wine 10. musrfit 1.12.1 (708f777b, ROOT 6.40.04) selected/grouped
controls give 20%/80%/50%. Asymmetry gives the same pooled G+R 50%, and its
independent G−R is 60%. Unequal-total pooled results are not generally an
arithmetic mean. The value controls do not establish matching error conventions.

Asymmetry corrects each acquired constituent before mapped pooling, with its
own applicable detector coefficients and supported exposure. It then aligns
whole bins using each resolved t0 on common observed support, retains supplied
count variances, and applies group-background preparation once to the requested
set. WiMDA 1.326 pools raw mapped counts before correction; this is a deliberate
ordering difference. In Mantid, standard ``MuonPreProcess`` prepares individual
period workspaces before grouping and pairing pool selected spectra.

For synthetic F/B counts (80000, 20000) and (10000, 90000), 0.1 μs native bins,
100000 frames per acquisition, and 0.05 μs coefficients, constituent correction
then pooling gives −12.7378265%; raw-first correction gives −13.2890365%.
musrfit 1.12.1 (708f777b, ROOT 6.40.04) exports −12.7378% for the equivalent
histogram occupancies using a stored scalar 200000 frames and 0.1 μs coefficients
(absolute tolerance 5×10⁻⁶ in fractional asymmetry). This checks correction order,
not acquired per-period exposure interpretation or error equivalence.

Structured shifted-count controls also agree with musrfit grouping and ADDRUN
on matched interior observations with explicit integer anchors, corrections off,
and native bins at the same tolerance. ADDRUN's edge and normaliser conventions
are not adopted. Asymmetry records fractional residuals without interpolation
and also checks common observed support and compatible native widths. One
shared explicit α/β belongs to the comparison; there is no implicit separate
calibration for each condition. Full cross-period/reference covariance remains
outside the marginal, conditional error contract.

For G+R, group-background subtraction acts once on the pooled observation;
Manual levels retain their requested-observation count units. Automatic reference
scaling uses summed evidenced sample exposure, while Manual scale applies once.
WiMDA's ordinary grouped Auto/Man/region background subtraction and its F/B G−R
path are distinct; FileBG acts earlier. Asymmetry applies the requested group
background to each independently reduced G−R set.

WiMDA 1.326 and musrfit 1.12.1 demodulate a formed scalar asymmetry in their
checked RRF workflows. Mantid ``RRFMuon`` rotates a prepared transverse pair.
Those interfaces do not define automatic synthesis of a pooled baseline from
unequal per-period transverse baselines. Asymmetry also checks that Red and Green
transverse baselines agree for pooled G+R rotation, removing the common pair
once. Unequal baselines make that rotation unavailable; Lab G+R and rotating
Red, Green, and G−R remain available. This is an Asymmetry safeguard. See
:doc:`/reference/rotating_frame_projection`.

Run addition and subtraction
-----------------------------

:doc:`/reference/run_arithmetic` compares the count operations and their limits.
Asymmetry retains signed coefficients and independent squared-weight variance
for ordinary and counterpart-period payloads. Raw independent unit sums retain
the Poisson limit; weighted/subtracted or supplied-noise results use retained
Gaussian variance and do not acquire a raw-Poisson origin from integer totals.
Required original detectors are corrected with their own facts before signed
weighting and pooling on whole-bin common observed good support. The raw pooled
histograms remain inspection views. Group background and a selected reference
act once on the pooled observation.

WiMDA 1.326 co-add/Co-sub and musrfit 1.12.1 addRun/ADDRUN provide the ordinary
count-addition comparison. Matched correction-off fit-time ADDRUN controls
agree on unit-addition and interior asymmetry values from whole-bin-aligned
counts within 5e-6 fractional asymmetry;
this establishes neither uncertainty nor endpoint equivalence. The tested
unequal-fact ADDRUN applies the primary RUN correction context to added data.
Asymmetry deliberately retains each input's correction facts and is invariant
under primary-run reversal for shared authored intent. WiMDA pools raw inputs
before correction. In Mantid, individual MuonPreProcess workspaces can be
prepared before merging; generic merging does not determine correction order.
Mantid 6.16 development Scale/Plus/Minus document the explicit independent
weighted-arithmetic analogue, without a native numerical check here. The owning
page links the public manuals and algorithms and distinguishes each operation.
WiMDA and standalone musrfit addRun retain master condition metadata. Asymmetry
also retains the first ordered input's displayed conditions and identity, with
finite active constituent ranges; missing master conditions remain unknown.
Mantid MergeRuns checks instrument names, spectrum counts, and units and offers
configured sample-log merging. Asymmetry uses complete recognised orientation
facts and evidenced implicit period permutations, retaining channel/data-order
assumptions for absent or incomparable identity. It does not reject on instrument
names alone or certify physical identity from matching shapes.

Asymmetry also checks flattened original observation reuse and invalid
coefficient/payload requests to protect independent-noise calculations. Known
nested repeated acquisitions or subtracted references are unavailable; a shared
target reference applied once after sample pooling remains supported. This does
not establish general covariance or reconstruct unknown processing history.

Pulse inputs and response origins
---------------------------------

WiMDA 1.326's **Double Pulse** and Mantid's supplied ``PulseOffset`` configure
separation. Asymmetry likewise requires fixed supplied separation for its
instantaneous count model; single-line post-arrival TF data cannot generally
identify it separately from amplitude, phase, and count scale. Source uncertainty
is not included in conditional fitted errors. Ordinary asymmetry fitting and
TF balance calibration do not gain a finite-width response from this setting.
See :doc:`/reference/count_domain_fitting` for the supported count targets and
:doc:`/reference/fourier_analysis` for the complex response and explicit origin
conversion used in the standalone Mantid kernel comparison. No fresh native
WiMDA or full Mantid pulse solve is claimed. An explicit musrfit 1.12.1
post-arrival two-component TF composition is a model control, not a verified
generic pulse feature.

Asymmetry also checks active finite-width input domains and rejects missing or
ambiguous widths rather than silently reporting identity compensation. These
checks concern an authored analysis request; they do not restrict file loading
or infer accelerator response from facility names or detector timing fields.


For Fourier compensation, Asymmetry retains the parabolic-plus-pion response
shared with its MaxEnt forward model. WiMDA 1.326's **Apply Frequency Response
Compensation** uses a Gaussian Fourier response; these are deliberately different
models (see the `WiMDA paper <https://www.isis.stfc.ac.uk/Pages/wimda-paper.pdf>`_
and :doc:`/reference/fourier_analysis`). The Mantid standalone MaxEnt kernel
comparison establishes the shared response after an explicit origin conversion,
not full native Fourier or MaxEnt reconstruction agreement. musrfit 1.12.1's
authored post-arrival TF composition does not test Fourier Power or its errors;
corresponding automatic Fourier compensation is unverified here.

The Power multiplier follows squared-amplitude algebra: for amplitude gain g,
amplitudes and their existing standard errors receive g, while Power and its
standard errors receive g². Asymmetry also checks a sampled amplitude-gain limit,
omitting the first positive frequency below the response floor and all following
bins from numerical consumers. This grid-dependent safeguard is not a physical
information-loss boundary or a recovery of response phase. Conditional errors
do not include uncertainty in the fixed pulse response or full spectral
covariance.
