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

The reference programs provide different ways to organise an analysis.
WiMDA offers grouped time and frequency workflows; musrfit uses authored
``.msr`` model and run descriptions; Mantid composes algorithms on workspaces.
Asymmetry exposes a Python analysis core with a desktop front end. The
comparison below concerns particular operations rather than a ranking of
programs or a collective file-format guarantee.

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

.. _comparison-spectral-conventions:

Normalisation and transformed observations
------------------------------------------

A group count trace and an F/B ratio are different observations. Dividing out
the muon lifetime creates a lifetime-corrected count signal, not a calibrated
polarisation. A finite-window level then defines a fractional residual for
Fourier analysis or a starting scale for a reconstruction. The
:doc:`reduction_pipeline` diagram shows these branches, and
:doc:`/reference/fourier_analysis` specifies the actual weights and scale limits.

The spectral descriptions below refer to WiMDA 1.326, musrfit 1.12.1
(revision ``708f777b``, ROOT 6.40.04), and Mantid algorithm-v1 documentation
with the 6.16 development snapshot (revision ``1967ccb4``). Formula and
manual behaviour are distinct from executed numerical agreement. Full native
Mantid FFT, PhaseQuad, and MaxEnt comparisons are unverified here; the separate
standalone pulse-kernel comparison does not establish them.

.. list-table:: Operations and conventions
   :class: reduction-conventions
   :header-rows: 1
   :widths: 25 38 37

   * - Observation / operation
     - Asymmetry
     - Reference convention and scope
   * - Grouped Fourier counts
     - Lifetime correction, fractional level estimated with :math:`1/\sigma`,
       optional :math:`1/\sigma` mean removal, and coherent-gain amplitude scale
     - WiMDA lifetime-corrects grouped counts, optionally removes a
       :math:`1/\sigma` mean, and scales the grouped spectrum by total events.
       Its F/B transform is a separate input.
   * - Theory-free single-histogram Fourier preparation
     - The grouped FFT uses the weighted level above
     - musrfit uses a lifetime-corrected arithmetic-mean level and supplied scale
       factor. This does not describe all musrfit fits or Fourier options.
   * - Lifetime removal and fitted constant
     - MaxEnt starts from an inverse-variance level and fits its own grouped
       phase/amplitude reconstruction model
     - In Mantid, RemoveExpDecay divides by a fitted constant after lifetime
       correction and subtracts one. With supplied errors the constant fit uses
       inverse-variance least squares; this is not a universal FFT convention
       or equivalence of MaxEnt reconstruction algorithms.
   * - Two measured transverse projections
     - Pointwise baseline/gain/sense rotation with retained shared counting noise
     - Mantid RRFMuon rotates two supplied components after explicit coordinate,
       phase, and unit conversion. PhaseQuad can prepare quadratures from a
       detector phase/amplitude table; it is a separate operation.
   * - One scalar F/B ratio
     - Complex carrier plus FIR low-pass display; fits retain original laboratory
       observations
     - WiMDA uses real demodulation and box smoothing; musrfit scalar RRF paths
       have their own packing, theory, and error conventions. Neither establishes
       four-program vector covariance agreement.

The level estimators can differ even for noiseless oscillations when the window
contains a nonzero oscillatory mean. Coherent-gain correction alone does not
remove that difference or calibrate detector efficiency. Padding refines the
sampled transform grid; it supplies no new observations. Neither peak height nor
magnitude area is universally invariant to window, damping, phase, or apodisation.

The `WiMDA homepage/manual <https://shadow.nd.rl.ac.uk/wimda/>`_ and
F. L. Pratt, Physica B **289–290**, 710 (2000), describe the grouped Fourier
and MaxEnt workflows. The `musrfit manual
<https://lmu.pages.psi.ch/musrfit-docu/user-manual.html>`_ distinguishes
single-histogram, asymmetry, and RRF analysis. In Mantid, see
`RemoveExpDecay v1 <https://docs.mantidproject.org/nightly/algorithms/RemoveExpDecay-v1.html>`_,
`MuonMaxent v1 <https://docs.mantidproject.org/nightly/algorithms/MuonMaxent-v1.html>`_,
and `RRFMuon v1 <https://docs.mantidproject.org/nightly/algorithms/RRFMuon-v1.html>`_.

Phase, clock, and geometry
~~~~~~~~~~~~~~~~~~~~~~~~~~

An oscillatory model phase belongs to a chosen clock and sign convention.
Asymmetry uses :math:`\cos(2\pi\nu t+\phi)` with μs, MHz, and model radians;
FFT manual tables and MaxEnt group controls use degrees. A positive manual FFT
phase rotates the complex spectrum by :math:`e^{-i\phi}`. The first retained
sample contributes a separate origin phase. Changing t0, source-response
origin, correction recipe, or observation window can change a phase estimate;
matching group IDs alone is insufficient to transfer it.

The manual FFT table and sampled **Auto Phase** estimates are distinct
from the fitted MaxEnt group-phase store. MaxEnt's **Use fitted phases** and
**Send phases to fit** exchange group phases with the grouped time-domain fit
using degree/radian conversion. They do not synchronise the FFT table. See
:ref:`fourier-phase-estimation` and
:doc:`/reference/grouped_time_domain_fitting` for the supported models and inputs.

WiMDA 1.326 provides manual degree-valued Fourier group phases and fitted
MaxEnt group phases. Their use belongs to those respective workflows; native
numerical transfer to Asymmetry's FFT phase is unverified here. musrfit's
``TFieldCos`` model uses :math:`\cos(2\pi\nu t+\pi\phi/180)`, with phase
in degrees, according to the `musrfit model manual
<https://lmu.pages.psi.ch/musrfit-docu/user-manual.html>`_. This names one model
convention, not every component or a verified manual Fourier-display exchange.

In Mantid, `CalMuonDetectorPhases v1
<https://docs.mantidproject.org/nightly/algorithms/CalMuonDetectorPhases-v1.html>`_
first determines a common frequency and then detector amplitudes and phases.
`PhaseQuad v1 <https://docs.mantidproject.org/nightly/algorithms/PhaseQuad-v1.html>`_
uses a detector phase/amplitude table to form two quadratures. Its detector
model is :math:`\cos(\omega t-\phi_i)`, with table phases in radians.
At the same origin this corresponds to :math:`\phi=-\phi_i` in Asymmetry's
plus-phase model, followed by conversion to degrees for its group controls.
A shifted origin requires the additional frequency-dependent clock conversion.
These are distinct
from a sampled common-phase FFT fit per group. A group label or phase table does
not establish orthogonal physical axes. Authored vector geometry and measured
timing remain explicit assumptions; see :ref:`vector-geometry-assumptions`.

The two-projection rotation uses :math:`x=P_x-b_x`,
:math:`y=(P_y-b_y)/g`, and :math:`\theta=2\pi\nu_{\rm RF}t+\varphi_{\rm RF}`:
:math:`x'=x\cos\theta+s y\sin\theta`,
:math:`y'=x\sin\theta-s y\cos\theta`.
The sense :math:`s` and second-axis sign must be mapped explicitly to Mantid's
real/imaginary convention, including nonzero phase. With Mantid inputs
:math:`R=x,I=s y` at the same :math:`\theta`, Asymmetry's outputs are
:math:`x'=R'` and :math:`y'=-I'`. Mantid RRFMuon uses radians for its
phase input; Asymmetry's frame phase controls use degrees. The authoritative
:doc:`/reference/rotating_frame_projection` describes the mapping, shared
acquisition/reference/background counting covariance, conditional errors, and
unequal-baseline pooled G+R safeguard. Lab G+R and rotating Red, Green, and G−R
retain their supported behaviour.

A single real ratio cannot supply two measured projections pointwise. Its
:doc:`/reference/rotating_frame` display instead mixes with a complex carrier
and filters. Same-bin quadrature covariance enters first-order magnitude errors;
filtered bins remain correlated and low-signal magnitude remains biased.
Correlated or anisotropic quadratures do not imply an exact Rician law. Corrected
marginal bars establish neither full cross-bin covariance nor an independent
joint likelihood; see :ref:`scalar-rrf-magnitude-errors` and
:ref:`vector-rotation-counting-errors`. The laboratory observations remain the scalar fit inputs;
grouped count fits retain the separate :ref:`grouped-count-model`.

Other analysis entry points
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Asymmetry provides :doc:`/reference/composite_models` for model expressions,
:doc:`/reference/fitting` for minimisers and conditional parameter errors,
:doc:`/reference/fit_wizard` for candidate-model comparison, and
:doc:`/reference/parameter_trending` for secondary fits to run parameters.
The model reference covers :doc:`/reference/fit_functions/index`;
:doc:`/reference/frequency_finishers` and :doc:`/reference/spectral_moments`
describe spectral estimators and derived quantities with their own assumptions.
:doc:`/reference/alc_mode`, :doc:`/reference/simulation`,
:doc:`/reference/user_functions`, and :doc:`/reference/negative_muon_analysis`
cover the separate scan, synthetic-data, plugin, and negative-muon workflows.
:doc:`/reference/project_files` describes saved analysis state and compatibility.
These are method pointers, without a claim that another program lacks an
analogous or extensible workflow.

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
--------------------------------

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
calibration for each condition. General cross-period/reference likelihoods remain outside the conditional
error contract; integrated scalars retain supported shared counting responses.

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

Integrated counting responses and fit support
---------------------------------------------

Asymmetry's count integral sums all selected native counts before classifying
the ratio. Its explicit time-bin mean adopts the ordinary usable-ratio policy,
with exclusive omission counts. Undefined ratios and defined zero-variance
results have no Gaussian weight. Supported signed observations are retained.
The first-order gradients and conditional calibration convention are described
in :doc:`/reference/alc_mode` and :ref:`asymmetry-counting-errors`.

A reused reference contributes one set of counting observations. Its signed
responses are combined before contraction, including within a scalar G−R.
For independent equal sample periods, each four bins F=600/B=400, subtracting
one reference of four bins F=60/B=40 at unity scale gives G−R=0 and standard
error 0.024343224778007384. This independently derived linear-propagation
control follows from cancellation of the identical reference responses; it is
not a native-program comparison. Unequal responses do not cancel in general.

Asymmetry also checks whether the requested independent-error fit reuses known
counting observations with nonzero cross-point covariance. That fit is
unavailable, while inspection, export, and independent subsets remain available.
Unknown identity retains an explicit distinct-observation assumption. This is
a bounded safeguard, not a general joint likelihood or calibration covariance.

WiMDA 1.326's dedicated ALC-column route and Mantid's ``AsymmetryCalc`` have
different error and sparse-count conventions. The checked musrfit 1.12.1
(708f777b, ROOT 6.40.04) positive-count packing comparison concerns 22 interior
points per case within 5×10⁻⁷ export tolerance; first/endpoints remain qualified.
These workflows do not establish a common reused-observation covariance
contract. No fresh native WiMDA ALC/RF replay or full Mantid execution is claimed,
and the analytic shared-reference control does not extend their agreement.

Shared counting noise in vector rotation
----------------------------------------

Asymmetry's vector reduction retains acquisition/reference-bin identity,
coefficients, and counting variance across projections. Its transverse rotation
propagates this covariance through the actual gain, sense, and phase. Range and
tail-background estimator influence survives rotation and display packing; cohort
tail-fit uncertainty uses retained acquisition variance while preserving fitted
levels. Errors remain conditional on calibration, baselines, gain, and correction
values. Corrected marginals do not establish an independent joint likelihood or
validate an overlapping-group MaxEnt objective.

In Mantid, `RRFMuon v1
<https://docs.mantidproject.org/nightly/algorithms/RRFMuon-v1.html>`_ provides a
comparable two-spectrum coordinate rotation after matching signs, phase, and
units. The v1 documentation associated with the 6.16 development snapshot is a
formula comparison, not native execution or shared-detector error agreement.
WiMDA 1.326 and musrfit 1.12.1 scalar RRF workflows use different inputs and
smoothing or packing; no equivalent covariance contract for these overlapping
ratios is established. Independent analytic and seeded Poisson controls support
Asymmetry's propagation. Acquired ISIS and PSI file execution, including musrfit
example ROOT files, checks the changed operation with authored groupings; it does
not establish acquired vector geometry or reference-program numerical agreement.
See :doc:`/reference/rotating_frame_projection` for the equations and limitations.

Scalar rotating-frame magnitude
--------------------------------

Asymmetry's scalar FIR display forms both quadratures from one asymmetry.
It propagates their signed same-bin covariance through squared filter weights
and includes the cross term in first-order **Magnitude** errors. Independent
identity-filter and measured linear-response controls check this accounting,
including phase invariance, heteroscedastic input errors, and finite masks.
The errors assume independent supplied scalar input noise; unknown input temporal
correlations, low-signal magnitude bias, and filtered-bin correlations remain.
Fits continue to use laboratory observations.

WiMDA 1.326 uses real demodulation, box smoothing, and a linear-error convention;
Asymmetry retains a separate core comparison mode for that convention.
musrfit 1.12.1 scalar RRF preparation and packing differ from this FIR, while
Mantid RRFMuon v1 rotates two supplied components. No new native-program magnitude
error agreement is established. Execution on unchanged ISIS HDF4/HDF5, PSI
BIN/MDU, and musrfit 1.12.1 example ROOT inputs checks the scalar operation with
authored pair/frame settings, not acquired physical truth or reference agreement.
See :doc:`/reference/rotating_frame` for equations, sources, and limitations.

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

Ordinary time-asymmetry models in Asymmetry offer an explicit fixed double-delta
response with equal incident strengths, survival weighting, and midpoint origin.
The complete source expression is evaluated only at live ages and normalised;
point evaluation does not claim integrated-bin or finite-width onset accuracy.
The setting follows models through wizard candidates and saved/replayed fits.
WiMDA 1.326's **Simulate Data** workflow accounts for **Double Pulse** and
reduces the pre-second-arrival count population. Its early-population
approximation is not established as numerically equivalent to a matched
two-birth count generator here. Asymmetry's ordinary response does not supply
that generator to the count-based pull diagnostic, which is unavailable for
positive separation. Mantid DoublePulseFit's fitting/error facilities and the
musrfit authored-model check do not establish repeated-simulation pull coverage.

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

Fixed supplied TF count response
--------------------------------

Asymmetry's TF alpha/beta calibration uses equal incident instantaneous births,
absolute muon survival, and complete age-local polarisation, with each flat
background added once. Beta is fixed at one or fitted as a backward amplitude
ratio. All four envelopes share the same source settings, observations, and
objective; minimum summed F+B cost selects a usable candidate. Counts are
predicted at packed-bin centres, without rate integration. Errors include all
fitted nuisances and are conditional on fixed source and selection assumptions.

WiMDA 1.326 **Double Pulse** F/B/alpha fitting is the beta-one precedent;
beta calibration is not verified WiMDA functionality. Mantid `DoublePulseFit v1
<https://docs.mantidproject.org/nightly/algorithms/DoublePulseFit-v1.html>`_
provides supplied shifted functions and weights, a forward-model analogue,
without establishing the same count calibration or covariance. musrfit 1.12.1
count models use an authored theory and reciprocal count balance; comparisons
must match amplitude units, phase orientation, lifetime, and objective.
Existing post-arrival shifted-TF polarisation agreement does not validate onset,
bin integration, or calibration errors. No four-program estimator or uncertainty
agreement is claimed. See :ref:`alpha-calibration` for the count equations.

Matched musrfit 1.12.1/ROOT 6.40.04 global-histogram likelihood controls on
rounded double-pulse counts agree within 0.0001 absolute alpha/beta for beta
fixed at one and free beta. The comparison uses 16 ns bins, raw bins 13–498,
zero fixed backgrounds, fixed lifetime, a shared free phase (native degrees
with a 180-degree backward offset), and reciprocal native count balance.
The native post-arrival damped-cosine representation absorbs the supplied
source into its amplitude, phase, and normalisation. This checks point estimates
after both births; it does not test native onset, bin integration, or error
equivalence. Independent count and full-nuisance information controls provide
the separate scientific targets.


Count-integral condition scans
------------------------------

Asymmetry projects capture completed baseline, peak, and RF fit results and
restore their numerical readouts and sampled overlays without optimisation.
Settings-only histories require explicit fitting for new results. This is an
Asymmetry history safeguard; the comparisons below do not establish identical
project-restoration behaviour in WiMDA, Mantid, or musrfit.


Asymmetry sums the selected corrected native F/B counts before forming
:math:`(F-\alpha B)/(\beta F+\alpha B)` for ordinary and RF scans. G−R is
:math:`A_G-A_R` with shared chosen calibration; G+R pools prepared counts before
the ratio. The explicit ``differential`` method is an equal-time-bin mean,
distinct from a field derivative. Windows include whole bins by native centres;
errors remain conditional on calibration. See :doc:`/reference/alc_mode`.

In Mantid's `PlotAsymmetryByLogValue
<https://docs.mantidproject.org/v6.11.0/algorithms/PlotAsymmetryByLogValue-v1.html>`_,
single-period Integral uses count-first ordering (algorithm v1, 6.11.0
documentation). Its dual-period detector-wise
contrast and Differential operation are different observables. For dual periods,
define :math:`c_F=(F_G-F_R)/(F_G+F_R)` and
:math:`c_B=(B_G-B_R)/(B_G+B_R)`; its Integral contrast is :math:`c_B-c_F`,
not :math:`A_G-A_R`. Its single-period Differential integrates pointwise
asymmetry along time rather than taking Asymmetry's equal-time-bin mean.
Integration's
histogram-edge bounds also differ from Asymmetry's centre selection. Full Mantid
execution is unavailable for this comparison.

WiMDA 1.326's dedicated ALC route starts from integrated acquisition columns;
arbitrary histogram-window and uncertainty equivalence is not established. Its
`public overview <https://shadow.nd.rl.ac.uk/wimda/>`_ describes the program;
no fresh native ALC/RF numeric replay supports this comparison. WiMDA's
**Differential ALC** differences adjacent table entries with its own gap cutoff
and per-kG scale. Asymmetry uses pair midpoints, positive spacing, an explicit
optional ``max_gap``, and fractional or percent units per selected G/K/run unit.
These field-table operations differ from Mantid's time Differential.

musrfit 1.12.1 (revision 708f777b, ROOT 6.40.04) packs counts before forming its
fitting ratio, as described in its `user manual
<https://lmu.web.psi.ch/musrfit/user/html/user-manual.html>`_. Executed positive-count
and alternating-count controls with α=β=1, fixed zero background, and packing 20
agree on 22 interior values/errors per case within 5 × 10⁻⁷ (ASCII precision).
The first packed value differs and the final nominal group is absent. Those
unresolved boundaries are excluded from agreement; no whole-output or
four-program numerical equivalence is claimed. Non-unit balance comparisons
must also match musrfit's reciprocal-alpha convention and fitting-coordinate
transformation.

Asymmetry propagates retained shared counting responses into each scalar, with
errors conditional on the chosen calibration and corrections. Its two-stage
baseline/peak inference propagates the fitted baseline's uncertainty with fixed
model and regions; this does not establish an identical covariance contract in
the other programs. Asymmetry also checks actual fit selections for unusable
Gaussian observations and known nonzero source covariance. An independent
selected window remains usable. It also restricts its scan fit adapters to
original observations: adjacent derivatives share their middle observation,
and marginal errors alone do not establish an independent likelihood.
Derivative display/export remains available, with unavailable errors for known
dependent input pairs. These are operation-specific Asymmetry support limits,
not universal prohibitions on derivative or correlated-data analysis.

Effective reduction records
---------------------------

Asymmetry captures an immutable effective recipe with core reductions and supported
transforms; see :doc:`/explanation/reduction_pipeline`. Work-directory, scan,
project-result, and supported export records retain the captured account. It
is distinct from raw replay, acquisition-content verification, and cache validity.
Private records retain locators; publishable recipe fields omit nested locators
and diagnostic prose without changing numerical identity.

musrfit 1.12.1 uses an explicit analysis configuration and writes fitted parameters
to its ``.msr`` result while ``.mlog`` retains the preceding parameter input
(`manual <https://lmu.pages.psi.ch/musrfit-docu/user-manual.html>`_). A bounded
native 1.12.1/ROOT 6.40.04 request/result-file check confirms that behaviour; it
does not establish acquisition hashing or general reduction equivalence. Mantid's
public `workspace-history script generation
<https://docs.mantidproject.org/nightly/algorithms/GeneratePythonScript-v1.html>`_
describes applied operations separately from saved data and external replay inputs.
This comparison has no matched native Mantid run. WiMDA 1.326 provides grouping and
correction annotations in reduced exports; its `overview/manual
<https://shadow.nd.rl.ac.uk/wimda/>`_ supplies workflow context, without a new native
execution or a claim of identical canonical identity safeguards.

Asymmetry also checks canonical recipe identity independently of dictionary order,
local paths, and display labels. The identity retains available source references;
it does not verify file contents or recover unknown historical settings.


Asymmetry also checks actual sampled observations, retained errors, usable support,
and interpretation when comparing saved single fits. Unknown or different input
histories remain available without a shared ranking or current residuals. This
safeguard is separate from model structure and recipe identity; it does not change
the fitting objective, information-criterion definitions, or TF calibration selection.

Asymmetry also checks an explicit scientific-computation revision before automatic
persistent reuse and current saved-result comparisons. This is an additional
compatibility safeguard, separate from recipe, observation, and source identity.
Historical results retain their old or unknown revision; ordinary documentation
and package-version changes do not invalidate calculations. The operation-history
and request/result-file behaviours described above do not establish an identical
automatic computation-revision contract in WiMDA, Mantid, or musrfit. This check
adds no muon-spin estimator and makes no new native numerical-equivalence claim.

Asymmetry also checks complete acquisition contents through an explicit cancellable
core comparison API. Captured expected identities and newly observed candidate
bytes remain separate; identical relocated bytes match independently of timestamp,
and legacy expectations remain unknown. Capture binds the fact to a stable new
read; a retained reference cache keeps the identity of its own counts. This is an
additional Asymmetry provenance safeguard, not an established identical policy in
WiMDA, Mantid, or musrfit. It changes no reduction estimator and makes no new native
numerical-agreement claim. Project reopening runs this comparison before
application, with explicit history, selected current-input replacement, and
cancellation choices for known changed inputs. Saved historical records and
calibration expectations remain intact; historical-only differences do not block
valid active inputs. API/CLI actions use the same owner-aware core operation.
These choices do not establish an identical replacement workflow in the other
programs.
