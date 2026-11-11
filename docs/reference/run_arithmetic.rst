.. _run-arithmetic:

Run arithmetic: co-add and co-subtract
======================================

Repeated measurements can be pooled by adding detector counts before forming
an asymmetry. A reference subtraction instead forms a difference of independent
counts, with its own propagated variance. Neither operation averages separately
reduced asymmetry curves.

In the Data Browser select two or more runs and choose **Co-add Selected**.
Ordinary co-add uses unit weights. **Subtract Selected (signed)…** subtracts
every selected run after the first at unit weight. **Subtract Reference Run…**
selects a reference with Automatic exposure scaling. **Separate Combined**
restores the constituent rows. There is no GUI editor for arbitrary run weights.


.. figure:: /_generated/screenshots/run_arithmetic.png
   :alt: Synthetic acquired runs with different clocks and correction facts combined in the Data Browser

   **Co-add Selected** accepts shared authored intent while retaining the sources'
   own clocks and file correction facts. The synthetic result contains only their
   common observed good support; **Separate Combined** restores its source rows.

Independent linear arithmetic
-----------------------------

The scriptable ``combine_runs`` operation accepts finite signed ``scales``.
For ``sign=+1`` all coefficients equal the supplied scales; for ``sign=-1``
the first coefficient retains its scale and each later coefficient is negated.
With independent counts :math:`N_j` and variances :math:`V_j`,

.. math::

   Y = \sum_j c_j N_j, \qquad V_Y = \sum_j c_j^2 V_j.

Genuinely raw independent Poisson inputs use :math:`V_j=N_j`; supplied
independent variance is retained. For example, :math:`10+2\times20=50` has
variance :math:`10+4\times20=90`. A unit sum with supplied variances 25 and 49
has variance 74. Sample/reference counts 100 and 20 with scales ``[2, 0.5]``
and ``sign=-1`` give 190; the sample coefficient is applied too.

Zero coefficients are recorded but do not enter compatibility, noise, or overlap
requirements. Asymmetry also checks that coefficients are finite and at least
one is nonzero: an invalid request raises ``CombineError`` without changing
its sources. Signed source counts require supplied independent variance for
pooling; a positive pooled total cannot establish missing source noise.
``Histogram.count_variance`` records unknown independent noise as NaN, distinct
from zero variance. Asymmetry also checks noise only on consumed detectors/bins
before correction or ratio/count fitting. Unknown noise there requires supplied
variance; unused raw context remains inspectable and does not block a selected
reduction. Fractional sources without supplied noise cannot become Poisson
observations by pooling to integer totals.

For full period payloads the same coefficients act on every counterpart.
Asymmetry also checks that contributing payloads have matching period counts,
detector counts, and native bin widths. Mixed ordinary/period payloads or
mismatched periods require explicit corresponding period selection; the
operation does not silently discard Green. Detector correspondence follows
channel order; matching array shapes do not establish physical detector identity.

Asymmetry also checks known direct source/acquired-period identity and shared
count buffers to prevent treating reused noise as independent. Equal run
numbers or equal count values alone do not establish overlap. Unknown identity
uses a recorded independence assumption. This is a direct-input safeguard,
not a covariance model for nested combinations or shared references.

Reduction, errors, and limits
-----------------------------

The result carries counts and canonical ``Histogram.count_variance`` through
normal reduction, integral/Fourier preparation, and counts-first crop/packing.
Weighted, signed, or supplied-noise observations cannot enter an exact raw
Poisson count likelihood merely because their final values are positive
integers. Independent raw unit sums retain the supported Poisson limit.
Gaussian count fits use retained variance, including variance below one,
without an arithmetic error floor; unusable uncertainties receive the existing
fit-input diagnostic. Ratio fits use their supported observation selection.
Errors remain conditional on fixed weights, alpha/beta, and correction facts.

Pooling differs from averaging even at :math:`\alpha=1`: the ratio is nonlinear
in counts. Runs with asymmetries 20% and 80% and totals 100 and 300 give pooled
65%, whereas their arithmetic mean is 50%.

Each original acquisition retains its file coefficients, supported deadtime
normaliser, supplied or unknown noise, and acquired clock. Reduction resolves
current authored grouping, correction, and timing intent against those facts.
It corrects required original detectors first, applies :math:`c_j` to counts
and :math:`c_j^2` to variance, and pools on common observed good support.
Whole-bin alignment records fractional timing residuals; it neither interpolates
nor pads missing observations. The combined raw histograms are inspection views,
not inputs to nonlinear correction or count-first replay. Ordinary runs,
corresponding periods, and mapped Red/Green sets use this preparation.

A shared explicit alpha/beta belongs to the requested result. Group background
is applied once after pooling; a selected reference contributes its own corrected
counts, clock, and squared-scale variance once. Missing required File deadtime
coefficients skip deadtime for the entire requested result. Zero terms, ignored
periods, and unconsumed detectors/bins cannot trigger availability or saturation
failures. Asymmetry also checks unknown original noise before nonlinear correction;
packing or integer-looking sums cannot supply missing noise evidence.

Only compatible evidenced exposures in a unit acquisition sum establish pooled
reference exposure. Arbitrary signed weights leave it unknown; a supported
sample/reference subtraction can retain its sample exposure. A whole-run
PSI/ROOT deadtime denominator of one is not a measured pulse exposure.
**Automatic** retains its visible unity assumption when exposure is unknown;
**Manual** retains the authored override. Correction uses the original
normalisers regardless of the inspection view's frame total.

Grouped Gaussian observations retain the prepared counts and noise. A fitted
count-loss model cannot replace several acquired normalisers with one surrogate;
that request receives an actionable fit-input error. TF calibration requires an
original acquisition; apply its shared calibration to the combination.

Inspection frame totals accumulate for co-add; reference subtraction retains
the sample frame total. Displayed field and temperature use the current good-frame
weighted summary with constituent values and ranges in provenance; this label
does not imply one physical sample condition.

Reference subtraction
---------------------

**Subtract Reference Run…** uses the existing **Automatic** exposure policy:
compatible evidenced exposures give the sample/reference ratio; unknown
exposure assumes unity and records that assumption. The core explicit signed
coefficients do not change the positive **Manual** background-reference control.
Both sample and reference variance contribute with the squared scale.

This produces an asymmetry of differenced detector counts, not a difference of
asymmetries. Negative counts remain inspectable with their retained noise and
are ineligible for raw Poisson fitting. A physical sample/holder contribution
belongs in an additive polarisation model; see
:doc:`data_reduction/backgrounds` for that distinction and reference-background
preparation, which is a separate reduction operation.

Provenance and projects
-----------------------

The combined run records its constituents, authored scales, sign, operation,
independence/correspondence assumptions, and negative-count tally. Source arrays
remain unchanged. Counts-first observation snapshots own their arrays and
resolved settings; an attached Run alone is not a replay recipe. Lightweight
fit records omit histogram replay buffers.

Projects retain combination definitions and reconstruct rows from sources.
Saved output arrays and fit records retain their recorded meaning without
automatic refitting. Numerical arithmetic identities prevent historical scores
from matching newly prepared observations. A definition without stored output
arrays reconstructs through current arithmetic.

Compared with other programs
-----------------------------

WiMDA 1.326 pools raw counts before correction and provides unit signed **Co-sub**; its **FileBG**
reference subtraction uses a frame ratio. These are count operations, not
asymmetry contrasts. WiMDA retains the master field and temperature, whereas
Asymmetry uses the summary described above. The `WiMDA manual/homepage
<https://shadow.nd.rl.ac.uk/wimda/>`_ describes the application; these checked
features do not establish identical uncertainty conventions.

musrfit 1.12.1 distinguishes standalone ``addRun`` from fit-time ``ADDRUN``.
The `musrfit manual
<https://lmu.web.psi.ch/musrfit/user/html/user-manual.html>`_ documents histogram
addition and whole-bin timing shifts. Matched correction-off fit-time ADDRUN
controls agree on unit-addition values and interior asymmetry values formed
from structured count inputs, to a tolerance of :math:`5\times10^{-6}` in fractional asymmetry. That does not
establish endpoint, error, arbitrary signed-weight, or likelihood equivalence.
At unequal correction facts the tested fit-time ADDRUN uses the primary RUN
correction context for added data; reversing the primary run changes the result.
Asymmetry deliberately uses each original's own correction facts, so its result
is independent of the primary run under shared authored intent. This unequal-fact
case is a deliberate difference, not native numerical agreement.

Mantid's documented 6.16 development `Scale v1
<https://docs.mantidproject.org/nightly/algorithms/Scale-v1.html>`_ and `Plus v1
<https://docs.mantidproject.org/nightly/algorithms/Plus-v1.html>`_ provide the
weighted-arithmetic analogue: multiplication scales errors and independent
addition adds variances. `Minus v1
<https://docs.mantidproject.org/nightly/algorithms/Minus-v1.html>`_ subtracts the
supplied quantity, so subtracting counts and subtracting asymmetries are
different requests. These are documented arithmetic comparisons, not native
Mantid numerical checks. `MuonPreProcess v1
<https://docs.mantidproject.org/nightly/algorithms/MuonPreProcess-v1.html>`_
can correct each workspace before merging. Generic merge alone does not choose
the correction order. Asymmetry's direct-overlap and likelihood safeguards
have the scope stated above; they are not attributed to these other programs.

References
----------

- S. J. Blundell, R. De Renzi, T. Lancaster, and F. L. Pratt, *Muon
  Spectroscopy: An Introduction* (Oxford University Press, Oxford, 2022).
- F. L. Pratt, Physica B **289–290**, 710 (2000).
