.. _background-ladder:

The Background Ladder: Which Stage Removes What
===============================================

Several different controls across Asymmetry carry the word "background", and
they act at different points in the pipeline on different representations of
the data. They are *not* alternatives to one another and they do *not* stack
into a double subtraction — each removes a different manifestation of the same
underlying quantity, at the stage where that manifestation appears. This page
is the map: what each stage removes, where it sits in the chain, and when to
turn it on. For the mechanics of any one stage, follow the cross-reference.

The physical quantity underneath most of these is a single thing: the
**steady, uncorrelated count rate** — particles that reach a detector without
being correlated with a muon decaying in the sample (decay positrons from
muons stopped upstream, room background, detector noise that survives the
coincidence logic). At a continuous source the rate is significant; at a
pulsed source the beam duty factor suppresses it to nearly nothing (Blundell,
De Renzi, Lancaster & Pratt, *Muon Spectroscopy: An Introduction*). One flat
rate in the raw histograms shows up as a baseline in the counts, a
zero-frequency feature in the FFT, a central peak in a zero-field MaxEnt
distribution, and a pedestal under the spectrum — so it is handled, where it
matters, at each of those stages.

The ladder at a glance
----------------------

.. list-table::
   :header-rows: 1
   :widths: 26 16 30 28

   * - Stage
     - Operates on
     - Removes
     - When
   * - Grouping background mode
       (:ref:`backgrounds <background-correction>`)
     - raw counts, pre-asymmetry / pre-FFT
     - the steady flat rate before the asymmetry ratio (one of four mutually
       exclusive modes, applied once)
     - enable per dataset at reduction time; feeds every downstream analysis
   * - Count-fit background nuisance
       (:doc:`../count_domain_fitting`)
     - raw counts, in the fit
     - a flat rate fitted *jointly* with the count model — independent of the
       grouping correction
     - when fitting raw counts and the grouping rate is not already trustworthy
   * - σ-clip baseline
       (:ref:`Robust baseline offset <robust-baseline-offset>`)
     - spectrum, post-FFT, display
     - the spectral noise floor (a positive pedestal of redistributed counting
       noise)
     - when measuring peak heights or areas above a true zero
   * - MaxEnt SpecBG
       (:doc:`../fourier_analysis`)
     - spectrum, display, ZF/LF
     - the zero-frequency central peak of a ZF/LF field distribution
     - ZF/LF field distributions with weak satellite structure buried under the
       central peak

Stage 1 — the grouping background mode (pre-asymmetry, pre-FFT)
---------------------------------------------------------------------

The grouping dialog's **Background** selector subtracts the steady flat rate
from the grouped forward/backward counts *before* the asymmetry ratio is
formed. It offers four modes — **Fixed values**, **Range average** (pre-t0),
**Tail fit** (late-time), and **Background run** — and they are **mutually
exclusive**: exactly one applies to a given dataset, through a single
correction chokepoint, and it is applied **once**. See
:ref:`background-correction` for how to choose among the four and for the
mathematics of each.

This is the only background stage that reaches *every* downstream analysis.
The asymmetry curve, the α estimators, the per-group time-domain fits and the
Fourier input all consume the same corrected group sums. In particular, the
FFT input is **rebuilt from the current grouping**, so the grouping
background mode applies to the transform too — there is no separate FFT-only
background control, and no second subtraction (see the FFT background
discussion in :doc:`../fourier_analysis` and the Fourier panel's *Background*
status line). A flat rate matters most here, because the
grouped FFT signal is lifetime-corrected by :math:`e^{t/\tau_\mu}` before the
transform: an unsubtracted constant becomes a growing ramp that dumps spurious
power into the low-frequency bins.

*When to enable.* Set a grouping background mode whenever the uncorrelated
rate is non-negligible — routinely for continuous-source data (Range average),
and for pulsed data analysed to long times (Tail fit), where even the small
residual rate biases weak late-time relaxation.

.. _background-ladder-count-fit-trap:

Stage 2 — the count-fit background nuisance (in the fit, on raw counts)
---------------------------------------------------------------------------

Count-domain fitting (:doc:`../count_domain_fitting`) fits a flat background
term ``bg`` jointly with the count model :math:`N_0 e^{-t/\tau_\mu}[1 + s A
P(t)] + \mathrm{bg}`. This term is fitted against the **raw counts**.

.. warning::

   **The grouping background correction never reaches the count fit.** The
   count fit always consumes raw histograms with deadtime applied — not the
   background-corrected group sums that the asymmetry and Fourier paths use. A
   user who believes the grouping correction has already removed the flat rate,
   and therefore fixes ``bg = 0``, will bias :math:`N_0` and α: the raw counts
   still contain the full background. Let ``bg`` float (or seed it from the
   grouping value) — do not assume Stage 1 has done the job for the count fit.

This is not a double subtraction; it is the *opposite* trap — a stage that
looks downstream of the grouping correction but is not. The two are
independent measurements of the same flat rate: Stage 1 stores it in the
grouping and applies it to the reduced data; Stage 2 measures it from the raw
counts the fit actually sees.

*When to enable.* Whenever you fit raw counts and want the background
determined from the same data and weighting as the rest of the count model —
the statistically clean route when a separate grouping estimate is unavailable
or untrusted. The fitted value is a legitimate measurement of the steady rate
and can be promoted into the grouping as a ``Fixed`` value (see
:doc:`../count_domain_fitting`).

Stage 3 — the σ-clip baseline (post-FFT, display)
-------------------------------------------------

A power or magnitude spectrum sits on a positive pedestal from the
redistributed counting noise — distinct from the flat *time-domain* rate of
Stages 1–2. The Fourier panel's robust **σ-clip baseline** estimates that
pedestal and subtracts it from the displayed spectrum, leaving sharp peaks
intact (they are rejected as outliers during the estimate). It is a
**display-channel** operation: it never alters the underlying transform, and
its converged width doubles as the noise estimate behind the signal-to-noise
readout. See :ref:`Robust baseline offset <robust-baseline-offset>` in
:doc:`../frequency_finishers` for the σ-clip mechanism and its pitfalls.

*When to enable.* Whenever peak heights or areas must be measured above a true
zero — comparing intensities across runs, integrating a line. It is orthogonal
to Stages 1–2: subtracting the time-domain rate flattens the low-frequency
ramp, while the σ-clip baseline removes the spectral noise floor that remains.

Stage 4 — MaxEnt SpecBG (post-reconstruction, display, ZF/LF)
-------------------------------------------------------------

In zero or longitudinal field the MaxEnt reconstruction returns a **field
distribution** :math:`p(B)` dominated by a strong zero-frequency central peak,
which can bury weak satellite structure. The **Zero-frequency background
(SpecBG)** control (ZF/LF mode only) subtracts a zero-centred pseudo-Voigt
model of that central peak from the *displayed* spectrum. Like the σ-clip
baseline it is **display-only** and never alters the reconstructed spectrum;
unlike it, it targets one specific feature — the ZF/LF central peak — rather
than a broadband pedestal. See :doc:`../fourier_analysis` for the SpecBG
controls.

*When to enable.* Only for ZF/LF MaxEnt field distributions where a central
peak is hiding the structure you care about. It is unrelated to the
transverse-field precession lines that the other stages serve.

Why they coexist
----------------

Because each stage acts on a different representation — raw counts, the fitted
count model, the spectral display channel, the reconstructed distribution —
turning on more than one does not subtract the same rate twice. The worst case
is mild redundancy: a pre-FFT-subtracted flat rate leaves a smaller
zero-frequency feature for SpecBG to model, not a negative artefact. The one
genuine trap is the count-fit asymmetry above — the grouping correction does
*not* propagate into the count fit, so its background must be handled there in
its own right.

See also
--------

* :ref:`Background correction <background-correction>` — the four mutually
  exclusive grouping modes and how to choose among them.
* :doc:`../count_domain_fitting` — the count model and its ``bg`` nuisance term.
* :doc:`../frequency_finishers` — the σ-clip baseline and other post-FFT
  conditioning.
* :doc:`../fourier_analysis` — the FFT background inheritance and MaxEnt SpecBG.

**References**

- S. J. Blundell, R. De Renzi, T. Lancaster, and F. L. Pratt, *Muon
  Spectroscopy: An Introduction* (Oxford University Press, Oxford, 2022) — the
  steady background count from uncorrelated detector hits at continuous
  sources, and its suppression at pulsed sources.
