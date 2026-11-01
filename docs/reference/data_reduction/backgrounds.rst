.. _background-correction:

Background correction
=====================

Two different things are called "background" in μSR, and they must not be
confused. The first is the **uncorrelated, time-independent count rate** in
the raw histograms — particles unrelated to a muon decaying in the sample
(for example decay positrons from muons stopped upstream) that survive the
coincidence logic. This page is about removing that rate from the counts
before the asymmetry is formed. The second is the **background asymmetry**
from muons that genuinely stop in the sample holder or cryostat — a real
muon signal, handled in the fit model (and calibrated with silver/hematite
reference measurements), not by the corrections here.

Subtracting a count-rate background matters because the asymmetry is a ratio:
a flat rate :math:`\eta` added to both groups pulls :math:`A(t)` toward zero
by a factor that grows as :math:`e^{t/\tau_\mu}` relative to the muon signal.
At a continuous source (PSI, TRIUMF) the uncorrelated rate is significant and
limits useful data to roughly 8 μs; at a pulsed source (ISIS) the beam duty
factor of about :math:`1.6\times10^{-3}` suppresses it to nearly nothing —
but "nearly nothing" can still matter when fitting weak relaxation out to
20–32 μs, which is exactly the regime pulsed sources exist for.

The grouping dialog's **Background** selector chooses the mode per dataset;
modes that cannot apply to the data type are greyed out. The correction is
applied to the grouped forward/backward counts before the asymmetry ratio,
with the subtraction's shared uncertainty propagated through the merged counts.

Choosing a mode
---------------

==========================  =====================================================
Data in hand                Mode
==========================  =====================================================
Continuous-source data      **Range average** — the pre-t0 bins measure the
                            uncorrelated rate directly
Pulsed-source data          **Tail fit** — no pre-t0 region exists; estimate
                            from the late-time spectrum
Dedicated reference run     **Background run** — sample-holder / silver /
                            laser-off measurement
Known constants             **Fixed values** (from a stored grouping)
==========================  =====================================================

Range average (pre-t0)
----------------------

The mean count over a window of bins *before* time zero, subtracted per
group — the musrfit convention, with its default window from
:math:`0.1\,t_0` to :math:`0.6\,t_0` and beam-period-aware trimming at PSI
and TRIUMF. Only continuous-source data have a pre-t0 region: muons arrive
continuously, so the histogram records the uncorrelated rate before each
muon's own clock starts. Pulsed files begin at the muon pulse, so this mode
is unavailable for them.

*When to use this.* The default for PSI/TRIUMF data. It is a direct
measurement, not a fit — prefer it over the tail fit whenever a pre-t0
region exists.

.. note::

   *Scripting note: the beam-period trimming engages by itself.* Trimming the
   window to a whole number of accelerator periods needs to know the facility,
   and :func:`~asymmetry.core.transform.background.apply_grouped_background_correction`,
   :func:`~asymmetry.core.transform.reduce.corrected_grouped_counts` and
   :func:`~asymmetry.core.transform.reduce.reduce_grouped_asymmetry` used to take
   a ``facility`` string defaulting to ``""`` — no facility, no trimming, no
   warning. The GUI passed it; a script that did not know to reduced the same run
   with an untrimmed window. The default is now ``None``, meaning *derive it*:
   pass ``metadata=run.metadata`` (the reduction functions receive histograms
   rather than a run, so they cannot fetch it themselves) and the label resolves
   through :func:`~asymmetry.core.transform.background.resolve_facility`, exactly
   as the GUI's explicit string always did. ``facility=""`` remains available as
   the deliberate "no trimming" opt-out, and any non-empty string still overrides.

Declared range endpoints are inclusive acquisition-bin integers. The window
must be ordered and contained in the aligned count record. For example,
``[4, 1]`` raises ``good-bin window [4, 1] is reversed.`` and ``[0, 9]``
on eight bins raises ``good-bin window [0, 9] is outside 8 bins.`` A malformed
pair raises ``background range requires two integer bin indices.`` The
same rules apply to grouped Fourier backgrounds. An absent range selects the
pre-t0 default; a declared invalid range raises an error.

Tail fit (late-time)
--------------------

Fits the late half of the good-data window with the muon decay plus a flat
rate,

.. math::

   C(t) = \left[ p_1\, e^{-t/\tau_\mu}\,
          \frac{\sinh(w/2\tau_\mu)}{w/2\tau_\mu} + p_2 \right] w,

where :math:`w` is the bin width (the bracketed factor averages the
exponential across each bin), and subtracts the flat rate :math:`p_2`. The
fit maximises the Poisson likelihood, which remains correct in the
late-time bins that hold only a handful of counts — the regime where
least-squares weighting fails. The fitted rate is reported with its
uncertainty, e.g. 0.23(11) counts/μs, and flagged when it is consistent
with zero.

*When to use this.* Pulsed-source (ISIS) data analysed to long times. The
expected result at ISIS is a rate consistent with zero — the duty factor
suppresses the background below measurability in most runs — so treat a
significantly non-zero rate as a diagnostic worth understanding (light
leak, detector noise, upstream stops) rather than as routine. Two caveats:
the asymmetry must have relaxed away by the fit window (a persistent
oscillation or slow relaxation biases :math:`p_2`), and the window needs
enough bins to constrain two parameters; both produce explicit failure
messages rather than silent numbers.

This is WiMDA's auto BG mode (and equivalent to Mantid's "Auto" flat +
exp-decay correction). WiMDA weights bins by :math:`\sqrt{N}` and deletes
bins holding ≤ 4 counts, which removes essentially the whole tail at fine
binning; the Poisson-likelihood fit needs no such surgery (study record:
divergence D4).

Background run
--------------

Subtracts a designated reference run — an empty sample holder, a silver
plate, or the laser-off partner of a photo-excitation measurement — from
the data, scaled by the ratio of good frames:

.. math::

   N_{\text{corr}}(t) = N(t) - \frac{F_{\text{sample}}}{F_{\text{ref}}}\,
   N_{\text{ref}}(t),
   \qquad
   \sigma^2 = N + \left(\frac{F_{\text{sample}}}{F_{\text{ref}}}\right)^2
   N_{\text{ref}}.

Good frames measure beam exposure, so the ratio puts both runs on the same
exposure footing. The reference is chosen from the loaded datasets (or
browsed from disk), recorded with the grouping, and re-resolved when the
project is reopened. Both runs receive identical deadtime treatment before
the subtraction, and each run's own time zero is used for alignment.

*When to use this.* Separating holder/mount signal when its *shape* (not
just a flat rate) must be removed — the reference subtraction removes the
full time-dependent spectrum of the unwanted component, which neither
constant mode can do. The cost is doubled statistical noise where the
reference dominates: the errors grow by the second term above. For
period-mode data (light on/off within one run), prefer period mapping —
periods share the exposure exactly, with no scaling assumption.

WiMDA's File BG subtracts the raw reference counts from deadtime-corrected
sample counts and leaves the error bars untouched; both are corrected here
(study record: divergences D6, D7).

.. _background-count-uncertainty:

Uncertainty of a subtracted background
--------------------------------------

An estimated constant level is one measurement shared by every corrected bin.
Let :math:`v_j` be the independent variance of raw count :math:`r_j`,
:math:`\sigma_\theta^2` the level-estimate variance, and
:math:`\kappa_j=\operatorname{cov}(r_j,\hat\theta)`. The sensitivity of a
corrected count to a subtracted level is :math:`g_j=-1`. A weighted sum has
variance

.. math::

   \operatorname{var}\!\left(\sum_j a_j c_j\right)
   = \sum_j a_j^2 v_j
   + \left(\sum_j a_j g_j\right)^2\sigma_\theta^2
   + 2\left(\sum_j a_j g_j\right)\left(\sum_j a_j\kappa_j\right).

For an output bin merging :math:`m` raw bins this becomes
:math:`\sum v_j+m^2\sigma_\theta^2-2m\sum\kappa_j`.
The last term accounts for overlap with the background-estimation window;
overlap is allowed. For a range mean the expression follows directly from
the raw counts' linear coefficients, including when the two windows coincide.
The corrected sum over its own estimation window is identically zero with
zero variance.

.. list-table:: Noise model by mode
   :header-rows: 1
   :widths: 17 31 52

   * - Mode
     - Independent variance
     - Shared level and raw-count covariance
   * - Fixed values
     - Observed raw counts
     - None; the declared level is exact.
   * - Range average
     - Observed raw counts
     - For estimation window :math:`W` with :math:`n_W` bins,
       :math:`\sigma_\theta^2=\sum_W r_j/n_W^2` and
       :math:`\kappa_j=r_j/n_W` on :math:`W`, zero elsewhere.
   * - Tail fit
     - Fitted Poisson means inside the fit window; observed counts outside
     - Expected-information variance and the linearised likelihood influence
       below; covariance is zero outside the fit window.
   * - Background run
     - :math:`r_j+s^2R_j`
     - None; reference bins are independent and frame ratio :math:`s` is
       treated as known.

For the tail fit, let :math:`\mu_j` be its fitted expected count,
:math:`J_j=\partial\mu_j/\partial(p_1,p_2)` and
:math:`I=\sum_j J_j^\mathsf{T}J_j/\mu_j`. Expected Fisher information
[Fisher1922]_ and first-order score propagation give

.. math::

   \sigma_\theta^2=w^2(I^{-1})_{22},\qquad
   \kappa_j=w(J_j I^{-1})_2,\qquad \theta=w p_2.

The fitted :math:`\mu_j` supplies the same Poisson variance basis as the
influence row. Combining model-based covariance with observed zero-bin
variances can produce an invalid joint covariance. Singular information
fails with ``Tail-fit information matrix is singular``; it does not declare
zero uncertainty. Shared count components are checked for a valid joint
covariance at construction.

Binning, count-integral asymmetry and count-ratio α use this same propagation.
For a differential mean asymmetry the weights are the per-bin asymmetry
derivatives divided by the number of averaged bins. Forward and backward
detector groups are treated as independent. Zero variance stays zero in the
count domain; asymmetry formation assigns the fractional error sentinel 1
when its denominator or propagated variance is zero.

These are **marginal, first-order** errors. A shared level also correlates
output bins: its covariance contribution within one group has rank at most
two. Fitters receive marginal error bars, not that full covariance. Tail-fit
linearisation assumes interior, identifiable fit parameters; near a
non-negative parameter bound its errors need not have calibrated coverage,
and marginal variances are not guaranteed conservative. Deadtime-corrected
counts use the Poisson approximation. Rebinning an already reduced curve
treats its points as independent and does not carry these shared components.

.. figure:: /_generated/screenshots/background_binned_uncertainty.png
   :alt: Grouping preview with tail-fit background subtraction and bunched asymmetry errors
   :width: 100%

   The asymmetry preview after **Tail fit** background subtraction and fixed
   bunching. Error bars include the level shared by each merged group of bins.

**References**

- S. J. Blundell, R. De Renzi, T. Lancaster, and F. L. Pratt, *Muon
  Spectroscopy: An Introduction* (Oxford University Press, Oxford, 2022).
- A. Suter and B. M. Wojek, Phys. Procedia **30**, 69 (2012).
- F. L. Pratt, Physica B **289–290**, 710 (2000).

.. [Fisher1922] R. A. Fisher, Philos. Trans. R. Soc. A **222**,
   309–368 (1922). `doi:10.1098/rsta.1922.0009 <https://doi.org/10.1098/rsta.1922.0009>`_.
