.. _background-correction:

Background Correction
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
with the subtraction's uncertainty propagated into the per-bin errors.

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

**References**

- S. J. Blundell, R. De Renzi, T. Lancaster, and F. L. Pratt, *Muon
  Spectroscopy: An Introduction* (Oxford University Press, Oxford, 2022).
- A. Suter and B. M. Wojek, Phys. Procedia **30**, 69 (2012).
- F. L. Pratt, Physica B **289–290**, 710 (2000).
