Frequency-domain conditioning
=============================

Once a grouped FFT spectrum exists, a handful of conditioning steps turn a raw
transform into something you can read a field off. They live in the
**Conditioning** and **Exclusions** sections of the Fourier panel and apply, in
order, to the averaged spectrum on its canonical frequency axis: pulse-response
compensation, a robust baseline offset, and frequency-range exclusions. The
display unit (MHz, Gauss, or Tesla) and the Real+Imag view are display choices
layered on top. None of these change the underlying transform — they are the
last mile between the FFT and the number you quote.

Field axis: reading a spectrum in Gauss
---------------------------------------

A precession line at frequency :math:`\nu` reports the local field through the
Larmor relation

.. math::

   \nu = \frac{\gamma_\mu}{2\pi}\,B, \qquad \frac{\gamma_\mu}{2\pi} = 135.538817\ \text{MHz T}^{-1},

so a transverse-field spectrum can be displayed against frequency or field
interchangeably. The unit selector above the frequency plot switches between
**Frequency (MHz)**, **Field (G)**, and **Field (T)**; the FFT and the maximum-
entropy reconstruction share one axis, so the same run peaks at the same field
in either view. For a transverse field of 200 G the Larmor line sits at
:math:`27.1` MHz, equivalently 200 G — the field axis simply relabels it.

*When to use this.* Switch to Gauss or Tesla whenever the physics is a field —
vortex-lattice internal-field distributions, knight shifts, applied-field
calibration. Stay in MHz when the physics is a frequency — hyperfine couplings,
F–μ–F entanglement, muonium pairs.

Pulse-response compensation
---------------------------

At a pulsed source the muons arrive spread over a finite pulse, so a signal
oscillating fast compared with the pulse width is averaged across the spread of
arrival times and its amplitude is suppressed: the pulse acts as a passband
filter that rolls the spectrum off at high frequency. Compensation divides each
bin by the pulse amplitude :math:`R(\nu)` — the same parabola-times-Lorentzian
lineshape the maximum-entropy forward model folds into its kernel, so the two
methods correct one physical pulse, not two approximations of it.

Because :math:`R(\nu) \to 0` near the first node of the pulse transform, the raw
correction :math:`1/R(\nu)` grows without bound. Asymmetry caps the gain (default
:math:`25\times`) and cuts the spectrum off at the node: beyond it the pulse has
erased the signal and no multiplication can recover it. This is the deliberate
difference from WiMDA, whose Gaussian :math:`\exp[(\pi f \tau)^2]` factor
amplifies high-frequency noise without limit. The pulse half-width defaults from
the run's instrument metadata; set it explicitly in the panel to override.

*When to use this.* Turn it on for ISIS (pulsed-source) transverse-field data
with lines above a few MHz, where the rolloff visibly suppresses peak heights.
Leave it off for continuous-source (PSI, TRIUMF) data, which has no pulse
envelope, and at low frequency where :math:`R(\nu) \approx 1`. *Pitfall:* the cap
and cutoff are physical limits, not bugs — if a line vanishes past the cutoff,
the pulse genuinely could not record it; reduce the pulse width only if you know
it.

.. _robust-baseline-offset:

Robust baseline offset
----------------------

A power or magnitude spectrum sits on a positive pedestal from the
redistributed counting noise. The baseline control estimates that pedestal and
subtracts it. **Robust σ-clip** (the default when enabled) iteratively re-
estimates the median and width of the inlier set, discarding points more than
:math:`\kappa` standard deviations away until the width converges; sharp peaks
are excluded as outliers and survive the subtraction intact. **WiMDA single-pass**
is the one-iteration, mean-location special case, kept for parity. The converged
width doubles as a baseline-noise estimate that feeds the signal-to-noise
readout.

*When to use this.* Enable it whenever you want peak heights or areas measured
above a true zero — comparing intensities across runs, integrating a line. The
robust mode is the right default; the single-pass mode reproduces a WiMDA
number when you need to match a legacy analysis. *Pitfall:* on a spectrum that
is mostly signal (a single dominant line filling the window) the inlier set is
small; widen :math:`\kappa` or leave the baseline off.

Frequency-range exclusions
--------------------------

The **Exclusions** table zeroes the spectrum inside up to ten symmetric
:math:`(\text{centre} \pm \text{half-width})` windows. Two conveniences sit on
top of the table:

- **Diamagnetic line.** Excludes a band centred on the muon Larmor frequency
  :math:`\gamma_\mu B / 2\pi` for the run's applied field, suppressing the
  unshifted diamagnetic peak so a weaker shifted line stands out.
- **PSI RF harmonics.** Fills the table with DC and the first five harmonics of
  50.63 MHz — the radio-frequency pickup that contaminates PSI continuous-source
  spectra.

*When to use this.* Exclude a known artefact (RF pickup, a saturating
diamagnetic line, a mains harmonic) that would otherwise dominate the plot or
bias a baseline or peak search. *Pitfall:* exclusion blanks data — never use it
to hide a feature you have not identified, and keep the half-widths tight so you
do not remove signal next to the artefact.

Real + imaginary view
---------------------

The **Real+Imag** display mode overlays the cosine (real) and sine (imaginary)
quadratures of the averaged spectrum on one axis. A correctly phased absorption
line is purely real with a flat imaginary part; residual structure in the
imaginary channel is the visual signature of an imperfect phase correction.

*When to use this.* Reach for it while tuning the phase — manual, per-group, or
the entropy optimiser — to see at a glance whether the real channel carries all
the line. For a final quantitative spectrum, switch back to a single channel.

Signal-to-noise readout
-----------------------

After an averaged FFT the panel reports the mean per-bin error and the peak
signal-to-noise — the largest ratio of spectrum amplitude to error across the
band, with the DC bin excluded so the average-subtraction spike does not
masquerade as signal. When the robust baseline is active its noise estimate
anchors the denominator. Use the peak S/N to judge whether a marginal line is
real before committing to a frequency-domain fit.

Burg all-poles pole scan (diagnostic)
-------------------------------------

The **Resolution (Burg)** display mode is a diagnostic, not a measurement. Where
the FFT models the spectrum as a sum of sinusoids (an all-zeroes transform), the
Burg method models it as an all-poles autoregressive process,

.. math::

   P(\nu) = \frac{P_m}{\left|1 - \sum_{k=1}^{m} a_k z^k\right|^2},
   \qquad z = e^{2\pi i \nu \Delta t},

with :math:`m` poles placed exactly where the data demand sharp lines. Because
:math:`m` can be far smaller than the number of time points, the method resolves
close lines from short windows that the FFT cannot. The pole count is chosen
automatically by minimising the Final Prediction Error (FPE) over the scan range
(default 2–40); if the optimum lands on an endpoint the log warns that the range
is too narrow.

What it is good for:

- **Qualitative super-resolution.** On a window of 48 bins at 50 ns
  (FFT resolution :math:`0.42` MHz), a :math:`0.25` MHz doublet collapses to a
  single FFT peak but the Burg estimate shows two — a quick way to ask "is that
  one line or two?" before committing to a fit.
- **A line-count hint.** The FPE-optimal pole count rises with the number of
  resolvable lines, so it suggests how many components a time-domain fit should
  include.

Its pathologies — why it is never the quantitative answer:

- **Spurious splitting and baseline peaks.** Pushing the pole count well above
  the FPE optimum splits strong lines and invents peaks in the baseline that are
  not in the data. The peak heights are not amplitudes and the areas are not
  populations.
- **Noise-dependent bias and small position offsets.** The estimate shifts with
  the noise realisation and nudges line positions; it carries no uncertainties.

Treat a Burg spectrum as a magnifying glass for spotting structure. Once you
know how many lines are present and roughly where, measure them with a
frequency-domain fit or with maximum entropy — those are the quantitative
methods (Blundell, De Renzi, Lancaster & Pratt, *Muon Spectroscopy*, §15.5;
Burg 1972).

.. _diamagnetic-line-removal:

Diamagnetic line removal
------------------------

In a transverse field the unshifted diamagnetic muon line is often the largest
feature and can bury a weaker shifted or radical line. **Remove diamagnetic
signal** fits a single damped cosine,
:math:`s(t)=A\cos(2\pi(\nu t+\phi))\,e^{-\lambda t}+c`, to the time-domain signal
*before* the FFT — seeding its frequency from the run's applied field so it locks
onto the diamagnetic line — and subtracts the oscillatory part. The fitted
frequency, converted to field, is reported in the log and overlaid on the
time-domain plot so you can judge the subtraction.

*When to use this.* Turn it on to reveal a shifted line sitting close to the
diamagnetic peak, or to read the applied field off the fitted frequency as an
independent calibration. *Pitfall:* the fit assumes a single damped cosine; if
several lines overlap the diamagnetic frequency the subtraction is approximate —
inspect the overlay, and prefer a full multi-line time-domain fit when the
removal leaves visible residual.

References
----------

- S. J. Blundell, R. De Renzi, T. Lancaster, and F. L. Pratt (eds.),
  *Muon Spectroscopy: An Introduction* (Oxford University Press, 2022), §15.5
  (frequency domain) and §14.2 (pulsed sources).
- Á. Sánchez-Monge, P. Schilke, A. Ginsburg, R. Cesaroni, and A. Schmiedeke,
  *Astron. Astrophys.* **609**, A101 (2018) — STATCONT robust continuum
  estimation by iterative σ-clipping.
