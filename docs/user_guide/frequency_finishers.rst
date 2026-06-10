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

References
----------

- S. J. Blundell, R. De Renzi, T. Lancaster, and F. L. Pratt (eds.),
  *Muon Spectroscopy: An Introduction* (Oxford University Press, 2022), §15.5
  (frequency domain) and §14.2 (pulsed sources).
- Á. Sánchez-Monge, P. Schilke, A. Ginsburg, R. Cesaroni, and A. Schmiedeke,
  *Astron. Astrophys.* **609**, A101 (2018) — STATCONT robust continuum
  estimation by iterative σ-clipping.
