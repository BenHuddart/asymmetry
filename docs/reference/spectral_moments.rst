Spectral moments
================

A muon field/frequency spectrum carries more than a peak position. Its *moments*
— the mean field, the RMS width, the skewness, the lineshape asymmetry — reduce
the whole distribution :math:`p(B)` to a handful of numbers that map directly onto
physics, and that can be trended across temperature or field like any fitted
parameter. The RMS width is the headline: for a type-II superconductor in the
mixed state the second moment of the vortex-lattice field distribution sets the
magnetic penetration depth, :math:`\langle\Delta B^2\rangle \propto 1/\lambda^4`,
and hence the superfluid density — see
:doc:`sc_penetration_depth`.

The **Spectral moments** control sits in the advanced stack of the Fourier panel
and under the reconstruction in the MaxEnt panel. Pick a unit, drag a range over
the line, and the readout updates live; **Send to trend** records the moments of
every selected run as a trendable series.

.. image:: /_generated/screenshots/spectral_moments_readout.png
   :alt: The Spectral moments readout in the Fourier panel, computed on the
         phase-corrected FFT of a YBCO vortex-lattice line, with the moments
         range and cutoff drawn over the spectrum.
   :width: 100%

*The* **Spectral moments** *readout, computed on the phase-corrected real FFT of
a YBCO vortex-lattice line. The spectrum is the asymmetric field distribution
p(B) — a sharp low-field peak and a long high-field tail — with the moments
range (shaded) and cutoff (dotted) drawn over it. The readout gives B_pk ≈
1987 G, the mean B_ave sitting above the peak, the RMS width, and the positive
skewness and asymmetry (β > 0) that are the vortex lattice's expected
signature.*

What each moment tells you
--------------------------

For a window over the line, above an amplitude cutoff, Asymmetry reports:

- :math:`B_{\mathrm{pk}}` — the peak field, refined by a parabola through the five
  points around the maximum.
- :math:`B_{\mathrm{ave}}` — the amplitude-weighted **mean** field; its shift from
  the applied field is the diamagnetic (or Knight) shift.
- :math:`\langle B_{\mathrm{ave}}-B_{\mathrm{pk}}\rangle` — how far the mean sits
  from the peak; a direct read on the line's asymmetry.
- :math:`B_{\mathrm{rms}}` about the mean and about the peak — the **width**. About
  the mean it is the standard deviation :math:`\sqrt{\langle\Delta B^2\rangle}`,
  the quantity that feeds :math:`\lambda`.
- **Skewness** — the third-moment asymmetry. Asymmetry reports both WiMDA's
  cube-root form :math:`\alpha=\operatorname{sign}(m_3)\,\sqrt[3]{|m_3|}/\sqrt{m_2}`
  and the standard standardised skewness
  :math:`\gamma_1=m_3/m_2^{3/2}`.
- **Asymmetry** :math:`\beta=(B_{\mathrm{ave}}-B_{\mathrm{pk}})/B_{\mathrm{rms,pk}}`
  — positive when the mean lies above the peak.

The vortex-lattice field distribution is the canonical use case: a sharp low-field
cutoff at the lattice's saddle-point field and a long tail to high field near the
cores give it a positive skew, so :math:`\beta>0` and :math:`\gamma_1>0` are the
expected signature, and their magnitude tracks the lattice geometry and its
disorder. The sign convention here matches WiMDA and the literature for that
distribution [1] [2] [3].

The range and the cutoff
-------------------------

Moments are **window-dependent**, so the window is always drawn on the plot — a
shaded *range* with draggable edges and a dotted *cutoff* line at the chosen
fraction of the peak. Drag them, or type exact values into the control; the choice
is recorded in the run's provenance so a trend is reproducible.

- The **range** isolates the line of interest and excludes neighbouring features.
  (This is a *range*, not an exclusion: it selects what to include.)
- The **cutoff** (a percentage of the peak) trims the wings and the spectral floor
  before the integral, so far-off baseline noise does not inflate the width.

Tighten the range toward the main line and the skewness and :math:`\beta` collapse
toward zero as the tail is excluded; raise the cutoff and the width narrows as the
wings drop out. There is no single correct window — report the one you used.

A caveat on apodised spectra
----------------------------

Apodisation broadens every line it smooths, so moments read from a filtered FFT
carry the filter as a systematic: the widths and skewness include the filter's
broadening, not just the sample's. When the active spectrum was computed with a
Lorentzian or Gaussian filter, the moments readout shows an amber caveat —

   *Apodised spectrum (lorentzian, τ = 1.8 µs): widths and skewness include the
   filter's broadening.*

— so a filtered reading is never silently mistaken for the unfiltered physics.
For quantitative widths, recompute the FFT with apodisation ``None`` (or
deconvolve the known filter contribution when reporting).

A caveat on :math:`B_{\mathrm{pk}}`
-----------------------------------

:math:`B_{\mathrm{pk}}` is the **fragile** member of the set. It is a parabola
fitted to five points around the discrete maximum; on a noisy or near-flat
spectrum the maximum hops between bins and the parabolic vertex can swing wide.
Everything built on it — :math:`\langle B_{\mathrm{ave}}-B_{\mathrm{pk}}\rangle`
and, especially, :math:`\beta` — inherits that fragility. The centroid, RMS
width, and skewness also depend on noisy cutoff membership. A small displayed
approximate error does not establish that any of these estimates is stable or
has calibrated coverage.

Uncertainties
-------------

The readout marks available errors **Approximate errors**. These are heuristic
errors under independent spectral-bin noise and frozen processing. Bootstrap
perturbs spectral bins with the supplied errors and reselects the cutoff support
and peak on each draw; it does not repeat the time/acquisition FFT or estimate
phase again. Parenthesised values, such as :math:`18.4(3)`, express that numerical
error estimate, not a calibrated confidence interval.

FFT cross-bin correlations, automatically estimated phase, upstream calibration
and background, and MaxEnt reconstruction uncertainty are not fully propagated.
Finite errors are multiplied by the square root of the padding factor when that
factor exceeds one. This scalar rescaling is heuristic: it can understate or
overstate errors, including with apodisation, and does not recover spectral
covariance or establish nominal coverage. The **Info** explanation describes these
limits. New computed recipes and trend rows retain the scope; **Parameters** shows
**Recorded warnings**, and TSV/GLE data exports retain each run's captured warning.
Saved results without a recorded scope retain that unknown provenance.

The central values describe the range-limited, cutoff-selected displayed spectrum.
Windowing, finite time support, conditioning, and reconstruction affect it; its
RMS width is not automatically the width of an unwindowed physical field
distribution. Neither these errors nor run-to-run scatter remove that distinction.

The scripting option ``uncertainty="propagate"`` uses the complete first-order
Jacobian with independent per-point errors [4], including normalization and
motion of the refined parabolic peak. It reports errors for the mean, both RMS
widths, :math:`\gamma_1`, the peak, the mean-minus-peak difference and
:math:`\beta`. The support above the cutoff and the discrete peak bin are held
fixed. A positive cutoff can make these errors too small when noisy points
cross its threshold; peak-bin changes are also outside the linearization.
Bootstrap instead re-finds both support and peak on each draw.

Cube-root :math:`\alpha` has no propagated uncertainty because its derivative
diverges at a symmetric line; use the propagated :math:`\gamma_1` uncertainty.
An unrefined or clamped peak leaves all peak-derived errors unavailable
(``NaN``), and zero-width support leaves every propagated error unavailable.
The same heuristic scalar padding rescaling applies to finite propagated and
bootstrap errors; neither mode propagates full spectral covariance.

Compared with other programs
----------------------------

WiMDA's inspected single-spectrum moment workflow supplies central values;
scatter across multiple entries is a different statistic from a propagated
single-spectrum error. See the versioned scope and public manuals in
:ref:`spectral-error-comparison`.

Which spectra qualify
---------------------

Moments are only meaningful for a **lineshape-faithful** spectrum: the MaxEnt
reconstruction or the phase-corrected real FFT. Power, magnitude, phase, Burg and
correlation modes are squared or diagnostic lineshapes that bias the width and the
skewness, so the control greys out for them with a note explaining why. Switch to
the phase-corrected real mode or the MaxEnt reconstruction to take moments.

Trending the moments
--------------------

**Send to trend** computes the moments of every **selected** run's spectrum, with
the current range/cutoff/unit, and records them as one computed series — one point
per run, indexed by field and temperature — that fits like any parameter series.
Re-sending the same selection replaces it rather than duplicating it. Fit
:math:`B_{\mathrm{rms}}(T)` with a superconductor model
(:doc:`sc_penetration_depth`) to extract :math:`\lambda(T)`, or trend the skewness
to follow a lattice transition.

When to fit the lineshape instead
---------------------------------

Moments are **model-free**: they summarise whatever :math:`p(B)` the spectrum
shows, which is exactly right for a first look, for tracking a width across a scan,
and for distributions with no clean analytic form. Fit the lineshape instead when
you have a physical model of :math:`p(B)` — a Brandt vortex-lattice distribution,
a Gaussian-broadened London model, a sum of diamagnetic and background lines — and
want its *parameters* with proper covariances, or when the tails are too noisy for
a stable third moment. The two are complementary: moments give the quick,
assumption-light trend; a lineshape fit gives the interpreted physics.

.. seealso::

   - :doc:`fourier_analysis` for the FFT and MaxEnt spectra;
   - :doc:`frequency_finishers` for the conditioning ladder that prepares them;
   - :doc:`parameter_trending` and :doc:`sc_penetration_depth` for fitting the
     resulting :math:`B_{\mathrm{rms}}(T)`.

References
----------

[1] E. H. Brandt, Phys. Rev. B **37**, 2349 (1988); Phys. Rev. B **68**, 054506
(2003).

[2] J. E. Sonier, J. H. Brewer, and R. F. Kiefl, Rev. Mod. Phys. **72**, 769
(2000).

[3] S. J. Blundell, R. De Renzi, T. Lancaster, and F. L. Pratt, *Muon
Spectroscopy: An Introduction* (Oxford University Press, Oxford, 2022).


[4] JCGM 100:2008, *Evaluation of measurement data — Guide to the expression
of uncertainty in measurement*, sections 5.1–5.2,
`doi:10.59161/JCGM100-2008E <https://doi.org/10.59161/JCGM100-2008E>`_.
