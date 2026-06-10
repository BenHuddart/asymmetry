.. _binning-modes:

Histogram Binning
=================

Counts arrive in bins of the TDC resolution (16 ns at ISIS, sub-ns at
continuous sources), far finer than most physics requires. Because the muon
ensemble decays as :math:`e^{-t/\tau_\mu}`, the Poisson error per raw bin
grows as :math:`e^{t/2\tau_\mu}` — the signal-to-noise halves roughly every
3 μs — so a single binning choice cannot serve both the densely-sampled
early times and the count-starved tail. Three modes are offered in the
grouping dialog's **Binning** control:

==================  =============================================================
Mode                Output bin width
==================  =============================================================
Fixed               bunching factor × raw width, everywhere
Variable            grows exponentially from *Initial bin* at t = 0 to
                    *Bin at 10 μs*
Constant error      grows as :math:`e^{t/\tau_\mu}` from *Initial bin* — equal
                    counts, flat error per bin
==================  =============================================================

All three are display/fit-input transformations: the raw histograms are
never modified, and changing mode (or any width) is always reversible. For
the non-fixed modes the *counts* are summed onto the output bins and the
asymmetry formed per output bin — at late times the raw bins hold zero
counts, where an asymmetry ratio per raw bin is undefined while summed
counts remain exactly Poisson.

Fixed
-----

The default. Every output bin merges the same number of raw bins.

*When to use this.* Oscillating (TF) data, Fourier analysis, and MaxEnt —
anything that needs uniform time sampling. Keep the bunching factor small
enough that the highest frequency of interest stays below the Nyquist limit
:math:`f_c = 1/(2\Delta t)`; rebinning is a low-pass filter, which can also
be used deliberately to suppress an unwanted high-frequency component.

Variable
--------

Width grows smoothly as

.. math::

   w(t) = w_0 \left( \frac{w_{10}}{w_0} \right)^{t/10\,\mu\text{s}},

set by the width at t = 0 and the width at 10 μs (WiMDA's two-knob
convention; defaults 0.08 μs and 0.25 μs).

*When to use this.* Relaxation data where the early-time shape matters (a
fast front, a Kubo–Toyabe dip) but the tail evolves slowly — fine bins
where the physics is fast, coarse bins where only the level matters. The
growth is gentler than constant-error mode, so it preserves more late-time
structure at the cost of growing error bars.

Constant error
--------------

Width grows at exactly the muon decay rate,

.. math::

   w(t) = w_0\, e^{t/\tau_\mu},

so the expected counts per output bin — and hence the Poisson error per
point — stay constant while the polarisation varies slowly. One knob: the
initial width sets the statistics level of every bin.

*When to use this.* Weak, slow relaxation followed to long times (20–32 μs
pulsed-source work): every plotted point carries equal statistical weight,
which is also the friendliest input for eyeballing weak trends. Two
caveats: late output bins become microseconds wide, so any structure faster
than the local width is averaged away — check with fixed binning first that
nothing oscillates; and the final bin, truncated by the good-data window,
carries fewer counts than the rest.

Notes
-----

Fourier and MaxEnt analyses require uniform sampling and always use fixed
binning regardless of this setting. Fits of the displayed curve use the
displayed binning — each point enters the χ² with its own error, so the
non-uniform widths are handled correctly — but note that heavily binned
input has less information about fast components than the raw data.

**References**

- S. J. Blundell, R. De Renzi, T. Lancaster, and F. L. Pratt, *Muon
  Spectroscopy: An Introduction* (Oxford University Press, Oxford, 2022) —
  rebinning schemes and counting statistics.
- F. L. Pratt, Physica B **289–290**, 710 (2000).
