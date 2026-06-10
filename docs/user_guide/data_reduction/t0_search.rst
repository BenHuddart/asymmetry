.. _t0-search:

Time-Zero Search
================

Everything downstream counts time from t0, the moment the muon spin starts
evolving in the sample. The value stored in a data file is a calibration
made by the instrument scientist; it is usually right, but the standard
advice applies — never rely on a stored t0 you did not record yourself.
A file converted from an old format, a run taken during commissioning, or a
detector with a shifted cable delay can all carry a wrong t0, which shows up
downstream as wrong frequencies in TF data and distorted early-time shapes.

The **Find t0** button in the grouping dialog estimates t0 from the
reference run's histograms and fills the override spinner. Nothing is
changed until Apply — the found value sits beside the file value for you to
compare, and the result line reports the detector-to-detector spread as a
health check (a few bins of spread is normal; tens of bins means dead
detectors or the wrong strategy).

Two strategies, chosen automatically from the data's facility:

**Continuous sources (PSI, TRIUMF) — prompt peak.** A single particle
triggering both the muon and positron counters produces a sharp spike at
zero time difference, good to a few tenths of a nanosecond. The estimate is
the maximum-count bin of each histogram — the same convention as WiMDA's
Search for T0 and musrfit's ``musrt0``.

**Pulsed sources (ISIS) — pulse-edge midpoint.** There is no prompt peak;
t0 is the *centre* of the muon pulse, found from the half-maximum point of
the histogram's rising edge. The first *good* bin is later still — analysis
must not start until the whole pulse has arrived (the t_good offset,
typically several bins at ISIS) — and the pulse width, not the bin width,
limits the usable frequency range (about 10 MHz at ISIS). WiMDA uses the
maximum bin at pulsed sources too, which lands at the pulse *peak* rather
than its centre; the midpoint convention here follows the textbook
definition.

*When to use this.* Files with missing or suspect t0 — old conversions,
commissioning data, instruments without calibration in the header — and as
a quick cross-check when an analysis produces a mysterious early-time
distortion or a TF phase that varies linearly with frequency (the signature
of a t0 error: a phase slope of q degrees per MHz corresponds to a t0 shift
of q/360 μs). For good files the search should reproduce the stored value
within a bin or two; a larger discrepancy is worth understanding before
trusting either number.

**References**

- S. J. Blundell, R. De Renzi, T. Lancaster, and F. L. Pratt, *Muon
  Spectroscopy: An Introduction* (Oxford University Press, Oxford, 2022) —
  time-zero and detector-phase calibration.
- A. Suter and B. M. Wojek, Phys. Procedia **30**, 69 (2012).
