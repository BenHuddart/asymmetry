.. _t0-search:

Time-zero search
================

Everything downstream counts time from t0, the moment the muon spin starts
evolving in the sample. The value stored in a data file is a calibration
made by the instrument scientist; it is usually right, but the standard
advice applies — never rely on a stored t0 you did not record yourself.
A file converted from an old format, a run taken during commissioning, or a
detector with a shifted cable delay can all carry a wrong t0, which shows up
downstream as wrong frequencies in TF data and distorted early-time shapes.

The grouping window's t0 row carries a read-only line, identical whichever
mode is selected, that shows the file value beside the detected one so you
never have to take either on faith:

.. figure:: /_generated/screenshots/grouping_window_t0_row.png
   :width: 70%
   :align: center
   :alt: The grouping window's t0 row with the file/detected/Δ line beneath
      it, reading "File: bin 40 · Detected: bin 40 (prompt peak, spread 0)
      · Δ +0".

   The t0 row (mode selector, spinbox, **Find t0**) and the line beneath it.
   Here the file header and the detected prompt peak agree exactly, so there
   is no verdict message.

The line is ``" · ".join(...)`` of the file value (``File: bin N``, or
``File: none (detected)`` when the run's header carried no t0 at all), the
detection (``Detected: …`` while a background scan is in flight, ``Detected:
unavailable`` if it failed, or ``Detected: bin M (strategy, spread S)``), and
the signed difference ``Δ ±d``. The detection itself runs off the GUI thread,
once per run, so switching between runs already scanned repaints instantly.

The detected bin and Δ are the file bin plus the **median per-detector shift**
— the median of ``detected_i − header_i`` over the detectors that resolved —
and the spread is the *range* of those shifts. On data where detectors
legitimately sit at different times (PSI headers carry a t0 per detector) the
median of the raw estimates and the group's header maximum are different
quantities, so comparing them reported a divergence that was only the
detectors' real stagger. A run whose every detector lands on its own header
therefore reads ``Δ +0`` with a spread of a bin or two, however far apart the
detectors themselves are.

**Verdict levels.** The messages are not on the line: when there is something
to say, a ``⚠`` button appears beside it, tinted amber for a warning and red
for an error, with every message in its tooltip and in a click-to-read popup.
A warning never blocks Apply (it colours the line and is echoed to the
analysis log when you Apply) — the possible messages are:

* *No time zero in the file header; using the detected value* — the file
  carried no usable t0 at all; resolution falls back to the detected
  consensus (this is the only place resolution *must* use the detected
  value rather than merely display it).
* *Time zero is bin N, outside the run's M bins* — the header t0 does not
  point inside the histogram.
* *Header time_zero disagrees with t0_bin; using t0_bin* — an ISIS file
  whose two redundant t0 fields disagree; the integer bin wins (see
  :doc:`/reference/loading_data`).
* *Detected t0 is bin D, file t0 is bin F — further apart than the N-bin
  tolerance* — the median shift exceeds the tolerance below, so the whole run
  sits away from its header.
* *Detectors 1, 2, … disagree with the other detectors' t0 shift by more than
  N bins* — those detectors moved by a different amount from the rest, named by
  1-based detector number. A detector that is far from the others but agrees
  with its own header is not an outlier.
* *Detector spread N bins — check the source type* — the per-detector shifts
  disagree with each other by more than four tolerances, usually because the
  source was misidentified as pulsed/continuous.
* *First good bin G is at or before the detected t0 (bin D)* — the analysis
  window opens on the muon arrival, so the prompt peak is inside the fitted
  asymmetry. This reads as a spuriously large early-time asymmetry rather than
  as an error, which is why it is worth saying out loud.
* *First good bin G is inside the muon pulse (peak at bin P)* — the pulsed
  equivalent: the window clears the pulse *centre* but not the pulse itself.
  Raise the t_good offset until it starts after the peak.

**Tolerances.** Every check above is judged against the *larger* of two
numbers: a floor per source family, and the measured width — in bins — of the
feature t0 was read off (the prompt peak's FWHM, or the pulse's 10 %→90 % rise).

The floors are 2 bins for a continuous source (PSI, TRIUMF) and 3 bins for a
pulsed one (ISIS). The continuous value follows from the prompt peak itself,
which fixes t0 to a bin or two, and is checked against PSI GPS data where
every detector's estimate lands within a bin of its header. The pulsed value
comes from a 1,245-file ISIS header survey
(``docs/porting/t0-determination/isis-header-index-base.md``): HiFi and MuSR
headers sit within a bin of the observed pulse, EMU headers run 1–2 bins late.

The measured width is what keeps those floors meaningful at any binning. A bin
index cannot name the centre of a prompt peak more precisely than the peak is
wide, and how wide that is *in bins* depends on the bin width: the same PSI
peak spans one bin at 1 ns binning and 4–11 at the 98 ps binning a modern GPS
run uses, so at 98 ps a few tenths of a nanosecond of jitter between detectors
— entirely inside the peak — would trip a bare 2-bin floor. The tolerance
therefore follows the data: a run whose peak is 6 bins wide is judged at 6
bins, an EMU run whose pulse rises over 5 bins at 5. A spread wider than four
tolerances on a single run's own detectors is a stronger signal than the
file/detected comparison alone — it means the search itself is confused, not
just that the header might be wrong.

**Find t0** is the one-shot fill for **Manual** mode: it runs the search on
the reference run and writes the equivalent offset into the mode's spinbox
(:doc:`/reference/detector_grouping` § Time-zero (t0) modes) — nothing is
applied until you press Apply. For a per-run search on every run in scope,
choose **Auto-detect** instead: it gives *each detector* its own detected t0,
the model musrfit's ``musrt0 -g`` writes, so a run whose detectors sit at
genuinely different times keeps that stagger instead of being shifted as one
block.

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
of q/360 μs). The line above already flags a discrepancy beyond the
tolerance, so a quiet line is itself the reassurance that the file value is
trustworthy.

**References**

- S. J. Blundell, R. De Renzi, T. Lancaster, and F. L. Pratt, *Muon
  Spectroscopy: An Introduction* (Oxford University Press, Oxford, 2022) —
  time-zero and detector-phase calibration.
- A. Suter and B. M. Wojek, Phys. Procedia **30**, 69 (2012).
