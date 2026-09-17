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
the signed difference ``Δ ±d``. Any verdict messages follow after an em-dash.
The detection itself runs off the GUI thread, once per run, so switching
between runs already scanned repaints instantly.

**Verdict levels.** A warning never blocks Apply (it only colours the line
and is echoed to the analysis log when you Apply) — the possible messages are:

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
  tolerance* — the consensus and the file value diverge by more than the
  tolerance below.
* *Detectors 1, 2, … disagree with their file t0 by more than N bins* — one
  or more detectors' own estimate diverges from their own file t0, named by
  1-based detector number.
* *Detector spread N bins — check the source type* — the per-detector
  estimates disagree with each other by more than four tolerances, usually
  because the source was misidentified as pulsed/continuous.

**Tolerances.** The divergence tolerance is 2 bins for a continuous source
(PSI, TRIUMF) and 3 bins for a pulsed one (ISIS). The continuous value
follows from the prompt peak itself, which fixes t0 to a bin or two, and is
checked against PSI GPS data where every detector's estimate lands within a
bin of its header. The pulsed value comes from a 1,245-file ISIS header
survey (``docs/porting/t0-determination/isis-header-index-base.md``): HiFi
and MuSR headers sit within a bin of the observed pulse, EMU headers run
1–2 bins late. A wider spread than that on a single run's own detectors is a
stronger signal than the file/detected comparison alone — it means the
search itself is confused, not just that the header might be wrong.

**Find t0** is the one-shot fill for **Manual** mode: it runs the search on
the reference run and writes the equivalent offset into the mode's spinbox
(:doc:`/reference/detector_grouping` § Time-zero (t0) modes) — nothing is
applied until you press Apply. For a per-run search on every run in scope,
choose **Auto-detect** instead.

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
