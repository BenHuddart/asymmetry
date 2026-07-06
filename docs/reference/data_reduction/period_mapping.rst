.. _period-mapping:

Period mapping
==============

Pulsed-source runs can interleave acquisition conditions frame by frame —
laser on/off at half the muon pulse rate in photo-μSR, RF on/off
alternating every few seconds, stepped delays in pump–probe sequences. Each
condition accumulates into its own *period* histogram set within one file,
sharing the beam exposure bookkeeping exactly. Two-period files load as a
single dataset with the Red/Green selector (Red, Green, G−R, G+R) in the
grouping dialog; this page covers runs with **more than two** periods.

The **Map periods…** button (shown beside the Red/Green selector when the
reference run has three or more periods) opens a matrix: one row per
period, with its good-frame count, and a three-way choice

=========  =====================================================
Red        summed into the red set
Green      summed into the green set
Ignore     left out entirely
=========  =====================================================

On Apply, a combined dataset is created whose red and green sets are the
count-level sums of the chosen periods (Poisson addition of histograms,
good frames added per set), and the full Red/Green machinery applies to it:
plot either set, or the G−R difference that isolates the effect of the
stimulus. The mapping is recorded with the grouping for provenance.
Defaults follow WiMDA's period-mapping form: period 1 to red, period 2 to
green, the rest ignored; with no green periods the result behaves as a
single-set run and the G±R modes do not apply.

*When to use this.* Multi-condition sequences where several periods share a
condition — e.g. a laser-delay scan where periods 1 and 3 are light-off and
2 and 4 are light-on at different delays: map {1, 3} → red and {2} → green
to compare one delay against the off reference, then remap for the next
delay without reloading anything. Because periods share exposure, the
difference needs no scaling — this is the preferred way to subtract an
off-state, ahead of background-run subtraction between separate runs.
Summing periods with *different* conditions into one set is statistically
fine but physically meaningless — the sums are labelled by your mapping,
not by what the hardware did, so record the period legend (the acquisition
log's period table) with the analysis.

Periods recorded as dwell (beam-off) intervals carry no muon data; the
acquisition marks them and they are fixed to Ignore.

**References**

- S. J. Blundell, R. De Renzi, T. Lancaster, and F. L. Pratt, *Muon
  Spectroscopy: An Introduction* (Oxford University Press, Oxford, 2022). —
  photoexcitation and RF alternating-frame acquisition.
- F. L. Pratt, Physica B **289–290**, 710 (2000).
