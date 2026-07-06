.. _detector-exclusion:

Detector exclusion
==================

A dead detector contributes nothing but dilutes a group's asymmetry
normalisation; a hot one (HV breakdown, light leak, noisy PMT) injects
counts uncorrelated with the muons and drags the asymmetry toward zero with
a spurious time dependence. Either way the cure is the same: drop the
detector from the analysis without touching the data.

The grouping dialog's **Exclude Detectors** field takes a WiMDA-style list —
``1,5,10-15`` — of 1-based detector ids; the detector layout editor offers
the same thing visually via its *Exclude mode* toggle (click a detector in
the schematic to strike it out). Excluded detectors are removed from every
group sum at reduction time: the raw histograms are untouched, no reload is
needed, the exclusion is recorded with the grouping in the project, and it
applies consistently to the asymmetry, alpha estimation, per-group fitting
and Fourier inputs (which all consume the same group sums).

Identifying a bad detector
--------------------------

The per-detector counts tell the story: a dead detector shows totals far
below its neighbours (often zero); a hot one shows an excess, and its
spectrum on a log scale deviates from the straight muon-lifetime line —
flat noise at all times rather than the exponential. The detector spread
reported by :ref:`Find t0 <t0-search>` is another quick flag: a detector
whose prompt peak sits far from the consensus has a timing problem even if
its rate looks normal.

*When to use this.* Exclude rather than correct whenever a detector's
problem is not a well-modelled inefficiency: deadtime correction can fix
rate-dependent counting loss, but nothing recovers a detector that lost HV
halfway through a run or counts light-leak noise. The statistical cost of
dropping one detector from a 32-or-more-detector group is a percent-level
increase in error bars; the systematic cost of keeping a bad one can be a
distorted asymmetry. After excluding, re-estimate alpha — the group balance
changes with the group membership.

WiMDA zeroes excluded detectors when the file is read (a reload per change);
applying the exclusion at grouping time gives the same reduction with the
raw data intact (study record: divergence D10).

**References**

- F. L. Pratt, Physica B **289–290**, 710 (2000).
