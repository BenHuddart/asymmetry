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

A required group emptied by exclusions is unavailable: the preview displays
**Preview unavailable:** and disables **Apply**. Removing an exclusion or
editing the authored membership restores availability. An exclusion cannot hide
an absent channel required by a preset. The numbers address acquired channels;
a pregrouped histogram cannot lose an individual physical constituent unless
that constituent's observations are separately available.

.. figure:: /_generated/screenshots/grouping_membership_exclusion.png
   :width: 100%
   :alt: Synthetic groups with shared channel 2 excluded and one effective channel per side.

   **Exclude Detectors** removes channel 2 from both authored groups; the status
   shows one effective channel on each side.

.. figure:: /_generated/screenshots/grouping_membership_unavailable.png
   :width: 100%
   :alt: Synthetic group emptied by exclusions, with an unavailable preview and disabled Apply.

   Excluding channels 1 and 2 empties the forward group. Its raw counts remain
   accessible while the grouping is edited.

Compared with other programs
----------------------------

WiMDA's **Group Table Edit** and grouping controls support numbered channel
selection [wimda_grouping]_. No WiMDA executable comparison is claimed.
Mantid detector selection and musrfit histogram lists explicitly select the
observations to sum; their recorded grouping checks and conventions are
summarised in :doc:`/reference/detector_grouping` and :doc:`/explanation/comparison`.

**References**

- F. L. Pratt, Physica B **289–290**, 710 (2000).

.. [wimda_grouping] F. L. Pratt, *WiMDA Manual*, v1.2 (2005),
   https://shadow.nd.rl.ac.uk/wimda/WiMDA%20Manual.doc.
