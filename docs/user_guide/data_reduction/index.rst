.. _data-reduction:

Data Reduction
==============

These pages document the reduction layer between the raw detector histograms
and the asymmetry curve: detector balancing (alpha), background handling,
histogram binning, time-zero determination, detector exclusion and
multi-period mapping. Every quantity here feeds *all* downstream analysis —
a weak alpha estimate or a missed background degrades every fit made from
the data — so each page states explicitly when its method applies and when
it does not.

.. toctree::
   :maxdepth: 2

   alpha_calibration
   backgrounds
   binning
   t0_search
   detector_exclusion
   period_mapping

All of the corrections described here are applied at reduction time and
recorded in the grouping settings: raw histograms are never modified, and a
project re-opened later reproduces the same reduction from the same recorded
choices.
