.. _run-arithmetic:

Run arithmetic: co-add and co-subtract
======================================

Two runs of the same measurement add to one run with twice the statistics; a
laser-on run minus its laser-off reference isolates the photo-excited signal.
Asymmetry does both at the **raw-count level** — it sums (or subtracts) the
detector histograms and reduces afterwards, never averaging the finished
asymmetry curves. The combined row that appears in the Data Browser is a
first-class run: it carries real histograms, so it can be regrouped,
deadtime-corrected, count-fitted, and Fourier/MaxEnt-transformed exactly like a
loaded run.

Both actions live in the Data Browser context menu. Select two or more runs and
choose **Co-add Selected**; or select one run and choose **Subtract Reference
Run…** to pick the reference to remove. "Separate Combined" restores the
constituents.

Why count-level, not curve-level
---------------------------------

The asymmetry and its error must see the *total* statistics. Summing counts
first and reducing once gives the correct pooled Poisson error; averaging two
reduced curves does not, unless the two runs happen to have identical
statistics. For forward/backward totals :math:`(F_1, B_1)` and
:math:`(F_2, B_2)` the count-sum route reduces :math:`(F_1+F_2,\, B_1+B_2)` to a
single Poisson error, while the curve-mean route forms
:math:`\tfrac12\sqrt{\sigma_1^2 + \sigma_2^2}` — equal to the pooled error only
when the runs match.

The gap grows with the imbalance. Co-adding a low-statistics run with one
carrying ten times the events, the old curve-mean error bar **over-estimates**
the combined error by 53 % relative to the correct pooled value; for two equal
runs the two routes agree to better than 0.1 %. The asymmetry *value* is
unchanged at :math:`\alpha = 1` (both routes are linear in the counts there),
but any nonlinear correction — deadtime, an :math:`\alpha \neq 1` balance, a
background subtraction — only comes out right when it acts on the summed counts.
This is why the combined row keeps its histograms rather than a finished curve.

Co-add
------

Counts add bin-by-bin per detector; detectors whose time-zero differs (PSI
multi-:math:`t_0` data) are aligned to a common bin before summing, as for a
single run's grouping. Good frames accumulate, so the deadtime normaliser sees
the combined exposure. Temperature and field are reported as the
**event-weighted mean** over the constituents (weighted by good frames, the
defensible summary of an inhomogeneous group), and the spread is recorded under
``temperature_spread`` / ``field_spread`` so a group spanning, say, 4.0(1)–10.0(1) K
is visible rather than hidden behind a single averaged number. Period-mode
(red/green) runs sum per period, preserving the two-period structure for the
G∓R reduction.

Co-subtract (reference run)
---------------------------

A reference subtraction forms

.. math::

   N_i^{\text{diff}} = N_i^{\text{sample}} - s\, N_i^{\text{ref}},

per detector bin :math:`i`, where :math:`s` is the good-frame ratio
sample/reference — the same exposure scale the background-run correction uses,
so two runs of unequal beam time are matched before subtraction. The two count
spectra are independent, so the errors add in quadrature,

.. math::

   \sigma_i^{\text{diff}} = \sqrt{N_i^{\text{sample}} + s^2 N_i^{\text{ref}}},

and the reduced asymmetry of the difference is formed from these propagated
errors rather than from a Poisson assumption that no longer holds. Bins that go
negative after subtraction are unphysical as expected counts; their count is
recorded in the combined run's provenance so an over-subtraction is not silent.

WiMDA performs the same count-level subtraction (``cosign = -1``) but leaves the
errors untouched; Asymmetry propagates them (see the porting study). WiMDA also
keeps the master run's temperature and field on a co-add, where Asymmetry
event-weights them and records the spread.

*When to use which.*

- **Co-add** — the same physical measurement repeated for more statistics. The
  result is one higher-statistics run; reduce, fit, or transform it normally.
- **Co-subtract (reference run)** — remove a signal carried by a separate
  exposure: laser-OFF from laser-ON in photo-μSR, or a known reference run.
  Frame-scaled, errors add in quadrature.
- **Background-run correction** (a reduction step, not a combined row; see
  :ref:`backgrounds <background-correction>`) — subtract a scaled background
  *as part of reducing one run*, when the background is a steady detector floor
  rather than a co-measured signal. It shares the same count-level subtraction
  arithmetic with co-subtract but does not create a new dataset.

Provenance and projects
------------------------

The combined run records its constituents (``combined_from``), the operation and
per-constituent weights/scales, the time-zero alignment, and any negative-count
tally under ``metadata["combination"]`` — mirroring the simulation provenance
block. Projects store only the *definition* (the source runs and the operation),
so reloading a ``.asymp`` recomputes the combined row from its sources through
the same count-level path; combined curves saved before this correction will
recompute to the statistically correct values on load.

References
----------

- S. J. Blundell, R. De Renzi, T. Lancaster, and F. L. Pratt, *Muon
  Spectroscopy: An Introduction* (Oxford University Press, Oxford, 2022).
