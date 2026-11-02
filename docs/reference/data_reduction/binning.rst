.. _binning-modes:

Histogram binning
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
never modified, and changing mode (or any width) is always reversible. In
all modes the *counts* are summed onto the output bins and the
asymmetry formed per output bin — at late times the raw bins hold zero
counts, where an asymmetry ratio per raw bin is undefined while summed
counts remain exactly Poisson.

Display bunching
----------------

The main toolbar's **Bunch:** control merges that many recorded output bins.
Its tooltip is ``Merge this many recorded output bins.`` The grouping dialog
sets the base binning; the toolbar composes with it. Fixed factor 4 and
**Bunch:** 2 therefore combine eight acquisition bins per displayed point.
The display factor is stored independently in each saved view mode.

For histogram-backed F–B asymmetry, **Fixed**, **Variable** and **Constant error**
all merge counts and their shared-background uncertainty before forming the
ratio. G±R views bin each period first, then combine their asymmetries and
independent variances. The Groups and Raw counts views sum the corresponding
count bins. A merge larger than the record leaves no complete output point.
Curves without histograms and rotated projections use the independent
curve-bin mean described below.

Good-bin window
---------------

The good-bin endpoints are inclusive, zero-based indices into the aligned
grouped counts. Both endpoints must be integers with
``0 <= first_good_bin <= last_good_bin < number_of_bins``. An absent first
endpoint selects bin 0; an absent last endpoint selects the final bin.
The selected window is applied before any binning mode.

A reversed window raises ``ValueError`` with, for example,
``good-bin window [7, 3] is reversed.`` An endpoint beyond a ten-bin record
raises ``good-bin window [3, 100] is outside 10 bins.`` Invalid windows are
rejected rather than clamped or replaced by the full record. A saved project
containing such a window still loads; reduction of the affected run raises
the error so the window can be corrected.

Forward and backward arrays must have equal one-dimensional shape. A
mismatch raises ``count arrays must have equal shape.`` Supplied count
variances must match that shape and be finite and non-negative; the
acquisition-bin width must be finite and positive. File-derived profile
windows use these same checks. Projects retain saved invalid windows when
loaded; reducing an affected dataset raises the named window error.

Uncertainty of merged counts
----------------------------

An estimated background level is shared by every raw bin. Merging m bins
therefore contributes m² times its level variance, together with the
covariance term for overlap with its estimation window. All binning modes
propagate these terms before forming the asymmetry ratio. Summing per-bin
error squares alone loses the shared contribution. See
:ref:`background-count-uncertainty` for the complete noise model.

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

Rebinning programmatically
--------------------------

A :class:`~asymmetry.core.transform.rebin.BinningSpec` selects the mode,
fixed factor and non-fixed widths. Its ``merge`` combines that many output
bins at the count level; a fixed factor of 4 with merge 2 equals factor 8.
An incomplete final group of output bins is dropped. Reduction records the
spec it used.

.. code-block:: python

   from dataclasses import replace
   from asymmetry.core.transform import binning_spec, reduce_grouped_asymmetry
   from asymmetry.core.transform import correction_flags_from_grouping

   grouping = run.grouping
   bins = replace(binning_spec(grouping), merge=4)
   flags = correction_flags_from_grouping(grouping)
   result = reduce_grouped_asymmetry(
       histograms=run.histograms, grouping=grouping, binning=bins,
       forward_idx=[0], backward_idx=[1], alpha=grouping["alpha"],
       use_deadtime=flags.use_deadtime, deadtime_mode=flags.deadtime_mode,
       use_background=flags.use_background, metadata=run.metadata,
   )

Choose the zero-based detector indices for the required pair. The returned
asymmetry and errors are in percent. The workflow and CLI ``reduce --rebin``
use the same count-level merge. For period sums and differences,
:func:`~asymmetry.core.transform.reduce.reduce_period_asymmetry` bins each
period first, then combines :math:`A_g\pm A_r` with variances added. Periods
must have identical output time arrays; mismatches raise ``ValueError``.

Absent binning keys use fixed factor 1 and widths 0.08/0.25 µs. Present invalid
values raise errors, for example
``binning mode 'Varaible' is not one of fixed, variable, constant_error.`` or
``bunching factor must be an integer >= 1; got 'abc'.`` Both widths must be
finite and positive. A saved invalid grouping payload can be loaded; consuming
it for a profile or reduction raises the named error. A non-positive width raises, for example,
``bin0_us must be finite and > 0; got -1.``

Curves without raw counts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`MuonDataset.merge_curve_bins <asymmetry.core.data.dataset.MuonDataset.merge_curve_bins>`
and :func:`~asymmetry.core.transform.rebin.merge_curve_bins` compute the mean
of consecutive curve points and add their independent variances. This is exact
for the mean of independent Gaussian measurements; it gives different values
and uncertainties from counts-first reduction for sparse or
background-subtracted counts. Workdir-stored curves, rotated-frame projections
and fit-stage bin merging use this explicitly named curve operation.

.. code-block:: python

   coarse = dataset.merge_curve_bins(8)

A factor of one returns a copy. A trailing partial group is dropped. The
per-point uncertainty is :math:`\sqrt{\sum\sigma^2}/factor`, on the input's
fraction or percent scale. The raw acquisition record remains unchanged.

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
