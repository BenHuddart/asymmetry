.. _simulation:

Synthetic runs and degraded statistics
======================================

Asymmetry can manufacture a complete synthetic run — per-detector count
histograms, grouping, provenance and a loadable NeXus file — from any fit
model, and can resample a measured run to a lower statistics level. Both
tools draw Poisson counts at the histogram level, so the synthetic data flow
through exactly the same reduction chain (grouping, α balance, error
propagation, binning) as beamline data, and everything downstream — fitting,
Fourier analysis, MaxEnt — behaves as it would on a real measurement.

Generate a synthetic run
------------------------

**When to use this.** Three situations call for synthetic data. *Fit
validation*: before trusting a fit on real data, check that the analysis
recovers known parameters — simulate from the fitted model, refit, and
compare. *Teaching*: produce realistic datasets for any relaxation or
oscillation function without beam time. *Method development*: manufacture
test inputs with controlled statistics for binning, Fourier or fitting
studies. It is not a substitute for calibration measurements — α, t0 and
backgrounds in a synthetic run are inputs, not measurements.

Open **File → Generate Synthetic Run…** with at least one run loaded. The
loaded run acts as the *instrument template*: the synthetic run copies its
detector count, time binning, per-detector t0, good-bin window, detector
grouping and α. The dialog seeds its model and parameter values from the
template run's current fit when one exists; otherwise pick any time-domain
model with **Edit Model…** (the same builder used by the fit panel).

Each detector in the forward group accumulates expected counts

.. math::

   N_d(t) = N_{0,d}\, e^{-t/\tau_\mu} \left[1 + a(t)\right] + b,

with the backward group seeing :math:`1 - a(t)`, where :math:`a(t)` is the
chosen model (the dialog works in percent, matching the fit panel),
:math:`\tau_\mu = 2.1969811` μs, and :math:`b` is an optional flat
background per bin. The α balance enters as the relative efficiency of the
two groups — forward detectors are weighted by :math:`2\alpha/(1+\alpha)`
and backward by :math:`2/(1+\alpha)` — so the reduction
:math:`(F - \alpha B)/(F + \alpha B)` recovers :math:`a(t)` with the
template's α restored. The recorded counts are independent Poisson draws of
these expectations; bins before t0 contain only the background. **Total
events** sets the expected event budget over the histogram window (the
realised total fluctuates, as it would on the beamline), and defaults to the
template run's own statistics so the synthetic noise level matches the data
you are used to looking at.

The result appears in the Data Browser as ``SIM 90001`` (and counting),
tinted and carrying a tooltip with its provenance: the generating model,
parameter values, seed, event budget and template run. The run's metadata
records the same information permanently. **Save as NeXus…** writes the run
as a standard ISIS muon NeXus file that reloads through *Open Data
File(s)…* with identical counts — this is also how a synthetic run persists
across sessions, since project files reference data files rather than
storing histograms.

Reproducibility is exact: the same template, model, parameters, events and
**seed** produce bit-identical counts on any machine. Leave *Fixed seed*
checked when you want a regenerable dataset (the seed is recorded in the
provenance either way); untick it to draw a fresh seed per generation.

Two caveats worth knowing. The ISIS NeXus format has no field for α, so a
*reloaded* synthetic file starts from the loader's default balance, exactly
as real data would — the generating α is recorded in the file's
``simulation`` group for reference. And the synthetic counts contain no
deadtime distortion, so the file's deadtimes are correctly written as zero;
do not enable deadtime correction for synthetic runs.

Degrade statistics
------------------

**When to use this.** Answering "would half the beam time have been
enough?" — for planning proposals, for checking that a marginal feature
survives at realistic statistics, or for testing how a fit degrades as
counts shrink. Because the thinned run is a new entry beside the original,
a single measurement yields a whole family of statistics levels.

Right-click a run in the Data Browser and choose **Degrade Statistics…**.
A factor :math:`f < 1` keeps each recorded count independently with
probability :math:`f` — binomial thinning. Thinning a Poisson process is
exactly Poisson, so the result is statistically indistinguishable from a
measurement :math:`f` times as long: per-bin errors grow as
:math:`1/\sqrt{f}`, with no approximation. The derived run appears beside
the source (which is never modified), badged with its factor, seed and
parent run, and can be saved as NeXus like any synthetic run.

A factor :math:`f > 1` is also accepted but is an extrapolation, not a
simulation: drawing :math:`\mathrm{Poisson}(f k)` from the recorded
:math:`k` inflates the variance by roughly :math:`1 + f` relative to a
genuinely longer measurement. Use it for qualitative "more statistics"
sketches only — never for quantitative sensitivity claims.

Scripting
---------

The same machinery is available without the GUI, in
:mod:`asymmetry.core.simulate`:

.. code-block:: python

   from asymmetry.core.io import load
   from asymmetry.core.io.nexus_writer import write_nexus_v1
   from asymmetry.core.simulate import degrade_run, reduce_run_to_dataset, simulate_run
   from asymmetry.core.fitting.composite import CompositeModel

   template = load("EMU00020722.nxs").run

   model = CompositeModel.from_expression("Oscillatory * Exponential")
   params = dict(model.param_defaults) | {"frequency": 1.35, "Lambda": 0.21}

   run = simulate_run(template, model, params, total_events=40e6, seed=7)
   dataset = reduce_run_to_dataset(run)      # browser-ready asymmetry curve
   write_nexus_v1(run, "SIM90001.nxs")       # loadable through load()

   half = degrade_run(run, 0.5, seed=1)      # "half the beam time"

Multi-group simulations (a different signal per detector group — phases
around a TF ring, for instance) use
:func:`asymmetry.core.simulate.simulate_run_from_group_signals` directly;
the dialog exposes the forward/backward case only.

A pull-distribution check makes the strongest validation pattern: simulate
many seeds from one parameter set, refit each, and histogram
:math:`(\hat\theta - \theta_{\mathrm{true}})/\sigma_{\hat\theta}`. For a
healthy analysis chain the pulls are standard normal; widths persistently
below one flag over-estimated errors and above one under-estimated errors.
Asymmetry's own test suite runs exactly this check.

References
----------

- S. J. Blundell, R. De Renzi, T. Lancaster, and F. L. Pratt (eds.),
  *Muon Spectroscopy: An Introduction* (Oxford University Press, Oxford,
  2022), Chs. 14–15.
- F. L. Pratt, Physica B **289–290**, 710 (2000) — WiMDA, whose Simulate
  and Degrade Statistics tools this feature parallels.
