.. _simulation:

Synthetic runs and degraded statistics
======================================

Asymmetry can manufacture a complete synthetic run — per-detector count
histograms, grouping, provenance, and a loadable NeXus file — from any fit
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

.. image:: /_generated/screenshots/simulate_dialog.png
   :alt: Generate Synthetic Run dialog with the built-in ideal pulsed F/B
      template and an EuO precession model
   :width: 100%

*The* **Generate Synthetic Run** *dialog opened with no run loaded, so the*
**Template run** *list falls back to the two idealised instruments — here*
**Built-in: Ideal pulsed F/B (ISIS-style)** *. The* **Model A(t)** *is a
damped Larmor precession (an Oscillatory × Exponential composite) at ν ≈ 22
MHz, a below-Tc EuO signal from the archetype gallery, with its parameter
table populated in percent. The* **Total events** *budget pre-fills from the
chosen instrument, the* **Generation** *mode is* **Forward/backward
asymmetry**\ *, and* **Fixed seed** *keeps the draw regenerable.*

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
storing histograms. A synthetic run that has not been saved this way is held
in memory only; saving a project that still references one raises a warning,
because the project will not be able to reload it.

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

Generation modes: asymmetry, counts, and periods
------------------------------------------------

**When to use this.** The **Generation** dropdown chooses *what kind* of
synthetic run to make. Leave it on the default for asymmetry work; switch it
when you need raw single-histogram counts for a count-domain fit, or a
red/green period structure.

*Forward/backward asymmetry* (default) is the run described above: the
antisymmetric :math:`\pm a(t)` split across the forward and backward groups,
reduced to an asymmetry curve. This is what feeds the standard fit panel and
the α-free forward/backward count fit, which needs the F/B pair to separate
the balance :math:`\alpha` from the amplitude.

*Count histograms (single-group)* imprints the **same** :math:`+a(t)` on every
group, so each detector group is an independent single-histogram measurement

.. math::

   N_g(t) = N_{0,g}\, e^{-t/\tau_\mu} \left[1 + a(t)\right] + b,

with no balancing backward detector. This is the data the **single-histogram**
count fit expects — fit any one group and recover its :math:`N_0`, the
amplitude inside :math:`a(t)`, and the flat background. Use it for
longitudinal-field / zero-field (LF/ZF) single-detector work, or to manufacture a clean test case for the count-domain
fit modes.

*Two-period (red/green)* generates two period histograms in one run, the way a
light-on/off or RF-on/off measurement is recorded. Red carries the full model;
the **Green amplitude (×)** factor scales green's signal — leave it at
:math:`0` for a flat reference period (the usual light-off case), so the
green-minus-red combination :math:`G - R` recovers the red signal cleanly. Each
period is an independent Poisson draw from one seed. The run lands in the Data
Browser with the full red/green machinery: pick the period or the
:math:`G \mp R` combination in the grouping dialog exactly as for a loaded
period-mode file. Save-as-NeXus flattens a two-period run to its red period —
period-mode synthetic runs live in memory and through the project, not on disk.

Built-in instrument templates
-----------------------------

**When to use this.** Teaching and quick experiments with no data to hand.
The dialog opens without a loaded run and offers two idealised instruments
that supply the geometry a real template would, so you can go straight from a
model to a synthetic run.

The two built-ins are the contrasting source archetypes of the field. The
**ideal pulsed F/B** mirrors an ISIS-style spectrometer — 32 forward and 32
backward detectors, 16 ns bins over a 32 μs window, and no uncorrelated
background, since a pulsed source has essentially none. The **ideal
continuous F/B** is a PSI-style pair with fine 1 ns binning over a short
10 μs window and a flat background of about 10 counts/bin/detector: the
time-independent uncorrelated background that characterises a continuous
source, where muons arrive one at a time and the time-zero clock runs free.
Select either from the *Template run* list; the event budget and background
spinners pre-fill with sensible values for the chosen instrument. Everything
downstream — generation, the NeXus round trip, refitting — behaves exactly as
for a loaded-run template.

The archetype gallery
---------------------

**When to use this.** A one-click route to a physically realistic dataset for
a named material, for teaching or for a sanity check that the analysis chain
reproduces textbook behaviour. **File → Simulate Preset** lists each
archetype; choosing one generates a badged synthetic run (or a whole scan)
from a fixed seed, so the same preset always yields the same data.

The presets and the physics they carry:

* **Ag — ZF Gaussian Kubo–Toyabe.** The canonical static nuclear-dipolar
  reference, :math:`\Delta = 0.39` μs\ :sup:`-1`.
* **Ag — LF decoupling series.** The same Kubo–Toyabe in longitudinal fields
  of 0, 10, 25 and 50 G, the textbook test that the dipolar fields are static.
* **EuO — ferromagnet through T**\ :sub:`c`. A zero-field temperature scan
  across :math:`T_c = 69` K: below it the muon precesses at a frequency that
  tracks the magnetic order parameter :math:`(1 - T/T_c)^\beta`; above it the
  signal is paramagnetic relaxation that broadens through the critical region.
* **PbF₂ — F-μ-F entanglement.** The dipolar beat pattern of a muon bound
  between two fluorine neighbours at :math:`r \approx 1.17` Å.
* **YBCO — transverse-field precession.** Knight-shifted Larmor precession in
  a 200 G transverse field.

Because each preset is a real synthetic run, you can refit it and recover the
input physics. Fitting the Ag preset, for instance, returns
:math:`\Delta = 0.389(1)` μs\ :sup:`-1`, and the F-μ-F preset returns
:math:`r_{\mu\mathrm{F}} = 1.170(1)` Å — the generating values, within the
counting errors. The parameter values follow Blundell *et al.* (see the
references); the preset tooltips name the relevant chapter.

Multi-group simulation
----------------------

**When to use this.** Synthesising a transverse-field ring whose detector
groups see the *same* precession at *different* phases — for instance to
manufacture test data for the multi-group time-domain fit, or to show how the
relative phase around a ring maps onto the geometry. **File → Generate
Multi-Group Run…** opens a per-group table of amplitude, relative phase and
relative count rate, over a shared normalised polarisation model.

The forward/backward dialog applies one signal with the α split; here each
group :math:`g` instead carries

.. math::

   a_g(t) = A_g \, P(t;\, \phi \to \phi + \phi_g),

the shared polarisation :math:`P(t)` scaled by the group's own amplitude
:math:`A_g` and shifted by its own phase :math:`\phi_g`. The amplitude owns
the overall scale and the count-rate weight owns the background, so the shared
model is a unit-amplitude, zero-baseline polarisation — exactly the contract
the grouped time-domain fit expects. If you have just run a grouped fit on the
selected run, the table is **seeded from that fit's per-group amplitudes and
phases**, so "simulate what I just fitted" is one click. The reduction
recovers each group's seeded signal bin-for-bin in expectation.

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

.. _pull-diagnostic:

Validating error bars: the pull distribution
---------------------------------------------

**When to use this.** After a fit, to ask the sharpest question you can ask
of any analysis chain: *are the error bars honest?* A fitted value is only as
trustworthy as its uncertainty, and the only way to test an uncertainty is to
repeat the measurement many times and see whether the scatter matches. The
pull diagnostic does exactly that, with synthetic repeats.

After a converged single fit, click **Pull diagnostic…**. The tool
re-simulates the fitted run many times at matched statistics, refits each copy
over the same window, and for every free parameter forms the *pull*

.. math::

   \mathrm{pull} = \frac{\hat\theta - \theta_{\mathrm{true}}}{\sigma_{\hat\theta}},

where :math:`\theta_{\mathrm{true}}` is the value you fitted and
:math:`\sigma_{\hat\theta}` the error each refit reports. For a sound chain the
pulls are standard normal. Read the histogram against its :math:`N(0, 1)`
overlay two ways: the **mean** should sit at zero (a non-zero mean is a
*bias* — the fit systematically misses), and the **width** should sit at one.
A width below one means the reported errors are too large; above one, too
small. The verdict line states this for each parameter, with the measured
width and its uncertainty, e.g. ``width 1.02(7)`` — well-calibrated.

Asymmetry's reduction uses exact Poisson error propagation for the asymmetry,
:math:`\sigma_A^2 = (1 - A^2)/(F + \alpha B)`-style rather than the
independent-numerator-and-denominator form, so the pull widths centre on one
with no fudge factor — the same check the test suite runs on every build.

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

No run to hand? Build one of the idealised instruments instead of loading a
file:

.. code-block:: python

   from asymmetry.core.simulate import build_builtin_template, simulate_run

   template = build_builtin_template("ideal_pulsed_fb")   # or "ideal_continuous_fb"
   run = simulate_run(template, model, params, total_events=40e6, seed=7)

The textbook archetypes are one call each, returning the run (or the whole
scan family), badged and ready to plot or fit:

.. code-block:: python

   from asymmetry.core.simulate_presets import build_preset_runs

   ag = build_preset_runs("ag_zf_kt")[0]          # single run
   euo_scan = build_preset_runs("euo_tscan")      # five temperatures

Count-mode and two-period synthesis have their own entry points. Count-mode
returns a run whose per-group histograms feed the single-histogram count fit;
two-period takes a :class:`~asymmetry.core.simulate.PeriodSpec` for red and
green and returns the loadable red/green payload.

.. code-block:: python

   from asymmetry.core.simulate import (
       PeriodSpec, simulate_count_run, simulate_two_period_run,
   )

   counts = simulate_count_run(template, model, params, total_events=40e6, seed=7)

   # Red carries the full model; green a half-amplitude variant (scale=0 makes
   # a flat reference period, so G−R recovers the red signal). Each period can
   # instead pass its own model/parameters/alpha/total_events.
   red = PeriodSpec(model, params)
   green = PeriodSpec(model, params, scale=0.5)
   periods = simulate_two_period_run(
       template, [red, green], total_events=40e6, seed=7,
   )

Multi-group simulations (a different amplitude and phase per detector group —
phases around a transverse-field (TF) ring, for instance) use
:func:`asymmetry.core.simulate.simulate_multi_group_run` with a
:class:`~asymmetry.core.simulate.GroupSignalSpec` per group; the
forward/backward dialog exposes only the single-signal α-split case.

.. code-block:: python

   from asymmetry.core.simulate import GroupSignalSpec, simulate_multi_group_run

   ring = CompositeModel(["Oscillatory"])        # normalised polarisation
   specs = [GroupSignalSpec(g, amplitude=0.2, relative_phase=phi)
            for g, phi in [(1, 0.0), (2, 1.57), (3, 3.14), (4, 4.71)]]
   run = simulate_multi_group_run(template, ring, specs,
                                  total_events=40e6,
                                  base_parameters={"frequency": 2.7})

A pull-distribution check makes the strongest validation pattern, and
:func:`asymmetry.core.pull_diagnostic.run_pull_distribution` packages it:
simulate many seeds from one parameter set, refit each, and histogram
:math:`(\hat\theta - \theta_{\mathrm{true}})/\sigma_{\hat\theta}`. For a
healthy analysis chain the pulls are standard normal; widths persistently
below one flag over-estimated errors and above one under-estimated errors.
Asymmetry's own test suite runs exactly this check.

References
----------

- S. J. Blundell, R. De Renzi, T. Lancaster, and F. L. Pratt,
  *Muon Spectroscopy: An Introduction* (Oxford University Press, Oxford,
  2022), Chs. 14–15.
- F. L. Pratt, Physica B **289–290**, 710 (2000) — WiMDA, whose Simulate
  and Degrade Statistics tools this feature parallels.
