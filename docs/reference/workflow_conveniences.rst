Workflow conveniences
=====================

A handful of beamline-honed conveniences that make the browser-and-plot loop
faster: run-quality columns, field read from the data log, a log-count
diagnostic, plain-text export, and a measuring cursor. Each is described below
with the question it answers.

Run-quality columns
-------------------

The Data Browser shows **Run**, **Title**, **T**, and **B** by default. Right-click
any column header and choose **Add column…** to add a run-quality column; the
chosen set is remembered with the project. Two are worth knowing:

- **Good Events (MEv)** — the total counts inside the good-bin range, summed over
  the forward and backward groups, in millions of events. This is the number that
  governs the statistical reach of a run: a fit's parameter errors scale roughly
  as :math:`1/\sqrt{N}`, so a run with 8 MEv reaches about :math:`\sqrt{8/2}=2\times`
  tighter than one with 2 MEv. It differs from **Counts (MEv)**, which sums *every*
  bin including the pre-:math:`t_0` and dead tail — Good Events is what actually
  enters the asymmetry.
- **Events/frame** — Good Events divided by the number of acquisition frames, a
  rate-independent measure of how much beam each run collected. Use it to spot a
  short or beam-starved run in a temperature series before you trust its fit.

Removing a column is the same menu (**Remove from Data Browser**).

Field from the data log
-----------------------

A run's **B** value normally comes from the file header — the setpoint. When the
magnet is ramping or drifting, the *logged* field is the honest one. **Options →
Use field from log** replaces the B column with the mean of the magnetic-field log
channel, exactly as **Use temperature from log** does for T. Log-sourced cells are
tinted and read-only (editing a measured mean is meaningless); turn the option off
to return to the header value and the editable cell.

The choice can be made per run from the run-information dialog, and for a co-added
run the field is the event-weighted mean of its constituents. When a run carries no
field log, the header value stands — there is no silent substitution.

As with temperature, the displayed B value is what a batch parameter trend plots
and exports: a field trend follows the logged field when this option is on
(see :ref:`trend-abscissa-coordinate`).

Log-count diagnostic
--------------------

Switch the time view to **Raw Counts** and tick **Log scale**. A pure muon-decay
histogram is

.. math::

   N(t) = N_0\,e^{-t/\tau_\mu} + b,

so on a logarithmic count axis it is a **straight line of slope**
:math:`-1/\tau_\mu`. That makes three faults jump out that are invisible on a linear
axis:

- a **mis-placed** :math:`t_0` kinks the line at early time;
- a **wrong background** :math:`b` bends the tail upward as the exponential dies and
  the constant takes over;
- **deadtime** at high instantaneous rate flattens the earliest bins.

None of these need a fit to see — the log-count view is the standard first look at a
freshly grouped run. Empty bins (zero counts) have no logarithm and are simply
dropped from the plot.

Plain-text data export
----------------------

The **Export…** button on the plot offers two destinations. *Export to GLE…* writes
the full ``.gleplot`` package as before. *Export plotted data (text)…* writes just
the data — no GLE script, no compile — for loading into any other program. Choose:

- **Data only** — time, asymmetry, and error columns;
- **Data + fit** — the data file plus a ``.fit`` sidecar of the fitted curve;
- **Fit only** — the fitted curve alone.

Tick **Limit to current x-range** to export only the window you are looking at. Every
file carries the same commented provenance header as the GLE sidecars — run number,
grouping, :math:`\alpha`, deadtime — so an exported spectrum stays self-describing.
Exporting one curve prompts for a filename; exporting several prompts for a folder.

A measuring cursor
------------------

Hovering over a plotted curve snaps the cursor to the nearest data point and reports,
in the status bar:

- the point itself, :math:`t` and :math:`A`;
- the **signal-to-noise** :math:`|A/\sigma|` there — a quick "is this feature real?"
  before you commit to a fit;
- a **parabolic peak**: the sub-bin position and height of a local maximum, fitted
  from the point and its two neighbours. This reads a line in an FFT or MaxEnt
  spectrum, or an avoided-level-crossing (ALC) resonance centre, to better than the bin spacing without
  setting up a fit;
- a **windowed average** :math:`\langle A\rangle \pm \sigma` over the visible
  x-range — the level of a baseline, an asymmetry plateau, or an ALC/repolarisation
  step, with the same error-on-the-mean a time-integral would give.

The readouts appear only on a single-curve view, where the snap is unambiguous; on
stacked subplots the cursor reports its raw position.
