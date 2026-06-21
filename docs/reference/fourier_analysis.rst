Fourier Analysis
================

.. image:: /_generated/screenshots/fourier_tf.png
   :alt: Frequency-domain Fourier spectrum of a YBCO vortex-lattice TF μSR signal
   :width: 100%

*Central plot in the* **Frequency** *domain showing the grouped Fourier*
*spectrum of a synthetic YBa₂Cu₃O₇₋δ vortex-state TF μSR run (TF = 200*
*mT, T = 10 K), zoomed to the Larmor peak at γ_μ·B_app ≈ 27.1 MHz. The*
*canonical asymmetric line shape — sharp low-field peak at the*
*saddle-point van Hove singularity, long high-field tail toward the*
*vortex cores — is the textbook signature of an isotropic triangular*
*vortex lattice (Brandt PRB 37, 2349, 1988; Sonier RMP 72, 769, 2000;*
*textbook Ch 8/9).*

Fourier analysis answers questions that the time domain answers slowly or
not at all: whether a signal contains three precession frequencies or two
(multi-site magnets, F-μ-F entanglement, muonium hyperfine pairs), and
what the internal-field distribution :math:`P(B)` looks like inside the
vortex lattice of a type-II superconductor in the mixed state
(:doc:`/workflows/superconductor_penetration_depth`). The frequency-domain
view is also the standard way to choose a sensible seed frequency before
attempting an oscillatory time-domain fit (:doc:`fit_functions/oscillation`).

Apodisation is the key practical knob. A Lorentzian filter sharpens an
exponentially damped line; a Gaussian filter is the natural choice for
nuclear-dipolar broadening; ``None`` is appropriate only when the
signal-to-noise is high enough that the line broadening from apodisation
would dominate the intrinsic widths. The default is Gaussian. Phase
correction follows the same logic: the entropy-optimised auto-phase mode
is robust for single-frequency signals; for multi-frequency or vortex
data, the manual per-group phase table or a future optimised phase fit
gives better control. Maximum entropy reconstruction (see `Maximum Entropy
Method`_ below) is the alternative when an FFT line is too weak or the usable
time window too short for the FFT to resolve it.

.. image:: /_generated/screenshots/apodisation_comparison.png
   :alt: Apodisation comparison on a YBCO vortex-lattice TF FFT
   :width: 100%

Implementation Summary
----------------------

Asymmetry currently exposes two Fourier workflows:

* the core Python API can transform a supplied time-domain dataset directly
* the desktop GUI uses a WiMDA-style grouped-detector workflow that rebuilds a
  grouped count signal for each included detector group before taking the FFT

In the GUI, the frequency-domain plot always shows the average of the
currently included groups rather than separate per-group spectra. Each loaded
dataset keeps its own Fourier phase-table state, so changing runs restores the
phase values, included groups, and auto-estimated markers associated with that
run.

Notation
--------

In this guide, frequency-domain quantities use standard symbols:

* :math:`A(t)` for asymmetry in the time domain
* :math:`\nu` for frequency (MHz)
* :math:`\lvert\mathcal{F}\rvert` for FFT magnitude

Fast Fourier Transform (FFT)
-----------------------------

The simplest method for frequency analysis:

.. code-block:: python

   from asymmetry.core.fourier.fft import fft_asymmetry
   from asymmetry.core.io import load
   
   dataset = load("data.nxs")
   
   # Compute FFT — returns (frequencies, real_part, magnitude)
   frequencies, real_part, magnitude = fft_asymmetry(dataset)
   
   # Plot
   import matplotlib.pyplot as plt
   plt.plot(frequencies, magnitude)
   plt.xlabel(r"$\\nu$ (MHz)")
   plt.ylabel(r"$|\\mathcal{F}|$")
   plt.show()

Filter / Apodisation
~~~~~~~~~~~~~~~~~~~~

The main Fourier workflow now follows WiMDA's filter model rather than the
older whole-trace window presets. The FFT panel exposes three apodisation
modes:

* ``None``
* ``Lorentzian``
* ``Gaussian``

along with:

* ``Filter start`` in microseconds
* ``Tau`` (the filter time constant) in microseconds

For a Lorentzian filter, Asymmetry follows WiMDA's two cases:

.. math::

   A_f(t) = e^{-t/\tau} A(t) \qquad \text{when } t_\mathrm{start} = 0

.. math::

   A_f(t) = \frac{1 + e^{-t_\mathrm{start}/\tau}}{1 + e^{(t - t_\mathrm{start})/\tau}} A(t)
   \qquad \text{when } t_\mathrm{start} > 0

The Gaussian mode uses the same WiMDA-style softened start with squared
arguments. ``Subtract average signal`` is applied before the filter, matching
WiMDA's default preprocessing path.

How The GUI FFT Is Built
~~~~~~~~~~~~~~~~~~~~~~~~

The docked Fourier workflow in the desktop application is intentionally more
specific than the low-level API:

* the FFT input is the grouped detector-count signal, not the forward/backward
  asymmetry trace
* the grouped signal is rebuilt from the current grouping and current bunching
  factor every time the transform is computed
* grouped-count inputs are corrected for the muon lifetime envelope before the
  later mean-subtraction and apodisation stages
* the active time-domain fit range is used as the FFT time window when a fit
  range is present
* zero padding is applied after the selected time window and preprocessing are
  fixed

This means the Fourier spectrum shown in the GUI is tied directly to the
current grouped analysis state, rather than being a generic transform of
whatever line is currently plotted in the time-domain view.

Background subtraction for the FFT input
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because the FFT input is rebuilt from the current grouping, the background
mode you set in the grouping dialog applies to the transform too — there is no
separate FFT-only background control. A flat uncorrelated background matters
here more than anywhere else: the grouped signal is lifetime-corrected by
:math:`e^{t/\tau_\mu}` before the FFT, so a constant rate :math:`b` becomes a
*growing* :math:`b\,e^{t/\tau_\mu}` ramp that dumps spurious power into the
low-frequency bins. Subtracting :math:`b` at the count level, before the
lifetime correction, removes that ramp and flattens the baseline.

Which background option when:

* **None** — the default, and the right choice at a pulsed source whenever the
  flat rate is consistent with zero (the ISIS duty factor suppresses it to
  "virtually unmeasurable", *Muon Spectroscopy* §14.3). Start here.
* **Tail fit** — fits a muon exponential plus a flat rate to the late-time
  counts and subtracts the flat part. This is the option for **pulsed data**,
  where there is no pre-t0 region: the spectrum starts at the muon pulse, so
  the background can only be read off the long counting tail. Asymmetry fits it
  by Poisson maximum likelihood (correct for the few-count late-time bins) and
  flags a result below two standard errors as consistent with zero — at ISIS
  expect exactly that, with a significantly non-zero value being a diagnostic
  (dark counts, light leaks, surviving upstream decays).
* **Pre-t0 range** — averages a band of pre-trigger bins. The natural estimate
  at a **continuous source**, where the muon arrives at a sharp prompt peak and
  the bins before it measure the background directly. Unavailable for pulsed
  files (no pre-t0 region to average).
* **Fixed** — subtracts a per-group constant you supply, for when the level is
  known independently.
* **Reference run** — subtracts a separately measured reference (sample holder,
  silver, laser-off) scaled by the good-frame ratio, for a *structured*
  background rather than a flat rate.

The same estimate is shared between the time-domain reduction and the FFT
input, so the value the grouping dialog previews is exactly what the transform
subtracts.

FFT Phase Modes
~~~~~~~~~~~~~~~

The GUI computes one complex grouped FFT and then derives the displayed curve
from that spectrum:

* ``(Power)^1/2``: magnitude :math:`|F|`
* ``Phase Spectrum``: raw spectral angle
* ``Cos``: real / cosine component of the uncorrected FFT
* ``Sin``: imaginary / sine component of the uncorrected FFT
* ``Phase``: WiMDA-style phase-corrected real projection
* ``phaseOptReal``: entropy-optimised real projection using musrfit's
  two-parameter optimiser backend

Only ``Phase`` uses the manual phase value, per-group phase table, and
``t0 Offset (μs)`` entry. ``phaseOptReal`` ignores those manual controls and
computes its own optimiser-driven correction instead.

Maximum Entropy Method
-----------------------

Maximum entropy (MaxEnt) reconstructs the frequency spectrum as the *least
committal* positive distribution consistent with the measured counts: it
maximises the spectral entropy subject to a :math:`\chi^2` constraint against
the data, weighting each bin by its own error. Compared with the FFT this can
resolve sharper lines from short or noisy time windows, at the cost of an
iterative refinement that you drive cycle by cycle and watch converge.

MaxEnt is a shipped feature (added in PR #16) and is available from both the
core Python API and the desktop GUI.

Engine API
~~~~~~~~~~

.. code-block:: python

   from asymmetry.core.io import load
   from asymmetry.core.maxent import maxent, MaxEntConfig

   dataset = load("data.nxs")

   config = MaxEntConfig(
       n_spectrum_points=2048,
       f_min_mhz=0.0,
       f_max_mhz=5.0,
       t_min_us=0.1,
       t_max_us=8.0,
   )
   result = maxent(dataset.run, config, cycles=10)

   import matplotlib.pyplot as plt
   plt.plot(result.frequencies_mhz, result.spectrum)
   plt.xlabel(r"$\nu$ (MHz)")
   plt.ylabel("MaxEnt amplitude")
   plt.show()

.. warning::

   **Pass** ``dataset.run``\ **, not the dataset.** ``maxent()`` operates on a
   :class:`~asymmetry.core.data.Run` — the raw-histogram object — *not* on the
   :class:`~asymmetry.core.data.dataset.MuonDataset` that ``load()`` returns.
   The dataset exposes the required ``Run`` as its ``.run`` attribute, so the
   call is ``maxent(dataset.run, config)``. Passing the ``MuonDataset`` itself
   raises a ``TypeError`` that names the fix (*"MaxEnt expects a Run … pass the
   dataset's .run"*).

.. warning::

   ``MaxEntConfig`` is a dataclass whose **first** positional field is
   ``n_spectrum_points`` — *not* ``f_min_mhz``. Always construct it with
   **keyword arguments**. A positional call such as ``MaxEntConfig(0.0, 5.0)``
   silently misconfigures the run, binding your intended frequency bounds to
   the wrong fields rather than raising an error. The time-window fields are
   ``t_min_us`` / ``t_max_us`` (microseconds), not ``t_min`` / ``t_max``.

``maxent(run, config, *, cycles=None, state=None, ...)`` returns a
``MaxEntResult`` with:

* ``frequencies_mhz`` — the frequency grid (MHz)
* ``spectrum`` — the reconstructed amplitudes on that grid
* ``diagnostics`` — per-cycle convergence metrics (``chi2``, ``entropy``,
  ``test``, ``sconv``, and the fitted phases/amplitudes/backgrounds)
* ``state`` — a resumable state object

``cycles`` sets how many outer refinement cycles to run. When ``cycles=None``
(the default) the engine runs ``config.outer_cycles`` cycles (``10`` by
default). Each call returns a usable spectrum; pass the returned ``result.state``
back in via ``state=`` to add further cycles incrementally rather than starting
over. Reusing a ``state`` after changing the configuration raises
``ValueError`` (``"MaxEnt state is incompatible with the current
configuration; restart."``) — build a fresh state for the new settings.

GUI Workflow
~~~~~~~~~~~~

Open the MaxEnt panel from **Frequency-domain → MaxEnt**: first select the run
in the F–B asymmetry / time view, then click the **MaxEnt** toolbar button.

The panel groups its controls as follows:

* **Groups** table — include each detector group and set its per-group phase.
* **Spectrum** — number of spectrum points and the default level (the flat
  prior the reconstruction starts from).
* **Window** — ``Auto window from field`` (derive the frequency window from the
  applied field), ``Half width (G)``, or an explicit ``Min`` and ``Max``
  frequency.
* **Time** — ``Start`` / ``End`` (µs) of the fitted window and a ``Binning``
  factor.
* **Cycle Refinement** — ``Inner iterations``; ``χ² target / N``; toggles to
  ``Fit phases`` / ``amplitudes`` / ``backgrounds`` / ``constant background``;
  and ``Use existing deadtime correction``.
* Cycle buttons — **+1**, **+5**, **+25** step the calculation by that many
  outer cycles; **Converge** runs a larger fixed batch in one go; **Restart**
  discards the current state; **Cancel** stops a running calculation.
* **Apply settings to selected runs** copies the current configuration to the
  other selected runs.
* A **Progress** area and a **Diagnostics** line reporting the latest
  ``Cycle``, ``χ²``, ``TEST``, and ``entropy``.

.. important::

   After changing **any** configuration — the frequency window, spectrum
   points, time window, binning, or the fit toggles — click **Restart** before
   running more cycles. Continuing into a stale state fails with *"MaxEnt state
   is incompatible with the current configuration."* Restart discards the
   resumable state so the next cycle button rebuilds it from the new settings.

Seeding a clean reconstruction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MaxEnt rewards a tightly scoped problem. To resolve a clean precession line:

* **Narrow the frequency window.** Uncheck ``Auto window from field`` and set a
  ``Max frequency`` just above the line you expect, so the reconstruction does
  not spend amplitude on empty high-frequency bins.
* **Set a sensible time window.** Start after t\ :sub:`0` and end where the
  signal has decayed into noise.
* **Keep the spectrum points and cycle count modest.** Fewer points and fewer
  cycles are usually enough to bring a single line out; reach for more only
  when a feature is genuinely unresolved.
* **Watch the Diagnostics line.** The ``χ²`` should fall toward the target and
  settle. If it stops improving — or starts climbing again under continued
  refinement — the fit is over-iterating; **Restart** and run fewer cycles
  (or with fewer spectrum points) rather than pushing it further.

As a worked example, the CdS shallow-donor TF 100 G run (run 20711) resolves
the 1.38 MHz Mu\ :sup:`+` line — :math:`\gamma_\mu \cdot 100\,\text{G}` — with a
frequency window capped a little above 1.38 MHz, a good time window, and a
modest number of cycles and spectrum points.

Reading the reconstruction overlay
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The spectrum is what MaxEnt is *for*, but the **time-domain reconstruction** is
how you check it. MaxEnt works by forward-modelling the raw counts: it asks what
each detector group's signal would look like if the data really came from the
current spectrum, and adjusts the spectrum until those model signals match the
measurements. The reconstruction overlay shows that comparison directly — the
measured signal and the model the spectrum predicts, group by group, in time.

After a run, tick **Show time-domain reconstruction** in the MaxEnt panel. The
plot workspace switches to a stacked view: one panel per detector group with the
data and the reconstructed model on the same axes, and a residuals strip beneath
showing the weighted residual :math:`(d-m)/\sigma`. The χ² printed above the
plot is the same number the Diagnostics line reports — it is computed from these
very residuals, so the overlay and the convergence figure can never disagree.

Tick **Combine groups on one axis** to overlay every group's data and model on a
single colour-coded panel above one shared residuals strip, instead of the
per-group stack. The combined view is the quicker read when you only want to see
whether the whole fit holds together; the per-group stack is better for tracing
left-over structure to a particular detector. Both report the same total χ².

This is the strongest single check of fit quality:

* **A good fit** sits on the data with residuals scattered evenly around zero
  and no left-over oscillation — the model has captured the precession.
* **Left-over structure in the residuals** (a coherent wiggle, a slow drift)
  means the spectrum is missing a line, has put it at the wrong frequency, or
  has the wrong phase for that group. Widen or re-centre the frequency window,
  re-seed the phases, and run again.
* **Residuals that blow up at early or late times** point at the time window or
  an instrumental artefact (deadtime at short times on a pulsed source; noise at
  long times) rather than at the spectrum itself.

The reconstruction is recomputed from the converged spectrum each run and is not
stored in the project file; re-open the spectrum and run a cycle to regenerate
it. Untick the box (or click the **MaxEnt** domain button) to return to the
spectrum.

Pulsed sources: the pulse-shape response
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At a pulsed source (ISIS) the muons arrive spread over a finite pulse of order
tens of nanoseconds, so a precession signal that oscillates fast compared with
the pulse is averaged across the spread of arrival times and its amplitude is
suppressed. Above roughly 5 MHz this distorts the recovered spectrum: a real
line at high frequency comes out with too little weight. Continuous sources
(PSI, TRIUMF) do not have this limit, so the correction is off by default.

The **Pulse shape** controls fold the instrument response into the forward
model, where it belongs — MaxEnt then reconstructs the *true* spectral weight
rather than the attenuated one. Choose:

* **Ignore** — no pulse shaping (continuous-source data).
* **Single pulse** — one proton pulse of the given **Half-width** (µs).
* **Double pulse** — the ISIS double-pulse structure, with the **Separation**
  (µs) between the two pulses.

The response is a parabolic proton-pulse transform rolled off by the pion
lifetime; the half-width and separation default to the ISIS values (about 50 ns
and 0.324 µs) because the loaders do not yet record them per run. The effect is
dramatic: on synthetic pulsed data with equal-amplitude lines at 1 and 7 MHz,
the recovered 7 MHz weight is suppressed by a factor of ~4 with the response off
and restored to roughly the 1 MHz weight with it on.

Excluding a time window
~~~~~~~~~~~~~~~~~~~~~~~~~

A laser flash, an RF burst, or a detector glitch can corrupt an interior stretch
of the histogram. Set **De-weight from / to** (µs) to down-weight that window in
the reconstruction. The points are *de-weighted* (their error bars are inflated
so they carry essentially no weight), not removed, so the time grid — and any
frequency resolution derived from it — is unchanged. This is why the control is
named for de-weighting rather than exclusion (contrast the count fit's hard
*Skip window*; see :doc:`exclusions`). Leave the fields blank to disable.

Field axis
~~~~~~~~~~~

A precession line at frequency ν corresponds to a local field B through
ν = γ_μ B / 2π, with γ_μ/2π = 135.5 MHz/T. The spectrum's **X Units** selector
(above the plot) switches the axis between frequency (MHz) and field (Gauss or
Tesla) without recomputing — handy when the science is a field distribution
(a vortex lattice, an internal-field spread) rather than a frequency.

Calibration: phases, deadtime, and ZF/LF mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MaxEnt needs one phase per detector group, and a good spectrum needs good
phases. The **Calibration** controls connect MaxEnt to the rest of the analysis.

**Phase exchange.** If you have already fitted the run with a grouped
time-domain fit, click **Use fitted phases** to seed the MaxEnt group phases
from that fit (converted from radians to degrees). After MaxEnt has refined the
phases, **Send phases to fit** writes them back to the grouped fit. A provenance
line records which direction the exchange went and when. (Phases are matched by
group id, so the forward/backward mapping never gets crossed.)

.. note::

   **The FFT per-group phases and the MaxEnt fitted-phase exchange are separate
   stores, by design.** The Fourier panel's per-group phase table (`FFT Phase
   Modes`_, with its auto-estimate) and this MaxEnt exchange (which swaps phases
   with the *grouped time-domain fit*) hold the same physical quantity — the
   per-group detector phase — but do not feed each other: refining phases in
   MaxEnt does not update the FFT phase table, and filling FFT phase estimates
   does not seed MaxEnt. Keep them in sync by hand if you switch estimators on
   the same run. A "use MaxEnt fitted phases" pull on the Fourier panel is a
   plausible convenience and will be added on the first concrete request for it;
   until then the two stores are independent.

**Deadtime.** At a pulsed source the detectors lose counts at early times where
the rate is highest, distorting the signal that carries the high-frequency
information. **Fit deadtime** estimates a per-detector deadtime from the
early-time count decay; **Apply to grouping** then promotes it to the run's
deadtime correction. The fit never changes the grouping on its own — you apply
it explicitly, so the provenance is clear.

**ZF/LF mode.** In zero or longitudinal field the two detectors (forward and
backward) measure the same relaxation with opposite phase. Select **ZF / LF
(two-group)** as the mode: include exactly two groups, and MaxEnt pins their
phases to 0° and 180° and ties their amplitudes through the run's α (the
detector-efficiency balance). The resulting spectrum is the **field
distribution** p(B) — for a static Gaussian Kubo–Toyabe relaxation it is broad
and centred near zero field rather than a sharp line. (The MaxEnt method is
tuned more for transverse-field rotation than for zero-field precession, so read
ZF spectra as distributions, not as resolved frequencies.)

**Zero-frequency background (SpecBG).** A ZF field distribution is dominated by
its strong central peak, which can bury weak satellite structure. Enable
**Zero-frequency background** (ZF/LF mode only) to subtract a zero-centred
pseudo-Voigt model of that central peak from the *displayed* spectrum — the
Gaussian and Lorentzian widths and the Lorentzian fraction shape the model. It
is display-only and never alters the reconstructed spectrum.

**Export.** **Export spectrum…** writes the spectrum as text (frequency MHz,
field G, density, with a parameter header); **Export log…** writes the per-cycle
convergence trace and the final per-group phases, amplitudes, and backgrounds.

.. _choosing-spectral-estimator:

Choosing a spectral estimator: FFT, MaxEnt, or Burg
---------------------------------------------------

Asymmetry offers three ways to turn a time-domain signal into a spectrum, and
they answer different questions. The first two are quantitative measurements;
the third is a diagnostic.

Use the **FFT** for **speed and linearity**:

* The signal-to-noise ratio is good
* Fast, one-shot computation is what you need
* You want a linear transform that maps directly onto the standard apodisation
  knobs, with amplitudes and areas you can trust

Use **MaxEnt** for **pulsed-source resolution and per-group phase handling**:

* The data is noisy or the usable time window is short, and you need to pull a
  weak line out at higher effective resolution
* You are at a pulsed source and want the finite-pulse response folded into the
  forward model (see `Pulsed sources: the pulse-shape response`_) rather than
  fighting the high-frequency rolloff afterwards
* You want one reconstruction that carries per-group phases and per-bin errors
  through to the spectrum (see `Calibration: phases, deadtime, and ZF/LF mode`_)
* You are willing to drive the iterative refinement and watch it converge

Use **Burg** (the *Resolution (Burg)* display mode) only as a **line-splitting
diagnostic**:

* You want to ask "is that one line or two?" — the all-poles estimate resolves
  close lines from a short window that collapse to a single FFT peak, and the
  pole count hints at how many components a fit should include
* You then read the count and rough positions and **measure** the lines with an
  FFT/MaxEnt spectrum or a frequency-domain fit — never off the Burg curve
  itself, whose peak heights are not amplitudes and which is prone to spurious
  splitting and noise-dependent position bias

The Burg method, its pole scan, and its pathologies are documented in
:doc:`frequency_finishers`. Once you have a spectrum from any estimator, fit it
quantitatively through :doc:`frequency_domain_fitting`.

The same line, two ways
-----------------------

The two quantitative estimators are most easily compared on one dataset. The
figures below show the transverse-field signal of a YBCO vortex lattice — whose
internal-field distribution :math:`P(B)` is intrinsically asymmetric, with a
sharp edge near the saddle-point field and a long tail towards the vortex cores
— reconstructed both ways.

.. image:: /_generated/screenshots/fourier_tf.png
   :alt: FFT spectrum of a YBCO vortex-lattice transverse-field signal
   :width: 100%

*The FFT: a fast, linear transform of the apodised asymmetry, framed on the
precession line.*

.. image:: /_generated/screenshots/maxent_ybco.png
   :alt: MaxEnt reconstruction of the same YBCO vortex-lattice signal
   :width: 100%

*The maximum-entropy reconstruction of the same run. Switch the frequency-domain
workspace to* **MaxEnt** *and click* **Compute MaxEnt** *(or a cycle button); the
spectrum appears once the iteration converges.*

Read on the same precession-frequency axis, the two are complementary: the FFT is
immediate and its areas map onto the apodisation settings, while the MaxEnt
reconstruction resolves the asymmetric line at higher effective resolution and
carries the per-group phases, the per-bin errors, and — at a pulsed source — the
pulse response through its forward model. Choose between them with the guidance
above.

Practical Example
-----------------

Complete workflow for frequency analysis:

.. code-block:: python

   from asymmetry.core.io import load
   from asymmetry.core.fourier.fft import fft_asymmetry
   import matplotlib.pyplot as plt
   
   # Load data
   dataset = load("data.nxs")
   
      # Compute FFT with WiMDA-style Gaussian apodisation and 4× zero-padding
   freq, real_part, magnitude = fft_asymmetry(
       dataset,
         window="gaussian",
         filter_start_us=0.2,
         filter_time_constant_us=1.5,
       padding_factor=4,
       t_min=0.1,
       t_max=10.0,
   )
   
   # Plot
   fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
   
   # Time domain
   ax1.errorbar(dataset.time, dataset.asymmetry, yerr=dataset.error, fmt='.')
   ax1.set_xlabel("Time (μs)")
   ax1.set_ylabel("Asymmetry")
   ax1.set_title("Time Domain")
   
   # Frequency domain
   ax2.plot(freq, magnitude)
   ax2.set_xlabel(r"$\\nu$ (MHz)")
   ax2.set_ylabel(r"$|\\mathcal{F}|$")
   ax2.set_title("Frequency Domain (FFT with Gaussian apodisation)")
   ax2.set_xlim(0, 50)  # Focus on 0–50 MHz
   
   plt.tight_layout()
   plt.show()

GUI Fourier Workflow
--------------------

The FFT spectrum does **not** compute automatically. Nothing is transformed
until you click ``Compute FFT``, and — exactly like the MaxEnt panel's cycle
buttons (see `Maximum Entropy Method`_) — that button sits below the fold in a
scrollable panel, so on a narrow dock it is easy to miss. A run you have not
computed shows a **centred prompt over the Frequency-domain plot** —
*"No FFT computed for this run. Configure the FFT parameters and click Compute
FFT."* — instead of a blank tab. That prompt is the expected
"not computed yet" state, not a broken transform, and it clears the moment a
spectrum is computed.

To compute an FFT spectrum:

#. **Select the run** in the Data Browser. Selecting a run populates the
   Fourier panel's ``Groups`` table (``Group 1`` / ``Group 2`` …). With no run
   selected the ``Groups`` table — and therefore the spectrum — stays empty.
#. **Bring up the Fourier panel** if it is not already visible (the toolbar
   ``FFT`` button raises it). The ``FFT`` button also switches the central plot
   to the **Frequency Domain** tab, so you see the spectrum area (with the
   compute prompt until you transform) rather than staying on the time view. It
   does **not** run the transform.
#. *(Optional)* **Tune the settings** — phase mode, apodisation, included
   groups, phases — using the controls described below.
#. **Click** ``Compute FFT``. This button is at the **bottom** of the panel,
   below the settings groups; on a narrow dock you must scroll the panel down to
   reach it. Only this click runs the transform and fills the
   **Frequency Domain** tab.

While the Frequency-domain tab has no computed spectrum, the plot shows the
compute prompt above and the status line reads ``No FFT computed for run <n>`` —
configure the parameters and click ``Compute FFT``.

Before you click ``Compute FFT``, the docked Fourier panel lets you tune the
WiMDA-first FFT workflow directly:

* choose an ``FFT Phase Mode`` first. The ``Info`` button beside it opens an
  in-app explanation of the formulas and interpretation of each mode
* keep ``Subtract average signal`` enabled to remove the WiMDA-style pre-FFT
  baseline term before filtering and transformation; this is on by default
* grouped Fourier inputs are also corrected for the muon decay envelope before
  the later mean-subtraction and apodisation steps, matching WiMDA's grouped
  ``Freq`` preprocessing more closely
* choose ``None``, ``Lorentzian``, or ``Gaussian`` apodisation, then tune the
  filter start and tau values to match WiMDA-style FFT shaping
* include or exclude detector groups in the ``Groups`` table before computing
  the transform
* in ``Phase`` mode, either type a single manual phase or enable
  ``Use per-group phase table`` and edit the per-group values
* choose ``Auto method`` (``Peak`` or ``Average``), then press
  ``Fill Phase Estimates`` to estimate one phase per detector group for the
  current dataset
* for averaged grouped spectra, optionally estimate WiMDA-style error bars from
  the spread across the selected group spectra

Fitting The Displayed Spectrum
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once a Fourier spectrum has been computed, the **Frequency** workspace can be
fit through the same **Fit** dock used for time-domain analysis.  The fit target
is the displayed real-valued spectrum.  V1 provides Gaussian and Lorentzian peak
components with constant or linear backgrounds; successful global frequency
fits send ``nu0`` and ``fwhm`` values, plus derived ``B0`` and ``Bwid`` field
equivalents, to the **Parameters** dock for trend analysis.  See
:doc:`frequency_domain_fitting` for the full workflow.

Phase Estimation Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~

Phase estimation in the GUI is explicit rather than always-on:

* estimates are only generated when you press ``Fill Phase Estimates``
* the selected ``Auto method`` determines how the single-group FFT phase is
  measured within the current phase-estimation window
* when a field value is available, that window is narrowed around the expected
  precession frequency; otherwise the positive-frequency FFT range is used
* the estimated values are written into the per-group phase table for the
  current dataset only

The phase colour cues in the table are also meaningful:

* blue: the phase value is currently active and will be used
* grey: the phase value is present but not currently active
* green: the phase value was auto-estimated for the current dataset via
  ``Fill Phase Estimates``

Asymmetry now makes two explicit design decisions for this workflow:

* Fourier transforms always use grouped detector-count signals rather than the
  FB asymmetry trace as the input source.
* The frequency-domain plot always shows the average of the currently included
  groups rather than separate per-group spectra.

The FFT therefore always uses the current grouped/bunched data definition. The
grouped detector-count signal is rebuilt with the current bunching factor
before the transform, and that grouped count signal is lifetime-corrected
before the later subtract-average and filter stages. The FFT bandwidth and
frequency spacing therefore track the effective binned timestep in the same way
they do in WiMDA.

The Fourier panel is scrollable so these controls remain usable when the dock
is narrow.

Frequency-Domain Plot Behaviour
-------------------------------

FFT output is shown on the dedicated ``Frequency Domain`` tab in the central
plot workspace. Each loaded run keeps its own frequency-domain view state in the
session, and a run with no computed FFT shows the centred compute prompt on that
tab — *"No FFT computed for this run. Configure the FFT parameters and click
Compute FFT."* — until you transform it (see `GUI Fourier Workflow`_). Entering
the Frequency-domain view with nothing computed **stays** on that tab and shows
the prompt; it no longer silently falls back to the time view, so the cue is
always where you are looking. The current x-range is preserved when switching
onto a run whose Fourier spectrum has not yet been computed. (The MaxEnt
spectrum tab behaves the same way, with a *"No MaxEnt spectrum computed…"*
prompt.)

The frequency viewer keeps canonical FFT x-data in absolute MHz or Gauss on the
axis itself. The main toolbar also provides an ``FFT X relative to field``
toggle, which recentres the x-limit controls around the applied
field/frequency of the selected run while leaving the plotted tick labels in
absolute units.
