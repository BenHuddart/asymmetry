Rotating-frame projection
=========================

.. image:: /_generated/screenshots/rotating_frame_projection.png
   :alt: The plot in the rotating frame: the Lab | Rotating switch on Rotating,
         the frame bar filled by Auto-detect, and stacked P′_x, P′_y and P_z
         subplots of a synthetic RF nutation run
   :width: 100%

*A synthetic two-period RF run on an idealised six-detector vector polarimeter,*
*shown in the rotating frame after* **Auto-detect…** *(see* `Worked example`_\ *).*
*The spin leaves* :math:`+z` *and nutates about* :math:`B_1 \parallel x'`\ *:*
:math:`P'_y` *carries the nutation,* :math:`P'_x` *stays at zero, and* :math:`P_z`
*swings through zero and back. The frame bar above the title holds the*
*setup and run fields; the badge above the top subplot names the frame.*

In a radio-frequency (RF) μSR experiment a coil adds a field :math:`B_1`
oscillating at the generator frequency :math:`\nu_\mathrm{RF}` to a static
field :math:`B_0 \parallel z`. Near resonance, :math:`\nu_\mathrm{RF} \approx
(\gamma_\mu/2\pi)B_0`, the muon spins turn away from :math:`z` — they *nutate*
— and the physics of the experiment (nutation, spin locking, echoes) lives in
a frame that turns about :math:`z` with the drive. In the laboratory the same
motion is a fast precession at :math:`\nu_\mathrm{RF}` whose envelope carries
the nutation, so the lab-frame :math:`P_x` and :math:`P_y` are hard to read and
harder to fit. The rotating frame is the standard tool of magnetic resonance
for exactly this reason [1]_ [2]_ [3]_.

When a grouping measures both transverse projections — the :math:`P_x` and
:math:`P_y` of EMU's vector-polarisation mode (:doc:`vector_polarization`) —
Asymmetry reaches that frame *exactly*. Each time bin's transverse pair is
rotated about :math:`z` by the angle the frame has turned through, giving
:math:`P'_x` and :math:`P'_y`, while :math:`P_z`, which a rotation about
:math:`z` leaves alone, is shown unchanged. No filter is involved: every bin is
transformed on its own, its error bar is propagated through the rotation, and
the bins stay independent. Display bunching is applied *after* the rotation,
which is valid for that reason, and the rotated curves can be fitted like any
other projection (`Fitting P′_x and P′_y`_).

This is a different tool from the :doc:`rotating-reference-frame display
<rotating_frame>` of a single forward–backward asymmetry. With only one
projection measured, the frame can be reached only by demodulating and
low-pass filtering, which correlates neighbouring points and is meant for
looking rather than fitting. That filtered display stays available for
single-pair data under **Options → Advanced → Rotating reference frame**; it is
not offered when the exact projection is.

Switching to the rotating frame
-------------------------------

Whenever the displayed projections include :math:`P_x` and :math:`P_y`, a
**Lab | Rotating** switch sits to the left of the projection chips. Clicking
**Rotating** relabels the transverse chips ``P′_x`` and ``P′_y`` and shows the
frame bar in the slot above the plot title; **Lab** returns to the laboratory
projections. The chips keep their tint, selection and memory in either frame.
When the toolbar is narrow the switch shortens to **Lab | Rot**, and when the
chips fold into a single button the switch moves into its menu as a **Frame**
section above the projections, with the button reading, for example,
``Rot · 2 of 3 ▾``. The switch is saved with the project's plot state.

The first time, nothing is drawn: the plot reads "Enter ν_RF to show the
rotating frame." and the bar's only editable field is **ν_RF**, outlined and
marked ``required``, under the hint "Enter ν_RF (the generator frequency) and
the B₁ axis. Auto-detect… proposes the rest and changes nothing until you
apply." Typing :math:`\nu_\mathrm{RF}` creates a frame on every displayed run,
with every other field at its default, and the rotated curves appear.
:math:`\nu_\mathrm{RF}` is the one value you must supply, because it is the
generator's frequency and is not recorded in the data files. Nothing else is
estimated until you ask.

The frame bar
-------------

The bar groups the frame's values by what they belong to.

**SETUP** fields describe the spectrometer and drive, and are shared by every
displayed run: an edit is written to each of them. Where the displayed runs
disagree, the field is empty and reads ``mixed``.

``ν_RF``
    The generator frequency, typed in MHz or, with the **MHz | G** toggle, as
    the field in gauss that resonates at it
    (:math:`\nu = (\gamma_\mu/2\pi)B`). The frame turns at this frequency.

``B₁ ∥`` **x** | **y**
    The lab axis of the linear RF field, which you know from the coil. Its
    sign carries no information for a linear field, and the choice between
    :math:`x` and :math:`y` is a 90° phase; what it fixes is which rotated
    component is parallel to :math:`B_1`. The y labels say so: with
    :math:`B_1 \parallel x`, :math:`P'_x` is labelled :math:`\parallel B_1` and
    :math:`P'_y` :math:`\perp B_1`.

``φ_RF`` (°)
    The phase of the RF drive at :math:`t_0`. At a fixed frequency and trigger
    it is a constant of the setup, so one value serves a whole series of runs.

Sense (``↻ −1`` | ``↺ +1``)
    The rotation sense of :math:`P_x + iP_y` ("Rotation sense of P_x + iP_y"),
    toggled by clicking. It is :math:`-1` when :math:`(P_x, P_y, P_z)` are
    right-handed about :math:`B_0`, the usual case, and :math:`+1` when the
    labels are left-handed about it — for example with :math:`B_0` reversed.

**RUN** fields (the header reads ``RUN`` and the run number) belong to the run
selected in the Data Browser:

``b_x``, ``b_y`` (%)
    The transverse baselines left after α: the constant parts of
    :math:`P_x` and :math:`P_y`, removed before the rotation. Each period of a
    two-period run keeps its own pair (`Periods`_).

``a_y/a_x``
    The transverse gain :math:`g`, the amplitude of :math:`P_y` over that of
    :math:`P_x`. It rescales :math:`P_y` so that the two transverse projections
    measure the polarisation on one scale.

Each field shows where its value came from. A default is grey italic, an
estimate written by **Auto-detect…** looks plain, and a value you typed
carries ✎ ("Typed by you — Auto-detect never proposes over it"). The frame is
saved on each run in the project, values and provenance together.

The badge above the top subplot names the frame the curves were drawn in, for
example ``Rotating frame · ν = 1.49 MHz · B₁ ∥ x′ · φ_RF = 40.1° · s = −1``,
so an exported figure describes itself; a value the displayed runs disagree on
reads, for example, ``φ_RF mixed``. When the plot panel is narrow the bar wraps
onto two rows, and below that the run fields fold into a ``Run`` *n* ``▾``
button that opens them in a popover, with **Auto-detect…** shortened to
**Detect…**.

Auto-detect
-----------

**Auto-detect…** estimates everything except :math:`\nu_\mathrm{RF}` and the
:math:`B_1` axis: "Estimate the sense, φ_RF, baselines and gain from the
displayed runs over the visible window; nothing changes until you apply." It
is enabled once every displayed run has a frame and the runs share
:math:`\nu_\mathrm{RF}` and the :math:`B_1` axis. It works from the time window
currently visible on the plot, which must span at least two turns of
:math:`\nu_\mathrm{RF}`; every sum it forms is weighted by inverse variance, so
the noisy late bins of a long window do not swamp the signal and there is no
need to zoom in first. It reads every period of each run on its own, never the
combination the plot is showing.

- **Baselines**, for each period of each run, are fitted beside the signal: a
  least-squares fit of :math:`b + c(t)\cos 2\pi\nu t + d(t)\sin 2\pi\nu t`, in
  which :math:`c(t)` and :math:`d(t)` are smooth cubic splines that absorb any
  slowly varying amplitude and phase. A plain mean over the window is pulled
  by the strong early turns, and an average over whole turns leaks the
  :math:`\nu \pm \nu_1` sidebands of a nutation; the fitted baseline suffers
  from neither.
- **Gain**, per run, is the ratio of the RMS transverse swings. In the lab the
  transverse polarisation turns in a circle, so the two projections should
  swing equally.
- **Sense**, from the runs together, is the side, :math:`+\nu` or
  :math:`-\nu`, of the complex spectrum of :math:`P_x + iP_y/g` that holds more
  power. The ratio of the two (as amplitudes) is the estimate's *contrast*.
- **φ_RF**, from the runs together and for that sense, turns the transverse
  polarisation onto the direction a nutation from :math:`+z` takes,
  :math:`\hat z \times \hat B_1` (`Conventions`_). The axis is the principal
  direction of the transverse polarisation in the frame; its sign comes from
  early times, when a spin leaving :math:`+z` first moves that way.

:math:`\nu_\mathrm{RF}` is never estimated. Under RF the transverse spectrum
is not a single line at :math:`\nu_\mathrm{RF}` but a sideband pair at
:math:`\nu \pm \nu_1`, which can lie far from :math:`\nu` when :math:`B_1` is
large, so a peak search would find a sideband rather than the drive.

.. image:: /_generated/screenshots/rotating_frame_review.png
   :alt: The Auto-detect rotating frame review with the sense and phase row,
         a gain row and a baseline row for each period, all pre-ticked
   :width: 75%

*The* **Auto-detect rotating frame** *review for the run above: the setup row,*
*then the run's gain and the baselines of its red and green periods, each*
*estimate beside the current value and its contrast.*

The estimate opens in the **Auto-detect rotating frame** review rather than
being applied. Its header names the :math:`\nu_\mathrm{RF}` and :math:`B_1`
axis it worked at, and each row sets the current value beside the estimate
(``0.0° → 40.1°``):

- **Sense, φ_RF** — one row for the runs together, since :math:`\varphi_\mathrm{RF}`
  is estimated for that sense, written to every run.
- ``Run`` *n* ``· gain`` — the run's gain.
- ``Run`` *n* — the run's baselines; a two-period run has one row per period,
  ``Run`` *n* ``· red`` and ``Run`` *n* ``· green``.

Rows still on defaults or earlier estimates start ticked; a row holding a value
you typed never does, and is marked ``typed``. An estimate whose contrast is
below 3 cannot be applied ("contrast … < 3 — cannot apply"): the data carry too
little transverse signal to say which way it turns. Nothing changes until
**Apply** *n* **changes**, which writes the ticked values as estimates; the
title strip then reads, for example, "✓ Applied 4 estimates · contrast 65".
**Cancel** leaves every frame as it was. If the estimate cannot be made — a
window shorter than two turns, for example — the status bar reads
"Auto-detect failed — see the log" and the log names the reason.

Periods
-------

On a two-period run (red is period 1, green period 2) each period keeps its
own baselines, with their provenance; the gain, sense and
:math:`\varphi_\mathrm{RF}` are shared. The distinction matters on RF data,
where the red and green transverse baselines can differ by about 0.1 % — for
example when the RF is on in one period and off in the other. A single pair of
baselines would leave the difference behind as a spurious oscillation at
:math:`\nu_\mathrm{RF}` in the rotated curves.

While the Grouping window's **RG Mode** shows **Red** or **Green**, the bar
shows and edits that period's baselines. **G minus R** and **G plus R** remove
the same combination of the period baselines as they apply to the data,
:math:`b_G - b_R` and :math:`b_G + b_R`, and the bar shows these derived
values read-only ("Derived from the periods' own baselines for this period
combination; edit them on a single period (red or green).").

Read a difference with care. A period with the RF off carries no transverse
signal, so when the RF is on in red and off in green, **G minus R** holds
minus red's transverse polarisation: the nutation appears turned over, into
:math:`-y'` for :math:`B_1 \parallel x'`. That is correct arithmetic, not a
wrong :math:`\varphi_\mathrm{RF}` — which is why **Auto-detect…** estimates from
each period separately and never from the combination on display.

Conventions
-----------

The rotation removes the baselines, puts the labelled projections on physical
axes, and then undoes the precession at :math:`\nu_\mathrm{RF}`:

.. math::

   z_\mathrm{phys}(t) = \bigl(P_x - b_x\bigr) + i\,(-s)\,\frac{P_y - b_y}{g},
   \qquad
   z'(t) = z_\mathrm{phys}(t)\,e^{+i(2\pi\nu_\mathrm{RF} t + \varphi_\mathrm{RF})},

with :math:`P'_x = \mathrm{Re}\,z'` and :math:`P'_y = \mathrm{Im}\,z'`. Here
:math:`s` is the sense, :math:`g` the gain, and :math:`b_x`, :math:`b_y` the
baselines of the displayed period (or their combination).

The muon's gyromagnetic ratio is positive, so its spin precesses in the
negative sense about a field along :math:`+z`, and :math:`e^{+i\theta}` undoes
that precession. With :math:`\varphi_\mathrm{RF}` the drive's phase at
:math:`t_0`, the co-rotating half of the linear RF field is static in the frame
and lies along the chosen axis. A spin leaving :math:`+z` then nutates about
:math:`B_1` towards :math:`\hat z \times \hat B_1`:

- with :math:`B_1 \parallel x'`, into :math:`+y'` — :math:`P'_x` is the
  component along :math:`B_1` (the spin-locked one) and :math:`P'_y` the one
  across it;
- with :math:`B_1 \parallel y'`, into :math:`-x'`.

On resonance the nutation runs at :math:`\nu_1 = (\gamma_\mu/2\pi)B_1`, with
:math:`B_1` the amplitude of the co-rotating component.

The frame turns at the generator's :math:`\nu_\mathrm{RF}`, so that
:math:`B_1` is static in it, not at the muon's Larmor frequency. During free
evolution — with the RF off, or between pulses — the spins turn at
:math:`\gamma_\mu B_\mathrm{local}`, so a local field away from resonance
appears in the frame as a slow drift at the offset
:math:`\gamma_\mu B_\mathrm{local}/2\pi - \nu_\mathrm{RF}`. That drift is
information about the field at the muon, not an error in the frame.

.. dropdown:: Error propagation

   With :math:`\theta = 2\pi\nu_\mathrm{RF}t + \varphi_\mathrm{RF}`,
   :math:`x = P_x - b_x` and :math:`y = (P_y - b_y)/g`,

   .. math::

      P'_x = x\cos\theta + s\,y\sin\theta, \qquad
      P'_y = x\sin\theta - s\,y\cos\theta,

   and each rotated bin carries
   :math:`\sigma'^2_x = \sigma_x^2\cos^2\theta + (\sigma_y/g)^2\sin^2\theta`
   and
   :math:`\sigma'^2_y = \sigma_x^2\sin^2\theta + (\sigma_y/g)^2\cos^2\theta`.
   Bins remain independent of one another. Within one bin, :math:`P'_x` and
   :math:`P'_y` share noise, with covariance
   :math:`\bigl((\sigma_y/g)^2 - \sigma_x^2\bigr)\sin\theta\cos\theta`, which
   vanishes when the two transverse projections have equal errors.

Fitting P′_x and P′_y
---------------------

A rotated subplot is a fit target like any projection: click it, and the
Single tab fits exactly the rotated curve drawn — in the run's current frame
and period combination — cropped to the fit range. The fit is saved among the
run's saved fits under its own projection, beside the run's lab-frame fits
(:ref:`saved-single-fits`), and its curve is drawn only on the rotated
subplot.

Each such fit records the frame its data were rotated in. If you later change
a value that alters the rotated curve — :math:`\nu_\mathrm{RF}`, the
:math:`B_1` axis, :math:`\varphi_\mathrm{RF}`, the sense, the gain, the
baselines the displayed period combination uses, or the combination itself —
the fit no longer describes the data on show. Its entry in the **Saved fits**
menu then carries ⚠, and while the form still describes it the line under its
name reads "Fitted in a different rotating frame — re-run the fit to update
it." Fitting again refreshes it. Provenance alone never makes a fit stale:
re-typing a value, or an estimate becoming a typed value, changes nothing that
was fitted.

The Batch tab fits a rotated projection across runs in the same way. A draft
fits whatever the plot's fit target shows, and follows it when you click
another subplot. Each member is rotated in its own run's frame, and the series
records the projection and each member's frame. Runs with no frame are listed
in the members, unticked and disabled, with the tooltip "No rotating frame —
set ν_RF in the rotating view". A series whose members' frames have since
changed shows ⚠ on its pill in the Parameters panel, with "Fitted in a
different rotating frame — re-run the fit to update it." on the tooltip, and
⚠ in the Batch tab's series menu. Rotated series are not yet offered to
:doc:`joint fits <joint_fit>`, where they are listed as ``Rotating-frame
series``.

No dedicated nutation model exists yet; the usual oscillation components
apply. Note what a fit of a damped oscillation at one frequency to
:math:`P'_y` measures: an *effective* nutation frequency, set by
:math:`\nu_1` together with anything that spreads or shifts it — the coil's
ring-up at the start of the pulse, the spread of :math:`B_1` over the sample,
and any detuning from resonance, which raises the nutation frequency to
:math:`\sqrt{\nu_1^2 + \Delta\nu^2}`. Treat it as :math:`\gamma_\mu B_1/2\pi`
only when those effects are known to be small.

Worked example
--------------

The figures on this page use a synthetic run with invented numbers, made for
the documentation: an idealised six-detector polarimeter with one detector
along each of :math:`\pm x`, :math:`\pm y` and :math:`\pm z`, in
:math:`B_0 = 110` G along :math:`z`, so that
:math:`\nu_0 = (\gamma_\mu/2\pi)B_0 \approx 1.49` MHz.

- **Red** (period 1) has the RF on at 1.49 MHz with :math:`B_1 \parallel x`
  and :math:`\nu_1 = 0.2` MHz (a co-rotating :math:`B_1` of about 15 G),
  damped at 0.2 μs⁻¹ as an inhomogeneous :math:`B_1` would damp it.
- **Green** (period 2) has the RF off: the spin stays on :math:`z` and relaxes
  slowly.
- In the lab, the drive's phase at :math:`t_0` is 40°, the labels have sense
  :math:`-1`, :math:`P_y`'s amplitude is 0.92 of :math:`P_x`'s, and the
  periods' baselines differ: :math:`(b_x, b_y) = (0.30, -0.20)` % in red and
  :math:`(0.22, -0.20)` % in green, so the RF raises :math:`b_x` by 0.08 %.

1. Select the run with the ``P_x``, ``P_y`` and ``P_z`` chips on, and click
   **Rotating**. The plot asks for :math:`\nu_\mathrm{RF}`.
2. Type ``1.49`` into **ν_RF** and leave **B₁ ∥** on **x**. The rotated curves
   appear in the default frame, with :math:`\varphi_\mathrm{RF} = 0` — still
   turning, because the frame's phase and the baselines are wrong.
3. Click **Auto-detect…**. The review proposes sense :math:`-1` and
   :math:`\varphi_\mathrm{RF} = 40.1°`, gain 0.919, red baselines
   (0.28, −0.21) % and green (0.23, −0.22) %, all with contrast 65 and all
   pre-ticked.
4. Click **Apply 4 changes**. :math:`P'_y` now shows the nutation into
   :math:`+y'`, :math:`P'_x` stays at zero within its errors, and :math:`P_z`
   swings through zero and back at the same frequency, both damped, as in
   the figure at the top of the page (bunched ×4 after the rotation).

Every recovered value lies within a few hundredths of a per cent, a tenth of a
degree, or a part in a thousand of what was put in. Changing
**RG Mode** to **G minus R** would then show the nutation turned over into
:math:`-y'`, as described under `Periods`_, without any change to the frame.

Assumptions and limitations
---------------------------

- **Both transverse projections are needed.** The switch appears only for
  groupings that declare :math:`P_x` and :math:`P_y`. A single
  forward–backward pair cannot be rotated exactly; use the filtered
  :doc:`rotating_frame` display for viewing such data.
- **The frame assumes** :math:`B_0 \parallel z` **and orthogonal transverse
  projections.** The rotation is about the :math:`P_z` axis. Detector pairs
  that only approximate orthogonal axes (see the limitations in
  :doc:`vector_polarization`) leave a corresponding error in the rotated
  components; the gain corrects a difference of scale, not of direction.
- **Rotating-wave approximation.** Only the co-rotating half of the linear RF
  field is static in the frame. The counter-rotating half turns at
  :math:`2\nu_\mathrm{RF}` in it, and its small effect on the spin dynamics
  (the Bloch–Siegert shift [4]_) is not part of the picture of a static
  :math:`B_1`; it is negligible when :math:`B_1 \ll B_0`.
- **Auto-detect assumes a nutation from** :math:`+z`. Its
  :math:`\varphi_\mathrm{RF}` puts the transverse polarisation on
  :math:`\hat z \times \hat B_1`, which is right for a spin starting along
  :math:`z` and driven about :math:`B_1`. For other preparations, type
  :math:`\varphi_\mathrm{RF}` from a reference run measured that way.
- **One or two periods.** A frame describes a run of one or two periods;
  runs recorded with more periods are not supported.
- **The rotated components share noise within a bin** (see the error
  propagation above), so fit them one at a time; this is one reason rotated
  series are not yet offered to joint fits.
- **No dedicated models yet.** Nutation, spin-locking and echo models are not
  in the fit-function library; the effective-frequency caveat under
  `Fitting P′_x and P′_y`_ applies to the general-purpose ones.

References
----------

.. [1] S. J. Blundell, R. De Renzi, T. Lancaster, and F. L. Pratt,
   *Muon Spectroscopy: An Introduction* (Oxford University Press, Oxford,
   2022) — polarisation projections and the rotating reference frame in μSR.

.. [2] M. H. Levitt, *Spin Dynamics: Basics of Nuclear Magnetic Resonance*,
   2nd ed. (Wiley, Chichester, 2008) — the rotating frame, resonance offset
   and nutation.

.. [3] I. I. Rabi, N. F. Ramsey, and J. Schwinger, Rev. Mod. Phys. **26**, 167
   (1954).

.. [4] F. Bloch and A. Siegert, Phys. Rev. **57**, 522 (1940).

Related topics
--------------

* :doc:`vector_polarization`
* :doc:`rotating_frame`
* :doc:`data_reduction/period_mapping`
* :doc:`fitting`
