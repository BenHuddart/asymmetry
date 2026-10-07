# Rotating-frame projection for vector polarisation

Status: started 2026-10-06 on `feat/rotating-frame-projection`; Phases 1–3 and
the Phase 4 docs done 2026-10-07. Mockup (Design
canvas, 7 artboards): https://claude.ai/artifact/97WPef8vq5t4vHPSn26LjC.

## Problem

In vector polarisation mode (EMU P_x, P_y, P_z; B₀ ∥ z), an RF or transverse
signal makes the transverse pair turn about z at the drive frequency. The
physics, such as nutation, spin locking and echoes, lives in the frame turning
with it. With both transverse projections measured, that frame is reached
exactly by a time-dependent linear combination:

    z′(t) = z_phys(t) · e^{+i(2πν t + φ_RF)},
    z_phys = (P_x − b_x) + i·(−s)·(P_y − b_y)/g

P′_x = Re z′, P′_y = Im z′, and P_z is unchanged by a rotation about z.

- s is the measured sense of the labels: −1 when (P_x, P_y, P_z) are
  right-handed about B₀, so that P_y is the physical y.
- g = a_y/a_x is the transverse gain.
- b_x and b_y are the baselines that remain after alpha.

The muon precesses negatively about +z. So with φ_RF the phase of the drive at
t0, a B₁ along the chosen lab axis sits on that axis of the rotating frame, and
a spin starting along +z nutates into +y′ (B₁ ∥ x′) or −x′ (B₁ ∥ y′).

There is no filter, every bin is transformed on its own, and rebinning after the
rotation is valid. That is unlike the filtered single-signal demodulation of
`core/transform/rrf.py`, which stays as it is.

This is the simplest version of the RF work studied in
`docs/plans/rf-rotating-frame.md`. That branch (`feat/rf-rotating-frame`: Bloch
engine, RF settings, RF vector fit window) is parked for a later revisit.

## Decisions (Ben, 2026-10-06)

- **D1 — A projection, not a fit tool.** The frame is a special projection of
  vector-polarisation groupings: a **Lab | Rotating** switch on the projection
  chip bar relabels the chips to P′_x, P′_y, P_z. It is offered whenever the
  grouping declares P_x and P_y, needs no RF settings, and is not gated behind
  Advanced.
- **D2 — One slot, two mechanisms.** Two bars use the same slot above the plot
  title; never both at once.
  - The exact rotation's frame bar appears with the Rotating switch.
  - The filtered Rotating-frame bar stays for single-pair data behind
    Options → Advanced, and is not offered in vector mode.
- **D3 — Parameters, grouped by what they belong to.**

  | Group | Parameter | Source |
  |---|---|---|
  | Setup | ν_RF (MHz ⇄ G) | The one required value: the user knows it from the generator |
  | Setup | B₁ axis (x \| y) | The user knows it from the coil. Its sign is meaningless for a linear field, and x vs y is a 90° phase |
  | Setup | φ_RF (°) | Phase of the drive at t0; a setup constant at fixed ν and trigger |
  | Setup | sense s (±1) | |
  | Run | b_x, b_y (%) | |
  | Run | gain a_y/a_x | |

- **D4 — Shared and per-run fields.** Setup fields are shared by the displayed
  runs and written to each of them; a field the runs disagree on reads "mixed".
  Run fields edit the run selected in the Data Browser.
- **D5 — Saved on the run.** Each run stores its frame, with each field's
  provenance (default, estimated, typed).
- **D6 — Nothing runs by itself.** Auto-detect… opens a review with each
  estimate beside the current value.
  - Rows still on defaults or earlier estimates are pre-ticked.
  - Typed values are never pre-ticked.
  - Estimates with contrast < 3 cannot be applied.
  - Nothing changes until Apply.
  - It uses the displayed runs, the current period and the visible window.
  - ν, s and φ_RF are estimated from the runs together; baselines and gain run
    by run.
  - *Phase 2 review:* ν_RF is not estimated at all. Under RF the transverse
    spectrum is a sideband pair at ν ± ν₁, which a check band about ν cannot
    hold when B₁ is large, so a ν check only echoed the typed value. Sense and
    φ_RF form one review row, because φ_RF is estimated for that sense. Every
    estimator sum is inverse-variance weighted, so the full window works
    without zooming in.
- **D7 — Rotated projections are fit targets.** Single and Batch fit P′_x, P′_y
  or P_z like any projection. The fit records its frame and goes stale when the
  frame changes. No new fit functions yet.
- **D8 — Rotate first, then bunch.** Display bunching applies after the rotation.
- **D9 — Baselines per period** (Ben, 2026-10-07, after measuring).
  - Each period of a run keeps its own transverse baselines b_x, b_y, with
    their provenance. Gain, sense and φ_RF stay shared.
  - A period combination's curve removes the same combination of the period
    baselines: green − red uses b_G − b_R, green + red uses b_G + b_R. The bar
    shows these derived values read-only.
  - Auto-detect reads every DAE period on its own, never the displayed
    combination. In green − red the RF-off period has no transverse signal, so
    the nutation appears turned over, into −y′, rather than φ_RF hiding it.
  - Evidence from the first RF dataset: with a bias-free estimator, the red
    and green baselines differ by up to about 0.1 %, beyond errors.
    - Driving the RF raises P_x's baseline by about 0.08 % in RF-on/off runs.
    - The two periods agree when both are driven, as in the echo runs.
    - P_y's shift across the field scan follows the detuning and changes sign
      through resonance.
    - The mechanism is open (RF pickup, or a P_z cross-talk of about 1 % in
      P_x).
  - Baselines are fitted beside the signal as b + c(t)·cos 2πνt +
    d(t)·sin 2πνt, with c and d cubic splines at 0.8 turns per knot.
    - A plain inverse-variance mean is pulled by the strong early turns, by up
      to about 0.08 %.
    - A whole-turn average leaks a nutation's ν ± ν₁ sidebands.

## Estimation (D6)

- **Baselines:** each run's b_x and b_y are the inverse-variance means over the
  window, since whole turns average to the baseline.
- **Gain:** g is the ratio of the RMS transverse swings. The lab-frame
  transverse polarisation is circular, so x and y swing equally.
- **ν and s:** the complex periodogram of (P_x − b_x) + i(P_y − b_y)/g, summed
  in power over the runs, peaks at s·ν. A positive peak means s = +1. Contrast
  is the peak over its mirror at −s·ν.
- **φ_RF:** the argument α of Σ z_phys e^{+i2πνt} is the direction of the
  transverse polarisation in the frame with φ = 0. A nutation from +z lies
  perpendicular to B₁, so φ_RF = target − α, with target 90° for B₁ ∥ x′ and
  180° for B₁ ∥ y′.

## Phases

1. **Core.** *Done 2026-10-06.* Checked on the research data: on the π/2 and π duration series and a continuous-RF run, P′_y shows the nutation into +y′, P′_x stays near zero, and per-run contrast is 4–10. Done when:
   - `core/transform/rotating_frame.py` holds the typed `RotatingFrame` record
     with per-field provenance, the exact rotation of a run's P_x/P_y datasets
     into P′_x/P′_y, and the estimators returning proposals with contrast;
   - the frame persists on the run in the project schema;
   - synthetic tests cover the conventions for both senses and both axes, the
     error propagation, and estimator recovery;
   - the estimators have been checked on the research data, outside the repo.
2. **GUI: switch, bar and plot.** *Done 2026-10-06.* On the research data the rotated overlay matches Phase 1 (P′_y nutates positive, P′_x stays near zero, P_z falls; per-run contrast 9–12 over 0–5 µs); fitting a P′ subplot is blocked until Phase 3.
   - The Lab | Rotating switch in the chip bar, including its short and folded
     forms.
   - The Setup | Run frame bar: mixed fields, provenance styling, the per-run
     popover when narrow.
   - The Auto-detect review popover.
   - Stacked P′ subplots with the frame badge, rotated before bunching.
3. **Fitting.** *Done 2026-10-07.* P′ projections as Single and Batch fit
   targets, with the frame in the fit's provenance and staleness when it
   changes.
   - A fit on P′_x or P′_y fits exactly the rotated curve drawn, cropped to the
     fit range, and keys under that label beside the lab fits.
   - The fit records a `FrameSnapshot`: the run's frame and its period weights.
     It is stale once the values that make the rotation change: ν, B₁ axis,
     φ_RF, sense, gain, the baselines the weights select, or the period mode.
     Provenance alone never makes a fit stale.
   - A stale single fit says so on the Saved fits row, and its menu entry
     carries ⚠. A stale series shows ⚠ on its pill, with the reason on the
     tooltip, and in the Batch tab's series menu.
   - A series records its `projection` (part of its identity) and each
     member's snapshot. Runs without a frame are listed disabled in the
     Batch tab's members.
   - Rotated series are not offered to joint fits yet.
   - On the research data, single damped-cosine fits of P′_y give
     ν₁ ≈ 0.33–0.38 MHz, and P_z gives ≈ 0.31 MHz. Neither is a good
     description (χ²ᵣ 1.5–2.6).
4. **Docs and gate.** *Docs done 2026-10-07.*
   - A new reference page, `docs/reference/rotating_frame_projection.rst`
     (Specialised modes), linked from `vector_polarization.rst`,
     `rotating_frame.rst`, the find-a-feature table and the glossary;
     schema v26 in `project_files.rst`; Batch-draft and stale-fit notes in
     `gui_usage.rst`.
   - Screenshot scenarios `rotating_frame_projection` (plot after
     Auto-detect) and `rotating_frame_review`, on the synthetic
     `make_rf_nutation_vector` run.
   - CHANGELOG.
   - validate, gui-smoke, docs.

Research data used for development stay outside the repo; tests are synthetic.
