# Project brief: maxent-completion

Umbrella: `wimda-parity-gap` · Wave A · Size L (3 phases)

## Motivation

The MULTIMAX engine and GUI shipped (PRs #16, #26), but the maxent study's
own Phase 1/2 items remain open, and the sweep confirmed they are the
highest-value MaxEnt gaps. The single most important is the **time-domain
reconstruction overlay** — the study called it "the strongest single
diagnostic of fit quality" and the engine already computes the projection
(`opus`); it just isn't exposed.

## WiMDA reference

`Wimdamax.pas`: time-domain reconstruction (`Timedom`, lines 1148–1163);
ISIS pulse-shape response (ignore/single/double pulse, half-width +
separation; parabolic proton-pulse FT × pion Lorentzian, lines 266–319);
ZF/LF two-group mode (phases pinned 0/180, F/B amplitudes tied via α, lines
404–408, 736–747) with `SpecBG.pas` zero-frequency lineshape subtraction;
deadtime fitting inside MaxEnt (`DEADFIT`, lines 867–937) with editable
phase/deadtime tables (`MaxEdit.pas`) and fitted-phase exchange with the
time-domain fit (`PhaseTableUnit.pas`); exclusion time window
(`readcontrol:112–116`); field-axis display + shift units
(`MaxControl.pas:157–314`); spectrum `.max` export + `.mlog` log.

## Scope & phasing

**Phase 1 — reconstruction overlay.** Expose per-group reconstructed time
spectra from `opus`, overlay on data in the plot workspace (per-group and
combined), include residuals. Extend `MaxEntResult`/diagnostics; persist the
toggle in the recipe.

**Phase 2 — pulsed-source correctness.** ISIS pulse-shape response in the
kernel (without it, amplitudes ≳5 MHz are distorted in pulsed data) with
pulse parameters defaulted from instrument metadata; exclusion time window
(σ-inflation, mirrors the engine's existing input model); field-axis/Tesla
display (shared units helper with `frequency-domain-finishers`).

**Phase 3 — calibration workflows.** Deadtime fitting inside MaxEnt;
editable per-group phase/amplitude/deadtime tables; **phase exchange** —
fitted phases from grouped time-domain fits seed MaxEnt and vice versa
(this absorbs the WiMDA slice of the `phase-auto-calibration` candidate);
ZF/LF mode + SpecBG zero-frequency background subtraction; spectrum/log
export.

**Out**: spectral deconvolution (`Sconv` — numerically hazardous per the
study; keep deferred); looseness/phase-acceleration knobs unless Phase 3
testing shows they're needed (suspected numerical-era cruft — decide in the
study with evidence).

## Current Asymmetry state

`core/maxent/engine.py` (full MULTIMAX with resumable state, χ²-plateau /
divergence guard), `maxent_panel.py`, recipe persistence. Phases/amps appear
in diagnostics but are not editable; no reconstruction output; MHz only.

## GUI/UX sketch

Reconstruction overlay as a plot-workspace toggle when a MaxEnt
representation is active (not a separate window). Tables as a tab in the
MaxEnt panel. "Use fitted phases" / "Send phases to fit" paired actions with
provenance labels (which fit, when). ZF/LF mode as a mode selector that
visibly constrains the group table.

## Physics-correctness notes

Pulse-shape response belongs in the forward model (OPUS kernel), not as a
post-hoc spectrum correction — follow WiMDA's placement. Mantid `MuonMaxent`
(same lineage) is the oracle for both pulse handling and deadtime fitting;
GPL — oracle only.

## Conflicts & dependencies

Primary surfaces: `core/maxent/engine.py`, `maxent_panel.py`, plot workspace
hook for the overlay. Wave A-disjoint, but: `spectral-moments` (Wave B)
touches `maxent_panel.py` — land before it starts; units helper shared with
`frequency-domain-finishers` — agree location early (suggest
`core/fourier/units.py`).

## Verification sketch

Overlay: reconstruction of synthetic single-line data matches the input
within noise; χ² shown equals engine χ². Pulse shape: synthetic pulsed data
with known frequency content — amplitude recovery flat vs frequency after
enabling. Deadtime fit: thin a run with known injected deadtime, recover it.
ZF mode: ZF run with known Kubo–Toyabe field distribution.
