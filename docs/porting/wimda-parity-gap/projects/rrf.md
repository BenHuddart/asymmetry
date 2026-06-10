# Project brief: rrf

Umbrella: `wimda-parity-gap` · Wave B · Size S–M · promotes the
`rrf-transform` candidate

## Motivation

Rotating-reference-frame display lets high-TF users see the slow envelope
(relaxation, beats) on top of a fast precession — standard practice for
vortex-lattice and Knight-shift work. WiMDA has display + fitting; Mantid
has `RRFMuon`; Asymmetry has nothing.

## WiMDA reference

Display: `PlotPar.pas:427–475` (frequency in MHz or Gauss, phase, smoothing
bin width), `Plot.pas:1652–1695,1712–1727` — demodulate FB asymmetry by
`2·cos(ωt+φ)` then box-smooth over the RRF bin; forces FB-asym mode.
Fitting: `RRFon` shifts component frequencies by `rotfreq` and phases by
`−rotphase` inside `MusrFun` so fits run in the rotating frame.

## Scope

- New Qt-free `core/transform/rrf.py`: `rrf_transform(time, asym, err,
  freq_mhz, phase_deg, smoothing)` returning the demodulated curve.
  **Physically-correct upgrade over the WiMDA port**: implement complex
  demodulation (multiply by `e^{−iωt}`, low-pass, take real/imag or
  magnitude) with a proper low-pass (windowed FIR or decimating mean with
  documented bandwidth) instead of WiMDA's `2·cos` + box smooth, which
  aliases the 2ω image imperfectly. Offer the WiMDA-equivalent mode for
  comparison; document both in the study.
- Plot integration: an RRF section in the time-domain plot controls
  (enable, ν in MHz⇄G, phase, bandwidth); works on the FB-asymmetry view;
  state persisted with plot state.
- **Fitting in the RRF** (decide in study): either (a) fit the demodulated
  curve with envelope models — simple, but error correlations from the
  low-pass need care; or (b) WiMDA's approach — fit the *raw* data with
  frequencies offset by the RRF (a parameter transform, statistically
  clean). Recommendation to evaluate (b) first: it's a small fit-layer
  feature (frequency offset) with exact statistics.

**Out**: per-detector RRF; RRF inside MaxEnt.

## Current Asymmetry state

Absent (`grep -ri rrf` empty outside docs). Candidate study folder:
`docs/porting/candidates/rrf-transform/`.

## GUI/UX sketch

Collapsible "Rotating frame" group in plot controls; ν entry with MHz⇄Gauss
toggle (γ_μ helper shared with the frequency-units work); the plot title/axis
annotates "frame: ν_RRF" so exported figures are self-describing.

## Conflicts & dependencies

Primary surfaces: new transform module + `plot_panel.py`. Wave B-disjoint
except `workflow-visualisation` (Wave C) also edits `plot_panel.py` —
sequenced before it. Mantid `RRFMuon` is the oracle (GPL — oracle only).

## Verification sketch

Synthetic single-frequency data: demodulation at exactly ν leaves the pure
envelope (matches the generating relaxation function to numerical
precision); off-resonance by δ leaves a δ beat; comparison of complex
demodulation vs WiMDA-mode on the same data documents the 2ω-image
suppression; high-TF corpus run (HiFi/HAL) envelope matches direct fit λ.
