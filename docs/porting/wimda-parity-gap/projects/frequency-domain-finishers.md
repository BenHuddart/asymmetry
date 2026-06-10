# Project brief: frequency-domain-finishers

Umbrella: `wimda-parity-gap` · Wave A · Size M–L (2 phases)

## Motivation

The FFT core reached WiMDA parity (and beyond, with entropy phase
optimisation), but a band of WiMDA's spectrum-conditioning and display
options remains, several of which matter for routine TF work at ISIS.
This project also carries the **Burg all-poles pole scan**, re-admitted by
decision 2026-06-10 as a diagnostic view.

## WiMDA reference

`FFTPar.pas` form + `Plot.pas` frequency paths: field-axis spectra
(`FFTb/AvFFTb`, `globals.pas:54`); frequency-range exclusion ×10 with
diamag-linked slot 1 and "use PSI RF harmonics" preset (50.63 MHz × 1–5 + DC;
`FFTPar.pas:327–377`, `Plot.pas:1993–2007`); fit-and-subtract diamagnetic
signal (5-param damped-cosine time-domain fit, `Plot.pas:1832–1890`);
frequency-response compensation (× `exp((πf·τ_pulse)²)` for the ISIS pulse
rolloff, `Plot.pas:1931–1944`); 2σ-clipped spectrum baseline offset
(`Plot.pas:1959–1984`); S/N-at-peak and average-error readouts
(`Plot.pas:1352–1385`); Burg MEM pole scan with FPE order selection
(`MaxEnt.pas`, `Plot.pas:1913–1927`); muonium/radical correlation spectrum
(`rmatch` Breit–Rabi pair matching → hyperfine axis, `Plot.pas:2149–2230`)
with radical cursor overlays. Plus the fourier study's own deferred items:
Tesla unit, real+imag combined view, N0-normalised single-histogram input.

## Scope & phasing

**Phase 1 — units and conditioning (core + wiring).** Field-axis display
(Gauss, and Tesla) for FFT spectra — conversion via existing γ_μ constants,
shared with the plot panel's frequency-units machinery; wire the existing
`exclude_frequency_ranges` core function into the Fourier panel with a
diamag-linked slot and a PSI-harmonics preset; spectrum baseline offset;
frequency-response compensation (pulse width from instrument metadata where
available); S/N readouts; real+imag view.

**Phase 2 — diagnostics.** Burg pole scan as a "Resolution (diagnostic)"
spectrum mode: Burg recursion + AR spectrum + FPE order scan (~100 lines
numpy in a new `core/fourier/burg.py`), sharing the panel's preprocessing
chain exactly as WiMDA does; docs state plainly what it is good for
(qualitative super-resolution, line-count hints) and its pathologies
(spurious splitting, no uncertainties) — never the quantitative result.
Fit-and-subtract diamagnetic signal. **Optional**: the radical correlation
spectrum if radical-chemistry users are anticipated — otherwise record as
the natural follow-on (it reuses the Breit–Rabi machinery already in
`core/fitting/muonium.py`).

**Out**: FB t=0 extrapolation (moot under Asymmetry's grouped-counts Fourier
source); N0 single-histogram normalisation only if it falls out naturally
(reassess in study — it interacts with `count-domain-fit-modes` Phase 1).

## Current Asymmetry state

`core/fourier/fft.py` (modes incl. `phase_opt_real`, exclusion function
unwired), `window.py`, `grouped.py`; `fourier_panel.py` with phase tables and
stale-MaxEnt awareness; MHz-only axis.

## GUI/UX sketch

Phase 1 items are additions to existing Fourier-panel groups (units combo
next to the existing frequency controls; an "Exclusions" collapsible section
with the preset button). The Burg mode joins the display-mode selector with a
clearly-labelled diagnostic badge and a first-use tooltip.

## Conflicts & dependencies

Primary surfaces: `core/fourier/` + `fourier_panel.py`. Wave A-disjoint.
Field/Tesla units logic should be written once (shared helper) since
`maxent-completion` needs the same axis — coordinate the helper's location
(suggest `core/fourier/units.py`) in both studies.

## Verification sketch

Field-axis: known-field TF run peaks at γ_μ·B. Pulse-rolloff compensation:
synthetic data convolved with the ISIS pulse shape recovers flat amplitude
after compensation. Burg: synthetic close doublet — document the window
length where FFT merges the lines but Burg resolves them, and the pole count
where spurious splitting begins (this characterisation *is* the docs
content). Exclusions: PSI run with RF harmonics before/after preset.
