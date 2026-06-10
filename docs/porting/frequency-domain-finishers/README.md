# Frequency-domain finishers — study

Umbrella: [`wimda-parity-gap`](../wimda-parity-gap/README.md) · Wave A,
project 5 · Size M–L (2 phases). Branch: `feat/frequency-domain-finishers`.

This is the mandatory study-first pass for the project briefed in
[`projects/frequency-domain-finishers.md`](../wimda-parity-gap/projects/frequency-domain-finishers.md).
It verifies that brief against the WiMDA Pascal source, reconciles the reused
APIs built by the just-merged `fourier-transform`, `units` and
`maxent-completion` work, and fixes the implementation plan.

## What this project finishes

The FFT and MaxEnt cores reached WiMDA parity (and beyond, with entropy phase
optimisation and the ISIS pulse-shape forward model). A band of WiMDA's
spectrum-**conditioning** and **display** options remained, several of which
matter for routine transverse-field (TF) work, plus the Burg all-poles pole
scan, re-admitted as a diagnostic by decision 2026-06-10.

- **Phase 1 — units & conditioning.** Field-axis display (verify-only — see
  below); frequency-range **exclusions** UI with a diamagnetic-linked slot and
  a PSI RF-harmonics preset; robust **spectrum baseline** offset; **pulse
  frequency-response compensation**; **S/N-at-peak and average-error**
  readouts; a **real+imag** combined display mode.
- **Phase 2 — diagnostics.** The **Burg all-poles** pole scan as a clearly
  badged diagnostic spectrum mode; fit-and-subtract **diamagnetic** signal
  removal. Optional / deferred: the muonium-radical **correlation spectrum**.

## Headline study findings

1. **Field-axis display is already built.** The frequency plot panel exposes a
   `Frequency (MHz) / Field (G) / Field (T)` selector
   ([`plot_panel.py:437`](../../../src/asymmetry/gui/panels/plot_panel.py))
   routed through `core.fourier.units`, with applied-field reference handling,
   shared by the FFT and MaxEnt frequency views (one `_frequency_plot_panel`
   instance). The brief's "MHz-only axis" line predates this. **Decision (Ben,
   2026-06-11): treat field-axis as done — verify-only**, and reallocate
   Phase-1 effort to the conditioning features. The γ_μ·B and FFT≡MaxEnt
   peak-agreement targets become verification tests, not new build.

2. **The reused APIs already exist and fit.** `exclude_frequency_ranges`
   ([`fft.py:422`](../../../src/asymmetry/core/fourier/fft.py)) takes
   `(centre_mhz, half_width_mhz)` pairs — exactly WiMDA's `RangeMid ± RangeWid`
   parameterisation; it is **unwired** and Phase 1 simply connects it to a panel
   section. `core/maxent/pulse.py` already models the ISIS pulse lineshape that
   the FFT compensation must invert — Phase 1 reuses its amplitude `R(ν)`
   rather than re-deriving WiMDA's simpler Gaussian factor. `units.py` is the
   single γ_μ source. The averaged-error machinery in
   `average_fourier_display_values` feeds the S/N readout. See
   [comparison.md](comparison.md) for the API-by-API reconciliation.

3. **The textbook covers the Burg method directly.** *Muon Spectroscopy*
   (OUP 2022) §15.5 presents the all-poles autoregressive (AR) method
   (its eqn 15.29, `P(ν)=a₀/|1+Σaₖzᵏ|²`), names Burg (1972) as the algorithm,
   describes FPE order selection, and states the exact advantages
   (super-resolution; no phase correction; works on short data sets) and
   pathologies (spurious splitting of strong features; spurious baseline peaks;
   small frequency offsets; errors not propagated) that the load-bearing Burg
   docs must convey. This is the canonical scientific anchor for Phase 2.

4. **Two genuine divergences from WiMDA are warranted** (full statement in
   [comparison.md](comparison.md)): the **pulse compensation** should invert the
   physically-grounded parabola×Lorentzian pulse response from `pulse.py` with a
   high-frequency guard, not WiMDA's unbounded `exp((πfτ)²)` Gaussian; and the
   **baseline** should default to **iterative** σ-clipping (the
   literature-standard robust continuum estimator — STATCONT, Sánchez-Monge
   et al. 2018) rather than WiMDA's single-pass 2σ clip, of which WiMDA's is the
   one-iteration special case.

## Documents

- [comparison.md](comparison.md) — WiMDA-source transcription of all eight
  features, the reused-API reconciliation, and every documented divergence with
  both behaviours stated.
- [implementation-options.md](implementation-options.md) — the settled options,
  the ordered two-phase plan, the file-by-file touch list, and recorded
  follow-ons. (Written after checkpoint-3 decisions.)
- [test-data.md](test-data.md) — the synthetic-first verification corpus.
- [verification-plan.md](verification-plan.md) — how each claim is verified.

## Scientific sources

Cited in full in [comparison.md](comparison.md). Primary: Blundell, De Renzi,
Lancaster & Pratt (eds.), *Muon Spectroscopy: An Introduction* (OUP, 2022),
§15.5 (frequency domain), §14.2 (pulsed sources), §4.4 (muonium), §12.4
(muoniated radicals). Burg (1972); Rainford & Daniell (1994); Skilling & Bryan
(1984); Riseman & Forgan (2003); Sánchez-Monge et al. (2018, STATCONT). GPL
references (Mantid, musrfit, WiMDA) are verification oracles only — never
vendored.
