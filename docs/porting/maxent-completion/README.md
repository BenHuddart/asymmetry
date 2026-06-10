# MaxEnt completion study

Status: study pass complete; implementation plan committed; awaiting go-ahead.

Slug: `maxent-completion` · Umbrella: [`wimda-parity-gap`](../wimda-parity-gap/)
(Wave A, project 6) · Size L (3 phases).

This study **extends** the implemented-engine study at
[`docs/porting/maxent/`](../maxent/) — it does not duplicate or re-litigate it.
That study chose the MULTIMAX algorithm with WiMDA as the behavioural contract
and shipped the joint multi-group engine (`core/maxent/engine.py`) and panel
(PRs #16, #26). This study covers the four highest-value items that engine
study explicitly deferred to its Phase 1/2/follow-on lists:

1. **Time-domain reconstruction overlay** — the strongest single diagnostic of
   fit quality; the engine's `opus` forward model already computes it, it is
   simply not exposed.
2. **ISIS pulse-shape response** in the forward model — without it, amplitudes
   above ~5 MHz are distorted on pulsed-source data.
3. **ZF/LF two-group mode** with SpecBG zero-frequency lineshape subtraction —
   the zero/longitudinal-field field-distribution workflow.
4. **Deadtime fitting + editable phase/amplitude/deadtime tables + phase
   exchange** with grouped time-domain fits — the MaxEnt calibration loop.

## The physics, briefly

A μSR frequency spectrum **is the distribution of local magnetic fields** p(B)
at the muon sites: a field B produces a precession line at ν = γ_μ B / 2π, with
γ_μ/2π = 135.5 MHz/T, so the frequency axis is convertible to field by
B = 2πν / γ_μ. MaxEnt is preferred over a plain FFT because it forward-models
the raw Poisson counts — no apodisation or zero-padding, structure placed only
where the data justify it, and instrumental effects (detector phase, time zero,
**pulse shape, deadtime**, background) folded directly into the model. These are
exactly the four gaps above. (Blundell, De Renzi, Lancaster & Pratt, *Muon
Spectroscopy*, OUP 2022, §15.5; the physics of p(B) in §5.1, §6.4, §9.5.)

The two source types matter for this work. On **pulsed sources** (ISIS) the
finite ~80 ns double proton pulse acts as a low-pass filter — useful frequencies
are limited to ~10 MHz and high-frequency amplitudes roll off (the pulse-shape
response, Phase 2, corrects the forward model for this); deadtime distortion is
worst at short times where the count rate peaks (the deadtime fit, Phase 3).
On **continuous sources** (PSI/TRIUMF) the frequency response is much wider and
deadtime is not a concern, so the pulse-shape response defaults to off
(continuous data) and single-pulse for ISIS. (*Muon Spectroscopy* §14.2–14.3.)

ZF/LF spectra (e.g. static Gaussian **Kubo–Toyabe** relaxation,
P_z(t) = 1/3 + (2/3)(1 − Δ²t²) exp(−Δ²t²/2)) correspond to a broad near-zero
field distribution rather than a sharp line, and the textbook notes the MaxEnt
method is tuned more for TF rotation than ZF precession — a caveat the user docs
will carry. (*Muon Spectroscopy* §5.1.)

## Scope decisions settled with Ben (2026-06-10)

| Question | Decision |
|---|---|
| Pulse-shape validation data / double-pulse | Synthetic-only validation; implement **both** single- and double-pulse, verified against the single-pulse limit of the double-pulse formula and flat-amplitude recovery. |
| Deadtime-fit promotion to grouping | **Suggest-only**: surface the fitted deadtime with a provenance label; the user explicitly applies it to the run grouping. Never auto-write. |
| ZF/LF generality | **Strict 2-group F/B parity**: exactly two selected groups, phases pinned 0/180, amplitudes α-tied, SpecBG display subtraction. |

Decisions inherited from the engine study (not reopened): single MULTIMAX
engine, WiMDA window mechanism, CODATA constants (τ_µ = 2.1969811 µs,
γ_µ/2π = 0.01355342 MHz/G), incremental convergence model, Burg MEM excluded,
generic `MaxEnt-v1` not exposed.

## Key finding — build on the projected-gradient V1, do not rewrite

The shipped engine is a **deterministic entropy-regularised projected-gradient
V1** with **real** cosine/sine forward–adjoint maps, not WiMDA's complex-FFT
3-direction Skilling–Bryan kernel (see [`comparison.md`](comparison.md) §0). The
brief is explicit that this project implements deferred items and does not
re-litigate engine decisions. Every new physics term (pulse shape, exclusion
σ-inflation, deadtime, ZF/LF tie) therefore folds into that real forward–adjoint
contract — feasible because the engine already exposes a cosine/sine component
split. This shapes the whole plan and is the single most important thing for the
implementing pass to internalise.

## Out of scope (rationale in `comparison.md` §9)

- **Spectral deconvolution (`Sconv`)** — numerically hazardous (`1/Sconv`
  divergence); deferred, as in the engine study.
- **Looseness / phase-acceleration knobs** — verdict **out** (not "decide
  later"): they target WiMDA's Skilling–Bryan kernel, which Asymmetry's V1 does
  not run, so they would be dead controls; the V1 has its own χ²-plateau guard.
- **Spectral moments** — the `spectral-moments` Wave B project (consumes this
  spectrum); the panel is kept modular so it can add a tab next wave.
- **Muonium-correlation display** — niche, not in this brief.

## Study files

- [`comparison.md`](comparison.md) — WiMDA ↔ Asymmetry behaviour for each gap,
  with divergences stated both ways (the canonical record).
- [`implementation-options.md`](implementation-options.md) — the chosen options,
  the full ordered 3-phase plan, file-by-file touch list, test plan, follow-ons.
- [`test-data.md`](test-data.md) — synthetic cases and the oracle strategy.
- [`verification-plan.md`](verification-plan.md) — staged validation per phase.

## Reference lineage (for doc citation; cite by name, never by equation number)

- Blundell, De Renzi, Lancaster & Pratt (eds.), *Muon Spectroscopy: An
  Introduction* (Oxford University Press, 2022) — the primary source.
- B. D. Rainford and G. J. Daniell, Hyperfine Interact. **87**, 1129 (1994).
- J. Skilling and R. K. Bryan, Mon. Not. R. Astron. Soc. **211**, 111 (1984).
- T. M. Riseman and E. M. Forgan, Physica B **289–290**, 718 (2000); **326**,
  226/230/234 (2003).
- F. L. Pratt, "WIMDA: a muon data analysis program for the Windows PC", Physica
  B **289–290**, 710 (2000).
- R. Kubo and T. Toyabe, in *Magnetic Resonance and Relaxation*, ed. R. Blinc,
  p. 810 (North-Holland, 1967); R. S. Hayano et al., Phys. Rev. B **20**, 850
  (1979).
- E. H. Brandt, Phys. Rev. B **37**, 2349 (1988); A. D. Hillier and R. Cywinski,
  Appl. Magn. Reson. **13**, 95 (1997) — vortex-lattice field distribution /
  penetration depth, the main quantitative consumer of the TF spectrum.
