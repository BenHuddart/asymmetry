# Spectral moments — verification plan

How each study claim and the eventual port get verified. Tests live beside the
behaviour: `tests/test_spectrum_moments.py` (core), `tests/porting/spectral-moments/`
(oracle), and the GUI/persistence tests in the existing fourier/project suites.

## Claims → checks

| # | Claim | Check |
|---|---|---|
| C1 | Gaussian line has zero skewness and `b_rms = σ` | §1.1 synthetic; assert `skewness≈0`, `beta≈0`, `b_rms_mean≈σ`, `b_pk≈b_ave≈B₀` |
| C2 | All moments correct on a skewed mixture | §1.2 closed-form mixture; assert `b_ave`, `b_rms_mean`, `skewness_g1`, `α`, sign of `β` vs analytic |
| C3 | Core matches WiMDA arithmetic | §2 transcribed oracle on one shared spectrum; agree ~1e-9 (D1 bug excluded) |
| C4 | β sign convention matches the literature | §1.3 vortex-lattice-like lineshape; assert `b_ave>b_pk`, `beta>0`, `skewness>0` |
| C5 | Parabolic peak refines, with edge guard | peak-on-a-coarse-grid: refined `b_pk` between bins; peak within 2 bins of an end → `peak_refined=False`, `b_pk` = discrete bin |
| C6 | Cutoff/range sensitivity is as documented | §1.4 sweep; tightening window → `skewness,beta→0`; raising cutoff → `n_sample↓`, `b_rms↓`; window captured in `recipe` |
| C7 | Uncertainties are real and expose `b_pk` fragility | §1.5 seeded bootstrap; finite, shrink with signal; `b_pk_err`/`beta_err` inflate on noisy flat-topped spectra |
| C8 | Empty window degrades gracefully | zero-amplitude / out-of-range window → `n_sample=0`, NaN moments, no exception; GUI greys readout |
| C9 | Eligibility guard greys ineligible modes | GUI test: set each Fourier mode; assert the moments group is enabled only for `phase_corrected`/`phase_opt_real`/MaxEnt reconstruction, disabled (with tooltip) for power/magnitude/phase/burg/correlation |
| C10 | Send-to-trend records a computed FitSeries | GUI test: compute + send; assert a `FitSeries` with `canonical_model=None`, member-per-spectrum, moment columns in `parameters`, `rep_type` of the source |
| C11 | Re-sending replaces, not duplicates | send same selection twice; assert one batch (same deterministic id); change the selection → new batch |
| C12 | `.asymp` round-trips moments series + recipe | save/load a project with a moments series; assert members, moment values, and `recipe` (range/cutoff/unit/mode/seed) survive |
| C13 | Live widget settings persist without schema bump | set range/cutoff/unit; save/load; assert restored; `CURRENT_SCHEMA_VERSION` unchanged (8); `restore_state({})` and missing-key tolerated |
| C14 | Window is visible on the plot | GUI test: eligible spectrum active → moment-window span + cutoff line drawn; ineligible → not drawn |
| C15 | Existing fourier/MaxEnt tests stay green | full `python tools/harness.py validate` (GUI under `QT_QPA_PLATFORM=offscreen`) |

## Ladder

1. `python tools/harness.py structural` — study layout + index entry.
2. `python tools/harness.py lint` — Ruff baseline.
3. `python tools/harness.py test -- tests/test_spectrum_moments.py` — core (C1–C8).
4. `python tools/harness.py test -- tests/test_spectral_moments_gui.py tests/test_project_*` — GUI + persistence (C9–C14).
5. `python tools/harness.py validate` — full suite incl. regression (C15).
6. `python tools/harness.py docs` — user-guide page builds clean.

## Risks & how the plan covers them

- **`b_pk` fragility** (the weakest moment): C5 pins the refinement + guard; C7
  makes the noise sensitivity *measurable* via bootstrap rather than asserted in
  prose; the user guide states the caveat.
- **Unit confusion** (G vs MHz): C12/C13 pin the recorded unit per series; `α`/`β`
  are invariant, only `B_*` rescale (asserted by converting a spectrum's axis and
  checking `α`,`β` unchanged, `B_*` scaled by `γ_μ`).
- **Eligibility regressions** (a new display mode silently becoming eligible): C9
  enumerates the mode set explicitly so a new mode defaults to ineligible.
- **Trend duplication** on re-send: C11 pins the deterministic batch id.
- **WiMDA divergences** (D1–D6): each is asserted or documented — D1 by the
  oracle excluding the bug, D2 by reporting `γ₁` alongside `α`, D3 by the
  per-member series tests, D4 by C7, D5 by C9, D6 by the invariance check.
</content>
