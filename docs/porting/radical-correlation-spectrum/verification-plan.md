# Muoniated-radical correlation spectrum — verification plan

How each claim is verified, smallest check first. All tests sit beside the
behaviour they protect and run under `python tools/harness.py validate`
(GUI tests need `QT_QPA_PLATFORM=offscreen`).

## Claim 1 — the pair relation is exact

**Claim:** `ν₁₂ + ν₃₄ = A_µ` from `muonium.py._tf_levels`, for any (B, A_µ).
**Test:** property test over a grid of (B, A_µ); assert `w12 + w34 == A` to
`< 1e-9` relative. Anchors the hyperfine axis. (Already demonstrated in the
study; promoted to a unit test in the correlation-core test module.)

## Claim 2 — the correlation peaks at the true A_µ (synthetic, gating)

**Claim:** a synthetic radical TF run with known (B, A_µ) yields a correlation
spectrum whose dominant peak is at A_µ.
**Test:** build the [test-data §1](test-data.md) cyclohexadienyl fixture
(A_µ = 514.4 MHz, B = 2900 G) via `core/simulate`; run
`compute_average_group_spectrum` with the `correlation` display mode; assert the
peak is within the spectrum's frequency resolution of 514.4 MHz, and that the
peak/median ratio comfortably exceeds the diamagnetic and noise background.
Repeat at B = 14500 G (field-independence) and for a two-radical mixture (two
peaks). This is the **primary correctness gate**.

## Claim 3 — parity with WiMDA `rmatch` / `CorrFn` (oracle)

**Claim:** the exact forward map agrees with WiMDA's `rmatch` to within the
documented approximation error, and `CorrFn` is ported faithfully.
**Test:** transcribed fixed-point checks ([test-data §2](test-data.md)): the
forward-map A matches `rmatch`-derived `|f₁+f₂|` to `~0.03 MHz` (and the true
A_µ to machine precision); `CorrFn` reproduces the Pascal at hand-computed
points including the `order = 0` fallback. Documents the rounded-constant
divergence rather than hiding it.

## Claim 4 — the A_µ axis is not field-converted

**Claim:** the correlation x-axis is labelled as a hyperfine coupling and is
excluded from the MHz/G/T field-unit selector.
**Test:** the correlation dataset's metadata carries the coupling x-label and
the diagnostic flag; a GUI test asserts the field-unit combo is disabled/ignored
for the correlation mode (so A_µ is never multiplied by `γ_µ`).

## Claim 5 — no regressions (must not change)

**Tests:**
- All existing `fourier`/`maxent` tests pass unchanged.
- `compute_average_group_spectrum` output for every non-correlation mode is
  identical before/after (a snapshot/equality guard).
- Project round-trip: a `.asymp` with a correlation `fourier_config` reloads to
  an identical spectrum (additive `GroupSpectrumConfig` keys; `to_dict`/
  `from_dict` symmetric).

## Claim 6 — documentation teaches the physics

**Claim:** the new user-guide page teaches radical-µSR to a non-specialist and
includes the mandatory TF-vs-ALC complementarity subsection, grounded in the
sources.
**Check (docs harness + review):** `python tools/harness.py docs` passes;
the page renders math (no raw LaTeX in prose), cites sources in APS-style lists,
uses the diagnostic "when to use" register, states uncertainties as `0.23(1)`,
cross-references the existing ALC page, and never cites the textbook's equations
by number. (Manual review against the owner's style + the §6 source checklist.)

## Ladder

```
python tools/harness.py structural
python tools/harness.py lint
python tools/harness.py test -- tests/test_fourier_correlation.py
python tools/harness.py docs          # documentation phase
python tools/harness.py validate      # full gate, each phase green (~2 min)
```
