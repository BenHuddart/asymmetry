# Verification plan (with expected values)

The full test plan, file-by-file, is in [`plan.md`](plan.md) §"Test plan". This
file states the **expected values** the assertions check. Three exactness
anchors (table spot-checks, ratio arithmetic) plus tolerance-based fit recovery.

Run tests with the worktree venv: `python tools/harness.py test -- tests/negmu/`.

## 1. Lifetime-table spot checks (EXACT)

`core/negmu/lifetimes.py` must transcribe Table C.1 (μs). These high-confidence
values (clean reads, cross-validated against WiMDA to ≤ rounding) are pinned
exactly; assert `tau_us(sym) == value` (float-equal at the stored precision):

| Symbol | Z  | τ (μs)   | Symbol | Z  | τ (μs)   |
|--------|----|----------|--------|----|----------|
| H      | 1  | 2.19480  | Fe     | 26 | 0.206    |
| Be     | 4  | 2.16747  | Cu     | 29 | 0.164    |
| C      | 6  | 2.030    | Br     | 35 | 0.133    |
| O      | 8  | 1.795    | Zr     | 40 | 0.110    |
| Na     | 11 | 1.204    | Nb     | 41 | 0.092    |
| Al     | 13 | 0.864    | Au     | 79 | 0.0728   |
| Si     | 14 | 0.759    | Pb     | 82 | 0.0747   |
| Ca     | 20 | 0.336    | Bi     | 83 | 0.0735   |

Plus structural assertions:

- `tau_us("Tl") == 0.0704` and `lifetime("Tl").symbol == "Tl"` — guards the
  WiMDA `'Ti'`→`'Tl'` symbol bug (`comparison.md` §5).
- `tau_us("Ne") == 1.461` (not 1.520) — guards the WiMDA Ne divergence.
- Every entry's `tau_us > 0` and strictly within `[0.05, 2.30]` μs.
- `lifetime("H").source == "SuzukiMeasdayRoalsvig1987"`.
- Lanthanide / Ru–In cluster entries (flagged in the plan) are **not** asserted
  to specific values, only `> 0` and present — their exact transcription is the
  implementer's to confirm against the textbook table.

## 2. Fit recovery (TOLERANCE-based)

Because recovered values depend on the seed, assertions are tolerance-based; the
generating parameters are exact. Synthesise with `simulate_capture_run`, fit
with `fit_capture_group` (τ fixed), `cost="poisson"`.

**Case 2a — two elements (C + O + decay-BG).** Template: 2 detectors, one group,
bin width 0.016 μs, 1024 bins, t0 at bin 0. Generating amplitudes (relative
populations, normalised to `total_events = 2.0e7`): C : O : decayBG = 5 : 3 : 2.
τ_C = 2.030, τ_O = 1.795, τ_decayBG = 2.1969811 μs. `seed = 0`,
`background_per_bin = 5.0`.

Expected after fit:
- `result.success is True`.
- Recovered `amp_C / amp_O` within **3σ** of the true ratio 5/3 = 1.6667 **and**
  `|Δ|/truth < 0.05`.
- Each recovered amplitude within `5 %` of its generating value.
- `reduced_chi_squared` inside the `assess_fit_quality` "good" band at ν =
  N_bins − N_free (verdict `"good"`, not `"poor"`/`"overdone"`).

**Case 2b — light + heavy (C + Fe + decay-BG), decade in τ.** Same template,
τ_Fe = 0.206; C : Fe : decayBG = 4 : 4 : 2; `seed = 1`. Same acceptance form;
additionally assert the Fe amplitude is recovered within `8 %` (its fast decay
gives fewer constraining bins).

**Case 2c — free-τ sanity.** Case 2a with `spec.free_tau = {"C"}`. Assert the
freed `tau_C` is recovered within `2 %` of 2.030 μs and its uncertainty is
finite and positive (the free-τ mechanism works).

**Case 2d — α-coupled F+B (Phase 2).** Forward+backward groups, shared C + O +
decay-BG, balance α = 1.25. Fit with `fit_capture_fb_alpha`. Assert recovered
`alpha` within 3σ of 1.25 and `|Δ| < 0.05`; shared amplitude ratios as in 2a.

**Case 2e — Gaussian cost parity.** Case 2a with `cost="gaussian"` converges and
recovers the ratio within `6 %` (looser; documents the cost difference).

## 3. Capture-ratio arithmetic (EXACT, hand-computed)

`capture_ratio_report` must reproduce these transcribed-by-hand numbers. Build a
synthetic `FitResult` directly (no fitting) with the stated parameters/covariance
and assert the report.

**Fixture A — independent (zero covariance).** amplitudes `amp_C = 1000`,
`amp_O = 500`; uncertainties `σ_C = 30`, `σ_O = 20`; covariance absent.
Reference element `O`:

    R = amp_C / amp_O = 1000 / 500 = 2.000
    σ_R = R · sqrt((σ_C/amp_C)² + (σ_O/amp_O)²)
        = 2 · sqrt((30/1000)² + (20/500)²)
        = 2 · sqrt(0.0009 + 0.0016) = 2 · 0.05 = 0.100

Expected: ratio `C/O = 2.000`, σ = `0.100` → reported **2.00(10)**.

**Fixture B — covariance-aware.** Same amplitudes/σ, plus `cov(amp_C, amp_O) =
+150` (ρ = 150/(30·20) = 0.25) supplied via `FitResult.covariance` /
`covariance_parameters`:

    σ_R = R · sqrt((σ_C/amp_C)² + (σ_O/amp_O)² − 2·cov/(amp_C·amp_O))
        = 2 · sqrt(0.0009 + 0.0016 − 2·150/(1000·500))
        = 2 · sqrt(0.0025 − 0.0006) = 2 · sqrt(0.0019)
        = 2 · 0.0435890 = 0.0871780

Expected: ratio `2.000`, σ = `0.0872` (4 s.f.) → reported **2.00(9)**. This
asserts the report uses the covariance when present (positive correlation
*reduces* the ratio error) and falls back to quadrature (Fixture A) when absent.

Assert both with `pytest.approx(..., rel=1e-6)`.

## 4. Reuse & no-GUI guards

- **No registration.** Assert the μ⁻ models are absent from the GUI registries:
  `for k in (...negmu labels...): assert k not in COMPONENTS and k not in MODELS`
  and that `core.negmu` imports without importing `asymmetry.gui` (Qt-free:
  reuse the existing core/GUI-isolation test pattern in `tests/`).
- **count_domain untouched.** A test asserting `core/fitting/count_domain.py` is
  not imported by `core/negmu` and that no `core/negmu` symbol shadows a
  `count_domain` public name (the modules are independent).

## 5. simulate_capture_run round-trip

- Expected (noise-free) histogram from `simulate_capture_run` integrates to
  ≈ `total_events` over the post-t0 window (within Poisson tolerance) and equals
  `Σ_i N_i exp(−t/τ_i) + bg` at the bin centres (assert against a direct numpy
  evaluation).
- Provenance: `run.metadata["synthetic"] is True`,
  `run.metadata["simulation"]["capture_mode"] is True`, components and seed
  recorded; deadtimes zeroed (inherited from `_sample_and_build_run`).
- A fixed seed reproduces the run bit-for-bit (two calls equal).

## 6. Per-phase acceptance

Each phase is green only when: `python tools/harness.py validate` passes (lint +
structural + full pytest), the new tests above for that phase pass, and
`python tools/harness.py docs` builds (Phase 5 / whenever the .rst lands). No GUI
smoke is required (no GUI surface). Acceptance criteria per work package are in
[`plan.md`](plan.md).
