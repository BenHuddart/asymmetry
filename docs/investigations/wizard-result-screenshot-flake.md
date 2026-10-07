# Investigation: `global_fit_wizard_result` screenshot fails on some CI runners

**Status: ROOT CAUSE FOUND; scenario made deterministic (2026-10-07).** The
docs-pages capture failed in about 40 % of runs with `StopIteration` at the
scenario's `b_key = next(...)` lookup: the recommendation held no split with
exactly Δ and B_L local. The search is not scheduling-dependent. It depends on
**which SIMD kernels NumPy dispatches on the runner's CPU**, because the
scenario's data made the role search start from ill-posed per-run fits.

## Evidence

A temporary probe workflow ran the scenario's
`build_global_fit_wizard_recommendation` call on eight `ubuntu-latest` runners,
three fresh processes each, with the same pip packages:

- Every runner was self-consistent: the same answer in every process.
- Pass or fail split exactly by `numpy._core._multiarray_umath.__cpu_features__`:
  every runner exposing AVX-512 (Xeon 8370C, 8573C, 6973P-C; EPYC 9V74 when
  the hypervisor exposed it) failed; every AVX2-only runner (EPYC 7763, EPYC
  9V74 without AVX-512) passed. macOS arm64 passes.
- On a failing host, the same call with `NPY_DISABLE_CPU_FEATURES` set to the
  host's AVX-512 features passed. That is the causal test.

Failing runs stop ~2.7 s in rather than ~8.5 s because the backward
elimination they take is shorter and fits fewer nodes.

## Mechanism

The default (separable) engine starts from the all-local node: one independent
LF Kubo–Toyabe + Constant fit per run (`_all_local_run_fit_results` →
`_best_single_run_fit_result`). At 50 G and 100 G (γ_μB_L/Δ ≈ 11 and 22) the
signal is decoupled: the polarisation is flat to (Δ/γ_μB_L)², so A_1, A_bg, Δ
and B_L are nearly degenerate within one run. There, every Migrad attempt
reports failure, even the ones that reach χ² ≈ 508 (50 G) and ≈ 447 (100 G),
and the ladder keeps the Simplex fallback that "succeeds" at χ² ≈ 635 and
≈ 476. Where Simplex stops on that flat valley moves with the last bits of
`exp`/`sin`/`cos`, which differ between NumPy's AVX2 and AVX-512 (SVML)
kernels:

| host     | 50 G anchor                       | 100 G anchor                      | surrogate's first share |
|----------|-----------------------------------|-----------------------------------|-------------------------|
| AVX2     | Δ 0.051, B_L 50.1, A_1 23.9       | Δ 0.091, B_L 48.7, A_1 24.1       | A_1 (IC 2126)           |
| AVX-512  | Δ 0.263, B_L 71.0, A_1 20.3       | Δ 0.209, B_L 69.4, A_1 20.4       | Δ (IC 2135)             |

From the AVX2 anchor the greedy chain shares A_1, then A_bg (reaching
{B_L, Δ} local, AICc 1998.4), and the flip neighbourhood finds the winner
{B_L} local, AICc 1964.2. From the AVX-512 anchor it shares Δ first, its
warm-started fit lands at AICc 2123.7 (χ² 2097.6, worse than the 1950.1 of the
nested {B_L}-local model, so it is itself mis-converged), and the walk stops
there. The answer is wrong in that run, and {B_L, Δ} local is never fitted.

## Resolution

The scenario now uses fields 0, 5, 10 and 25 G (γ_μB_L/Δ ≈ 0, 1, 2, 5), where
every per-run fit converges under Migrad from every seed. On the same eight
runners, AVX-512 and AVX2 alike, the ranking is identical to ~1e-9 AICc.
The narrowest decision in the chain is 0.02 AICc, about four orders of
magnitude above the cross-host spread. The figure's story changed with it: B
now costs +4.0 AICc and its per-run Δ sits on A's shared value.

## Open core issues (not fixed here)

- `_best_single_run_fit_result` prefers a successful Simplex over a failed
  Migrad endpoint 127 χ² lower, so the all-local anchor on a decoupled run is a
  poor, host-dependent minimum.
- The elimination chain can accept a node whose χ² exceeds that of a nested,
  less flexible node already in the cache; the certificate check only compares
  a child with its direct parent.
- `RunEstimate.at_bound` is documented as parameter names, but both callers
  pass `_bound_hit_names` display strings (`"Delta at lower bound"`), so the
  "never propose a railed parameter" rule in `_separable_backward_elimination`
  and `surrogate._eligible_names` never fires.
