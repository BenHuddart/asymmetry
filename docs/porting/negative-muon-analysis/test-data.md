# Test data

## No μ⁻ corpus

No negative-muon elemental-analysis runs exist locally or in the project corpus
(`project_testing_corpus`). Acquiring real ISIS μ⁻ data is a hard prerequisite
for *validation* and is the promotion trigger for any GUI. **All testing in this
plan is synthetic.** This is stated loudly in the WIP disclaimer and the docs.

## Why the existing simulator needs an additive function

`core/simulate.py` synthesises forward/backward and single-histogram count data
by imprinting an asymmetry on a **single muon-decay envelope**:

    N_d(t) = N0_d · exp(−t/τ_μ) · [1 + a_d(t)] + b      (expected_counts, line ~462)

The μ⁻ capture histogram is a **sum of exponentials at different rates**,
`Σ_i N_i exp(−t/τ_i) + bg`, which is not `(single envelope) × (1 + a)` for any
physical `a(t)` (the same reason `count_domain` cannot fit it — `comparison.md`
§3). Coercing it through the `group_signals` seam would require an `a(t)` that
divides out the τ_μ envelope and references the simulator's internal
normalisation — fragile and dishonest.

**Decision (Ben):** add a new public `simulate_capture_run` to `core/simulate.py`
that builds the multi-exponential expectation **directly** and **reuses** the
existing seeded-Poisson sampling + `Run`/provenance assembly
(`_sample_and_build_run`). This honours "use `core/simulate`, do not write a
bespoke generator," produces a real `Run` (so the `(dataset, group)` and
α-coupled F+B fit paths are exercised end-to-end), and keeps the sampling law and
provenance identical to every other synthetic run. Signature, normalisation, and
provenance contract are specified in [`plan.md`](plan.md) §"WP1.4".

## Synthetic cases (used by the verification plan)

All cases use a small template `Run` (a few detectors, fixed bin width, one
detector group), a fixed `seed`, and a large event budget so Poisson noise does
not dominate recovery. Generating parameters are stated numerically in
[`verification-plan.md`](verification-plan.md); the headline cases:

- **Two-element identification** — components C (τ = 2.030 μs) + O (τ = 1.795 μs)
  + decay-BG (τ = τ_μ), well-separated lifetimes, known amplitude ratio.
- **Light+heavy separation** — C (2.030) + Fe (0.206) + decay-BG, spanning a
  decade in τ.
- **α-coupled F+B** — the same components on a forward and a backward group with
  a known balance α, recovered by the Phase-2 fit.
- **Capture-ratio arithmetic** — a fixture with hand-computed amplitudes and
  covariance feeding `capture_ratio_report`, checked against exact arithmetic
  ([`verification-plan.md`](verification-plan.md) §3).

## Lifetime-table spot checks

A subset of high-confidence Table C.1 values (cross-validated against WiMDA) is
pinned exactly as a transcription guard — see
[`verification-plan.md`](verification-plan.md) §1. Lanthanides and the
period-5 Ru–In transition cluster are **excluded** from spot-checks (their exact
element↔value assignment in Table C.1 must be confirmed by the implementer
against the textbook; flagged in the plan's table).
