# Simulate mode: verification plan

The feature is its own oracle: simulation from known parameters followed by
the real reduction + fitting chain must recover those parameters with the
claimed statistics. Every check below lands as a pytest (corpus checks
skip-if-missing), strictly stronger than anything WiMDA had (WiMDA's
simulation is unseeded and was never statistically verified).

**Outcome (2026-06-10): all sections implemented and green** —
`tests/test_simulate.py` (§1, §4), `tests/test_nexus_writer.py` (§2–3 +
corpus), `tests/test_simulate_dialog.py` (§5); 53 tests, full validate
1817 passed. One refinement against §2 as written: the refit χ²ᵣ band is
centred on the analytic expectation E[(1−A²)/(1+A²)] rather than 1,
because the shipped asymmetry error formula propagates F ± αB as
independent and so over-estimates σ_A at |A| > 0 — a chain-wide property
this suite exposed (recorded as a follow-on in
implementation-options.md), not a simulation defect. Fits are also
windowed to the healthy-count region (t ≤ 8 μs), since √n errors from
observed counts bias χ² in the ≲ 1 count/bin tail.

## 1. Forward-model correctness (unit level)

- **Envelope normalisation**: with the signal off (a ≡ 0, b = 0), the
  expected counts summed over all detectors and bins equal the requested
  total events to within the histogram-window truncation correction
  (1 − e^{−T/τ_μ}); per-bin means follow N₀·e^{−t/τ_μ} (regression against
  the analytic value, then a Poisson z-test on the sampled histograms over
  many bins).
- **α split**: simulated F/B group totals satisfy N_F/N_B = α within
  counting errors; the reduced asymmetry of a zero-signal simulation is
  flat at 0 (the α in generation and the α in reduction cancel exactly).
- **Signal forwarding**: with Poisson sampling replaced by the expectation
  (internal hook or huge-N limit), the reduced asymmetry reproduces the
  generating model A·P(t) bin-by-bin (rtol ≲ 1e-10 in the expectation mode).
- **Per-detector t0**: staggered template t0 bins produce histograms whose
  signals align in *time* (not bin index) after reduction.
- **Pre-t0 bins**: contain exactly the background rate (zero by default).
- **Determinism**: same seed → bit-identical `Run` (counts arrays equal);
  different seeds → different draws; provenance metadata records the seed,
  model expression, parameters, and template identity.

## 2. Round trip: simulate → NeXus → reload → refit

- **File identity**: write the synthetic run, reload through `NexusLoader`;
  per-detector counts arrays identical, bin width / t0 / good bins /
  grouping / good_frames / title / field / temperature as written;
  `nexus_version == "v1"`; the `/run/simulation` provenance group survives
  into `metadata["nexus_fields"]`.
- **Refit recovery (single seed)**: simulate from a known parameter set
  (α = 1 template; see comparison.md divergence 9 for why), reload, fit the
  same model seeded *away* from the truth: every recovered parameter within
  3σ of the generating value, χ²ᵣ in the textbook band 1 ± 2·√(2/d).
- **Corpus end-to-end** (skip-if-missing): fit a TF model to the
  Ferromagnetic-nickel HDF5 run, simulate from the fitted parameters with
  matched event total, save, reload, refit — recovered parameters within
  errors of the generating ones, and the synthetic dataset's per-bin errors
  match the real run's at equal events (same reduction chain ⇒ same error
  model).

## 3. Pull distribution over many seeds (the headline statistical check)

For a fast model (`Exponential + Constant`, modest bins/events for runtime):
simulate ≥ 100 seeds, fit each, form pulls
(fitted − true)/σ_fitted per parameter. Assert mean consistent with 0
(|mean| < 4/√N) and variance consistent with 1 (within the χ² interval for
N samples). This verifies the *entire* chain — Poisson generation, grouped
reduction, the Mantid-model error formula, and iminuit's covariance — as one
statement: errors are neither over- nor under-estimated. Runtime budget
≲ 20 s (tiny histograms; the suite is parallelised).

## 4. Degrade statistics

- **Mean**: thinned counts ≈ factor × original (global ratio within
  counting error).
- **Error scaling**: per-bin asymmetry errors of the thinned run scale as
  1/√f relative to the original (median ratio over the good-bin window).
- **Exact thinning law** (f < 1): for a synthetic source with known λ, the
  thinned bins pass a variance/mean ≈ 1 Poisson check (binomial thinning of
  Poisson is Poisson); document and test the over-dispersion of the f > 1
  `Poisson(k·f)` branch (variance/mean ≈ 1 + f against the *true* λf).
- **Determinism + provenance**: seeded, repeatable, derived-run metadata
  records factor/seed/source run; original run untouched.

## 5. GUI (offscreen)

- Dialog opens with a loaded template, seeds its model/parameters from the
  current per-run fit state when present, falls back to builder defaults
  otherwise; Generate adds a badged dataset to the Data Browser; the badge
  marks it synthetic; Save-as-NeXus writes a file the loader accepts.
- Degrade context action produces a new badged derived run; the original's
  counts are unchanged.
- Both respect the conftest widget-cleanup conventions
  (`QT_QPA_PLATFORM=offscreen`, no leaked `MainWindow`).

## 6. Harness ladder

`structural` and `docs` green at the study milestone; full
`python tools/harness.py validate` green at the end of every implementation
phase (run from the project venv; ~2 min).
