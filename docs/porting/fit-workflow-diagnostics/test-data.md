# Test data: fit-workflow-diagnostics

All verification data is **synthetic and in-repo** (via `core/simulate`) or
tabulated constants — no external corpus is required for the core suite. Any check
that uses the WiMDA Muon-School corpus gates on an env var with `pytest.skip` (the
`tests/test_psi_loader.py` pattern), so CI on the public repo stays green without it.

## MINOS vs HESSE — deliberately asymmetric likelihood

- **Generator:** a low-statistics zero-field Kubo–Toyabe run from
  `core/simulate` with a **fixed seed** (low counts → the relaxation/Δ parameter
  sits in a non-parabolic region; near a positivity bound its likelihood is visibly
  skewed). KT is the canonical example because Δ near zero is the textbook
  asymmetric case.
- **Expectation:** for the skewed parameter, `minos_errors[name] = (lo, hi)` with
  `|lo| ≠ |hi|` and the HESSE σ sitting between them in the **documented direction**
  (for a parameter pushed against a lower bound, the upper MINOS excursion exceeds
  the lower). For a well-determined parameter (e.g. the asymmetry amplitude in a
  high-stat run) MINOS ≈ HESSE to a few percent — a control assertion.
- **Count-domain α:** run `fit_fb_alpha` with MINOS on a simulated F/B pair; assert a
  finite asymmetric interval on α and that the promote payload (`alpha_error`) is
  still the **scalar HESSE** value (overlay-only invariant).

## χ² band — tabulated chi² quantiles

- **Oracle:** independent tabulated χ² quantiles (e.g. for ν where the 95% two-sided
  band is well known, cross-check `band_low`/`band_high` against published values),
  plus the exact `scipy.stats.chi2.ppf` the helper uses. The helper already has unit
  tests (PR #32); new tests assert the **wiring** — that the same fit produces an
  *identical* verdict dict from every surface (single panel, grouped, global, model
  dialog) for the same `(χ², ν)`.
- **Verdict-boundary fixtures:** construct `(χ², ν)` triples landing just inside each
  region — CDF just below `(1−R)/2` (overdone), in band (good), just above `(1+R)/2`
  (poor) — and assert the verdict label, for R = 0.95.

## Chain-seeding — EuO temperature scan through Tc

- **Data:** the EuO PSI `.bin`/`.mdu` T-scan already used elsewhere in the corpus
  (gated by env var) **and** a synthetic T-scan from `core/simulate` (fixed seed)
  whose order parameter collapses through a transition, so the core suite needs no
  external data.
- **Expectation:** with **static/average** seeds, fits near and above the transition
  either fail to converge or wander (the seed is far from the post-transition basin);
  with **chain-from-previous** seeding (ordered by temperature), each fit starts in
  its neighbour's basin and the scan converges monotonically through Tc. Assert lower
  failure count / tighter χ²ᵣ spread for the chained run vs the static run on the
  synthetic scan; the external EuO check is a gated confirmation.
- **Contract check:** assert chained seeds are re-normalised — amplitude≡1,
  background≡0 in the seed handed to member N+1 regardless of member N's fitted
  amplitude/background.

## Abort — mid-series cancellation

- **Data:** any multi-member synthetic series (≥3 members) from `core/simulate`.
- **Expectation:** a `cancel_callback` that returns `True` after the first member
  raises `FitCancelledError`; the driver records **no** `FitSeries`/`FitSlot`; a
  subsequent fit of the same series with no cancel completes normally (the next fit
  is unaffected). An in-fit abort test: a `cancel_callback` flipping `True` during a
  single long fit raises and yields no `FitResult`.

## FitLog — round-trip provenance

- **Data:** a synthetic single fit + a synthetic grouped series.
- **Expectation:** the enriched `fit_result_summary` carries `quality`,
  `uncertainties_asymmetric`, `model_name`, `fit_range`, `npar`, `ndof`,
  `provenance`; a `.asymp` save/load round-trips all of them on `FitSlot.result` and
  `FitSeries.results_by_run`; `FitLog.format_record(record)` returns a block
  containing the run, model, verdict, and (when present) the `+hi −lo` MINOS columns.
  Export writes that text to a chosen path and re-reads identically.

## Notes

- Fixed seeds everywhere (`core/simulate` accepts a seed) so assertions are
  deterministic across runs and machines.
- No `Date.now()`-style nondeterminism in core: the FitLog `timestamp` is injected
  by the GUI layer; core formatting takes the timestamp as data, so formatter tests
  pass a fixed string.
