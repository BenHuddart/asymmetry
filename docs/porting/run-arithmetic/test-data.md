# Test data: run-arithmetic

## Synthetic (primary, no external corpus needed)

`core/simulate.py` with fixed seeds is the primary oracle — it gives exact
control over statistics and ground truth.

- **Pull test (co-add distributional identity).** Simulate N runs of the same
  model at events E each (distinct seeds) and one run at N·E events. Co-add the
  N and reduce; the pulls of (co-added − single-N×) against the combined error
  must be ~N(0,1). Confirms count-sum co-add is distributionally identical to a
  single longer run.
- **Co-subtract of identical runs → zero.** Simulate one run twice with the
  *same* seed (identical counts); co-subtract → asymmetry identically 0 with
  errors ≈ √2 × the single-run error (variances add, equal exposure → scale 1).
- **Two-period co-add.** `simulate_two_period_run` (PeriodSpec) → co-add two
  such runs; verify per-period histograms summed and `period_*` keys intact;
  G∓R reduction on the combined run matches summing each period then reducing.
- **Negative-count guard.** Co-subtract two runs where the reference exceeds
  the sample in some bins; assert the guard counts negative bins and the
  variance radicand stays ≥ 0.
- **Event-weighted metadata.** Two runs at different T and unequal good frames;
  assert combined `temperature` is the frame-weighted mean and
  `temperature_spread` brackets both.

## Corpus (gated on `$ASYMMETRY_*` env, `pytest.skip` when absent)

Following the `tests/test_psi_loader.py` pattern — external data gates on an
environment variable and skips cleanly when unset; committed code never embeds
a local path.

- **Co-add-then-reduce vs WiMDA arithmetic.** A CdS or EuO pair from the Muon
  School corpus: load two runs, co-add, reduce, and compare the asymmetry to
  WiMDA's count-level co-add of the same pair (values agree at α=1; error bars
  follow the pooled-Poisson rule, RA2/RA8).
- **Quantified low-count correction.** The same pair (or a deliberately
  low-count slice) gives the headline number: percentage difference between the
  old curve-mean error bar and the new pooled error bar (≈30 % at a 10:1
  statistics ratio; exact figure quoted in the user guide).
- **Two-period photo-µSR silicon.** A period-mode silicon run from the corpus
  exercises the period-summation path on real data.

## Regression gate

- **CdS** and **EuO** existing reductions must still load and reduce; co-add
  results change deliberately (RA8) — the affected test expectations are
  updated in the same commit, with the change called out.
- `.asymp` round-trip: save a project with a combined dataset, reload, and
  confirm `rebuild_combined_dataset` reproduces the combined row through
  `combine_runs` (silent recompute).
</content>
