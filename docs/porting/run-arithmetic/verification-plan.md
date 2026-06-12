# Verification plan: run-arithmetic

## Claims and how each is checked

| Claim | Verification | Source |
|---|---|---|
| Count-sum co-add == single longer run (distributionally) | Pull test, N synthetic runs vs one N×-events run; pulls ~N(0,1) | synthetic |
| Co-add value unchanged, error corrected | Compare old curve-mean vs new pooled error on a low-count synthetic pair; quantify | synthetic + corpus |
| Co-subtract variances add | Identical-run subtract → 0 ± √2·σ; mismatched seeds → propagated error matches `subtract_scaled_counts` | synthetic |
| F9: co-subtract uses the chokepoint | Code review + a test asserting the subtract path calls `subtract_scaled_counts` / `resolve_background_reference` (no parallel impl) | code |
| Negative-count guard | Subtract with reference > sample; guard counts negatives, radicand ≥ 0 | synthetic |
| Period summation via periods.py | Two-period co-add reuses `sum_period_histograms`; `period_*` keys correct; G∓R matches | synthetic + corpus |
| Event-weighted T/field + spread (W3) | Unequal-frame, unequal-T pair → frame-weighted scalar + spread brackets both | synthetic |
| Combined row is first-class (RA8) | Regroup, deadtime-correct, count-fit and MaxEnt a combined row through existing paths; assert no `histograms=[]` failure | synthetic |
| `.asymp` round-trip (W11) | Save/reload project with a combined dataset; `rebuild_combined_dataset` reproduces it via `combine_runs` | unit + GUI |
| WiMDA arithmetic parity | Co-add-then-reduce vs WiMDA on a corpus pair; record divergences (errors) in comparison.md | corpus (gated) |
| Corpus regression | CdS + EuO still reduce; updated co-add expectations | corpus (gated) |

## Quantified co-add correction (deliverable)

Compute, on a real low-count pair, the percentage change in the combined error
bar from curve-mean → pooled Poisson, and the asymmetry-value agreement. Put
the number in `docs/user_guide/.../run_arithmetic.rst` and the PR body.

## Validation ladder

```
.venv/bin/python tools/harness.py test -- tests/test_combine.py
QT_QPA_PLATFORM=offscreen .venv/bin/python tools/harness.py test -- tests/test_data_browser_combine.py
.venv/bin/python tools/harness.py docs        # user-guide builds clean
.venv/bin/python tools/harness.py validate     # full gate, GUI offscreen
```

## Final-stage gates

1. Full `validate` green (GUI offscreen); user-guide docs build clean.
2. `/code-review` high effort over the branch diff; fix confirmed correctness
   issues + cheap cleanups, record uncertain/out-of-scope as follow-ons; commit
   as a separate "code-review fixes" commit; re-run validate.
3. Rebase onto origin/main (sibling Wave B PRs); resolve append-only shared
   files additively; re-validate if code changed.
4. Push + open PR (no auto-merge).

## Recorded follow-ons (out of scope here)

- Symmetric two-run / N-run signed co-subtract (decision 1).
- In-batch co-add during sequential fitting (fit-workflow-diagnostics).
- Co-subtract across multi-period runs beyond the reference-run case.
</content>
