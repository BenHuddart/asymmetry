# Beta Correction — Verification Plan

## Core identities (tests/core/test_transforms.py or a new focused file)

1. **β = 1 regression (exactness).** For arbitrary counts and α:
   `compute_asymmetry(f, b, alpha=a, beta=1.0)` returns bit-identical
   (value **and** error) arrays to the pre-port two-argument call. Same for
   `compute_asymmetry_with_count_errors` and `binned_fb_asymmetry`.
2. **musrfit-form equivalence.** Random positive (f, b), random (α, β):
   our value equals the transcribed musrfit formula
   `(α_m f − b)/(α_m β_m f + b)` with `α_m = 1/α`, `β_m = β`
   (rtol ~1e-12). Pins the convention mapping in `comparison.md`.
3. **Ground-truth recovery.** Noise-free synthetic two-detector model
   (`test-data.md`): reduction with the true (α, β) returns `A₀,f·P(t)`.
4. **Error scaling.** `σ(β) / σ(β=1) == (1+β)/2 · D(1)²/D(β)²` per bin
   (D = βF + αB), i.e. the derived closed form
   `σ = |α|(1+β)√(FB(F+B))/D²`; count-error variant analogous.
5. **Sentinels and guards.** One-sided bins keep σ = 1; `βF + αB = 0` bins
   return (0, 1); lenient grouping read maps β ∈ {NaN, inf, 0, −2, "x"} → 1.0.
6. **Independence invariant.** The α estimators' outputs are unchanged by the
   grouping's β (β must not reach `estimate_alpha*` — the corrected-counts
   builder stops before the asymmetry step).
7. **Integral observable.** `integrate_asymmetry` with β matches
   `compute_asymmetry` on the summed counts (both methods), preserving the
   shared-error-model contract.

## Persistence (tests/core/test_grouping_profiles.py + project round-trip)

8. **Emission-only-when-≠1.** `GroupingProfile.to_dict()` contains no `beta`
   key at β = 1 (existing profiles byte-identical); β = 0.9 round-trips
   through `to_dict`/`from_dict` and `profile_from_payload`.
9. **Resolution.** `resolve_effective_grouping` writes `grouping["beta"]`
   only when ≠ 1; the reduction honours it end-to-end (dataset differs from
   the β = 1 reduction).
10. **Instrument heal keeps β** (scalar, instrument-independent).

## GUI (tests/gui/test_grouping_dialog.py + preview pane tests)

11. **Card wiring.** The β card exists (blue stage colour), its value field
    round-trips draft → payload → profile, and editing β marks the dialog
    dirty exactly like α.
12. **Compare ghost.** `compare_stage="beta"` produces a ghost that differs
    from the solid when β ≠ 1 and draws no ghost when β = 1 (unconfigured
    stage rule); payload-invariance — the compare never mutates
    `_current_grouping_payload` (the correction-order study's trap test,
    extended to β).
13. **Pager/chip.** The β stop appears in pipeline order after α and only
    when configured, matching `_correction_stage_active`.

## Ladder

- Iterating: focused files above via
  `python tools/harness.py test -- tests/core/test_transforms.py` etc.
- After core changes: `python tools/harness.py test --tier fast`.
- Before handing back: `python tools/harness.py validate` once, plus
  `python tools/harness.py structural` (docs screenshot map, test layout)
  and `lint`.

## Outcomes (fill in during the implementation pass)

- [ ] Core identities 1–7 green.
- [ ] Persistence 8–10 green.
- [ ] GUI 11–13 green.
- [ ] `validate` green at the standard tier.
