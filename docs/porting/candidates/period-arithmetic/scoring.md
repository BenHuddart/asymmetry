# Period arithmetic: scoring

## Impact (1-5)

**Score: 3**

- Critical for ISIS pulsed-beam users.
- Less impactful for PSI continuous-beam users (most of musrfit's
  audience).
- Unlocks correct multi-period analysis where currently we silently
  drop periods.

## Ease (1-5)

**Score: 3**

- IO layer extension straightforward.
- `combine_periods` is ~50 lines numpy.
- GUI integration requires data-browser changes (per-run combo box)
  which is moderate complexity.
- Edge cases (background subtraction conventions, period
  numbering across instruments) need careful documentation.

## Score = impact × ease = **9**

Tier: **Next**. High value for the ISIS audience; should land in the
6-9 month horizon.
