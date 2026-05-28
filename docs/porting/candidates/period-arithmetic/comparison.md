# Period arithmetic: comparison

| Aspect | Mantid | WiMDA | musrfit | Asymmetry |
|---|---|---|---|---|
| Multi-period load | ✅ workspace group | ✅ via `period-mapping` | ◐ via separate `.msr` runs | ◐ default period only |
| Sum / subtract syntax | ★ `"1+2-3"` strings | ◐ red/green form controls | ❌ | ❌ |
| Per-period grouping | ✅ | ✅ via radio buttons | ❌ | ❌ |
| Reference | `MuonProcess`, `LoadMuonNexus*.cpp` | `PeriodMappingUnit.pas` | | `core/io/nexus.py` |

## Proposed Asymmetry implementation

- Extend `core/io/nexus.py` to expose all periods, not just the
  default one. Return a list of `MuonDataset` instances tagged with
  `metadata["period"]`.
- Add `core/transform/periods.py` with:
  ```python
  def combine_periods(datasets: list[MuonDataset],
                     expression: str) -> MuonDataset:
      """Combine periods using a +/- expression like '1+2-3'."""
  ```
- GUI surface: a per-run "Periods" combo box in the data browser
  with the most common expressions pre-populated.

## Edge cases

- Period mismatch in shape (different bin counts): raise with a
  clear error.
- Background subtraction conventions vary across instruments;
  document one canonical interpretation and surface the others as
  options.
