# Period selection — cross-program comparison

## How each program models periods

| Program | Period model | Selection / combination | Notes |
| --- | --- | --- | --- |
| **WiMDA** | Period histograms inside one raw file | "RG box" radio set: Red, Green, G−R, G+R | Red = first period, Green = second period. Difference/sum formed on the reduced asymmetry. |
| **Mantid** | One workspace per period in a `WorkspaceGroup` | `MuonProcess`/`Plus`/`Minus`; "SummedPeriodSet"/"SubtractedPeriodSet" | General N-period; combination is explicit workspace arithmetic. |
| **musrfit** | `RUNS` / `addrun` at the msr level | Sum/append of runs; RF/ALC handled per run block | No single red/green control. |
| **Asymmetry (before)** | 2-period file collapsed to one `MuonDataset`; 3+ → `list` | GUI-only RG box in `grouping_dialog.py` | Logic duplicated in the GUI; not scriptable. |

## Asymmetry's existing loader behaviour (the contract to preserve)

`src/asymmetry/core/io/nexus.py`:

- `_split_period_counts` splits the raw `counts` dataset: 2-D `[det, bin]` →
  one period; 3-D `[period, det, bin]` → one array per period.
- `_build_period_datasets` reduces **each period independently**: per-period
  grouping (`apply_grouping`), `compute_asymmetry(alpha=1.0)`, ×100 to percent,
  then the good-bin window. t0/good-bin/grouping are the same across periods;
  `good_frames` and `dead_time_us` are split per period.
- For a **2-period** file, `_combine_two_period_datasets` merges the two into a
  single `MuonDataset` whose visible arrays are the **Red** (period 1) spectrum,
  and stashes both periods on `run.grouping`:
  - `period_histograms = [red_hist, green_hist]` (index 0 = red, 1 = green),
  - `period_good_frames`, `period_dead_time_us` (per period),
  - `period_mode` (default `"red"`).
- For **3+** periods the loader returns a `list[MuonDataset]`, one per period
  (`period_number` 1..N).

## The RG arithmetic (from the GUI, now the contract for core)

From `mainwindow.py` (`_reduce_asymmetry_for_dataset` + `_period_histograms_for_mode`):

- **Red**: period index 0. **Green**: period index 1.
- **G − R**: `green_asym − red_asym`, error `sqrt(green_err² + red_err²)`,
  arrays truncated to the common length, red's time axis used.
- **G + R**: `green_asym + red_asym`, same error rule.
- Selecting a period also swaps in that period's `good_frames` and
  `dead_time_us` before reduction.

## Physical mapping (photo-µSR)

For the photo-µSR silicon experiment the guide maps **light-OFF → Green**
(period 2) and **light-ON → Red** (period 1). Confirmed empirically on run
103277: the Red period relaxes substantially more than the Green period (see
[verification-plan.md](verification-plan.md)). The core API keeps the neutral
`red`/`green` labels; the light-OFF/ON mapping is documented, not baked in,
because it is experiment-specific (RF/ALC have no light state).
