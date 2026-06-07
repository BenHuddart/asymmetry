# Period (red/green) selection — porting study

Status: **implemented** (core API + GUI refactor + photo-µSR validation).

## Feature

Pulsed-source muon runs can be recorded in *period mode*: a single `.nxs` file
holds several period histograms — for example light-OFF / light-ON in a
photo-µSR experiment, RF-on / RF-off, or ALC steps. Analysts need to select a
single period (e.g. "give me the light-OFF spectrum") and/or combine the two
(green − red, green + red).

This is the WiMDA "RG box" capability. Before this study it existed **only** in
the Asymmetry GUI (`grouping_dialog.py` + `mainwindow.py`), which violated the
engineering invariant that analysis behaviour lives in `asymmetry.core` and the
GUI wraps it. The port moves the rule into core and refactors the GUI to call
it, and adds a scriptable entry point so period-mode data can be analysed from
scripts.

## Reference programs

- **WiMDA** — the "RG box" (Red / Green / G−R / G+R) in the grouping window.
- **Mantid** — `LoadMuonNexus`/`MuonPreProcess` expose periods as separate
  workspaces in a workspace group; arithmetic is done with
  `MuonProcess`/`Plus`/`Minus`. (Period *combination* arithmetic, not a single
  "RG" control.)
- **musrfit** — handles RF/ALC period structure at the `RUNS`/`addrun` level;
  no single red/green widget.

See [comparison.md](comparison.md) for the cross-program comparison and
[implementation-options.md](implementation-options.md) for the design chosen.

## Outcome

- New core module `src/asymmetry/core/io/periods.py`:
  - `select_period(data, period)` — scriptable per-period `MuonDataset`.
  - `load(filepath, period=...)` convenience kwarg in `asymmetry.core.io`.
  - `period_count` / `period_labels` / `resolve_period_index` helpers.
  - `select_period_histograms` + `combine_period_asymmetry` — the low-level
    rule now shared with the GUI (single implementation).
- GUI `mainwindow.py` refactored to call the core helpers instead of holding
  the logic.
- Validation against photo-µSR silicon (runs 103277–103298): see
  [verification-plan.md](verification-plan.md) and
  [`validate_photomusr.py`](validate_photomusr.py).
