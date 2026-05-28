# Simulate mode: comparison

| Aspect | WiMDA | musrfit | Mantid | Asymmetry |
|---|---|---|---|---|
| GUI dialog | ✅ `Simulate.pas` form | ❌ | ❌ | ❌ |
| Hand-edit workflow file | ❌ | ✅ via `.msr` template | ✅ via Python script | ◐ via `archetypes.py` (dev-only) |
| Output as live dataset | ❌ (writes to disk) | ❌ (writes to disk) | ✅ (workspace in ADS) | n/a |
| Per-bin Poisson noise | ✅ | ✅ | ✅ via `Stats=Poisson` | ◐ available in archetypes helper |
| Histogram-level synthesis | ✅ | ✅ | ✅ | ◐ via `_build_run_with_detector_asymmetries` |

## Reference implementation: WiMDA's `Simulate.pas`

Inputs:
- Theory function name (from the `musr-function-registry`)
- Parameter values
- Time range (t_min, t_max, bin width)
- Counts per bin
- Asymmetry amplitude

Output:
- Synthetic count histograms saved to disk in a chosen format
- Side-effect: the same form lets the user re-load the synthetic run
  into the main analysis pipeline

## Proposed Asymmetry implementation

New module: `src/asymmetry/core/simulate.py` containing:

```python
def simulate_asymmetry(
    model: CompositeModel | ModelDefinition,
    parameters: ParameterSet,
    *,
    time: NDArray[np.float64] | None = None,
    n_points: int = 480,
    t_max_us: float = 8.0,
    counts_per_bin: float = 1e5,
    seed: int = 0,
    metadata: dict | None = None,
) -> MuonDataset:
    ...
```

New GUI dialog: `src/asymmetry/gui/windows/simulate_dialog.py`. Reuses
the existing `FitFunctionBuilderDialog` for model selection, then
adds parameter spin-boxes (one per model parameter), time-axis
controls, counts-per-bin slider, and "Create dataset" → adds to
data browser.

The synthesis helpers already in
`docs/screenshots/data/archetypes.py` (`_build_run_with_detector_asymmetries`,
`_poisson_errors`) are the natural backend. Promote them into
`core/simulate.py` as supported public API.

## Edge cases the study should document

- Composite models with bound parameters — clamp the dialog
  spin-boxes to the registered min/max.
- Numerical safety when `A_max → 100%`: `_poisson_errors` already
  clips, but the dialog should warn.
- Reproducibility — every simulated dataset should record its
  `seed`, model expression, and parameter dict in
  `MuonDataset.metadata["simulation"]` so a project file can
  reload the simulated run later.
