# Phase auto-calibration: comparison

| Aspect | Mantid | musrfit | WiMDA | Asymmetry |
|---|---|---|---|---|
| Algorithm | shared-physics global fit over early-time TF data | manual / external | manual / phase table | manual |
| Output format | TableWorkspace (detector → (A, φ)) | phase column in `.msr` | UI phase-table editor | n/a |
| Fit model | damped cosine with shared f, λ | n/a | n/a | n/a |
| GUI integration | Phase Calculation tab in Muon Analysis | n/a | phase-table dialog | n/a |
| Reference | `Framework/Muon/src/CalMuonDetectorPhases.cpp` | | `src/PhaseTableUnit.pas` | |

## Algorithm sketch

```python
def calibrate_phases(
    dataset: MuonDataset,
    *,
    fit_window_us: tuple[float, float] = (0.0, 4.0),
    initial_frequency_mhz: float | None = None,
) -> dict[int, float]:
    """Return a {group_id: phase_radians} map by globally fitting
    a damped cosine to each detector group's early-time signal.

    The frequency and damping rate are shared across groups; per-group
    amplitude and phase fit locally.
    """
    ...
```

The implementation can reuse the Multi-Group Fit machinery: load the
dataset with grouping, set the model to `Oscillatory + Constant`,
classify (f, λ) as Global and (A, φ, A_bg) as Local, run the fit
over a restricted time window, harvest the per-group φ values.

## GUI surface

Two entry points:

1. **Grouping dialog "Calibrate phases" button** — applies the
   estimate to the active dataset; outputs a phase table that
   feeds the asymmetry calculation.
2. **Multi-Group Fit window "Seed phases from calibration"** —
   uses the calibrated phases as initial guesses before the
   user's chosen fit model runs.

## Edge cases

- Datasets with no clear precession (ZF, very low TF): the fit
  fails to find a frequency; raise a meaningful exception and
  fall back to zero phases with a warning.
- Frequency ambiguity (signal contains two close frequencies):
  output the dominant one; document that the caller can pin
  `initial_frequency_mhz` manually.
- Group with very low statistics: report the per-group fit
  uncertainty and surface it in the phase table.
