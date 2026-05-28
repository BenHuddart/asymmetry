# MINOS error analysis: comparison

| Aspect | Mantid | musrfit | WiMDA | Asymmetry |
|---|---|---|---|---|
| Hessian (symmetric) errors | ✅ default | ✅ HESSE | ✅ default | ✅ default |
| MINOS asymmetric errors | ◐ via minimiser opt-in | ★ explicit MINOS command | ❌ | ❌ |
| Likelihood profile (SCAN/CONTOUR) | ◐ via post-processing | ✅ SCAN, CONTOUR | ❌ | ❌ |
| Per-parameter `+err` / `-err` display | ◐ in output workspace | ★ in STATISTIC block | ❌ | ❌ |
| Backend | Levenberg-Marquardt or generic | Minuit2 | FITE LM | iminuit (Minuit2 wrapper) |

## Implementation comparison

**musrfit's MINOS** is invoked from the COMMANDS block:

```
COMMANDS
MIGRAD
HESSE
MINOS
SAVE
```

Each command writes its result back to the `.msr` STATISTIC block;
MINOS appends `+err` / `-err` columns. The user sees them on next
file load.

**Mantid's MINOS** is reached via:

```python
Fit(Function=..., InputWorkspace=..., Output="out",
    Minimizer="Simplex", Errors="MINOS")
```

The output workspace table includes asymmetric errors.

**Asymmetry's current path** uses `iminuit.Minuit.migrad()` followed
by `Minuit.errors`, returning symmetric Hessian uncertainties. The
proposed change adds an `engine.fit(..., minos=True)` parameter that
runs `Minuit.minos()` after MIGRAD and packs the result into
`FitResult.errors_minos`.

## API sketch

```python
@dataclass
class FitResult:
    parameters: ParameterSet
    uncertainties: dict[str, float]            # Hessian (existing)
    errors_minos: dict[str, tuple[float, float]] | None  # new
    ...

class FitEngine:
    def fit(self, dataset, model_fn, params, *, minos: bool = False) -> FitResult:
        ...
        if minos:
            m.minos()
            errors_minos = {
                name: (m.merrors[name].upper, m.merrors[name].lower)
                for name in params.names
            }
```

## UI surface

- Fit panel: add a "MINOS errors" checkbox below the "Fit" button.
- Parameter table: when MINOS errors are present, render the value
  cell as `0.392 +0.011 / -0.013` instead of `0.392 ± 0.012`.
- Result text: extend the success summary to include MINOS rows
  when available.

## Edge cases the study should document

- MINOS can fail (parameter hit bound, fit didn't converge). Fall
  back to Hessian errors with a warning log.
- MINOS is ~10x slower than HESSE on typical fits — make sure the
  GUI shows a progress indicator.
- Fixed parameters should be excluded from MINOS automatically.
