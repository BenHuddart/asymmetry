# Moments analysis: comparison

| Aspect | WiMDA | musrfit | Mantid | Asymmetry |
|---|---|---|---|---|
| Available | ✅ `Moments.pas` | ❌ | ❌ | ❌ |
| Moments computed | m₀, m₁, m₂, α-width, β-width, peak position | n/a | n/a | n/a |
| Cutoff & range controls | ✅ form-driven | n/a | n/a | n/a |
| Per-run averaging | ✅ | n/a | n/a | n/a |

## Algorithm sketch

```python
def moments(freq: NDArray, spectrum: NDArray, *,
            cutoff: float | None = None,
            f_min: float | None = None,
            f_max: float | None = None) -> dict[str, float]:
    """Return m0, m1, m2, alpha_width, beta_width, peak_freq.

    Standard textbook definitions:
    m_n = ∫ (f - f0)^n · S(f) df / ∫ S(f) df
    alpha_width = sqrt(m2)
    """
```

Window selection is the only subtlety: WiMDA uses an interactive
cutoff slider. Asymmetry can expose the same as a Fourier-panel
control with sensible auto-defaults.
