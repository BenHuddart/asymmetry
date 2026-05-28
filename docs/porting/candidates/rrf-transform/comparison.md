# RRF transform: comparison

| Aspect | Mantid | musrfit | WiMDA | Asymmetry |
|---|---|---|---|---|
| Algorithm exposure | ★ standalone `RRFMuon` algorithm | ◐ alternative run type | ❌ | ❌ |
| In-phase + quadrature output | ✅ | ✅ | n/a | n/a |
| Low-pass filter | ✅ user-configurable | ◐ implicit via sample rate | n/a | n/a |
| Reference-freq input | explicit parameter | from RUN block | n/a | n/a |
| Integration with fit panel | ◐ post-RRF workspace → general Fit | ✅ first-class run type | n/a | n/a |

## Algorithm sketch

```python
def rrf(
    dataset: MuonDataset,
    reference_freq_mhz: float,
    *,
    lowpass_us: float | None = None,
) -> tuple[MuonDataset, MuonDataset]:
    """Return (in_phase, quadrature) RRF-demodulated views.

    The reference frequency is typically γ_μ · B_app for vortex-lattice
    or Knight-shift workflows. lowpass_us defaults to half the period
    of the reference frequency.
    """
    omega = 2.0 * np.pi * reference_freq_mhz
    complex_signal = dataset.asymmetry * np.exp(-1j * omega * dataset.time)
    # Low-pass: simple FIR rectangular average over `lowpass_us` window
    ...
    in_phase = MuonDataset(
        time=dataset.time,
        asymmetry=complex_signal.real,
        error=dataset.error / np.sqrt(2),
        metadata={**dataset.metadata, "rrf_reference_mhz": reference_freq_mhz,
                  "rrf_component": "in_phase"},
    )
    quadrature = MuonDataset(..., metadata={..., "rrf_component": "quadrature"})
    return in_phase, quadrature
```

## Edge cases to document

- Aliasing at `2 ω_ref`: the low-pass cutoff must be below
  `2 ω_ref` minus a margin.
- Phase convention: musrfit and Mantid disagree on the sign of the
  imaginary component; document the choice and verify against one
  of the reference outputs.
- Error propagation: the `/sqrt(2)` factor in the per-bin error
  is the standard reduction from independent ±90° components.
