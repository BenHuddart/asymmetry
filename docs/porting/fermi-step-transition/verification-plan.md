# Verification plan

1. Formula: plateaus, midpoint, 10/90 % levels at Tc ∓ ln 9 · dT.
2. Reference parity: Mantid `SmoothTransition` code, Khasanov 2020 Eq. 4, tanh
   form — agreement to 1e-12 relative.
3. Numerical safety: dT at its 1e-6 K floor over a grid containing T = Tc gives
   finite values with warnings promoted to errors.
4. Registry: temperature scope only, "Critical behaviour" category, parameter
   names/formula string, metadata (signed plateaus, positive dT floor, indexed
   LaTeX), mathtext-safe LaTeX (registry-wide `test_latex_preview`).
5. Seeds: rising and falling steps, unsorted input with NaN, flat trace, repeated
   component suffixing, merge into `suggest_model_seeds`.
6. End to end: fit from the data-aware seeds recovers the synthetic parameters.

All of the above live in `tests/core/test_fermi_step_parameter_model.py` (plus the
registry-wide tests in `test_parameter_models.py`, `test_parameter_metadata.py`,
`test_latex_preview.py`).
