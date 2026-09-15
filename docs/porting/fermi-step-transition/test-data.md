# Test data

No real-data corpus entry yet; the model is verified against transcribed
formula oracles and synthetic data.

- **Mantid oracle**: `SmoothTransition.cpp` `function1D` transcribed
  (`test_fermi_step_matches_mantid_smooth_transition_code`).
- **Literature identities**: Khasanov 2020 Eq. 4 and the tanh form
  (`test_fermi_step_matches_khasanov_wtf_form`, `..._tanh_form`).
- **Synthetic wTF step**: A1 = 0.112, A2 = 0.223, Tc = 50.3 K, dT = 0.8 K on a
  1 K grid, 0.002 Gaussian noise (seed 7) — seed accuracy and fit recovery.
- **Falling, unsorted step with a NaN abscissa** — seed robustness.

Candidate real data: any wTF temperature scan through a magnetic transition
(e.g. a future corpus scenario); none is currently in `docs/screenshots/data`.
