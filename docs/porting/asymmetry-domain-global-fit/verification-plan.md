# Verification plan

## Unit / behavioural (CI)

`tests/test_asymmetry_global_fit.py` (synthetic, deterministic):

1. **Global recovery** — inject a known shared `lambda` and per-dataset `amp`
   across ≥ 2 noisy exponential datasets; assert the fitted global value and each
   local value land within tolerance of truth, and `success` is true.
2. **Constraint tightening** — assert σ(global) from `fit_global` < σ(global)
   from independent single-dataset fits on the same data.
3. **Single-dataset equivalence** — `fit_global([ds], …)` matches
   `FitEngine().fit(ds, …)` on fitted value and reduced χ².
4. **Combined reduced χ²** — for well-fit synthetic data with the true σ,
   combined χ²ᵣ ≈ 1 within a generous band; dof equals
   `ΣN_d − N_free_global − Σ N_free_local_d`.
5. **Edge cases** — mismatched param names → `ValueError`/`KeyError`;
   global/local overlap → `ValueError`; fixed param held constant; bounds
   respected; zero/non-finite errors rejected.
6. **Mapping + broadcast inputs** — keyed mapping of datasets and a single
   broadcast `ParameterSet` both work and key results correctly.

Run:

```bash
python tools/harness.py test -- tests/test_asymmetry_global_fit.py
```

## Structural / lint

```bash
python tools/harness.py structural
python tools/harness.py lint
```

`structural` confirms the new module keeps the core import boundary (no Qt /
matplotlib / `asymmetry.gui`).

## Docs

```bash
python tools/harness.py docs
```

Builds the new asymmetry-domain global-fit user-guide page (placed distinctly
from the in-flight count-domain `global_fit_wizard.rst` docs PR to avoid a
conflict).

## Oracle

Ground truth is the injected synthetic parameters; the statistical-tightening
claim is verified relatively (global σ < independent σ) rather than against an
external number. No external corpus is required to re-run the verification.
