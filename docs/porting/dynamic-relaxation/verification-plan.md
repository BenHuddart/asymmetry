# Dynamic relaxation: verification plan

How the implementation PR will be proven correct and "to the standard of the
existing relaxation functions" (clearly referenced; parameters with units; clean
equation + physical description in the info helper).

## 1. Correctness (pytest — `tests/test_fitting_models.py` + new cases)

- **Analytic limits L1–L9** (test-data.md §1) asserted to tight tolerances.
- **Property tests** (§2): finiteness, envelope bounds, ν- and B_L-monotonicity,
  continuity of the ν→0 limit (dynamic == static within 1e−4).
- **Internal cross-checks** (§3): Keren(ZF) == Abragam form; dynamic-KT vs Keren
  in the fast regime.
- **Round-trip fits** (§4) via `FitEngine` / `global_fit` recover known params.
- **Grid independence:** dynamic-KT result stable (< 0.5 %) when the internal
  time step is halved.

## 2. Registration & metadata (the "high standard" checklist)

- Present in **both** registries: `MODELS` (models.py) and `COMPONENTS`
  (composite.py); `CompositeModel.from_expression("DynamicGaussianKT + Constant")`
  builds and evaluates.
- `to_model_definition().param_info` returns a `ParamInfo` for every parameter
  with the **correct unit** (Δ,a,σ → µs⁻¹; ν → MHz; B_L → G; A → %) and a
  non-empty description.
- `latex_equation` renders (assert non-empty, parses) and the **info-helper**
  applicability note exists in `component_docs.py`.
- Each component **description cites its paper** (Hayano 1979 / Uemura 1985 /
  Keren 1994 / Abragam 1961) — assert the citation substring is present.
- Fit-wizard "static vs dynamic KT" portfolio now offers the dynamic branch.

## 3. Repo validation ladder (tools/harness.py)

```
python tools/harness.py structural
python tools/harness.py lint
python tools/harness.py test -- tests/test_fitting_models.py tests/test_composite*.py
python tools/harness.py validate     # full suite before PR (slow; ~30–60 min)
```

GUI smoke (`gui-smoke`) to confirm the new components appear in Build Fit Function
and the info helper renders the equation + description.

## 4. Numerical-accuracy oracle (optional, reference-only)

Cross-plot against Mantid `DynamicKuboToyabe` on identical synthetic inputs
(GPL-3 → comparison only, no code copied). Agreement target: < 1 % over 0–8 µs
for representative (Δ, ν, B_L).

## 5. Documentation

- Update the user guide LF/KT page to cover the dynamic regime.
- Promote the corpus worked-example stubs (Copper, Ionic motion) once the models
  land, replacing the "blocked by missing model" notes.
- Flip this study's `index.json` status study → implemented when the PR merges.

## Done = all of: limits pass · round-trips recover · metadata/citation checklist
green · harness `validate` green · info helper renders.
