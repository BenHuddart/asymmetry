# Verification plan

## Ladder

1. `python tools/harness.py structural` — study layout + index.json entry.
2. `python tools/harness.py lint` — Ruff baseline (src/tests/tools).
3. `python tools/harness.py test -- tests/test_sc_brandt_field_width.py` —
   the new synthetic + analytic-anchor suite.
4. `python tools/harness.py validate` — full suite before PR.

## Acceptance criteria

- Analytic anchors (see `test-data.md`) all pass:
  `g(0)=1` ties to `lambda_nm_to_sigma_us`; `g(1)=0`; powder = single/√3;
  quadrature background; monotonic decrease; guarded λ/B_c2/b≥1.
- Synthetic round-trip recovers λ within a few % and B_c2 when constrained.
- Registered components appear in `component_names_for_x("field")` and fit
  through `fit_parameter_model` like the other `SC_*` models.
- No Qt/matplotlib/`asymmetry.gui` imports in the new core code.
- Docs: `docs/user_guide/sc_penetration_depth.rst` gains a field-dependence
  section with the Brandt formula, the new model names/params, and a worked
  σ(B₀) snippet.

## Manual corpus validation (reported in PR, not CI)

Fit the LiFeAs powder field sweeps and report fitted λ_ab vs the published
195(2) / 244(2) nm, plus the B_rms plateau cross-check (1.91 / 1.22 mT vs
Fig. 1). Record Sample 2's field-induced-magnetism caveat (σ_M ∝ B₀^½ not
modelled here).

## Risks / open questions (resolved in CONSULT)

- σ-domain vs B_rms-domain → recommend σ (matches existing pipeline).
- Powder factor as a second registered component vs prose-only → recommend
  second component (motivating example is a powder).
- Background channel default → quadrature, default 0.
