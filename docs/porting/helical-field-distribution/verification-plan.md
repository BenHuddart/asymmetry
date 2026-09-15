# Verification plan

1. Accuracy: powder and crystal lines agree with the helix-phase average to
   1e-11 across both evaluation paths, phases and orientations; small ratios
   agree with the B² reference to 1e-10.
2. Limits and identities: r = 0 (J₀, J₀/H₀ with phase), r = 1 (cosine), the
   closed-form W₀, magic-angle crystal = powder, `HelicalPowder` at r = 0 =
   `OverhauserPowder`, continuity across the r = 1e-6 switch, ratio > 1
   relabelling, arbitrary time order and negative times, t = 0 normalisation.
3. Registry: ZF geometry, magnetism class, moderate cost, `phase` fixed by
   default, parameter metadata for `theta_h`/`phi_h`, composite parse and
   serialisation round trip, docs placement and applicability text
   (registry-wide tests).
4. Cost (benchmark, not a test): per-curve time on the grids in `comparison.md`.

Steps 1–3 live in `tests/core/test_helical.py` plus the registry-wide tests in
`test_parameter_metadata.py`, `test_fit_function_docs.py` and
`test_latex_preview.py`.
