# Verification Plan: Model Function Parity

Acceptance is the harness ladder green at each phase
(`python tools/harness.py validate`, ~2 min; `.venv/bin/python`;
`QT_QPA_PLATFORM=offscreen` for GUI tests) plus the checks below. Oracle
values and datasets: [test-data.md](test-data.md).

## Phase 1 — functions

1. **WiMDA-transcribed oracles** (new tests in
   `tests/test_wimda_parity_components.py`-style module): `Polynomial`,
   `PowerLawQuadBG`, `MuRepolarisation` reproduce the transcribed
   `fitfunctions.pas` values (test-data §1.1–1.3) to ≲1e-12 where the forms
   are identical, with each documented divergence (comparison.md D1–D7)
   asserted as a behavioural test, not prose.
2. **Polynomial exact round-trip**: coefficients recovered ≲1e-6 on exact
   data, full quintic and quadratic-with-fixed-tail variants.
3. **MuRepolarisation physics**: B₀ derived from `constants.py` equals
   A/(γₑ+γ_μ) (vacuum-Mu check ≈ 1585 G); exact-synthetic recovery of
   (a_Mu, A_hf, a_Dia); limits y(0) = a_Mu/2 + a_Dia, y(∞) → a_Mu + a_Dia.
4. **Composite recipes**: Arrhenius constant-delta identity (D4 pinned
   numerically); `OrderParameter + Constant` ≡ WiMDA `func5` on the physical
   domain incl. clamp at/above Tc; LorentzianLCR ≡ WiMDA peak term.
5. **EuO regression**: `OrderParameter + Constant` on the EuO runs
   (2928–2943) trend reproduces Tc = 69.2(1) K, β = 0.417(7), α = 1.23(5)
   within quoted uncertainties (scripted via the core API against the
   fitted-trend fixtures, not a GUI test).
6. **Registry/docs enforcement**: new components carry applicability text
   (`component_docs.py`) with no inline citations, APS-style entries in
   `PARAMETER_MODEL_REFERENCES`, ParamInfo for every parameter, correct
   scopes (`Polynomial`/`PowerLawQuadBG` common, `MuRepolarisation` field);
   the existing docs-policy tests (`test_fit_function_docs.py`) pass over
   the enlarged registry.
7. **User-guide build**: `python tools/harness.py docs` clean with the new
   `parameter_trending.rst` sections (each new function gets a result-first
   "when to use this" register; WiMDA migration notes incl. the
   2-Lorentzian coefficient non-transfer and the eV→meV delta).

## Phase 2 — machinery

1. **Error modes**: behavioural tests per mode (test-data §1.7), incl. the
   floor-only-in-Column rule (D10) and Percent zero-y masking (D9).
2. **Estimate from scatter**: fixed-point equivalence test (test-data §1.6)
   — one-pass result equals the explicitly-iterated WiMDA scheme's limit;
   χ²ᵣ verdict suppressed in this mode in the dialog.
3. **Union multi-range**: synthetic λ(T) divergence with the critical region
   excluded recovers the generating parameters (test-data §1.8); mask
   semantics (OR, fallback, empty-union guard) unit-tested; EuO λ(T)
   qualitative corpus check.
4. **χ² quality helper**: band values match `scipy.stats.chi2.ppf` oracle
   table; verdict thresholds match WiMDA semantics (CDF two-sided at
   R = 0.95); ν = 0 and failed-fit edge cases; helper importable without Qt
   (core purity).
5. **GUI**: `ModelFitDialog` offscreen tests — error-mode selector drives
   `fit_parameter_model` inputs; window editor round-trips
   `ModelFitRange.windows`; χ² verdict label + teaching tooltip render;
   cross-group dialog unaffected (no new controls there).
6. **Core purity**: no Qt/matplotlib imports introduced into
   `asymmetry.core` (structural harness).

## End state

- Full `validate` green after each phase; milestone commit per phase; no
  push (standing instruction).
- Study docs updated with final decisions and verification outcomes;
  follow-ons recorded in implementation-options.md (results-table export,
  arbitrary X if not reached, .asymp persistence of model fits, cross-group
  GUI exposure, generic quadrature combinator).
