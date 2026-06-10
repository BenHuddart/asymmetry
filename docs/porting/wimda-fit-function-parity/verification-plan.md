# Verification Plan: WiMDA Fit Function Parity

## Per-component acceptance criteria

Every new component must pass, before merge:

1. **Shape/finiteness**: output shape = input shape; finite for t ∈ [0, 32] µs
   across the full default parameter ranges and at parameter bounds.
2. **t = 0 normalization**: component (excluding amplitude) evaluates to 1 at
   t = 0 (MS-Intro §15.4: all relaxation functions satisfy G(0) = 1), except
   where a phase parameter makes cos φ ≠ 1 by construction.
3. **Limit identities** (the specific ones listed in `test-data.md`):
   reductions to existing verified components at parameter limits agree to
   ≤ 1e-9 absolute (closed forms) or ≤ 1e-6 (numerical solvers/quadrature).
4. **Golden values**: agreement with the independent reference implementation
   to ≤ 1e-8 (closed forms) / ≤ 1e-4 relative (grid-based solvers) on a fixed
   parameter/time lattice.
5. **WiMDA agreement** (where a WiMDA trace exists): ≤ 1e-3 absolute on the
   normalized polarization, except where a deliberate departure is documented
   in `comparison.md` (deg→rad, positive frequency, quadrature scheme, Kadono
   re-derivation, powder average) — those record the expected deviation
   instead.
6. **Registry hygiene** (extends existing structural tests): entry in
   `COMPONENTS` with category and domain; `ParamInfo` (with unit and
   `default_min`) for every parameter; applicability text in
   `component_docs.py` citing MS-Intro equation numbers or the flagged source;
   `latex_equation` and `formula_template` present.
7. **Serialization round-trip**: `CompositeModel` containing the component
   survives `to_dict`/`from_dict` and a `.asymp` save/load.
8. **Fit recovery**: synthetic data generated from the component + noise is
   fitted by the engine recovering input parameters within 2σ (smoke test per
   component, seeded).
9. **Performance**: cached components evaluate a 2000-point trace in < 50 ms
   warm / < 2 s cold (same budget the dynamic KT port used); cache keys
   quantized per existing patterns.

## Convention checks (one-off)

- Phase handling: all new oscillatory components use radians and the
  positive-frequency convention; cross-check against `muonium.py`'s documented
  convention by fitting the same synthetic trace with WiMDA-convention phase
  converted.
- Gyromagnetic constants come from `core/utils/constants.py` only (no inline
  0.01355342-style literals); structural grep test.

## GUI verification

- Component picker shows the agreed categories; every new component appears
  exactly once; info button shows the applicability text (manual smoke +
  existing picker unit tests extended).
- Parameter table renders units and enforces `default_min` bounds.
- `gui-smoke` harness passes.

## Process

- Implementation tracked against this study; each landed component ticks a row
  in a checklist appended to `README.md` at implementation time.
- Full `python tools/harness.py validate` green before PR.
- Update this study's `comparison.md` with measured WiMDA deviations and the
  resolution of the Kadono expression discrepancy; update `index.json` status
  `study → implemented` in the implementation PR.
