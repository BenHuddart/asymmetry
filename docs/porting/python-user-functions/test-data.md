# Test data — python-user-functions

This feature needs **no external corpus**: every behaviour is exercised
with synthetic plugins and synthetic data generated in-test. The WiMDA
sources cited in [comparison.md](comparison.md) (`$WIMDA_SRC/src`) are
needed only to re-trace the study, never by the test suite — so no
env-var-gated tests (the `tests/test_psi_loader.py` /
`ASYMMETRY_MUSRFIT_DATA` pattern) are required here.

## Probe grids (validation)

- Time domain: `np.linspace(0.0, 32.0, 257)` µs — includes t = 0 (catches
  1/t singularities) and late times (catches overflow in exponents).
- Frequency domain: `np.linspace(0.0, 50.0, 257)` MHz.
- Parameter models: the x grids already used by parameter-model tests
  (temperature/field spans including 0).

## Synthetic plugin fixtures

Written by tests into a `tmp_path` directory passed to
`load_user_functions(directory=...)`:

| Fixture | Purpose |
|---|---|
| `good_relaxation.py` | registers a valid component; appears in registry + report |
| `keren_clone.py` | the shipped example plugin (same file as the tutorial listing) |
| `bad_signature.py` | wrong arity → designed `UserFunctionError` at load |
| `nan_output.py` | NaN on the probe grid → designed error at load |
| `name_collision.py` | re-registers `Keren` (and a `PARAMETER_MODEL_COMPONENTS` name, cross-registry) → rejected, registries untouched |
| `missing_metadata.py` | empty description/formula → designed error |
| `missing_domain.py` | no/invalid domain tag → designed error |
| `raises_on_import.py` | module-level exception → captured in report, other files still load |

## Bit-for-bit parity reference

The `Keren` parity test compares the plugin-registered clone against the
built-in `COMPONENTS["Keren"].function` with `np.array_equal` (exact, not
allclose) on the probe grid over a parameter sweep
(`Delta ∈ {0.2, 0.5, 1.5}`, `nu ∈ {0.1, 1, 10}`, `B_L ∈ {0, 20, 100}` G),
then through an end-to-end fit of synthetic Keren data (same seeds → same
result) and an `.asymp` save/load round-trip.

## Degrade-path data

A project written in-test with the user component registered, then
reloaded in a session where it is not: the fit-slot model dict must
round-trip byte-identically through load → save with the plugin absent.
