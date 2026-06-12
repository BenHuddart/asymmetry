# Verification plan — python-user-functions

All checks run in the standard suite (no external data, no env gates).

## 1. Keren parity through the plugin path

- Registration: the example plugin registers via `register_component`;
  definition fields (params, defaults, category, domain) match intent.
- **Bit-for-bit**: plugin clone vs `COMPONENTS["Keren"].function` with
  `np.array_equal` on the probe grid across the parameter sweep in
  [test-data.md](test-data.md).
- Picker: the clone appears in the time-domain component set
  (`components_for_domain("time")`) flagged `user=True`.
- Fit: identical synthetic-data fit results (values and uncertainties)
  for built-in vs clone models.
- Persistence: `.asymp` round-trip of a model using the clone restores it
  exactly while the plugin is registered.

## 2. Load-failure behaviour (designed errors at load, never mid-fit)

For each bad fixture (bad signature, NaN on probe grid, name collision
with a built-in, cross-registry name collision, missing metadata,
missing/invalid domain tag, module raising on import):

- `load_user_functions` returns a report entry carrying the designed
  error; nothing raises out of discovery.
- All registries are byte-identical to their pre-call state (atomic
  validation; isolation).
- Remaining good files in the same directory still load.

## 3. Guards and docs-enforcement tests with user components loaded

- `domain_library` import-time guard passes (registration is post-import
  and domain-validated, so `FREQUENCY_COMPONENT_NAMES` semantics hold).
- The three docs-enforcement tests in `tests/test_fit_function_docs.py`
  pass with user components registered — exemption **by the `user` flag**,
  not name lists (W17).
- `test_name_collisions_resolve_by_registry_kind` passes unchanged (N4
  scheme keeps the grandfathered `Constant` behaviour).

## 4. Project round-trip with the component absent (W1)

- Load an `.asymp` referencing an unregistered user component: model opens
  with named zero-valued placeholders; no silent model substitution
  (regression test against the old `restore_state` fallback).
- Fitting blocked with a message naming the missing components.
- Save again: serialized model dict byte-identical to the loaded one.
- Re-register the plugin, reload: model fully live again.

## 5. Registry isolation

- Facade refuses overwrites of built-ins and of other user components.
- A failed registration mutates nothing.
- `user=True` definitions never replace or shadow `user=False` ones.

## 6. GUI surfaces

- Picker shows the User badge for `user=True` components (offscreen Qt
  test); component-info dialog shows plugin-supplied applicability and
  references through the kind-aware lookup.
- Setup → "User functions…" dialog lists loaded/failed sources from the
  report.
- Startup log lines emitted for failures (gui-smoke stays green).

## 7. Whole-suite

- `python tools/harness.py validate` green (lint + structural + full
  pytest, GUI tests under `QT_QPA_PLATFORM=offscreen`).
- `python tools/harness.py docs` green with the tutorial page in the
  toctree.

## Results (2026-06-12, implementation complete)

All seven sections verified; the suite additions live in
`tests/test_user_functions.py` (facade, discovery, degrade, Keren parity —
34 tests) and `tests/test_user_functions_gui.py` (fit-tab degrade, badges,
load-report dialog — 8 tests), with the three docs-enforcement tests in
`tests/test_fit_function_docs.py` now sweeping built-ins by the `user`
flag.

- §1 Keren parity: bit-for-bit (`np.array_equal`) on a 1025-point grid
  across the Δ/ν/B_L sweep; **exact** fit-result equality (values and
  uncertainties) against the built-in; picker presence; persistence
  round-trip. The example plugin loads through the real discovery path.
- §2 All seven bad-fixture classes produce the designed
  `UserFunctionError`/report entries; registries verified byte-identical
  after each failure; sibling files still load.
- §3 Guard + docs tests green with user components registered; exemption
  verified to be flag-based (the test asserts the user category is *not*
  in `CATEGORY_PAGES`).
- §4 Round-trip with the component absent: model dict bit-identical
  through load → save; placeholder evaluates to zero; both fit tabs block
  with the designed message; re-registration revives the model. The old
  silent default-model fallback is regression-tested.
- §5 Overwrites refused in both directions; atomicity tested.
- §6 Badge, info-dialog provenance, Setup dialog, log lines all covered;
  gui-smoke green.
- §7 `validate` green (see PR); `docs` builds clean (only pre-existing
  screenshot warnings).
