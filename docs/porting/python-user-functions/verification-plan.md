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

## Results

(filled in at implementation completion)
