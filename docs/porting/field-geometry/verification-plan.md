# Field geometry — verification plan

For the future implementation pass (not this study). The change is small and
data-driven, so verification is mostly golden-value assertions on the loader.

## Unit tests (beside the loader)

Add to the NeXus loader test module. Each loads a golden file and asserts the
resolved metadata:

1. **TF mislabel fixed** — `EMU00018850.nxs`:
   `field_state == "TF"`, user-facing geometry == "Transverse",
   `detector_orientation == "Longitudinal"`, `field_geometry_source == "field_state"`.
2. **TF at zero field, not ZF** — `MUSR00044991.nxs`:
   `field_state == "TF"`, geometry == "Transverse" (NOT "Zero field"),
   despite `field == 0`.
3. **Genuine LF** — `56426.nxs`: `field_state == "LF"`, geometry == "Longitudinal".
4. **Fallback to orientation** — `emu00124218.nxs` (no field state):
   `field_state == ""`, geometry == "Longitudinal",
   `field_geometry_source == "orientation"`.
5. **Synthetic ZF unit test** — craft an in-memory/temp h5py file with
   `magnetic_field_state == "ZF"` and assert geometry == "Zero field" (no real
   ZF file exists in the corpus).
6. **Blank/unknown state** — crafted file with `magnetic_field_state == "n/a"`
   (or empty) → falls through to orientation; does not raise.
7. **Orientation preserved** — assert `detector_orientation` is present and
   distinct from the geometry on a TF/L file.

## Regression / parity checks

- **No existing test breaks**: run `python tools/harness.py test --
  tests/test_*nexus*` (and any GUI/project tests that read `field_direction`).
- **Project round-trip**: if `field_direction` is persisted in `.asymp`, load an
  old project and confirm it still opens (schema/back-compat). Grep first:
  `grep -rn field_direction src/ tests/`.
- **GUI surface**: confirm the panel that shows field direction now shows the
  field-state-derived geometry, with the orientation visible separately (or at
  least not lost).

## Cross-program parity note

There is **no reference-program oracle** for the corrected behaviour — Mantid,
musrfit, and WiMDA all do something different (none reads `magnetic_field_state`).
So verification is against the **file contents themselves** (the
`magnetic_field_state` string is ground truth), not against another program's
output. Document this explicitly in the implementation PR so reviewers don't
expect Mantid-identical labels.

## Validation ladder

```bash
python tools/harness.py structural
python tools/harness.py lint
python tools/harness.py test -- tests/<nexus test file>
```

GUI-affecting change → also `python tools/harness.py gui-smoke`.
