# Field geometry — implementation options & recommendation

This is the study pass. No code changes were made. Below are the options weighed
for the future implementation pass, and the recommendation.

## The candidate fix (from the task)

> Prefer `sample/magnetic_field_state` for geometry; keep detector orientation as
> a separate field; fall back to orientation only when the field state is absent.

## Options considered

### Option A — Match Mantid exactly (keep using `orientation`)
Keep deriving the user-facing geometry from detector `orientation`, as today and
as Mantid does.

- **Pro:** Mantid-parity; zero new fields.
- **Con:** Does **not** fix the bug. 1731/2050 corpus files stay mislabelled
  "Longitudinal" for TF runs. Mantid only gets away with this because it *names*
  the value `main_field_direction` (instrument axis) and never presents it as the
  per-run TF/LF/ZF experiment state — Asymmetry does present it that way.
- **Verdict:** Rejected — preserves the conflation.

### Option B — Replace `field_direction` source with `magnetic_field_state`
Read geometry only from `sample/magnetic_field_state`; drop the orientation read.

- **Pro:** Correct for modern ISIS files; simplest "fix the value" change.
- **Con:** Loses the orientation value entirely (used for default grouping, and
  the only signal on PSI / legacy files that lack `magnetic_field_state` — 137
  corpus files + all PSI). Regresses information Mantid keeps.
- **Verdict:** Rejected — throws away the legacy fallback and the instrument axis.

### Option C — Two distinct fields + prefer-state-with-fallback (RECOMMENDED)
Refine the candidate fix:

1. **Keep** the orientation read, but rename the metadata key to its true meaning
   — `detector_orientation` (or `main_field_direction`, matching Mantid). This is
   an **instrument-build** property, not the experiment geometry.
2. **Add** a new metadata key `field_state` populated from
   `sample/magnetic_field_state` (V1: `run/sample/...`; V2: `raw_data_1/sample/...`),
   normalised to `'TF'`/`'LF'`/`'ZF'` (uppercase, 2-letter).
3. The **user-facing field geometry** = `field_state` when present and non-blank,
   else fall back to the orientation-derived value, **and record which source was
   used** (provenance flag, e.g. `field_geometry_source ∈ {field_state, orientation}`).
4. **Never** synthesise ZF from a zero applied-field magnitude. Trust the string.
   (`MUSR00044991.nxs` is `'TF'` at 0 G.)

- **Pro:** Fixes the bug; keeps Mantid-parity for the orientation value and for
  the legacy/PSI fallback; strictly more correct than any reference program;
  provenance makes the value auditable in the GUI.
- **Con:** Two new metadata keys + GUI surface for them; a little more code than B.
- **Verdict:** **Recommended.** This is the candidate fix, refined with (a) an
  explicit rename of the orientation value, (b) a provenance flag, and (c) the
  "trust the string, never infer ZF from field=0" rule.

## Detailed recommendation for the implementation pass

### Reading & normalisation
- V1 (`_load_v1`): read `sample/magnetic_field_state`
  (`entry/sample/magnetic_field_state`).
- V2 (`_load_v2`): read `entry/sample/magnetic_field_state` (note: `sample` is
  already fetched at [`nexus.py:293`](../../../src/asymmetry/core/io/nexus.py)).
- Normalise: strip + uppercase; accept `'TF'`/`'LF'`/`'ZF'`. Treat empty / `'n/a'`
  / unknown as **absent** (fall through to orientation), not as an error.
- Keep `_normalise_orientation` for the orientation value; also handle lowercase
  `'l'` (already does via `.upper()`) — the corpus contains `'l'`.

### Metadata shape (proposed)
- `detector_orientation`: "Longitudinal"/"Transverse" (was `field_direction`).
- `field_state`: "TF"/"LF"/"ZF" or `""`.
- `field_direction` (the user-facing geometry, kept as the existing key for GUI
  compatibility): mapped from `field_state` when present
  (`TF`→"Transverse", `LF`→"Longitudinal", `ZF`→"Zero field"), else the
  orientation value.
- `field_geometry_source`: `"field_state"` | `"orientation"` | `""`.

> Decide during implementation whether to repurpose the existing `field_direction`
> key (less GUI churn) or introduce a new key and migrate GUI usages. Grep GUI +
> project-serialisation for `field_direction` first; it may be persisted in
> `.asymp` projects, which would make a rename a schema concern.

### Fallback ladder (per file)
1. `sample/magnetic_field_state` present & recognised → use it; source=`field_state`.
2. Else detector `orientation` present → map L/T; source=`orientation`.
3. Else unknown (`""`); do **not** guess from field magnitude.

PSI `.bin`/`.mdu` and HDF4 v1 NeXus: no `magnetic_field_state` → step 2/3. HDF4
remains unreadable by h5py and is out of scope ([[project_hdf4_loader_gap]]).

### Things to explicitly NOT do
- Do not infer ZF from `magnetic_field == 0`.
- Do not assume `'ZF'` exists in ISIS files — it does not appear in the corpus;
  handle it if present but don't synthesise it.
- Do not drop the orientation value — keep it as `detector_orientation`.

### Open questions for implementation
- Is `field_direction` persisted in the `.asymp` project schema? (affects rename)
- Does any GUI panel or fit-recommendation logic branch on `field_direction`
  today? If so, switching its meaning from orientation→field-state could change
  recommendations — verify intended.
- Confirm the official NXmuon enumeration for `magnetic_field_state` (spec not
  available locally — see comparison.md caveat).
