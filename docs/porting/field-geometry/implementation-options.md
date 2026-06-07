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
   **else `None`/unknown — do NOT fall back to the orientation-derived value.**
   (Decision 2026-06-07, user preference — see below.)
4. **Never** synthesise ZF from a zero applied-field magnitude. Trust the string.
   (`MUSR00044991.nxs` is `'TF'` at 0 G.)

- **Pro:** Fixes the bug; keeps the orientation value (as a distinct
  `detector_orientation` field) for whoever needs the instrument axis; strictly
  more correct than any reference program; never presents a guessed geometry.
- **Con:** Two new metadata keys + GUI surface for them; a little more code than B.
- **Verdict:** **Recommended.** This is the candidate fix, refined with (a) an
  explicit rename of the orientation value, (b) **no orientation fallback for the
  user-facing geometry — return unknown instead**, and (c) the "trust the string,
  never infer ZF from field=0" rule.

### Decision: no orientation fallback for the user-facing geometry

The original candidate fix fell back to the orientation-derived value when
`magnetic_field_state` was absent. **Rejected** (user preference, 2026-06-07):
falling back to orientation reintroduces exactly the conflation this fix removes —
on EMU/MuSR the banks read `'L'` regardless of the applied-field geometry, so a
fallback value of "Longitudinal" would be a misleading guess, not a real reading.

Instead, when `field_state` is absent the user-facing geometry is **`None` /
"Unknown"**. The orientation is still recorded separately as `detector_orientation`
(its honest meaning, the instrument main-field axis), so no information is lost —
it is simply not promoted to stand in for the experiment geometry.

Affected files (this corpus): the 137 files lacking `magnetic_field_state` — all
ARGUS runs (76) and the EMU "Ferromagnetic nickel" set (61) — would report an
**unknown** field geometry rather than "Longitudinal". This is the intended,
honest outcome. All PSI `.bin`/`.mdu` and HDF4 v1 files would likewise report
unknown geometry (they carry no `magnetic_field_state`).

## Detailed recommendation for the implementation pass

### Reading & normalisation
- V1 (`_load_v1`): read `sample/magnetic_field_state`
  (`entry/sample/magnetic_field_state`).
- V2 (`_load_v2`): read `entry/sample/magnetic_field_state` (note: `sample` is
  already fetched at [`nexus.py:293`](../../../src/asymmetry/core/io/nexus.py)).
- Normalise: strip + uppercase; accept `'TF'`/`'LF'`/`'ZF'`. Treat empty / `'n/a'`
  / unknown as **absent → unknown geometry** (do not fall through to orientation),
  not as an error.
- Keep `_normalise_orientation` for the orientation value; also handle lowercase
  `'l'` (already does via `.upper()`) — the corpus contains `'l'`.

### Metadata shape (proposed)
- `detector_orientation`: "Longitudinal"/"Transverse" (was `field_direction`).
- `field_state`: "TF"/"LF"/"ZF" or `""`.
- `field_direction` (the user-facing geometry, kept as the existing key for GUI
  compatibility): mapped from `field_state` when present
  (`TF`→"Transverse", `LF`→"Longitudinal", `ZF`→"Zero field"), else **`None` /
  "Unknown"** (NOT the orientation value).
- (`field_geometry_source` provenance flag is no longer needed: the geometry is
  either from `field_state` or unknown — there is no second source to disambiguate.)

> Decide during implementation whether to repurpose the existing `field_direction`
> key (less GUI churn) or introduce a new key and migrate GUI usages. Grep GUI +
> project-serialisation for `field_direction` first; it may be persisted in
> `.asymp` projects, which would make a rename a schema concern.

### Resolution (per file) — for the user-facing geometry
1. `sample/magnetic_field_state` present & recognised (`TF`/`LF`/`ZF`) → use it.
2. Else **unknown** (`None`/"Unknown"). Do **not** fall back to orientation and do
   **not** guess from field magnitude.

(The detector `orientation` is still read independently and stored as
`detector_orientation` for every file that has it — it is just not used to resolve
the geometry.)

PSI `.bin`/`.mdu` and HDF4 v1 NeXus: no `magnetic_field_state` → step 2 (unknown).
HDF4 remains unreadable by h5py and is out of scope ([[project_hdf4_loader_gap]]).

### Things to explicitly NOT do
- **Do not fall back to detector orientation** for the user-facing geometry —
  return unknown instead (it would be a misleading guess: banks read `L`
  regardless of applied field).
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
