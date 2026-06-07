# Field geometry (TF / LF / ZF) determination — study

**Status:** implemented. The study pass (investigation) is below; the loader
change has since landed in `src/asymmetry/core/io/nexus.py` per the recommendation
in [implementation-options.md](implementation-options.md) (Option C, no
orientation fallback). See the "Implementation outcome" section at the end.

**Slug:** `field-geometry`
**References studied:** Mantid, musrfit, WiMDA (all from local source checkouts
under `~/Source/`), plus the ISIS pulsed-muon NeXus (`pulsedTD`) files in the
WiMDA Muon School corpus.

## Why this study exists — the Asymmetry bug

Asymmetry's NeXus loader sets the user-facing `field_direction` from the
**detector-bank orientation**, not from the per-run applied-field state:

- V1: [`nexus.py:172-175`](../../../src/asymmetry/core/io/nexus.py) reads
  `instrument/detector/orientation` → `_normalise_orientation` → `field_direction`.
- V2: [`nexus.py:299-300`](../../../src/asymmetry/core/io/nexus.py) reads
  `instrument/detector_1/orientation` the same way.
- [`_normalise_orientation` (nexus.py:1058-1065)](../../../src/asymmetry/core/io/nexus.py):
  first char `L`→"Longitudinal", `T`→"Transverse", else passthrough.
- `sample/magnetic_field_state` (`'TF'`/`'LF'`/`'ZF'`) is **never read**.

Detector `orientation` and `magnetic_field_state` are different physical things:

| | what it describes | who sets it |
|---|---|---|
| `instrument/detector*/orientation` | where the detector banks physically sit — an **instrument-build** property (the main-field *axis* the banks straddle) | fixed per instrument build |
| `sample/magnetic_field_state` | the **applied-field geometry for this run** (TF/LF/ZF) | per run, by the experimenter |

On EMU/MuSR the banks read `'L'` *even for transverse-field runs*, so Asymmetry
currently mislabels every TF calibration run "Longitudinal". The correct value is
present in the file.

### Ground-truth (this corpus)

`.venv/bin/python` + h5py over the 2050 readable HDF5 `.nxs` files in
`~/Documents/WiMDA muon school/` (all `definition = 'pulsedTD'`):

| `(magnetic_field_state, detector orientation)` | count | meaning |
|---|---|---|
| `('TF', 'L')` | 1731 | **mislabelled** — TF run, banks read L |
| `('TF', 'T')` | 145 | TF run, orientation happens to agree |
| `('LF', 'L')` | 37 | LF run (fields 560–840 G), agree |
| `(absent, 'l'/'L')` | 137 | all ARGUS (76) + EMU nickel set (61), **no field state** → geometry reported **unknown** |
| (HDF4, unreadable by h5py) | 1913 | out of scope — see [[project_hdf4_loader_gap]] |

- `EMU00018850.nxs`: `magnetic_field_state='TF'`, `magnetic_field=20 G`,
  `detector_1/orientation='L'` — the canonical mislabel.
- `MUSR00044991.nxs`: `magnetic_field_state='TF'` at **`magnetic_field=0 G`** —
  proves you **must not** infer ZF from a zero field value; trust the string.
- **No `'ZF'` string appears in any of the 2050 files.** ISIS in this corpus
  records only `'TF'`/`'LF'`; zero-field runs are not tagged `'ZF'` here.

## Headline finding across the reference programs

**None of WiMDA, musrfit, or Mantid reads `sample/magnetic_field_state` to drive
field geometry.** See [comparison.md](comparison.md) for cited detail. In short:

- **Mantid** — uses the **detector `orientation` string** (first char `t` →
  "Transverse", else "Longitudinal"), i.e. *exactly the same source Asymmetry
  uses today*. It stores this as a run log named **`main_field_direction`** and
  uses it only to pick the default detector grouping on MUSR/CHRONUS — never for
  fitting or asymmetry maths. It has no ZF category and never reads
  `magnetic_field_state`.
- **musrfit** — does not classify at all. It stores a free-text `Setup`, a
  free-text `Orientation`, and a numeric field value, all informational (plot
  titles). The experimenter declares TF/LF/ZF implicitly by which theory function
  they write in the `.msr` fit file. It reads `magnetic_field_state` at the raw
  HDF level but discards it.
- **WiMDA** — does not maintain a TF/LF/ZF label that drives analysis. The user
  picks detector grouping (`.grp`) and analysis mode manually. It reads the
  applied-field *value* and, on ISIS, an instrument `a_selected_magnet`
  (`'L'`/`'T'`/`'A'`) flag used **only** to choose which field column to print in
  the logbook. It does not read `magnetic_field_state` at all.

## What this means for Asymmetry

The naive reading ("the established programs read `magnetic_field_state`, so
Asymmetry should too") is **false** — none of them do. But that does not vindicate
Asymmetry's current behaviour, because Asymmetry is doing something subtly
different from all three: it derives a value from `orientation` and then **labels
it `field_direction` and presents it to the user as the experiment's field
geometry**. Mantid derives the same value but names it `main_field_direction`
(the instrument axis) and never claims it is the per-run TF/LF/ZF state.

So the bug is a **semantic/labelling conflation**, and the data genuinely carries
the better signal. The recommendation (detailed in
[implementation-options.md](implementation-options.md)) is to:

1. Keep the orientation-derived value, but rename it to its true meaning
   (`detector_orientation` / `main_field_direction`, matching Mantid).
2. Add a separate `field_state` read from `sample/magnetic_field_state`.
3. Make the user-facing geometry come from `field_state` when present, and be
   **`None`/"Unknown" when it is absent** — do **not** fall back to the
   orientation-derived value (decision 2026-06-07: that fallback would
   reintroduce the conflation, since the banks read `L` regardless of the applied
   field). The orientation is still kept as a separate `detector_orientation`
   field, so nothing is lost.
4. Never infer ZF from a zero field magnitude — trust the string.

This is strictly more correct than any reference program while keeping the
orientation value as a distinct field.

## Files in this study

- [comparison.md](comparison.md) — per-program field/path/fallback table with
  source citations.
- [implementation-options.md](implementation-options.md) — options for the
  loader change and the recommendation (confirm/refine the candidate fix).
- [test-data.md](test-data.md) — the ground-truth files and inspection command.
- [verification-plan.md](verification-plan.md) — how a future implementation pass
  should be verified.

## Implementation outcome

The loader change landed in `src/asymmetry/core/io/nexus.py` (both `_load_v1` and
`_load_v2`):

- New helpers `_normalise_field_state` (→ `'TF'`/`'LF'`/`'ZF'`/`''`) and
  `_field_direction_from_state` (→ "Transverse"/"Longitudinal"/"Zero field"/`''`).
- Three metadata keys are now emitted:
  - `field_state` — raw code from `sample/magnetic_field_state` (or `''`).
  - `field_direction` — **repurposed**: now the user-facing geometry derived
    *only* from `field_state`; `''` when the state is absent (no orientation
    fallback).
  - `detector_orientation` — the instrument-axis value (the old `field_direction`
    source), kept distinct.
- GUI: the run-info dialog gained a separate **"Detector Orientation"** row
  (`run_info_dialog.py`), and `detector_orientation` is a selectable browser
  column (`data_browser.py`). Per the 2026-06-07 UX decision, an absent field
  state shows **blank** (not "Unknown"), the geometry reads as **words**, and the
  detector orientation is surfaced separately.

Tests: `tests/test_nexus_loader.py` covers TF-overrides-L-orientation (V1+V2), LF,
ZF, absent-state→blank (no fallback), and blank/`n/a`→unknown. Verified end-to-end
against real corpus files: `EMU00018850.nxs` (TF/L → Transverse),
`MUSR00044991.nxs` (TF at 0 G → Transverse, not ZF), `56426.nxs` (LF →
Longitudinal), `emu00124218.nxs` (no state → blank, orientation Longitudinal).
