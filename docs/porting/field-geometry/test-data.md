# Field geometry — test data

## Corpus

ISIS pulsed-muon HDF5 NeXus files (`definition = 'pulsedTD'`) in
`~/Documents/WiMDA muon school/` (see [[project_testing_corpus]]). 2050 files are
readable by h5py; a further 1913 are HDF4 and unreadable (out of scope —
[[project_hdf4_loader_gap]]).

## Inspection command (reproducible)

Run with the project venv:

```bash
cd ~/Documents/"WiMDA muon school"
/Users/bhuddart/Source/Asymmetry/.venv/bin/python - <<'PY'
import h5py, glob, os
from collections import Counter
c = Counter()
for f in glob.glob("**/*.nxs", recursive=True):
    try:
        with h5py.File(f, 'r') as h:
            e = h[list(h.keys())[0]]
            def g(p):
                try:
                    v = e[p][()]
                    if hasattr(v, '__len__') and not isinstance(v, (str, bytes)): v = v[0]
                    return v.decode() if isinstance(v, bytes) else v
                except Exception: return None
            fs = g("sample/magnetic_field_state")
            ori = g("instrument/detector_1/orientation") or g("instrument/detector/orientation")
            c[(str(fs), str(ori))] += 1
    except Exception:
        c[("HDF4/unreadable", "")] += 1
for k, v in c.most_common(): print(f"{v:5d}  {k}")
PY
```

## Observed distribution (2026-06-07)

| `(magnetic_field_state, orientation)` | count |
|---|---|
| `('TF', 'L')` | 1731 |
| `('TF', 'T')` | 145 |
| `(absent, 'l')` | 76 |
| `(absent, 'L')` | 61 |
| `('LF', 'L')` | 37 |
| HDF4 / unreadable | 1913 |

No `'ZF'` value appears in any file.

## Golden ground-truth files (for the implementation pass)

| File | `magnetic_field_state` | `magnetic_field` | `detector_1/orientation` | Tests |
|---|---|---|---|---|
| `Basics/data_hdf5/EMU00018850.nxs` | `TF` | 20 G | `L` | TF run, banks read L → must label **Transverse** (the headline mislabel) |
| `Basics/data_hdf5/MUSR00044991.nxs` | `TF` | **0 G** | `L` | TF at zero field → must **not** be inferred ZF; label **Transverse** |
| `56426.nxs` (Superconductivity) | `LF` | 560 G | `L` | genuine LF → label **Longitudinal**; agrees with orientation |
| `emu00124218.nxs` | *absent* | 0 G | `l`/`L` | no field state → **fall back** to orientation ("Longitudinal"); provenance=`orientation` |

Locate each: `find ~/Documents/"WiMDA muon school" -iname "<name>"`.

## Gaps in test coverage

- **No `'ZF'` example available** — cannot golden-test the ZF mapping from real
  data; will need a synthetic/edited file or a unit test with a crafted value.
- **No readable PSI `.bin`/`.mdu` with a field-state equivalent** here for the
  fallback path — PSI files have no `magnetic_field_state` by design; the
  fallback to orientation/absent is the expected behaviour.
- HDF4 v1 NeXus cannot be opened by h5py, so the V1 `run/sample/...` path cannot
  be exercised against this corpus.
