# Time-integral asymmetry — test data

## What the feature needs

Unlike most ports, the headline observable lives at the **series** level, so
validation needs **two** kinds of input:

1. **Single-run reduction** — any loadable run, to check the per-run
   `(value, error)` scalar against a hand-computed integral and against Mantid
   `PlotAsymmetryByLogValue` run-by-run.
2. **A field scan** — a *series* of runs taken at stepped field (ideally a real
   LF repolarisation or ALC scan), to check assembly, ordering by field, and the
   resulting curve shape (monotonic recovery for repolarisation; a resonant dip
   for ALC).

## Corpus

ISIS pulsed-muon HDF5 NeXus files in `~/Documents/WiMDA muon school/` (see
[[project_testing_corpus]]; 2050 h5py-readable `pulsedTD` files, the HDF4 set is
out of scope — [[project_hdf4_loader_gap]]). PSI `.bin`/`.mdu` (EuO, Chemistry)
also load.

### Finding a field scan in the corpus (reproducible)

The field-geometry study already enumerated `magnetic_field_state` /
`magnetic_field` across the corpus. Reuse that approach to group runs into
candidate scans: **same instrument + same temperature + `field_state = LF` (or
TF for QLCR-style) + stepped `magnetic_field`** is a field scan. Run with the
project venv:

```bash
cd ~/Documents/"WiMDA muon school"
.venv/bin/python - <<'PY'
import h5py, glob
from collections import defaultdict
scans = defaultdict(list)
for f in glob.glob("**/*.nxs", recursive=True):
    try:
        with h5py.File(f, 'r') as h:
            e = h[list(h.keys())[0]]
            def g(p):
                try:
                    v = e[p][()]
                    if hasattr(v, '__len__') and not isinstance(v,(str,bytes)): v=v[0]
                    return v.decode() if isinstance(v, bytes) else v
                except Exception: return None
            inst = g("instrument/name")
            T    = g("sample/temperature")
            B    = g("sample/magnetic_field")
            fs   = g("sample/magnetic_field_state")
            rn   = g("run_number")
            if B is not None:
                scans[(str(inst), str(fs), round(float(T),1) if T else None)].append((float(B), rn, f))
    except Exception:
        pass
# A group with many distinct fields at one T is a field scan.
for k, pts in sorted(scans.items(), key=lambda kv: -len(set(p[0] for p in kv[1]))):
    fields = sorted(set(p[0] for p in pts))
    if len(fields) >= 5:
        print(f"{k}: {len(fields)} fields  {fields[:8]}{'...' if len(fields)>8 else ''}")
PY
```

The selected scan (instrument, temperature, run-number range, field list) must
be **recorded in this file once identified**, so the verification is reproducible
and pinned — exactly as the period-selection study pinned photo-µSR runs
103277–103298 and the field-geometry study pinned its distribution.

> **TODO (study → implementation):** run the command above, pick one repolarisation
> (LF) scan and, if present, one ALC/QLCR scan, and record their run numbers,
> temperature, and field values here.

## Golden references

- **Mantid oracle** — run `PlotAsymmetryByLogValue` (both `Type=Integral` and
  `Type=Differential`) over the same run list with `LogValue` = the field log,
  matching `TimeMin/TimeMax`, `Alpha`, grouping, and Red/Green. Capture the
  output `(log_value, asymmetry, error)` table as the golden file. Mantid is the
  primary numerical oracle because Asymmetry's `compute_asymmetry` already shares
  its error model. (Mantid is GPL-3 — usable as an external oracle only; do **not**
  copy code into MIT Asymmetry, same rule as the MaxEnt study.)
- **WiMDA cross-check** — for the `method="integral"` path with `alpha=1.0`, the
  result must equal WiMDA's `(F−B)/(F+B)` over the same good-bin window. Use a
  WiMDA ALC run as a secondary check; expect agreement to counting precision.
- **Synthetic check** — a constructed forward/backward pair with a known constant
  asymmetry over a flat window gives an exact analytic integral and error, for a
  loader-independent unit test (mirrors how `test_period_selection.py` uses
  controlled inputs).

## Edge cases to cover with data

- A run **missing the field log** → excluded from the scan with a recorded
  reason (Mantid skips such runs); the rest of the scan still builds.
- **Dual-period** run reduced under each `period_mode` (red/green/diff/sum).
- **Differential vs Integral** divergence where the asymmetry is *not* flat over
  the window (the two methods agree only for a flat asymmetry).
- A TF run at **0 G** present in the field list (per field-geometry finding,
  MUSR00044991 exists) — must not be special-cased as ZF; it is a legitimate scan
  point at B = 0.
