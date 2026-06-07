# Period selection — test data

## Photo-µSR in silicon (primary validation case)

Location: `~/Documents/WiMDA muon school/Semiconductors/Photo-muSR in silicon/Data_hdf5/`

- Runs **103277–103298** — period mode (HIFI). Two periods per file:
  light-OFF = **Green** (period 2), light-ON = **Red** (period 1).
- Run **103299** — TF calibration (for α; single period).

Workflow guide: `CarrierRecombinationSilicon 2026.docx`; summary in
`.../ANALYSIS_asymmetry.md`.

Observed on `HIFI00103277.nxs` via the new API:

- `period_count == 2`, `period_labels == ["red", "green"]`.
- Red and Green share the same time axis (2040 points), T = 291 K, B = −100 G.
- Red (light-ON) relaxes substantially more than Green (light-OFF) — the
  expected physical signature.

## Other period/multi-period corpora (generalisation, not validated here)

- **Chemistry / Benzene** — RF resonance (period structure).
- **ALC in TCNQ** — ALC scan (multi-period).

These exercise the 3+ period `list` path and the integer-period selector; the
photo-µSR set exercises the 2-period red/green path.

## Unit-test fixtures

`tests/test_period_selection.py` builds a synthetic two-period `MuonDataset`
in memory (no file dependency) so the period-selection contract — correct
period extracted, errors on bad input, scalar/label access, provenance
preserved, GUI/core agreement, and G−R/G+R arithmetic — runs in CI without the
WiMDA corpus. A NeXus round-trip is exercised only when a sample period-mode
file is available.
