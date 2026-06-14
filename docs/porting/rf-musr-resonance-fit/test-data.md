# Test data

## Synthetic fixtures (in `tests/test_rf_musr_resonance.py`)

All synthetic data is generated from the model itself at the paper couplings, so
the tests are self-contained (no corpus files required in CI):

- **Aₚ = 0 reduction:** `mu_proton_levels(800, 514.78, 0)` vs
  `muonium._tf_levels(800, 514.78)` — midpoints match to 1e-6 MHz; pair
  splittings equal `γ_p·B`.
- **Resonance fields:** `rf_resonance_fields(514.78, 124.6, 218.5)` ≈
  `(893.9, 796.7) G`, both inside the corpus window (560–1080 G).
- **Transition selection:** sweep all 28 level pairs over `[1, 2000] G`; only
  `E₇−E₅` and `E₈−E₆` resonate at 218.5 MHz inside 500–1100 G.
- **Direct inversion:** forward `(A_µ, A_p) → (B₁, B₂)`, then least-squares
  inversion of those exact fields recovers `A_µ = 514.780`, `A_p = 124.600`
  (machine precision), start-independent over the finite-resonance basin.
- **Curve round-trip:** generate `rf_resonance_mup` over 80 fields, fit the
  composite model back → `A_µ`, `A_p` to 1e-2.
- **Robustness:** pathological params `(0,0)`, `(1e6,1e6)`, `(−5,−5)`,
  `ν_RF=0` all yield finite curves (no raise).
- **Analytic under-splitting:** exact split ≈ 97 G > analytic split (≈ 78 G).

## Corpus data (benzene RF technique)

`C:\Users\benhu\iCloudDrive\Documents\WiMDA muon school\Chemistry\Muon spectroscopy of benzene\`

- **Raw runs:** `data/RF resonance/56426.nxs … 56462.nxs` (37 DEVA/MUT@ISIS
  runs) + HDF5 mirror `data_hdf5/RF resonance/`. Red-Green mode, RF = 218.5 MHz,
  293.0 ± 0.1 K, applied fields ≈ 560–1080 G. Each run = one field point; the
  field scan is built from the per-run (Green − Red) integral asymmetry.
- **Reference paper:** `RFpaper.pdf` (McKenzie 2013).
- **Ground truth:** that folder's `GROUND_TRUTH.md` §3D / §11.

### Paper-graded targets (McKenzie 2013, Table 1; C₆H₆Mu in benzene, RF-µSR)

| Quantity | Value |
|---|---|
| ν_RF | 218.5 MHz |
| **A_µ** | **514.78(4) MHz** |
| **A_p** | **124.6(14) MHz** |
| Temperature | 293.0 ± 0.1 K |

### Figure-digitised observables (Fig. 3a, liquid benzene; ±~5 G read error)

| Quantity | Value |
|---|---|
| Left resonance dip | 773 G |
| Right resonance dip | 865 G |
| Splitting | ~92 G |
| Mean field | ~819 G |

Note (from the corpus ground truth): the y-axis ("rf asymmetry ×10⁻³") is
instrument/grouping-dependent — treat dip **depths** as relative; only the
field **positions / splitting / widths** are robust digitised observables, and
even those carry figure-tracing + absolute-calibration error larger than the
nominal ±5 G read error. The authoritative grading values are the paper Table 1
hfccs, not the read-off fields.
