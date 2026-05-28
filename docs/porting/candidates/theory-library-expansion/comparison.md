# Theory library expansion: function-by-function comparison

| Function | Mantid | musrfit | WiMDA | Asymmetry | Reference |
|---|---|---|---|---|---|
| Keren | ✅ `Keren.cpp` | ❌ | ❌ | ❌ | Keren PRB 50, 10039 (1994) |
| Meier | ✅ `Meier.cpp` | ❌ | ❌ | ❌ | Meier PRB 17, 1739 (1978) |
| Abragam | ❌ | ✅ `PTheory::Abragam` | ❌ | ❌ | Abragam, *Principles of Nuclear Magnetism* (1961) |
| Bessel | ❌ | ✅ `PTheory::Bessel` | ◐ via registry | ❌ | Le Bras et al. PRB 41, 4030 (1990) |
| MuoniumDecouplingCurve | ✅ `MuoniumDecouplingCurve.cpp` | ❌ | ❌ | ❌ | Patterson, RMP 60, 69 (1988) |
| SpinGlass | ❌ | ✅ `PTheory::SpinGlass` | ❌ | ❌ | Uemura et al. PRB 31, 546 (1985) |
| SuperconductorVortexLattice | ❌ | ✅ `PTheory::SkewedGss` (skewed Gauss approx.) | ❌ | ◐ parameter-domain only | Brandt PRB 37, 2349 (1988) |

## Implementation notes per function

### Keren
- Form: closed-form Bessel-function combination
- Cost: O(N) numpy evaluation, single special-function call per bin
- Validation: Mantid regression curves

### Meier
- Form: closed-form muonium-style hyperfine model with magnetic
  exchange parameter
- Cost: O(N) evaluation
- Subtlety: ambiguous parameter conventions across literature —
  document the chosen convention explicitly

### Abragam
- Form: `exp(-σ² ν⁻² [exp(-ν·t) - 1 + ν·t])` (envelope) × cosine
- Cost: O(N)
- Already partially implementable via composite expressions — the
  point is to expose it as a single named component for the wizard

### Bessel
- Form: `J₀(2π·f·t) · exp(-λ·t)`
- Cost: O(N), one scipy `j0` call per bin
- Used heavily in incommensurate magnetism studies

### MuoniumDecouplingCurve
- Form: field-dependence curve (output is *not* a time series; it
  is a parametric model in the parameter-trending registry)
- Lives in `core/fitting/parameter_models.py`, not `models.py`

### SpinGlass
- Form: time-dependent depolarisation with a stretched-Gaussian
  field distribution
- Cost: O(N), single special-function call

### SuperconductorVortexLattice (time-domain)
- Form: time-domain analogue of the Brandt skewed P(B), evaluated
  via inverse cosine transform
- Subtlety: Asymmetry already has a parameter-domain `SC_TwoGap_SS`;
  this is the *time-domain forward model* needed for fitting
  raw TF signals in the mixed state

## Test-data strategy

Generate synthetic data from the reference implementation for each
function, save as a small CSV in `tests/porting/theory-library-expansion/`,
then assert that Asymmetry's output matches within `1e-6`.
