# Fourier Transform Implementation Options

## Option 1: Keep A Magnitude-First FFT API And Add GUI-Only Phase State

Pros:

- smallest UI-only edit
- no existing core callers need to change

Cons:

- phase remains disconnected from the owning transform logic
- blocks later detector-phase and phase-table work
- repeats the current design weakness

Assessment: reject.

## Option 2: Add A Phase-Aware Complex-Spectrum Core API First

Pros:

- makes phase correction an explicit core concern
- preserves existing `fft_asymmetry` callers by keeping the current wrapper
- gives later detector-phase work a stable seam
- easy to verify with synthetic cosine signals

Cons:

- still only supports manual phase in the first slice
- does not yet implement per-detector or automatic phase workflows

Assessment: choose for the first implementation slice.

## Option 3: Jump Directly To Per-Detector Phase Tables

Pros:

- closer to Mantid and musrfit end state
- addresses the detector-phase requirement directly

Cons:

- crosses core, grouping, GUI, and project-persistence boundaries at once
- requires a broader decision on detector-table storage and editing UX
- harder to validate incrementally

Assessment: defer until the complex-spectrum seam exists.

## Selected Direction

Implement in this order:

1. add a complex-spectrum helper in `src/asymmetry/core/fourier/fft.py`
2. add manual phase rotation in that core helper and preserve the current
   wrapper API for existing callers
3. add WiMDA-style `t0` offset handling in the same core seam
4. expose manual phase and `t0` entry in
   `src/asymmetry/gui/panels/fourier_panel.py`
5. extend tests to lock the WiMDA phase contract
6. introduce a central plot workspace with explicit `Time Domain` and
   `Frequency Domain` tabs so both representations can coexist in the main
   window
7. keep the existing time-domain plot behavior on the time tab and route FFT
   output to the frequency tab
8. add a frequency-tab x-axis unit toggle between `Frequency (MHz)` and
   `Field (G)` using a display-only conversion over canonical MHz data
9. add grouped FFT output controls in `src/asymmetry/gui/panels/fourier_panel.py`:
   per-group inclusion toggles with averaged output across the selected groups
10. make the Fourier dock content scrollable so the WiMDA-style grouped
   controls remain usable in the docked layout
11. add post-transform controls that fit the docked frequency-tab workflow:
   a main-window relative field/frequency axis toggle, excluded frequency bands, and
   WiMDA-style average-error estimation for averaged grouped spectra
12. follow with WiMDA automatic phase estimation tied to the grouped controls

This first direction (steps 1–12) is now largely complete. The WiMDA-first
implementation pass has delivered the core complex-spectrum API, all five
WiMDA-style display modes, grouped FFT with per-group phase tables, the
frequency/time dual-tab workspace, and the field-relative x-axis toggle.

## Viewer Decision

Adopt explicit central tabs for time-domain and frequency-domain plots rather
than a toolbar-only domain toggle.

Rationale:

- preserves separate axis limits and controls for each representation
- makes the coexistence model explicit to the user
- avoids overloading the existing time-domain `1/2/3` saved-view controls
- matches Mantid's stronger precedent for explicit frequency-domain context

First-slice constraints:

- frequency tab is view-only
- frequency tab keeps pan/zoom, overlay, annotations, labels, and export
- time-only controls such as bunching, fit range, and polarization selection do
  not appear in the frequency tab
- project persistence stores viewer state, not computed FFT arrays

---

## musrfit Feature Decision Matrix

This section records a decision for each musrfit feature: whether it supersedes
the WiMDA approach, coexists alongside it, or is deferred/rejected.

### Phase Correction Family

| Approach | Source | Status | Decision | Rationale |
|---|---|---|---|---|
| Single-constant phase `c₀` | WiMDA | **Implemented** | Keep as default | Familiar to existing users; sufficient for most datasets |
| `t₀` offset term `f·t₀` | WiMDA | **Implemented** | Keep | Corrects detector timing directly in frequency domain |
| Per-group phase table | WiMDA | **Implemented** | Keep | Essential for averaging across groups with different phases |
| Peak / average auto-estimation | WiMDA | **Implemented** | Keep | Good first estimate; fast and intuitive |
| Linear slope `c₁·(i/N)` | musrfit | Not yet implemented | **Implement as optimizer backend only; do not expose as a user control** | musrfit itself uses `c₁` only in the `phaseOptReal` automatic optimizer, never in manual or CLI paths. Exposing `c₁` as a standalone control would present users with a parameter that has no physical interpretation and no estimation method; `t0_offset_us` already covers the linear-in-frequency manual slope case with a physically motivated parameterization. |
| Entropy + penalty optimizer | musrfit | Not yet implemented | **Add as `phaseOptReal` mode** | Provides automatic two-parameter phase correction; different in kind from WiMDA's single-frequency peak method |

**On superseding**: musrfit does not use linear phase exclusively. All musrfit
manual, CLI, and interactive paths use a single constant (c₁ = 0 implicitly).
Linear phase is the internal optimization variable for `phaseOptReal` only —
users never set `c₁` directly in musrfit either. The correct boundary is:
implement `c₀ + c₁·(i/N)` as the optimizer's internal parameterization; keep
the existing `c₀ + 2πf·t₀` form for all other paths. Do not unify the APIs.

The existing `t0_offset_us` already handles the physically motivated
linear-in-frequency slope for users who know their detector timing offset. The
optimizer's `c₁` is an abstract variable that is found by minimization, not
entered by hand. These serve different purposes and should remain separate.

### Preprocessing / Lifetime Correction

| Approach | Source | Status | Decision | Rationale |
|---|---|---|---|---|
| Grouped-count lifetime correction `exp(t/τ_μ)` | WiMDA | **Implemented** | Keep as default | Appropriate for grouped detector sums; preserves counts-based input |
| N₀ normalization (subtract + divide by mean) | musrfit | Not yet implemented | **Add as advanced option** | Required for single-histogram FFT inputs; produces asymmetry-like trace; unreliable at low field where mean is not a stable N₀ estimate |

**On superseding**: These two paths apply to different input types. Grouped
asymmetry data from two-detector pairs does not need N₀ normalization because
the asymmetry has already been formed. N₀ normalization is primarily useful
when doing FFT directly on individual histogram counts, which Asymmetry does
not currently support. This is therefore not a competition: it is a new
capability for a new use case.

### Background Subtraction

| Approach | Source | Status | Decision | Rationale |
|---|---|---|---|---|
| Subtract error-weighted mean of signal window | WiMDA | **Implemented** | Keep as default | Fast and automatic; good for oscillatory signals |
| Range-based background (average of pre-signal bins) | musrfit | Not yet implemented | **Add as advanced option** | More physically motivated; use when the pre-signal region is accessible and different from the signal-window mean |
| Explicit background constant | musrfit | Not yet implemented | **Add as advanced option** | Useful when background is known from a separate measurement |

**On superseding**: Range-based background estimation is strictly more accurate
than mean-of-signal-window when a clean pre-signal region exists. However,
grouped muSR asymmetry data typically has t₀ at or near the start of the
measured window, leaving little or no pre-signal region. For most grouped
inputs, the current approach is appropriate. Add range-based as an option for
expert workflows or single-histogram inputs.

### Apodization

| Approach | Source | Status | Decision | Rationale |
|---|---|---|---|---|
| WiMDA-style Lorentzian / Gaussian with explicit tau and start time | WiMDA | **Implemented** | Keep | More controllable and reproducible than qualitative levels |
| musrfit qualitative levels (None / Weak / Medium / Strong) | musrfit | — | **Do not adopt** | Implicit window width makes results harder to reproduce or compare; users working from publications need explicit time-constant values |

**On superseding**: WiMDA's explicit tau/start approach is better for
reproducibility and documentation. musrfit's qualitative levels are a UX
simplification that trades precision for convenience. Do not port them.

### Output Modes

| Mode | Source | Status | Decision | Rationale |
|---|---|---|---|---|
| `(Power)^1/2` | WiMDA | **Implemented** | Keep; keep WiMDA name | Magnitude; WiMDA label clarifies this is NOT squared power |
| `Phase Spectrum` | WiMDA | **Implemented** | Keep; keep WiMDA name | Spectral angle; WiMDA name avoids confusion with corrected "Phase" mode |
| `Cos` | WiMDA | **Implemented** | Keep | Raw real; distinguishable from phase-corrected real |
| `Sin` | WiMDA | **Implemented** | Keep | Raw imaginary; distinguishable from phase-corrected imaginary |
| `Phase` | WiMDA | **Implemented** | Keep; keep WiMDA name | Phase-corrected real; WiMDA's "Phase" ≈ musrfit's "real" after correction |
| `phaseOptReal` | musrfit | Not yet implemented | **Add as sixth mode** | No WiMDA equivalent; requires entropy optimizer backend |
| `power` (`\|F\|²`) | musrfit | — | **Do not adopt** | musrfit's `power` = `\|F\|²` would confuse users expecting WiMDA's `\|F\|` |
| `real+imag` simultaneous | musrfit | Not yet implemented | **Consider** | Useful for phase correction quality inspection; deferred |
| `imag` (corrected imaginary) | musrfit | Not yet implemented | **Consider** | Completes the quadrature pair; less commonly needed but analytically useful |

**Critical naming note**: musrfit's "phase" mode = WiMDA's "Phase Spectrum"
(both are `atan2(Im, Re)`). musrfit's "real" mode ≈ WiMDA's "Phase" mode
(both are phase-corrected real projection). These collisions must be documented
in any UI or help text that describes musrfit-sourced features.

### X-Axis Units

| Unit | Source | Status | Decision | Rationale |
|---|---|---|---|---|
| MHz | WiMDA / musrfit | **Implemented** | Keep | Primary frequency unit |
| Gauss | WiMDA / musrfit | **Implemented** | Keep | Conventional field unit for muSR |
| Tesla | musrfit | Not yet implemented | **Add** | Low-cost; standard SI unit; commonly used in publications |
| Mc/s | musrfit | Not yet implemented | **Defer** | Rarely used in practice; only adds value for very niche workflows |

### Per-Detector FFT

| Approach | Source | Status | Decision | Rationale |
|---|---|---|---|---|
| Per-detector phase tables and FFT | musrfit / Mantid | Not yet implemented | **Defer** | Requires a broader detector-phase contract and significant changes to the grouping and metadata model; not needed for current parity target |

---

## Recommended Follow-On Implementation Order

Based on the decisions above, the recommended sequence for post-WiMDA-parity
implementation work is:

1. **`phaseOptReal` mode with entropy optimizer** — adds musrfit's most
   distinctive phase capability; requires porting the `PFTPhaseCorrection`
   algorithm (entropy + penalty, SciPy `minimize` as the Python equivalent of
   Minuit2).

2. **Linear phase slope `c₁` (optimizer backend)** — implement `c₀ + c₁·(i/N)`
   as the internal parameterization used by the entropy optimizer when
   `phaseOptReal` mode is active. Do not expose `c₁` as a panel control; the
   existing `t0_offset_us` covers the physical timing-offset use case.

3. **`imag` / `real+imag` output modes** — add the phase-corrected imaginary
   quadrature as an optional display mode; enables full complex spectrum
   inspection.

4. **N₀ normalization** — add as a toggle for single-histogram FFT inputs;
   keep grouped-count path as default.

5. **Range-based background subtraction** — add pre-signal bin range input to
   the Fourier panel.

6. **Tesla x-axis unit** — minor UI addition; no algorithmic change.

---

## Follow-On Work and Recommendations (Summary)

- **Phase Correction**: Retain the single-constant `c₀` (plus `t0_offset_us`)
  model for all manual and interactive phase paths — this is also the norm in
  musrfit's own manual and CLI paths. Implement `c₀ + c₁·(i/N)` only as the
  internal backend for the `phaseOptReal` entropy optimizer. Do not expose `c₁`
  as a standalone user control; musrfit itself never allows users to set `c₁`
  directly in manual mode. The existing `t0_offset_us` parameterization covers
  the physically motivated linear-in-frequency slope; `c₁` is an abstract
  optimization variable intended to be found by a minimizer, not entered by hand.

- **Preprocessing**: Add musrfit's N₀ normalization and flexible background
  subtraction as advanced options. WiMDA-style grouped-count lifetime
  correction remains default; musrfit normalization is appropriate for
  single-histogram inputs or expert users dealing with baseline drift.

- **Output Surface**: Add `phaseOptReal` as a sixth display mode. Maintain
  WiMDA naming for the existing five modes to avoid the dangerous musrfit
  naming collisions (`phase` ≠ `Phase`; `real` ≠ `Phase`). Document the
  cross-program equivalences explicitly.

- **Apodization**: Keep WiMDA-style explicit tau/start control. Do not adopt
  musrfit qualitative levels.

- **Per-detector FFT**: Defer until the grouping metadata model is extended to
  support per-detector phase tables.

- **Testing**: Expand synthetic and reference-backed tests to cover musrfit-style
  preprocessing and phase correction, ensuring parity and correctness for both
  modes.

- **Documentation**: Update user and developer docs to clarify which features
  are WiMDA-parity, which are musrfit-advanced, and the naming equivalences
  between the two programs.
