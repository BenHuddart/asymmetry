# Fourier Transform Study

Status: WiMDA-first implementation complete; musrfit advanced features under
study

This study records how Fourier transform workflows are implemented in WiMDA,
musrfit, Mantid, and the current Asymmetry codebase before porting additional
frequency-domain behavior into Asymmetry.

The immediate porting risk was detector phase handling. The reference programs
do not treat FFT as a transform of one already-grouped asymmetry trace only.
They keep explicit phase controls around the transform path, and Mantid and
musrfit both retain per-detector phase information that can later be combined
or phase-corrected. The WiMDA-first pass addressed the core phase contract and
grouped FFT workflow. The next pass focuses on musrfit features not present in
WiMDA.

## Current Decision

The WiMDA-parity implementation is largely complete. The following features
have been delivered:

- core complex-spectrum API in `src/asymmetry/core/fourier/fft.py`
- manual phase rotation and WiMDA-style `t0` offset in the core seam
- all five WiMDA-style display modes: `(Power)^1/2`, `Phase Spectrum`, `Cos`,
  `Sin`, `Phase`
- WiMDA-style grouped-count lifetime correction before mean subtraction and
  apodisation
- grouped FFT averaging with per-group inclusion and per-group phase tables
- WiMDA-style Lorentzian / Gaussian / None apodization with explicit tau and
  start time
- automatic phase estimation (peak and power-weighted average methods)
- time-domain and frequency-domain dual-tab workspace
- frequency-tab x-axis unit toggle between MHz and Gauss
- field-relative x-axis display mode
- WiMDA-style average-error estimation for averaged grouped spectra
- in-app info surface that documents the FFT mode formulas with rendered
  equations
- scrollable Fourier dock suitable for the full grouped-FFT control surface

The following musrfit and Mantid features remain as follow-on work:

- musrfit linear phase correction (`c₁` slope term)
- musrfit entropy-based automatic phase optimizer (`phaseOptReal` mode)
- musrfit N₀ normalization for single-histogram FFT inputs
- musrfit range-based background subtraction
- phase-corrected imaginary quadrature output mode (`imag`)
- Tesla x-axis unit
- per-detector phase tables (Mantid / musrfit)

## Scope

- WiMDA FFT controls and per-group phase workflows
- musrfit Fourier preprocessing and phase-corrected transform options
- Mantid FFT plus phase-table workflows in frequency-domain analysis
- Current Asymmetry core FFT, GUI controls, and project-state seams

## Study Files

- `comparison.md`: implementation comparison across all four programs,
  including cross-program output mode name map and algorithm descriptions
- `implementation-options.md`: candidate ways to close the current gaps,
  including a feature decision matrix for musrfit features
- `test-data.md`: synthetic and reference-backed comparison cases
- `verification-plan.md`: focused validation for the implementation pass

## Current Asymmetry Baseline

- Core owner: `src/asymmetry/core/fourier/fft.py`
- Windowing owner: `src/asymmetry/core/fourier/window.py`
- Grouped signal builder: `src/asymmetry/core/fourier/grouped.py`
- GUI entry point: `src/asymmetry/gui/panels/fourier_panel.py`
- Existing tests:
  - `tests/test_fourier.py`
  - `tests/test_fourier_reference_methods.py`
  - `tests/test_gui_panels_basic.py`
  - `tests/test_project_schema.py`

Current Asymmetry behavior supports:

- FFT of one processed `MuonDataset`
- optional time cropping before transform
- WiMDA-style apodization windows (Lorentzian / Gaussian / None with tau and
  start time)
- simple zero-padding
- explicit complex-spectrum access
- manual phase rotation (`c₀` constant)
- WiMDA-style `t0` offset phase rotation
- WiMDA-style FFT mode selection over that complex spectrum: `(Power)^1/2`,
  `Phase Spectrum`, `Cos`, `Sin`, and `Phase`
- WiMDA-style grouped-count lifetime correction before later mean subtraction
  and apodisation
- grouped FFT averaging with optional WiMDA-style average-error estimates
- per-group inclusion toggles and per-group phase tables
- a frequency-axis toggle for absolute vs field-relative display
- MHz ↔ Gauss x-axis unit conversion
- an in-app info surface that explains the FFT mode formulas with rendered
  equations

Current Asymmetry behavior does not yet support:

- linear phase slope (`c₁` frequency-dependent correction)
- musrfit entropy-based automatic phase optimization (`phaseOptReal` mode)
- musrfit N₀ normalization (subtract and divide by estimated baseline)
- musrfit range-based or explicit background subtraction
- phase-corrected imaginary quadrature display
- Tesla or Mc/s x-axis units
- per-detector phase tables
- Mantid-style phase-table import or phase-quad workflows

## WiMDA vs musrfit Feature Mapping

See `comparison.md` for the full output mode name map and algorithm
descriptions. The most important naming collisions to be aware of:

| WiMDA label | musrfit label | Are they the same thing? |
|---|---|---|
| `Phase` (corrected real) | `real` | **Yes** — same formula, different names |
| `Phase Spectrum` (spectral angle) | `phase` | **Yes** — same formula, different names |
| `(Power)^1/2` = `\|F\|` | *(none)* | musrfit `power` = `\|F\|²`; **not the same** |

See `comparison.md` §"Cross-Program Output Mode Name Map" for the full table.

| Feature | WiMDA | musrfit | Asymmetry (current) | Recommendation |
|---|---|---|---|---|
| Phase family | Single constant `c₀` | Linear `c₀ + c₁·(i/N)` in optimizer only; constant `c₀` in all manual/CLI paths | Single constant `c₀` | Keep single-constant default; implement linear only as `phaseOptReal` optimizer backend; do not expose `c₁` as a panel control |
| Phase optimizer | Peak / power-weighted | Entropy + penalty (Minuit2) | Peak / power-weighted | Add entropy optimizer for `phaseOptReal` |
| Lifetime correction | `exp(t/τ_μ)` on counts | `exp(t/τ_μ)` + N₀ norm | `exp(t/τ_μ)` on counts | Add N₀ norm as advanced option |
| Background | Mean of signal window | Range-based or explicit | Mean of signal window | Add range-based as advanced option |
| Apodization control | Explicit tau + start | Qualitative levels | Explicit tau + start | Keep WiMDA style |
| Output `(Power)^1/2` | `\|F\|` | *(N/A; musrfit `power` = `\|F\|²`)* | `\|F\|` | Keep WiMDA style and name |
| Output corrected real | `Phase` | `real` | `Phase` | Keep WiMDA name |
| Output spectral angle | `Phase Spectrum` | `phase` | `Phase Spectrum` | Keep WiMDA name |
| Output entropy-opt real | *(N/A)* | `phaseOptReal` | *(N/A)* | Add as sixth mode |
| Per-detector FFT | No | Yes | No | Defer |

No code changes are made in the current documentation pass. Only documentation
and recommendations are updated.

## Advanced Follow-On Functionality

These areas stay covered by the comparison harness but are not part of the
current WiMDA-parity target:

- musrfit-style lifetime correction with `N0` renormalization before FFT
- musrfit linear or optimized phase families beyond a single manual phase plus
  `t0` term
- WiMDA diamagnetic subtraction and FFT-background correction, which still need
  a dedicated design for where that preprocessing should live in Asymmetry
- Mantid detector phase-table quadrature reconstruction (`PhaseQuad`)
- Mantid-style imported detector phase tables

## Candidate Port Seams

1. Core spectrum contract: make the complex spectrum explicit and phase-aware.
2. Fourier panel state: persist phase mode and manual phase alongside the
   existing window, padding, and display settings.
3. Main-window workspace: split time-domain and frequency-domain viewing into
   separate central tabs while keeping a shared plotting implementation.
4. Grouping boundary: attach future per-group or per-detector phase tables to
   the same metadata boundary that already carries detector-specific reduction
   choices.
5. Verification boundary: use synthetic oscillatory datasets to lock phase
   rotation behavior before adding GUI-triggered transform execution.
