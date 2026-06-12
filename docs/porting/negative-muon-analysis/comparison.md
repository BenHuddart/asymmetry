# Comparison: WiMDA `NegMuAnalyse` vs the Asymmetry μ⁻ API

Reference: `$WIMDA_SRC/src/NegMuAnalyse.pas` (2474 lines) + hooks in
`Analyse.pas` and `PlotPar.pas`. Object-Pascal source studied as an **oracle
only** (GPL). Adapt the physics and data; discard the form structure.

## 1. The model

WiMDA's `NegMuAnalyse` fits the raw decay-electron count histogram of a μ⁻ run
to a sum of up to **five elemental components** plus a **decay background**, each
a single exponential with its own amplitude:

    N(t) = Σ_{i=1..5} N_i · exp(−t/τ_i)  +  N_bg · exp(−t/τ_μ)  (+ flat bg)

- `τ_i` — the element-characteristic muonic-atom **total disappearance
  lifetime**, seeded from the built-in table (`mystrings`), shown in the grid's
  `Tau` row.
- `N_i` — per-element amplitudes. In forward/backward (`fgFB`) mode the grid
  carries `NF`/`NB` (forward/backward amplitudes) and an `alpha` detector
  balance; in single-group modes it carries one `N` per element.
- `τ_μ` — the free-μ⁻ decay-background lifetime; WiMDA sets
  `pp.Lifetime := 2197.03` ns (`NegMuAnalyse.pas:140`).

The asymmetry-style fit functions of ordinary μSR play no role in the basic
capture analysis — the observable is the **count decay**, not a polarization.
(The optional polarisation function — `None`/`LorGau`/`Diamagnetic` — multiplies
the model only in the μ⁻SR slice; §7.)

**Asymmetry adaptation:** a multi-exponential raw-count model
`N(t) = Σ_i N_i·exp(−t/τ_i) + bg`, with `τ_i` fixed at the literature value by
default (any τ freeable), the "decay background" being just another component
with `τ = MUON_LIFETIME_US`. Built and fitted in the new `core/negmu/` package.

## 2. Parameters (WiMDA hard-coded slots → Asymmetry named params)

WiMDA stores the fit in the global `p[]` array at fixed indices:

| WiMDA slot        | Meaning                              | Asymmetry name           |
|-------------------|--------------------------------------|--------------------------|
| `p[213..217]`     | forward amplitudes, elements 1–5     | `amp_<symbol>` (forward) |
| `p[219..223]`     | backward amplitudes, elements 1–5    | `amp_<symbol>` (backward)|
| grid `Tau` row    | per-element lifetime seeds           | `tau_<symbol>` (fixed)   |
| grid `Decay BG`   | free-μ⁻ background amplitude/τ        | `amp_decayBG`, `τ_μ`     |
| grid `alpha`      | F/B detector balance (`fgFB`)        | `alpha`                  |

The hard-coded slots and the `Fixed: array of array of boolean` grid are
**discarded** — Asymmetry uses named `Parameter`/`ParameterSet` entries with the
ordinary fix/limits/link machinery. Component naming is by **element symbol**
(`amp_C`, `tau_C`, …) plus the reserved label `decayBG`.

## 3. Why `core/fitting/count_domain` does NOT fit this (key reuse finding)

`count_domain.fit_single_histogram` / `fit_fb_alpha` fit

    N(t) = N0 · exp(−t/τ_μ) · [1 + s·A·P(t)] + bg

— a **single** muon-decay envelope `exp(−t/τ_μ)` modulating one polarization
`P(t)`. There is exactly one decay rate. The μ⁻ capture histogram is a **sum of
exponentials at different rates** (`τ_i` ranging from ≈2.2 μs for light elements
down to ≈0.07 μs for Pb); it cannot be written as one envelope times `(1 + a)`
for any physical `a(t)`. Therefore:

- **`count_domain`'s public fitters are not reusable** for the capture model.
- Its `_poisson_cash` / `_gaussian_chi2` cost functions **are private** and the
  module is **off-limits to modify** (scope decision). The new fitter therefore
  **replicates the ~6-line Cash statistic** and reuses everything else from the
  shared engine (`drive_minuit`, `FitResult`, `Parameter`/`ParameterSet`).

The standard `FitEngine.fit` is also not a fit: it minimises a Gaussian
`LeastSquares` over **asymmetry** data, not a Poisson cost over **raw counts**.
A composite of `ExponentialRelaxation` components expresses the functional *form*
`Σ A_i exp(−Λ_i t)`, but only as a percent-asymmetry model fitted with the wrong
statistics — see `implementation-options.md` §"Reuse audit / adapt-vs-new".

## 4. The capture-ratio report

`RatioButtonClick` (`NegMuAnalyse.pas:455–620`) computes amplitude ratios of a
chosen element of interest against others, separately for forward and backward:

    forward:  p[213+i] / p[213+j]        backward:  p[219+i] / p[219+j]

reported via `messagedlg` as e.g. `C/O: F = 2.000  B = 1.950`. The ratio of
capture amplitudes is the **relative capture probability** → elemental
composition. This is a **pure derived quantity** over fitted amplitudes; in
Asymmetry it is a small function `capture_ratio_report(fit, spec, …)` returning a
plain dataclass with covariance-aware uncertainties — **no new results
framework**.

## 5. Lifetime table — literature verification (mandatory)

WiMDA's table (`mystrings`/`myelements`, `NegMuAnalyse.pas:104–120`) is **not
adopted as-is**. It was verified against the standard compilation and three
concrete problems were found:

1. **Length mismatch.** `mystrings` declares **69** values but `myelements`
   declares **67** symbols. The two trailing lifetime values (`'0.076'`,
   `'0.07'`) have no element — orphan data.
2. **Symbol bug.** `myelements[66] = 'Ti'` — but position 66 in the sequence
   `…Au, Hg, ?, Pb` is **thallium**: it must read `'Tl'`. (Titanium already
   appears correctly at position 22.) A copy of the WiMDA "do not oracle against"
   ledger pattern.
3. **Value divergences** from the literature compilation, e.g. Ne 1.520 μs
   (WiMDA) vs **1.461 μs** (Suzuki/Measday/Roalsvig — and physically more
   sensible, ≈ the fluorine value, not above it); Zn 0.169 vs **0.161**; Sr
   0.142 vs **0.132**; Ba 0.072 vs **0.0949**; plus ~1–4 % differences across
   many mid-Z elements. WiMDA also lacks all **lanthanides** (its list jumps
   Ba → Hf).

**Authoritative source adopted:** *Muon Spectroscopy: An Introduction* (Blundell
et al., OUP 2022), **Appendix C, Table C.1**, "The lifetime of negative muons
(in nanoseconds) implanted in various elements", whose values are explicitly
"obtained by combining the measurements listed in" **T. Suzuki, D. F. Measday &
J. P. Roalsvig, *Phys. Rev. C* 35, 2212 (1987)**, with the free-muon lifetime
quoted as 2196.981(2) ns (RPP 2016). The full transcribed table (in μs, with σ,
WiMDA cross-value, and divergence flags) is the deliverable constants table in
[`plan.md`](plan.md) §"Element lifetime table". The decay-background τ_μ reuses
the existing `MUON_LIFETIME_US = 2.1969811 μs` (= 2196.981 ns).

> Bug-ledger addition (for `wimda-parity-gap/decision-record.md` §3 style):
> *NegMuAnalyse element table — `mystrings` length 69 ≠ `myelements` length 67;
> `myelements[66]='Ti'` is a typo for `'Tl'`; several lifetime values diverge
> from Suzuki/Measday/Roalsvig 1987 (Ne, Zn, Sr, Ba notably) and the whole
> lanthanide block is absent. Do not oracle the table against WiMDA.*

## 6. No GUI exposure — mechanism

The GUI fit-function pickers
(`gui/panels/fit_function_builder.py`, `model_fit_dialog.py`, `alc_panel.py`)
enumerate **only** the `COMPONENTS` and `MODELS` registries, via
`core/fitting/domain_library.components_for_domain` / `models_for_domain`. The
μ⁻ models live as **plain builder functions in `core/negmu/`** and are **never
passed through `registration.insert_definition`** (the single insertion path for
both registries). They are therefore invisible to every GUI surface **by
construction** — no separate registry, no excluded-from-picker flag, no risk of
accidental exposure. (The μ⁻SR polarisation functions in the Phase-4 slice are
likewise package-local and unregistered.)

## 7. Set-as-BG and the μ⁻SR polarisation slice

- **Set-as-BG** (`SetBgButtonClick`): evaluate the fitted model for the
  *unwanted* components and subtract from the displayed histogram, leaving the
  signal of interest. In Asymmetry this is a histogram-level subtraction of the
  evaluated unwanted-component model — `residual = counts − Σ_{i∈unwanted}
  N_i·exp(−t/τ_i)`. It reuses the Phase-1 model builder to evaluate the unwanted
  components; the subtraction itself is elementwise (cf. run-arithmetic's
  `subtract_scaled_counts` at the histogram level). **Phase 3.**
- **μ⁻SR polarisation function** (`None`/`LorGau`/`Diamagnetic` × `a0`/`λ`/`freq`/
  `phase`): the optional multiplier for μ⁻ spin-rotation work, where ≥5/6 of the
  spin polarization is lost in the muonic cascade (Blundell et al. 2022, §22.1).
  Package-local asymmetry-style functions, unregistered. **Phase 4.** The
  `PlotPar.pas` μ⁻-lifetime decay-correction hook is GUI plot territory and is
  **out of scope** (API-only) — recorded as a follow-on.

## 8. Left behind (deliberate non-ports)

- Hard-coded `p[213..223]` slots and the `Fixed` boolean grid → named params.
- The ~700-line GLE export generator → out (PDF/standard export supersedes;
  consistent with `decision-record.md` §1 "in-app GLE editor: drop").
- Commented-out blocks, the `Button3Click` grid-geometry code, `messagedlg`
  reporting → replaced by typed return values.
- WiMDA's `pp.Lifetime` mutation of the global plot lifetime → not needed; the
  decay-BG τ is an explicit component parameter.
- μ-XRF / muonic X-ray elemental analysis (the *other* μ⁻ technique) → out, as
  ever (`decision-record.md`; this is the lifetime method only).
