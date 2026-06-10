# Implementation Options: Model Function Parity

## Architecture (settled by prior ports and the umbrella brief)

- New functions are entries in `PARAMETER_MODEL_COMPONENTS`
  (`core/fitting/parameter_models.py`): pure-numpy evaluation, `ParamInfo`
  registration in `core/fitting/parameters.py`, applicability text +
  `PARAMETER_MODEL_REFERENCES` in `core/fitting/component_docs.py`,
  user-guide sections in `docs/user_guide/parameter_trending.rst`, tests
  beside behaviour.
- Machinery: core changes in `parameter_models.py` (+ a new shared
  `core/fitting/fit_quality.py`); GUI exposure only in
  `gui/panels/model_fit_dialog.py` (decision 2026-06-10 — cross-group dialog
  untouched, follow-on).
- Session-only: no `.asymp` schema changes (decision 2026-06-10 — model fits
  are not persisted today; persistence is a follow-on).
- Physics source: *Muon Spectroscopy: An Introduction* (Blundell, De Renzi,
  Lancaster, Pratt; OUP 2022); no equation numbers in user-facing text;
  APS-style reference lists.

## Proposed components (Phase 1)

| Component | Params (defaults) [limits] | Scope | Notes |
|---|---|---|---|
| `Polynomial` | c0 (0), c1 (1), c2–c5 (0) [unbounded] | common | monomial basis in absolute x; unused orders fixed at 0 by the user; `formula_template` shows the full quintic |
| `PowerLawQuadBG` | a (1), n (1), BG (0) [BG ≥ 0] | common | `hypot(a·\|x\|^n, BG)`; same \|x\| guard as `PowerLaw` |
| `MuRepolarisation` | a_Mu (15), A_hf (4463 MHz) [> 0], a_Dia (5) | field | (½+x²)/(1+x²) with x = B/B₀, B₀ = A_hf/(γₑ/2π + γ_μ/2π) from `constants.py`; x in G |

Composite recipes documented (not coded) in `parameter_trending.rst`:
`Arrhenius + Arrhenius` (with eV→meV and the 0.089 % constant note),
`OrderParameter + Constant`, `LorentzianLCR + LorentzianLCR + Polynomial`
(with the coefficient non-transfer warning). Deferred design note: generic
quadrature combinator (`⊕` in the composite grammar) — recorded in
comparison.md §1.3.

## Machinery design (Phase 2)

### Error modes

Core: an `ErrorMode` enum + small pure transform
`apply_error_mode(y, yerr, mode, value) -> yerr'` feeding
`fit_parameter_model`:

- `COLUMN` (default): propagated errors, stabilisation floor retained.
- `PERCENT`: σᵢ = (pct/100)·|yᵢ|; no floor; zero-σ points masked (existing
  mask), documented.
- `ABSOLUTE`: σᵢ = const; no floor.
- `NONE`: σᵢ = 1 (unit weights); no floor.
- `SCATTER` ("estimate errors from scatter"): unit-weight fit + post-hoc
  parameter-error rescale by √(χ²/dof) — the fixed point of WiMDA's
  Estimate iteration (equivalence proof in comparison.md §2.1, tested per
  test-data §1.6). χ²ᵣ carries no goodness information in this mode; the
  quality verdict is suppressed with an explanatory tooltip.

GUI: a compact "Errors" combo box + value field (enabled for
Percent/Absolute) in the dialog's data section, applying to all ranges of
the dialog (WiMDA semantics — the mode describes the data, not the model).

### Union multi-range

`ModelFitRange` gains `windows: list[tuple[float, float]] | None = None`.
Mask = OR over windows when present, else the existing (x_min, x_max). Core
mask construction factored into a helper shared by single and cross-group
paths (cross-group *core* inherits the capability; its GUI stays as-is).
GUI: per-range "+ window" affordance adding (min, max) spin-box rows under
the existing range row, mirroring the current range-row idiom.

### χ² quality helper (shared)

New `core/fitting/fit_quality.py`:

```python
@dataclass(frozen=True)
class FitQuality:
    verdict: Literal["good", "poor", "overdone"] | None  # None if dof < 1
    chi2_reduced: float
    band_low: float    # χ²ᵣ target band at confidence R
    band_high: float
    confidence: float  # R, default 0.95, clamped [0.5, 0.999]

def assess_fit_quality(chi_squared: float, dof: int, confidence: float = 0.95) -> FitQuality
```

Implemented with `scipy.stats.chi2` (CDF for the verdict, ppf for the band)
— exact WiMDA verdict semantics (two-sided test), better numerics.
Qt-free; `fit-workflow-diagnostics` (Wave B) reuses it for the time-domain
fit panel. Displayed in `ModelFitDialog` next to the existing χ² label, with
a teaching tooltip (what the band means at this ν; why χ²ᵣ below the band
suggests overestimated errors or overfitting; verdict assumes real errors —
suppressed for Scatter/None modes).

### Stretch: arbitrary X column

Only if all the above is green with session budget left: `x_key` extended to
any trended parameter (param-vs-param), scoping degrading to `common`.
Otherwise recorded as a follow-on (see below).

## Decisions

| # | Question | Decision (2026-06-10) |
|---|---|---|
| 1 | Persist new machinery state in `.asymp`? | No — session-only; persistence of model fits recorded as follow-on (Ben) |
| 2 | Expose error modes / multi-range in cross-group dialog? | No — `ModelFitDialog` only; core generic; follow-on (Ben) |
| 3 | Error-floor interaction with explicit modes | Floor applies in **Column mode only**; Percent/Absolute/None/Scatter bypass it (explicit user choices honoured verbatim; WiMDA has no floor) |
| 4 | Multi-window GUI idiom | Inline "+ window" rows: each range row gains a button adding (min, max) spin-box pairs beneath it, mirroring the existing range-row idiom; no fixed cap |
| 5 | χ²-band confidence R | Fixed 0.95 in the dialog; `assess_fit_quality(confidence=...)` parameterised in core, clamped [0.5, 0.999] like WiMDA's `Rgoodfit`; GUI exposure deferred to `fit-workflow-diagnostics` |
| 6 | Error-mode selector scope | Whole dialog (one "Errors" combo + value field in the data section; the mode describes the data series, not the model — WiMDA semantics) |

All decisions agreed with Ben 2026-06-10.

## Implementation plan

Implementation can start cold from this section plus comparison.md and
test-data.md. Branch: `feat/model-function-parity` (off `main` 19f242b).
Each phase ends with `python tools/harness.py validate` green
(`.venv/bin/python`; GUI tests need `QT_QPA_PLATFORM=offscreen`) and a
milestone commit. No push, no PR.

### Phase 1 — functions

1. `src/asymmetry/core/fitting/parameter_models.py`:
   - `_polynomial(x, c0..c5)` — plain monomial sum; register `Polynomial`
     (scope `common`; defaults c0=0, c1=1, c2–c5=0; all unbounded;
     `formula_template`/`latex_equation` show the quintic).
   - `_power_law_quad_bg(x, a, n, BG)` — `np.hypot(a*|x|^n, BG)` with the
     same 1e-12 |x| floor as `_power_law`; register `PowerLawQuadBG`
     (scope `common`; BG ≥ 0).
   - `_mu_repolarisation(x, a_Mu, A_hf, a_Dia)` — B₀ derived from
     `constants.py` (γₑ/2π + γ_μ/2π in MHz/G, computed in the module, not
     hard-coded); r = (x/B₀)²; y = a_Mu·(0.5+r)/(1+r) + a_Dia; register
     `MuRepolarisation` (scope `field`; A_hf > 0 hard minimum, default
     4463 MHz; a_Mu default 15, a_Dia default 5).
2. `src/asymmetry/core/fitting/parameters.py`: ParamInfo entries for c0–c5,
   BG, a_Mu, A_hf (MHz), a_Dia (unicode/latex/GLE labels, units,
   `default_min` where bounded).
3. `src/asymmetry/core/fitting/component_docs.py`: applicability text for
   the three components (diagnostic "when to use this" register, rendered
   symbols, no inline citations) + `PARAMETER_MODEL_REFERENCES` entries
   (APS style; *Muon Spectroscopy: An Introduction* by name for
   MuRepolarisation, plus original literature where applicable).
4. Tests (new `tests/test_wimda_model_function_parity.py`, pattern of
   `test_wimda_parity_components.py`): oracle tables from test-data
   §1.1–1.4, Polynomial round-trip, MuRepolarisation B₀/limit/recovery
   checks, recipe identities (Arrhenius delta D4, OrderParameter+Constant
   ≡ func5, LorentzianLCR ≡ WiMDA peak), docs-policy assertions for the new
   names.
5. EuO regression (verification-plan §1.5) against the PR #15 numbers.
6. `docs/user_guide/parameter_trending.rst`: sections for the three new
   functions (template: existing "Magnetic Order Parameter" section) + a
   "WiMDA model-function migration" subsection holding the three composite
   recipes with the D2/D4 warnings.
7. `validate` green → milestone commit.

### Phase 2 — machinery

1. `core/fitting/parameter_models.py`:
   - `ErrorMode` (`StrEnum`: `column`, `percent`, `absolute`, `none`,
     `scatter`) + `apply_error_mode(y, yerr, mode, value) -> NDArray | None`
     (pure; returns the σ array to fit with, or unit weights).
   - `fit_parameter_model` gains `error_mode: ErrorMode = ErrorMode.COLUMN`
     and `error_value: float | None = None`; floor applied only for
     `COLUMN`; `SCATTER` = unit-weight fit then multiply
     `uncertainties` by √(χ²/dof) and flag the result
     (`ParameterModelFitResult.error_mode` field for the GUI).
   - `ModelFitRange.windows: list[tuple[float, float]] | None = None`;
     factor a `range_mask(x, fit_range)` helper (OR over windows, fallback
     to x_min/x_max, empty/inverted windows rejected with a clear error);
     use it in `fit_parameter_model` / `evaluate_parameter_model_fit` and
     the cross-group data path (core inherits; cross-group GUI unchanged).
2. New `core/fitting/fit_quality.py`: `FitQuality` dataclass +
   `assess_fit_quality(chi_squared, dof, confidence=0.95)` per the design
   above (scipy.stats.chi2; Qt-free; docstring records WiMDA semantics and
   the fit-workflow-diagnostics reuse contract).
3. `gui/panels/model_fit_dialog.py`:
   - "Errors" combo (Column/Percent/Absolute/None/Estimate from scatter) +
     value field (enabled for Percent/Absolute) in the data section;
     plumbed into `_run_fit`'s `fit_parameter_model` call.
   - Per-range "+ window" button adding (min, max) spin-box rows; rows
     round-trip `ModelFitRange.windows`; remove-window affordance.
   - Quality verdict label next to the existing χ² text driven by
     `assess_fit_quality`, colour-coded (good/poor/overdone), teaching
     tooltip; suppressed (with explanation) for `scatter`/`none` modes and
     dof < 1.
4. Tests: error-mode behaviours (test-data §1.7), scatter fixed-point
   equivalence (§1.6), union multi-range synthetic + mask unit tests
   (§1.8), fit_quality oracle table + edge cases (§1.5), offscreen GUI
   tests (selector → core inputs; window editor round-trip; verdict label),
   EuO λ(T) qualitative multi-range check.
5. `docs/user_guide/parameter_trending.rst`: "Weighting and error modes"
   and "Fit windows" subsections (result-first; when to use scatter
   estimation and its caveat; why to exclude the critical region and how
   the verdict reads).
6. `validate` + `docs` green → milestone commit; update study README status
   and this file with outcomes.

### Stretch (only if Phases 1–2 green with budget left)

Arbitrary X column: extend the dialog's x-source to any trended parameter
(param-vs-param), `component_names_for_x` degrading to `common` scope for
non-field/temperature x. If not reached: stays follow-on #2.

**Outcome (2026-06-10): not reached — stays follow-on #2.** Scoping showed
the x-axis selection is woven through `fit_parameters_panel.py` (~4 kloc:
x-combo, trend assembly, state save/restore, GLE labels), so param-vs-param
is a panel-level feature, not a dialog tweak. The core is already x-agnostic
(`fit_parameter_model` takes any x; `component_names_for_x` degrades to
`common` for unknown keys), so the follow-on is GUI-only.

## Verification outcomes (2026-06-10)

- Phases 1 and 2 implemented as planned; full `validate` green after each
  (1787 → 1810 passed); docs build clean.
- All WiMDA-transcribed oracles pass to ≲1e-12 where forms are identical;
  divergences D1–D10 asserted behaviourally (D11/D12 are design facts).
- **EuO regression**: headless core pipeline (PSI runs 2928–2943,
  `Oscillatory*Exponential + Constant` per run on t = 0–8 µs, trend fitted
  with `OrderParameter`) gives Tc = 69.24(9) K, β = 0.409(7), α = 1.18(4) —
  within combined uncertainties of the PR #15 GUI extraction
  (Tc = 69.2(1) K, β = 0.417(7), α = 1.23(5)); the ν(T) trend is frozen as a
  fixture in `tests/test_wimda_model_function_parity.py` so the regression
  runs without the corpus.
- Scatter-mode equivalence with WiMDA's Estimate iteration is *tested* by
  explicitly iterating the WiMDA scheme
  (`test_scatter_mode_is_fixed_point_of_wimda_estimate_iteration`).
- MuRepolarisation recovers B₀ = A/(γₑ+γ_μ) (vacuum Mu 1585.0 G from
  `constants.py`) and round-trips (a_Mu, A_hf, a_Dia) on exact synthetics;
  no clean isotropic-Mu corpus B-scan exists (test-data.md §2), so the
  synthetic oracle is the quantitative record.

## Post-implementation review fixes (2026-06-10)

A 7-angle recall-biased review surfaced 10 confirmed findings, all fixed:

1. **Persistence (premise correction)**: model fits DO reach `.asymp` via the
   fit-parameters panel state — `windows`, `error_mode`, `n_points` are now
   serialized in both `fit_parameters_panel._serialize_model_fits` and the
   global-parameter-fit-window serializer (lenient `parse_fit_windows`
   reader; legacy state loads with defaults).
2. Cross-group dialog: inherited error-mode combo and "+ Window" controls are
   suppressed via `_supports_error_modes`/`_supports_windows` class flags
   (the cross-group backend honours neither); hidden "active" checkboxes now
   stay hidden across UI rebuilds (`_post_rebuild_ranges_ui` hook).
3. Window/bounds edits invalidate the stored result (status + χ²/verdict
   labels reset); `fit_parameter_model` returns a failed result (never
   raises) for invalid windows, and `evaluate_parameter_model_fit` skips
   them, so plotting paths cannot crash on mid-edit state.
4. Panel overlay/GLE sampler spans the window-union envelope via the shared
   `effective_range_bounds` helper; `.fit` sidecars record `fit_windows`.
5. Scatter mode with ν < 1 reports indeterminate errors (empty
   uncertainties + message) instead of collapsing them via the ndof clamp.
6. Quality verdict is silent for results with unknown `n_points` (legacy /
   cross-group bridge) instead of claiming "no degrees of freedom"; ν now
   uses `ParameterSet.free_parameters`.
7. Percent error mode: zero/negative value falls back to 1 % instead of
   masking every point.
8. `scipy.stats` imported at `fit_quality` module level (lazy in-function
   import cost a measured ~0.9 s freeze on the UI thread at first verdict).
9. Stale quality label cleared on parameter edits.
10. MuRepolarisation reuses `G_E_MHZ_PER_G + G_MU_PER_G`-style constants and
    `VACUUM_MUONIUM_A_HF_MHZ` from `core/fitting/muonium.py` (B₀ now
    bit-identical to the Breit-Rabi components; default A_hf 4463.302, was a
    stale 4463.0).

## Follow-ons (recorded regardless of stretch outcome)

1. **Send model-fit results to the results table** — model-fit outputs
   (per-range parameters + errors + verdict) exported as rows so trending
   recurses (supersedes WiMDA's second-level Model Fit Table).
2. **Arbitrary X column** (if not reached as stretch).
3. **Persist model fits in `.asymp`** (schema bump, append-only).
4. **Cross-group dialog exposure** of error modes + union ranges.
5. **Generic quadrature combinator** in the composite grammar.
6. **`python-user-functions`** (Wave B) generalises this registry — land
   after this project.
