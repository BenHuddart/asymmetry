# Muoniated-radical correlation spectrum — implementation options & plan

Settled after checkpoint-1 (pre-study) and checkpoint-3 (post-study), Ben,
2026-06-11. This is the cold-implementable plan: chosen options, ordered phases
(each ending `validate`-green and main-mergeable), the file-by-file touch list,
the test plan, and recorded follow-ons.

## Settled decisions

| # | Decision | Choice | Rationale |
|---|---|---|---|
| D1 | `rmatch` vs exact map | **Exact Breit–Rabi forward map** through `muonium.py._tf_levels` (scan A_µ; pair sum = A_µ exactly). `rmatch` kept only as a test oracle. | Physical-correctness rule; reuses `muonium.py` (no re-derivation); no rounded constants; exact peak. ([comparison.md §2,§4.1](comparison.md)) |
| D2 | GUI exposure | **Badged display-mode radio** in the FFT Phase-Mode group ("Correlation (radical) — specialist"), revealing a small control group on select — the Burg precedent. | Folds into existing derived-display-mode machinery; un-prominent; zero new panel region. |
| D3 | Hyperfine-axis label | **"Muon hyperfine coupling A_µ (MHz)"**; the axis is **excluded** from the MHz/G/T field-unit selector. | Pedagogically explicit; A_µ ≠ γ_µ·B so field conversion is meaningless. ([comparison.md §5](comparison.md)) |
| D4 | Controls exposed | **Reference field override** (default = run field) **+ CorrFn order** (default 2). Axis range/step **auto** from Nyquist + resolution. | Matches WiMDA's `CorrField`/`CorrOrder`; axis range needs no manual control. |
| D5 | Averaging | **Correlate the averaged spectrum** (`CorrFn(mean S(ν₁₂), mean S(ν₃₄))`). Per-group "Corr" = select one group. True per-group `AvCorr` → follow-on. | Matches the averaged-channel architecture; better noise; documented divergence. ([comparison.md §4.1](comparison.md)) |
| D6 | Verification | **Synthetic-first** (`core/simulate` radical run → peak at known A_µ) + WiMDA `rmatch`/`CorrFn` oracle. | No real radical data needed/available. ([test-data.md](test-data.md)) |
| D7 | Worked example | **Cyclohexadienyl** (Mu + benzene), A_µ = 514.4(1) MHz. | Canonical; citable (textbook §19.4 Ex. 19.8; McKenzie 2013). |

## Phase plan

### Phase 1 — core (new `correlation.py` + pipeline seam)

1. **`src/asymmetry/core/fourier/correlation.py`** (new, ~80 lines numpy):
   - `corr_fn(y1, y2, order=2)` — verbatim port of `Plot.pas:1387-1394`
     (`order=0 → |y1·y2|`).
   - `breit_rabi_pair(field_gauss, a_mhz) -> (nu12, nu34)` — thin wrapper over
     `muonium.py._tf_levels` returning `(w12, w34)` with `w12+w34 == a` by
     construction. (Reuses `muonium.py`; does **not** re-derive.)
   - `correlation_spectrum(freqs, power, *, field_gauss, order=2, a_axis=None)`
     → `(a_mhz, corr)`: for each A on the hyperfine axis, get the exact pair,
     **linearly interpolate** `power` at `ν₁₂` and `ν₃₄` (np.interp; zero outside
     range, mirroring WiMDA's `i2<=nf` guard), combine with `corr_fn`. Default
     `a_axis` = uniform grid from `~2·diamag` up to the spectrum's max
     resolvable A (≈ Nyquist), at the spectrum's frequency resolution.
2. **`fft.py`**: append `"correlation"` to `_DISPLAY_ALIASES`
   (`"correlation"`, `"correlation (radical)"` → `"correlation"`); add
   `_YLABELS["correlation"]`. Keep the block additive at the end.
3. **`spectrum.py`**:
   - `GroupSpectrumConfig`: add `correlation_reference_field_gauss: float|None =
     None` and `correlation_order: int = 2` (additive in `to_dict`/`from_dict`).
   - `compute_average_group_spectrum`: detect `is_correlation`; compute the
     averaged **power** spectrum on the normal path, then transform it with
     `correlation_spectrum` (resolve reference field: config override → run
     field via `_reference_field_gauss`). Emit a correlation dataset with
     `x_label = "Muon hyperfine coupling A_µ (MHz)"`, `y_label`,
     `fourier_diagnostic = True`, `correlation_axis = True`,
     `fourier_correlation_field_gauss`. Skip post-FFT conditioning (like Burg).
4. **Tests** `tests/test_fourier_correlation.py`: Breit–Rabi `w12+w34=A`
   property; `corr_fn` fixed points + `order=0`; `rmatch` oracle agreement
   (~0.03 MHz) and exact-A_µ; synthetic cyclohexadienyl run → peak at 514.4 MHz
   (gating), second field, two-radical; `from_dict`/`to_dict` round-trip.

`python tools/harness.py validate` green; milestone commit.

### Phase 2 — GUI (badged radio + revealed controls)

1. **`gui/panels/fourier_panel.py`**:
   - Add `_correlation_radio` to the Phase-Mode `QButtonGroup`, WARN/accent
     styling + tooltip, label "Correlation (radical) — specialist"; register in
     `_DISPLAY_*` maps and `_current_display_mode`/`_set_display_mode`.
   - Add a revealed control group (enabled on toggle, like the Burg order spins):
     a reference-field `QLineEdit`/`QDoubleSpinBox` (Gauss; placeholder = run
     field) and a `CorrOrder` `QSpinBox` (default 2).
   - `get_state`/`restore_state`: persist the reference field and order.
   - Feed `correlation_reference_field_gauss` / `correlation_order` into the
     `GroupSpectrumConfig` built in `mainwindow._on_compute_fourier`.
2. **Axis handling**: when the active dataset has `correlation_axis`, the
   frequency-plot panel must **disable/ignore** the MHz/G/T field-unit selector
   and use the dataset's `x_label` (the A_µ axis is never γ_µ-converted).
3. **Tests** (`QT_QPA_PLATFORM=offscreen`): selecting the radio enables the
   controls and produces a correlation dataset; state round-trips; the field-unit
   selector is inert for the correlation axis.

`validate` green; milestone commit.

### Phase 3 — documentation (first-class deliverable)

New pedagogical user-guide page (e.g. `docs/user-guide/radical-correlation.md`
or the project's user-guide format) teaching radical-µSR to a non-specialist:

- **What a muoniated radical is** and how it forms (Mu addition to C=C / ring /
  C=O; β-muon; isotropic A_µ). Grounded in McKenzie 2013 §1.1/Fig. 1 and
  textbook §12.4.
- **Why a radical gives a line PAIR** and how the pair encodes A_µ (Breit–Rabi,
  high-TF `A_µ = ν₁₂ + ν₃₄`; rendered math, no equation numbers).
- **What the correlation spectrum's hyperfine axis means** and when to use it
  (the diagnostic "when to use": high field, liquids, resolvable precession,
  good yield; pitfalls — high-TF only, needs prompt radical + continuous source,
  diamagnetic line excluded, spurious low-order pairs).
- **Mandatory subsection** "TF correlation spectrum vs ALC — complementary
  routes to radical hyperfine couplings" (per the brief; content verified in
  [comparison.md §6](comparison.md)): both measure couplings from orthogonal
  directions (TF→A_µ; ALC Δ₁→A_µ cross-check, Δ₀→nuclear couplings + dipolar →
  orientation/dynamics); where each shines; the TF-then-ALC workflow; a concrete
  cross-reference to Asymmetry's existing ALC mode
  (`core/transform/integral.py`, `core/fitting/field_scan.py`, the ALC page).
- **Worked example:** cyclohexadienyl, A_µ = 514.4(1) MHz.
- Owner's style: result-first; rendered math; uncertainties `0.23(1)`; APS-style
  reference list; never cite the textbook's equations by number.
- Add to the user-guide toctree (additive, at end of block).

`python tools/harness.py docs` + `validate` green; milestone commit.

## File-by-file touch list

| File | Change |
|---|---|
| `src/asymmetry/core/fourier/correlation.py` | **new** — `corr_fn`, `breit_rabi_pair`, `correlation_spectrum` |
| `src/asymmetry/core/fourier/fft.py` | append `correlation` alias + ylabel |
| `src/asymmetry/core/fourier/spectrum.py` | config keys + derived-mode branch + correlation metadata |
| `src/asymmetry/gui/panels/fourier_panel.py` | badged radio + revealed controls + state |
| `src/asymmetry/gui/.../plot_panel` (frequency) | inert field-unit selector for the correlation axis |
| `src/asymmetry/gui/.../mainwindow` | pass correlation config fields |
| `tests/test_fourier_correlation.py` | **new** — core + oracle + synthetic gate |
| `tests/` GUI | radio/controls/state/axis tests |
| `docs/user-guide/radical-correlation.*` + toctree | **new** pedagogical page |
| `docs/porting/index.json` | this study's entry (additive) |
| `docs/porting/frequency-domain-finishers/{comparison,implementation-options}.md` | cross-link §9 follow-on → this study |

## Test plan

See [verification-plan.md](verification-plan.md). Gates: Breit–Rabi identity;
synthetic cyclohexadienyl peak at A_µ; `rmatch`/`CorrFn` oracle; no regression in
other Fourier/MaxEnt modes; project round-trip; docs harness + style review.

## Implementation status (all three phases landed)

- **Phase 1 (core):** `core/fourier/correlation.py` (`corr_fn`, `breit_rabi_pair`,
  `correlation_spectrum`); `correlation` display alias in `fft.py`; the derived
  mode + `correlation_reference_field_gauss`/`correlation_order` recipe keys in
  `spectrum.py`; `tests/test_fourier_correlation.py` (32 tests).
- **Phase 2 (GUI):** badged "Correlation (radical) — specialist" radio + revealed
  reference-field / order controls in `fourier_panel.py`; config wiring in
  `mainwindow.py`; correlation-axis lock of the field-unit selector + distinct
  x-label in `plot_panel.py`; recipe keys added to `core/project/schema.py`
  whitelist; GUI tests in `tests/test_gui_panels_basic.py`.
- **Phase 3 (docs):** `docs/user_guide/radical_correlation.rst` (pedagogical page
  + mandatory TF-vs-ALC complementarity subsection, cross-referencing
  `alc_mode`), added to the user-guide toctree.
- All phases `validate`-green.

## Recorded follow-ons (not in scope this sitting)

- **True per-group `AvCorr`** (per-group `CorrFn` then average) for bit-parity
  with WiMDA, if a user needs it (D5).
- **Hyperfine-axis range/step controls** (D4) if manual zoom is requested.
- **Low-field correlation** (textbook eqn 4.64, the ν₁₂/ν₂₃ pair) — only if a
  low-field radical use-case appears.
- **Link to `muonium-radical-hyperfine` fit workflow** — once a fitted A_µ
  exists, seed/confirm it against the correlation peak.
