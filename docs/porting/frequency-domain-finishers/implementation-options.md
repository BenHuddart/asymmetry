# Frequency-domain finishers — implementation options & plan

Settled after the study pass and checkpoint-3 decisions (Ben, 2026-06-11).
Implementation may start cold from this document. Conventions: worktree
`.venv/bin/python`; harness ladder with full `validate` green per phase;
GUI tests under `QT_QPA_PLATFORM=offscreen`; tests beside behaviour with
WiMDA-transcribed oracles where parity is claimed; no push / no PR.

## Decisions (checkpoint-3)

| Topic | Decision |
|---|---|
| Field axis | **Verify-only** — already built in `plot_panel` (`FieldUnit`). No new build; add verification tests. |
| Exclusions | **Wire** existing `exclude_frequency_ranges`; panel section + diamag-linked slot + PSI preset. |
| Pulse compensation | **Invert `R(ν)` from `pulse.py`**, guard = **cap gain (default 25×) + hard cutoff at the first node of G** (where R→0). Not WiMDA's unbounded Gaussian. |
| Baseline | **Iterative σ-clip** (median location, κ=2.0 default, capped iterations); 1-iteration = WiMDA single-pass parity. |
| S/N readout | **Extend** existing per-bin peak S/N + mean error; guard empties; DC-excluded peak search. |
| Real+Imag view | **New display mode**, overlay of cos/sin quadratures (Phase 1). |
| Burg | **Display-mode entry "Resolution (Burg)" + diagnostic badge**, default FPE scan **2–40 poles**, boundary warning (Phase 2). |
| Diamag removal | **Pre-FFT time-domain damped-cosine fit-and-subtract**; report fitted field to reference control **+ time-domain overlay** of the fitted line (Phase 2). |
| Correlation spectrum | **Deferred** follow-on. |
| N₀ single-histogram | **Deferred** follow-on. |

## Architecture

All post-FFT conditioning is a single pure-core function operating on the
**averaged display channel** and the canonical **MHz** frequency axis, inserted
into `compute_average_group_spectrum`
([`spectrum.py:184`](../../../src/asymmetry/core/fourier/spectrum.py)) right after
the average is formed (~line 276), in WiMDA's order: **pulse compensation →
baseline → exclusions**. The diamag fit-and-subtract is a *pre-FFT* time-domain
step (before `fft_complex_asymmetry`). Burg is a parallel spectrum builder that
consumes the same preprocessed time-domain signal. Keeping it all core-side
preserves the AGENTS.md core/GUI split and the project-recompute contract (the
GUI and `.asymp` reload run the identical recipe).

New config lives on `GroupSpectrumConfig` (spectrum.py); panel state in
`fourier_panel.get_state/restore_state`; persisted keys appended to
`_FOURIER_RECIPE_KEYS` (schema.py) — small, additive, at the end of the block.

---

## PHASE 1 — units & conditioning

Ends `validate`-green, main-mergeable. Steps in dependency order.

### 1.1 Core: conditioning module `core/fourier/conditioning.py` (new)
- `iterative_sigma_clip_baseline(values, *, kappa=2.0, max_iter=10) -> (baseline, noise_sigma)` —
  median/σ over inliers, iterate to σ-convergence; `max_iter=1` reproduces
  WiMDA single-pass. Returns the offset to subtract and the converged σ (for S/N).
- `pulse_compensation_gain(freqs_mhz, *, half_width_us, separation_us, n_pulses, max_gain=25.0) -> gain` —
  `1/R(ν)` from `pulse.pulse_amplitude_phase`, clipped to `max_gain`, **zeroed
  (or held) at and above the first node** of `R` (first index where R falls
  below a small floor). Returns the per-bin multiplicative gain.
- `apply_spectrum_conditioning(freqs_mhz, display, error, *, compensation, baseline, exclusions) -> (display, error)` —
  orchestrates compensation → baseline → exclusions (reusing
  `exclude_frequency_ranges`) on display **and** error; pure, no Qt.
- Reuse, do not re-derive: imports `pulse_amplitude_phase`,
  `exclude_frequency_ranges`. (A parallel pulse model is a review defect.)

### 1.2 Core: config + spectrum seam
- `GroupSpectrumConfig` (spectrum.py:54): add fields with safe defaults —
  `exclusion_ranges: list[tuple[float,float]] = []`, `exclude_enabled=False`,
  `diamag_exclusion=False`, `baseline_mode="none"` (`none|sigma_clip|wimda`),
  `baseline_kappa=2.0`, `pulse_compensation=False`, `pulse_half_width_us=0.0`
  (0 ⇒ metadata default), `pulse_separation_us=0.0`, `pulse_n_pulses=1`,
  `pulse_max_gain=25.0`. Extend `to_dict`/`from_dict` (append at end).
- `compute_average_group_spectrum`: after the averaged display/error are formed
  (~line 276), build the diamag exclusion centre from the run's reference field
  (γ_μ·B via `units.gauss_to_mhz`), resolve the pulse half-width from metadata
  when `pulse_half_width_us==0`, and call `apply_spectrum_conditioning`. For the
  new **Real+Imag** mode, retain both quadratures (return cos as `asymmetry`,
  sin in a metadata-carried companion array or a second dataset field — see 1.4).

### 1.3 Core: display mode plumbing
- Register `"Real+Imag"` (alias `real_imag`) in `fft.py` `_DISPLAY_ALIASES`/
  `_DISPLAY_MODES`, `spectrum.py` `_YLABELS`, and `fourier_display_values`
  (returns the real part; the imag companion is produced alongside in the
  averaged path). Keep WiMDA's five + `phaseOptReal` untouched.

### 1.4 GUI: Fourier panel
- New **"Conditioning"** collapsible group: pulse-compensation checkbox + width
  field (placeholder = metadata default) + max-gain; baseline mode combo
  (None / σ-clip / WiMDA single-pass) + κ; an **"Exclusions"** subsection with a
  ≤10-row editable table (centre MHz, half-width MHz), a "diamag-linked" toggle
  (row centre = reference field, read-only), and a **"PSI RF harmonics"** preset
  button (DC + 50.63×{1..5}, width = `2/τ` fallbacks per WiMDA).
- Add the **Real+Imag** radio to the FFT Phase Mode group.
- Extend `get_state`/`restore_state` with the new keys; extend the Info dialog
  with the new mode and conditioning formulas (rendered math).
- `set_average_summary`: surface baseline-noise S/N when σ-clip is active;
  guard empty/zero-error; exclude DC from the peak search.

### 1.5 GUI: controller + plot
- `mainwindow._on_compute_fourier`: pass the new panel state into
  `GroupSpectrumConfig`; for Real+Imag, plot the imag companion as a secondary
  overlay trace on `_frequency_plot_panel` (simple overlay; primary = real).

### 1.6 Schema
- Append the new keys to `_FOURIER_RECIPE_KEYS` (schema.py:112). No version
  bump needed (additive, defaulted) — confirm migration leaves old projects
  loading with defaults; add the round-trip test.

### 1.7 Phase-1 tests (`tests/test_fourier_finishers.py`, new)
Field axis (peak at γ_μ·B; FFT≡MaxEnt peak field); exclusions (band-zero, PSI
preset centres, diamag slot tracks reference); pulse compensation (flatten,
monotone uncompensated rolloff, guard bounds gain); baseline (σ-clip removes
offset/keeps peaks/converges; 1-iter == WiMDA); S/N (peak vs analytic, mean
error, empty guard); Real+Imag (both quadratures). Extend `test_project_schema`
for the new recipe keys. Reuse-asserting: compensation test imports
`pulse_amplitude_phase`.

### 1.8 Phase-1 docs
`docs/user_guide/frequency_finishers.rst` (new, added to
`docs/user_guide/index.rst` toctree): pedagogical, result-first, rendered math,
APS refs in a list, a "when to use this" register per feature (units,
exclusions, pulse compensation, baseline, S/N, real+imag). `python tools/harness.py docs` green.

### 1.9 Milestone
`validate` green → commit `frequency-domain-finishers Phase 1: field-axis
verify, exclusions wiring, pulse compensation, robust baseline, S/N, real+imag`.

---

## PHASE 2 — diagnostics

Ends `validate`-green, main-mergeable.

### 2.1 Core: `core/fourier/burg.py` (new, ~100 lines numpy)
- `burg_coefficients(signal, order) -> (ar_coeffs, power)` — Burg recursion
  (reflection coeff `κ_k = 2Σfb/Σ(f²+b²)`, `P_k = P_{k-1}(1-κ²)`, Levinson
  update, forward/backward residual update). Transcribed from WiMDA `memcof`.
- `ar_power_spectrum(ar_coeffs, power, freqs_mhz, dt_us) -> spectrum` —
  `√(P/|1-Σaₖe^{-2πikνΔt}|²)`, amplitude convention. From `evlmem`.
- `fpe_order_scan(signal, orders) -> (best_order, fpe_by_order)` —
  `FPE_m = P_m(N+m)/((N-m)·P₀(N+1)/(N-1))`, argmin log₁₀; boundary flag.
- `burg_spectrum(signal, freqs_mhz, dt_us, *, order_range=(2,40)) -> (spectrum, best_order, hit_boundary)`.

### 2.2 Core: wire Burg into the spectrum path
- New display mode `"Resolution (Burg)"` (alias `burg`). In
  `compute_average_group_spectrum`, when selected, run `burg_spectrum` on the
  **same preprocessed time-domain grouped signal** the FFT uses (build the
  signal, apply the same average-subtract/filter, then Burg instead of rfft),
  averaged across groups; force power-like output; carry `best_order` /
  `hit_boundary` into metadata for the badge/warning.

### 2.3 Core: diamag fit-and-subtract
- `core/fourier/diamag.py` (new) or a helper in `conditioning.py`:
  `fit_and_subtract_diamagnetic(dataset, *, seed_field_gauss) -> (clean_dataset, fitted_field_gauss, fitted_params)`.
  Damped-cosine model `A·cos(2π(f t+φ))e^{-λt}+c`, fit with the existing engine
  (iminuit), seed `f = γ_μ·B`; subtract from the time signal; return the fitted
  field. Pre-FFT, before `fft_complex_asymmetry`, gated by config.

### 2.4 GUI
- Add "Resolution (Burg)" radio with a diagnostic badge (accent colour, tooltip
  per the standing rule: qualitative super-resolution / line-count hint; never
  the quantitative result). On compute, show `best_order` and the boundary
  warning. Pole-range control defaults 2–40.
- Add "Remove diamagnetic signal" checkbox; on success write the fitted field to
  the reference-field control and overlay the fitted damped-cosine on the
  time-domain plot (`_plot_panel`).

### 2.5 Phase-2 tests (extend `test_fourier_finishers.py`, + `test_burg.py`)
Burg: doublet resolved where FFT merges (record window length for docs); FPE
tracks line count; boundary warning; spurious-splitting onset (record order for
docs); numerical oracle vs transcribed WiMDA `memcof`/`evlmem`. Diamag: fitted
field ≈ truth and reported back; subtraction flattens the line. Corpus smoke
(skip-gated): fitted field ≈ applied field.

### 2.6 Phase-2 docs
Extend `frequency_finishers.rst` with the **Burg** section (load-bearing:
state plainly what it is good for — qualitative super-resolution of close lines
from short windows; FPE-optimal pole count as a line-count hint — and its
pathologies — spurious splitting, noise-dependent bias, small position offsets,
no uncertainties; cite *Muon Spectroscopy* §15.5 and Burg 1972) and a
**diamagnetic removal** section. `docs` green.

### 2.7 Milestone
`validate` green → commit `frequency-domain-finishers Phase 2: Burg all-poles
diagnostic + diamagnetic fit-and-subtract`.

---

## File-by-file touch list

**New**
- `src/asymmetry/core/fourier/conditioning.py` (P1)
- `src/asymmetry/core/fourier/burg.py` (P2)
- `src/asymmetry/core/fourier/diamag.py` (P2)
- `tests/test_fourier_finishers.py` (P1, extended P2)
- `tests/test_burg.py` (P2)
- `docs/user_guide/frequency_finishers.rst` (P1, extended P2)

**Edited (small, additive)**
- `src/asymmetry/core/fourier/spectrum.py` — config fields, conditioning seam, Burg/diamag dispatch, Real+Imag.
- `src/asymmetry/core/fourier/fft.py` — register Real+Imag / Burg display aliases.
- `src/asymmetry/core/fourier/__init__.py` — export new public functions.
- `src/asymmetry/gui/panels/fourier_panel.py` — Conditioning/Exclusions sections, new radios, state, Info dialog.
- `src/asymmetry/gui/mainwindow.py` — pass new state; Real+Imag overlay; diamag field write-back + time-domain overlay; Burg badge/warning.
- `src/asymmetry/core/project/schema.py` — append to `_FOURIER_RECIPE_KEYS`.
- `docs/user_guide/index.rst` — toctree entry.
- `docs/porting/index.json` — study entry (this commit).

## Recorded follow-ons
- **Radical correlation spectrum** (Breit–Rabi `rmatch` → hyperfine axis) — defer; reuse `core/fitting/muonium.py` relations when promoted.
- **N₀-normalised single-histogram FFT input** — defer; interacts with count-domain PR #41.
- **Per-detector FFT** and **FB t=0 extrapolation** — out of scope (rationale in comparison.md).
- **Field-axis probe override** (¹⁹F/¹H γ) — `units.py` already supports it; expose only if radical work lands.

## Implementation status (both phases landed)

Both phases are implemented and validate-green. Deviations from the plan, with
rationale:

- **Diamagnetic fit** uses `scipy.optimize.curve_fit` (not the iminuit engine)
  for the one-off damped cosine: it is self-contained, and the fit is made
  robust by normalising the signal to unit scale and bounding the frequency to a
  window around the applied-field seed (the raw grouped-count signal is large and
  has a growing noise envelope, which otherwise lets the fit alias to a spurious
  high frequency). The fitted field is reported to the log and overlaid on the
  time-domain plot via `plot_panel.set_diamagnetic_overlay`.
- **Real+Imag** carries the imaginary quadrature in spectrum metadata
  (`fourier_imag`) and overlays it on the frequency plot via
  `plot_panel._overlay_fourier_imag`; the FFT preprocessing was extracted into
  `fft.prepare_fft_time_signal` so the Burg path shares the exact same input as
  the FFT (no parallel reimplementation).
- **Burg** is wired as the `Resolution (Burg)` display mode; its global maximum
  can sit on the DC/baseline peak, so the field-axis verification searches a
  window around the expected Larmor line. The documented pathology that the
  characterisation test pins is the proliferation of **spurious baseline peaks**
  at excessive pole count (the most robust, deterministic signature of the
  textbook "spurious splitting / baseline peaks" failure mode).

Both follow-ons (radical correlation spectrum, N₀ single-histogram FFT) remain
deferred as planned.

## Risks / watch-items
- Conditioning must stay on the canonical MHz axis; the plot panel owns
  unit display. Don't double-convert.
- `_FOURIER_RECIPE_KEYS` additions must default-load old projects (no version
  bump); verify with the round-trip test.
- Burg consumes the *same* preprocessed signal as the FFT — share the build
  path, don't fork preprocessing.
- Keep `mainwindow.py` / `schema.py` diffs minimal and additive (shared-touch
  files with other Wave-A work).
