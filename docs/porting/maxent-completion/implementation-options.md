# MaxEnt completion — implementation plan

This is the actionable plan. It is written to be started **cold** from the
committed docs: it names files, line anchors, the order of work, and the tests
that gate each phase. Read [`comparison.md`](comparison.md) (WiMDA ↔ Asymmetry
behaviour, §0 on the V1 kernel) and [`README.md`](README.md) (scope decisions)
first. All paths are under the worktree
`~/Source/Asymmetry-worktrees/maxent-completion/`. Use the worktree's
`.venv/bin/python`.

## Chosen options (settled with Ben, 2026-06-10)

| Choice | Decision |
|---|---|
| Pulse-shape validation | Synthetic-only; implement single **and** double pulse |
| Deadtime promotion | Suggest-only; explicit apply to grouping, with provenance |
| ZF/LF generality | Strict 2-group F/B parity |
| Reconstruction overlay home | **Plot-workspace view token** (new time-domain view, gated like `maxent`) |
| Units helper API | **Simple converter functions** in `core/fourier/units.py` |
| ZF/LF table constraint | **Mode selector hard-constrains** (exactly two F/B groups; phases pinned read-only; α from grouping; run blocked otherwise) |
| Schema versioning | **No bump**; add data-affecting fields to `_state_signature` only |

## Architecture spine (applies across phases)

1. **Build on the projected-gradient V1.** Do not rewrite the inner loop. New
   physics folds into the **real cosine/sine forward–adjoint** maps
   (`_project_forward` `engine.py:637`, `_project_forward_components` `:660`,
   `_project_adjoint` `:685`) and the outer nuisance fit (`_fit_group_nuisance`
   `:909`). The pulse response enters as a per-frequency complex weight applied
   to the cosine/sine projections (see Phase 2). This is the load-bearing
   decision; everything else is plumbing.

2. **Config is the contract.** Every new knob is a field on `MaxEntConfig`
   (`engine.py:123`) with a default, serialised in `to_dict`/`from_dict`
   (`:146–207`, which already defaults missing keys → old projects load
   unchanged, no schema bump). The panel exposes it via `get_state`/
   `restore_state` (`maxent_panel.py:310`/`:339`), which flow straight into
   `maxent_config()` (`:337`). Data-affecting fields get added to
   `_state_signature` (`engine.py:761`) so a resumed state restarts on change.

3. **Keep `maxent_panel.py` modular.** The `spectral-moments` Wave B project
   adds to this same panel next wave. Prefer additive `QGroupBox`/tab sections;
   if introducing a `QTabWidget`, keep the existing controls in a "Run" tab and
   add new tabs ("Tables", etc.) so moments can add its own cleanly.

4. **Shared files: small, additive, end-of-block.** `mainwindow.py` (menu/view
   hooks), `core/project/schema.py` (no change expected), `docs/porting/
   index.json` (append-only), user-guide toctrees (append at end of block),
   `core/utils/constants.py` (add `PION_LIFETIME_US`).

## Phase 1 — reconstruction overlay

**Goal.** Expose the per-group reconstructed time spectra the engine already
computes (`opus`), overlay them on the grouped time-domain data in the plot
workspace (per-group + combined) with a residuals strip, extend the result /
diagnostics, and persist the toggle in the recipe.

### Steps

1. **Engine — expose the reconstruction.** Add a function
   `reconstruct_group_signals(result_or_state, maxent_input) -> dict[int,
   ReconstructedGroup]` in `core/maxent/engine.py` that wraps `opus(spectrum,
   maxent_input, phases=…, amplitudes=…, backgrounds=…)` (`:708`) and packages,
   per group: `time_us`, `data` (the engine's normalised signal
   `group.signal`), `model` (the `opus` prediction), `sigma`, `mask`, and
   `residual = (data − model)/sigma`. Reuse `MaxEntInput.groups` for the data /
   time / σ arrays. Add per-group χ² and the total (must equal
   `MaxEntResult.metadata["maxent_chi2"]` — assert in tests). Keep the existing
   `opus` untouched.
   - `MaxEntResult` is frozen and holds `state` + `metadata`; the reconstruction
     is cheap and deterministic from `state.spectrum` + `maxent_input`, so
     compute it on demand rather than storing arrays (consistent with
     representations caching transient datasets, not arrays).
   - Persist a compact reconstruction summary (per-group χ², n_obs) into
     `MaxEntResult.metadata` / diagnostics so the GUI can show χ² without
     recomputing.

2. **Representation — new time-domain reconstruction view.**
   - Add `RepresentationType.TIME_MAXENT_RECON = "time_maxent_recon"` to
     `core/representation/base.py:26` and map it to `"time"` in `DOMAIN_OF`
     (`:41`).
   - Add a `TimeMaxEntReconstruction` representation in
     `core/representation/time.py` modelled on `TimeGroups`
     (`time.py:115–148`, which returns one dataset per group via
     `build_grouped_time_domain_datasets`). Its `compute()` reads the MaxEnt
     recipe (same `recipe["maxent_config"]` block the `FrequencyMaxEnt` rep
     uses, `frequency.py:103`), rebuilds the input, and returns per-group
     `MuonDataset`s carrying both the data curve and the model curve (use
     dataset metadata to tag `role="data"`/`role="model"` and the residual), so
     the plot layer can draw model-on-data + residuals. `recompute_on_load =
     False` (matches MaxEnt — expensive, recomputed on demand).
   - Wire it into the representation factory / `container.make_representation`
     (same place `FrequencyMaxEnt` is constructed).

3. **Plot workspace — new view token + renderer.**
   - Add `"reconstruction"` to `_VIEW_TOKENS` (`plot_workspace_panel.py:20`);
     it is a **time-domain** token (not in `_FREQUENCY_VIEWS`).
   - Gate availability exactly like `"maxent"`: in
     `_on_maxent_worker_finished` (`mainwindow.py:4966`) add `"reconstruction"`
     to the available views alongside `"maxent"` after a successful run.
   - Render: extend `plot_panel.plot_grouped_time_domain_subplots`
     (`plot_panel.py:2250`) — or add a sibling `plot_maxent_reconstruction` —
     to draw, per group, the data points + the model line + a residuals strip
     beneath (matplotlib `subplots` with a height-ratio split, or a twin
     stacked axis). A **combined** view overlays the selected-group mean (or a
     representative group) in one axis. Residual plotting is net-new in
     `plot_panel.py` (grep confirms no residual axis today) — add it here.

4. **Panel — overlay toggle.** Add a "Show reconstruction overlay" checkbox to
   the MaxEnt panel; persist it as `MaxEntConfig.show_reconstruction` (default
   `True` per the engine study's recommendation that it is the strongest
   diagnostic). Add the field to `MaxEntConfig` + `to_dict`/`from_dict` and to
   panel `get_state`/`restore_state`. (Toggle gates whether the
   `"reconstruction"` view is auto-selected after a run; the view token remains
   available regardless.)

### Touch list (Phase 1)

- `core/maxent/engine.py` — `reconstruct_group_signals`, `ReconstructedGroup`
  dataclass, metadata summary, `show_reconstruction` config field + signature.
- `core/representation/base.py` — enum + `DOMAIN_OF` entry.
- `core/representation/time.py` — `TimeMaxEntReconstruction`.
- `core/representation/factory.py` / `container.py` — construction wiring.
- `gui/panels/plot_workspace_panel.py` — view token.
- `gui/panels/plot_panel.py` — reconstruction+residuals renderer.
- `gui/windows/mainwindow.py` — gate the view in `_on_maxent_worker_finished`;
  select it on toggle.
- `gui/panels/maxent_panel.py` — overlay checkbox + state keys.
- `tests/test_maxent.py` (or new `tests/test_maxent_reconstruction.py`) +
  an offscreen panel/plot test.

### Tests (Phase 1) — gate: `validate` green

- **S1** reconstruction-within-noise + **χ² equals engine χ²** (exact).
- per-group reconstruction arrays equal `opus(spectrum, …)` exactly.
- recipe round-trip with `show_reconstruction` (**P1** subset).
- offscreen: view becomes available post-run; renders per-group + combined +
  residuals without crashing; shown χ² == result metadata χ².

## Phase 2 — pulsed-source correctness

**Goal.** ISIS pulse-shape response in the forward model (single/double),
interior exclusion window via σ-inflation, and field-axis (Gauss/Tesla) display
via the new units helper.

### Steps

1. **Constants.** Add `PION_LIFETIME_US = 0.026` to `core/utils/constants.py`
   (alongside `MUON_LIFETIME_US = 2.1969811`, `:33`).

2. **Pulse-shape kernel.** New module `core/maxent/pulse.py`:
   - `pulse_response(frequencies_mhz, *, half_width_us, separation_us,
     n_pulses) -> (p_cos, p_sin)` returning the per-frequency real weights
     `P_c(ν)=CONVOL_R(2πν)`, `P_s(ν)=CONVOL_I(2πν)` from `comparison.md` §2.
     `n_pulses=0` → `(ones, zeros)`. DC: `p_cos[ν=0]=1, p_sin[ν=0]=0`. Use
     CODATA `τ_µ`, `τ_π`. (ω in rad/µs with ν in MHz, t in µs.)
   - Pure function, unit-tested directly (S2 kernel asserts) — no full MaxEnt
     run needed for the kernel checks.

3. **Fold into the forward/adjoint.** Thread an optional pulse weight pair into
   `_project_forward`/`_project_forward_components`/`_project_adjoint` and the
   `opus`/`tropus`/`_fit_group_nuisance`/`_residual_gradient_payload` callers.
   The forward model becomes
   `amp·[P_c(ν)·(C@f) − P_s(ν)·(S@f)] + bg` — i.e. scale the cosine projection
   by `P_c` and the sine projection by `P_s` before combining (the engine
   already computes `(C@f, S@f)` in `_project_forward_components` `:660`; extend
   the single-cosine `_project_forward` `:637` to the two-component form when a
   pulse is active, falling back to the current path when not). The adjoint
   transposes the same weighting. Because the weight is diagonal in frequency,
   this is `O(n_freq)` extra work, no kernel rebuild.
   - The lifetime envelope `E(t)` is already applied upstream
     (`build_group_signal_dataset(apply_lifetime_correction=True)`,
     `engine.py:573`); do **not** re-apply it.

4. **Config.** Add `pulse_mode` (`"ignore"|"single"|"double"`, default
   `"ignore"`), `pulse_half_width_us`, `pulse_separation_us` to `MaxEntConfig`
   (+ to_dict/from_dict + signature — pulse settings change the model, so they
   must invalidate resumed state). Default `pulse_half_width_us ≈ 0.05` (50 ns),
   `pulse_separation_us ≈ 0.324` (ISIS), but seed from instrument metadata where
   the loader exposes it — **research item:** check whether `nexus.py` captures
   ISIS pulse width/separation; if so, default from it, else from these
   constants. Record the finding in `comparison.md`.

5. **Exclusion window.** Add `exclude_t_min_us`, `exclude_t_max_us` (default
   `None`). In `build_maxent_input` (`engine.py:582`), for points inside the
   interior window, **inflate σ** by a large factor (mirroring the engine's
   input model — multiply `normalized_sigma` by e.g. `1e8`) rather than dropping
   them; keep the head/tail `t_min`/`t_max` as masking. The mask stays full
   length (FFT/grid length intact). Add to signature.

6. **Units helper** `core/fourier/units.py` (shared with
   `frequency-domain-finishers` — API recorded below). Add a field-axis unit
   selector to the MaxEnt panel (MHz / Gauss / Tesla) persisted as
   `MaxEntConfig.field_axis_unit` (display-only; does not change the engine).
   The spectrum plot relabels the x-axis and converts tick values via the
   helper. Replace the engine's private `_field_to_frequency_mhz`
   (`engine.py:118`) with a call into the helper (keep behaviour identical).

### Units helper API (recorded — `frequency-domain-finishers` reuses verbatim)

`core/fourier/units.py`, pure functions on the existing CODATA constants
(`MUON_GYROMAGNETIC_RATIO_MHZ_PER_T = 135.538817`, `GAUSS_TO_TESLA = 1e-4`):

```
class FieldUnit(StrEnum):  # "mhz" | "gauss" | "tesla"
    MHZ; GAUSS; TESLA

mhz_to_gauss(mhz) -> gauss          # mhz / (γ/2π · 1e-4)
gauss_to_mhz(gauss) -> mhz          # gauss · γ/2π · 1e-4
mhz_to_tesla(mhz) -> tesla          # mhz / (γ/2π)
tesla_to_mhz(tesla) -> mhz          # tesla · γ/2π
gauss_to_tesla(g) / tesla_to_gauss(t)   # · 1e-4 / · 1e4
convert(value, frm: FieldUnit, to: FieldUnit) -> float   # via MHz pivot
axis_label(unit: FieldUnit) -> str                       # "Frequency (MHz)" / "Field (G)" / "Field (T)"
frequency_resolution_mhz(bin_width_us, n_spectrum_points) -> float   # 1/(2·Δt·N)
```

All operate on scalars and numpy arrays (numpy-friendly). γ/2π is the muon
value; the gyromagnetic ratio is a parameter with the muon default so the
sibling project can pass the fluorine/proton constants if needed.

### Touch list (Phase 2)

- `core/utils/constants.py` — `PION_LIFETIME_US`.
- `core/maxent/pulse.py` — new pulse-response module.
- `core/maxent/engine.py` — pulse fold-in across forward/adjoint/opus/tropus/
  nuisance/gradient; exclusion σ-inflation; new config fields + signature;
  units-helper call.
- `core/fourier/units.py` — new shared helper.
- `gui/panels/maxent_panel.py` — pulse-mode selector + width/sep fields;
  exclusion-window fields; field-axis unit selector; state keys.
- `gui/panels/plot_workspace_panel.py` / `plot_panel.py` / `mainwindow.py` —
  field-axis relabel/convert on the spectrum plot.
- `tests/test_maxent.py`, `tests/test_fourier_units.py`, offscreen panel test.

### Tests (Phase 2) — gate: `validate` green; Phase-1 tests pass

- **S2** flat-amplitude recovery (enabled) vs roll-off (disabled); single-pulse
  limit of double-pulse; DC values. Optional Mantid `start.py` kernel oracle
  (skipif-guarded).
- **S3** interior exclusion: recovery + grid-length-unchanged.
- `test_fourier_units.py`: MHz↔Gauss↔Tesla round-trips, resolution helper.
- **P1/P2**: recipe round-trip with pulse+exclusion+units; signature forces
  restart on pulse-mode / exclusion change.

## Phase 3 — calibration workflows

**Goal.** Deadtime fitting inside MaxEnt (suggest-only promotion), editable
per-group phase/amplitude/deadtime tables with phase exchange, ZF/LF 2-group
mode + SpecBG, and spectrum/log export.

### Steps

1. **Deadtime fit (DEADFIT equivalent).** Add a deadtime nuisance to
   `_fit_group_nuisance` (`engine.py:909`), gated by a new
   `fit_deadtime` config flag (default off). Per group, fit the
   non-paralysable deadtime against the normalised residual using the
   linearised model from `comparison.md` §4, reformulated against the
   **pre-normalised group counts** (thread the per-group count scale / frames /
   bin-width through `MaxEntGroupInput` so the `∝ counts²` term and the physical
   µs conversion are well-defined). Report per-group deadtime in physical µs in
   the diagnostics. Add `fit_deadtime` to the signature.
   - **Suggest-only promotion**: a panel action "Apply deadtime to grouping"
     writes the fitted per-group deadtime into the run grouping's deadtime
     correction, stamped with provenance (run, cycle count, timestamp via the
     caller — scripts pass time in, no `Date.now()` in core). Never auto-write.

2. **Tables tab + phase exchange.** Add a "Tables" tab to the MaxEnt panel (or a
   `QGroupBox` if not converting to tabs) surfacing per-group phase (°),
   amplitude, and fitted deadtime (µs). The per-cycle dicts already arrive via
   `set_diagnostics` (`maxent_panel.py:444`, payload `diagnostics["phases"][-1]`
   etc.) keyed by group id — render the latest, editable for phase/amplitude.
   - **"Use fitted phases"**: read the grouped time-domain fit's per-group
     `relative_phase` (radians) via `group_specs_from_grouped_fit`
     (`core/simulate.py:833`) or `fit_panel.grouped_simulate_seed_for_run`
     (`fit_panel.py:4082`); seed `MaxEntConfig.group_phase_degrees` with
     `rad2deg`, matched **by group id**. Stamp a provenance label
     ("from <fit label>, <timestamp>").
   - **"Send phases to fit"**: write MaxEnt phases (°) back to the
     `TIME_GROUPS` representation's grouped-fit `relative_phase` params as
     radians (`deg2rad`), matched by group id, with a provenance label.
   - Unit conversion at every boundary is the main correctness trap (S6).

3. **ZF/LF mode + SpecBG.**
   - Add `mode` (`"general"|"zf_lf"`, default `"general"`) to `MaxEntConfig`.
     A panel **mode selector** hard-constrains (settled): ZF-LF requires exactly
     two included groups (F/B); their phase cells auto-set to 0/180 and become
     read-only; `fit_phases` disabled; α read from the run grouping and shown;
     the run is blocked with a clear message if not exactly two groups.
   - In `_fit_group_nuisance`, when `mode == "zf_lf"`, after the per-group
     amplitude/background least-squares, apply the α-tie
     `x[B] = (x[F]+x[B])/(1+α); x[F] = α·x[B]` to amplitudes and backgrounds
     (`comparison.md` §3), and hold phases at 0/180. Add `mode` to the signature.
   - **SpecBG**: new `core/maxent/specbg.py` `subtract_zero_frequency(spectrum,
     frequencies, *, gauss_width, lorentz_width, lorentz_fraction) -> spectrum`
     implementing the zero-centred pseudo-Voigt of `comparison.md` §3 (carry the
     `×1.201` constant, documented as empirical; anchor to the lowest in-window
     bin). **Display-only**: operate on a copy of the spectrum dataset for the
     field-distribution view, never the engine spectrum. Add SpecBG width/
     fraction fields to the panel (shown only in ZF/LF mode).

4. **Export.** Spectrum text export (two-column frequency/field + density, with
   a parameter header) and a run log (per-cycle χ²/entropy/TEST + final
   phases/amps/deadtimes), modern CSV-like, **on demand** (never auto-save). Add
   panel actions + small `core/maxent/export.py` helpers.

### Touch list (Phase 3)

- `core/maxent/engine.py` — deadtime nuisance + count-scale threading;
  ZF/LF α-tie + phase pinning; `fit_deadtime`/`mode` config + signature.
- `core/maxent/specbg.py`, `core/maxent/export.py` — new modules.
- `core/transform/deadtime.py` or grouping write-path — suggest-only promotion
  target (reuse the existing grouping deadtime field; do not invent a new one).
- `gui/panels/maxent_panel.py` — Tables tab, phase-exchange actions + provenance
  labels, ZF/LF mode selector + group-table constraint, SpecBG fields, export
  actions. **Keep modular** (spectral-moments adds here next wave).
- `gui/panels/fit_panel.py` / `core/fitting/grouped_time_domain.py` — read path
  for fitted phases (already exists) and the write-back target.
- `gui/windows/mainwindow.py` — menu/action hooks (minimal, additive).
- `tests/test_maxent.py` (+ deadtime/zf/specbg/exchange tests), offscreen tests.

### Tests (Phase 3) — gate: `validate` green; Phases 1–2 pass; **docs build clean**

- **S4** injected-deadtime recovery (physical µs; corrected-run ≈ 0; suggest-only
  does not mutate grouping).
- **S5** ZF Kubo–Toyabe: α-tie obeyed, phases pinned, spectrum broad near zero;
  SpecBG on a copy.
- **S6** phase-exchange round-trip (rad↔deg, by id, provenance attached + survives
  project round-trip).
- **P1** full recipe with every new field; **P2** signature restart for ZF/LF +
  deadtime.
- offscreen: tables tab; exchange actions + provenance; ZF/LF constrains the
  table; export produces well-formed files.

## User-facing documentation (part of the deliverable, each phase)

New pages under `docs/user_guide/` following the `fit_functions/` template
(result-first physics prose; rendered math; uncertainties as 0.23(1); APS refs
in lists; a "when to use this" register per feature). Add to the relevant
toctree (append at end of block). Cite *Muon Spectroscopy* (Blundell, De Renzi,
Lancaster & Pratt, OUP 2022) by name; never cite its equations by number.

- **Phase 1**: "Reading the MaxEnt reconstruction overlay" — what the overlay
  and residuals tell you about fit quality; the spectrum *is* p(B), B = 2πν/γ_µ.
- **Phase 2**: "Pulsed vs continuous sources and the pulse-shape response" — why
  amplitudes above ~10 MHz roll off on ISIS, when to enable single/double pulse,
  the exclusion window for glitches; field-axis units. Cite §14.2–14.3, §15.5.
- **Phase 3**: "MaxEnt calibration: phases, deadtime, and ZF/LF mode" — the
  phase-exchange workflow, deadtime fitting and when to promote it, ZF/LF mode
  for Kubo–Toyabe field distributions and the TF-vs-ZF caveat, SpecBG. Cite
  §5.1, §15.3, Exercise 15.1.

## Recorded follow-ons / candidate-entry notes

- **`phase-auto-calibration` candidate**: this project absorbs its WiMDA slice
  (fitted-phase ↔ MaxEnt exchange). Note that in the candidate entry when
  touching `docs/porting/` (the umbrella's portfolio table already routes it
  here).
- **`spectral-moments` (Wave B)**: consumes this spectrum and adds to
  `maxent_panel.py`; this plan keeps the panel modular for it. No code here.
- **Mantid kernel oracle**: if `MaxentTools/start.py` imports standalone, the
  skipif-guarded oracle test is a cheap future hardening — record the import
  result in `comparison.md` §10 during Phase 2.
- **Instrument-metadata pulse defaults**: if `nexus.py` does not currently
  capture ISIS pulse width/separation, defaulting from constants is acceptable
  for this project; capturing them from the loader is a small follow-on for the
  data-loading family.
- **Looseness / phase-accel knobs**: verdict **out** (not deferred) — see
  `comparison.md` §9. If a real convergence pathology appears in Phase 3
  testing, fix it in the engine's existing χ²-plateau/divergence guard, not via
  a resurrected knob; record any such fix here.

## Risk register

| Risk | Mitigation |
|---|---|
| Pulse fold-in subtly wrong (sign/convention) | Unit-test the kernel at DC and the single-pulse limit first; flat-amplitude recovery (S2) is the integration check; optional Mantid `start.py` oracle. |
| Deadtime unit round-trip (normalised ↔ physical µs) | Thread explicit count-scale/frames/bin-width; S4 corrected-run-returns-0 anchors it. |
| Radians↔degrees phase-exchange bug | Convert at every boundary; S6 round-trip; match by group id, never row index. |
| Residual plotting is net-new in `plot_panel.py` | Build on the existing stacked-subplot renderer (`plot_panel.py:2250`); keep it a separate method. |
| Panel grows monolithic (moments comes next) | Tab structure ("Run"/"Tables"); additive sections; no cross-coupling. |
| Resumed state silently uses stale settings | Every data-affecting field added to `_state_signature` (`engine.py:761`); P2 asserts the restart. |
