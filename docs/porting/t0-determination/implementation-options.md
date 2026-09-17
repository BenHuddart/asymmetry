# t0 determination — findings, recommendations and options

## A. Findings, ranked by consequence

Each item names the file:line, the consequence, and the fix class. "Confirmed"
means reproduced in this study, not just read.

| # | Finding | Evidence | Consequence | Class |
|---|---|---|---|---|
| F1 | Fresh drafts mislabelled **Manual**: `_t0_policy_from_payload` compares `t0_bin` (F/B group max) against `max(detector_t0_bins)` over all detectors | `src/asymmetry/core/project/profiles.py:920-927` vs `core/io/root.py:604`, `psi.py:1065`; confirmed on a 15-detector GPS run (group max 1606, all-detector max 1614 ⇒ Manual 1606); the project's stored profiles carry it | Every fresh PSI/ROOT file with an out-of-group detector opens in Manual; the absolute value then becomes a real shift for other runs or after a group change | correctness |
| F2 | Policy honoured only by reduction: grouped Fourier, MaxEnt, count-domain fits align on file t0 | `core/fourier/grouped.py:308-320`, `gui/mainwindow.py:8302-8304`, `core/fitting/grouped_time_domain.py:292,346-349`; only `reduce.py:247-296` and `grouping.py::group_forward_backward` read `EFFECTIVE_DETECTOR_T0_KEY` | A Manual/Auto shift changes the asymmetry plot and fits but not the Fourier spectrum, MaxEnt, or count fits of the same run | correctness |
| F3 | MaxEnt time axis from the *shifted* `grouping["t0_bin"]` while counts are file-aligned | `core/maxent/engine.py:583-590` | Axis and data disagree by the policy delta | correctness |
| F4 | Count-fit t0 promotion sign inverted | model `t_eval = time + t0` (`core/fitting/count_domain.py:744, 960`); `promote.py:107-108` adds `round(t0/w)`; confirmed: stored t0 +3 bins ⇒ fitted +1.9 bins ⇒ promoted 103 → 105 (truth 100) | Promote t₀ doubles the error instead of removing it | correctness |
| F5 | Missing header t0 silently becomes bin 0 | `psi.py:1019,1526`, `root.py:578`, `nexus.py:914` | Unlabelled garbage alignment; musrfit warns loudly in the same case | correctness / UX |
| F6 | NeXus payload: `t0_bin = histograms[0].t0_bin`, no `detector_t0_bins` | `nexus.py:816-828` | Per-detector `time_zero` arrays (ISIS v2 can carry them) never reach profiles or the histogram-count fingerprint (`schema.py:978`) | consistency |
| F7 | `nexus_writer` mixes file `time_zero` with effective `corrected_time`/`first_good_bin` | `core/io/nexus_writer.py:129-133, 188-196` | A run saved under Manual/Auto is internally inconsistent | correctness |
| F8 | NeXus loader-time dataset axis = file `corrected_time`; re-reduction uses `(k − t0)·w` | `nexus.py:740-744, 943-958` | Possible half-bin difference between first display and every later re-apply | consistency |
| F9 | Manual stores an absolute bin; applied as `value − file_common_t0(run)` per run; docs call it an offset | `profiles.py:1307,1316`; `docs/reference/detector_grouping.rst:455-458` | Multi-run profiles shift each run by a different amount; meaning changes when groups change | semantics |
| F10 | No file-vs-detected display or warning | `gui/windows/grouping/dialog.py:2371-2389` shows one value per mode; `t0_search.rst:15-17` claims side-by-side | The user's stated requirement is unmet and the docs are wrong | UX |
| F11 | Plot-panel good-window mask uses detector 0's `time_axis` | `gui/panels/plot_panel.py:6268-6272` | Mask misplaced by `common − t0_0` bins on staggered-t0 data | UX |
| F12 | `promote_t0_to_grouping` shifts `t0_bin` but not `first_good_bin`/override; `_apply_t0_policy` writes search provenance before its `delta == 0` return | `promote.py:115`; `profiles.py:1313-1318` | Good window drifts; stale provenance keys | consistency |
| F13 | PSI/ROOT good window derived from per-detector offsets over **all** detectors while `common_t0` is over F/B only | `psi.py:1077-1080`, `root.py:614-629` | An excluded/unused detector can set the run's first good bin | consistency |
| F14 | `rebin.py:196-201` docstring says "left-edge" stamps; arithmetic is bin-centre (matches WiMDA/musrfit) | comparison.md §5 | Doc-only | docs |
| F15 | `Histogram.good_bin_start` docstring says "offset from t0"; loaders store absolute | `core/data/dataset.py:32-33` | Doc-only | docs |
| F16 | Corpus t0-recovery tests use hard-coded `~/Documents/...` paths | `tests/io/test_t0_search.py:118-123` | Not runnable elsewhere | tests |
| F17 | ISIS `t0_bin`/good-bin attributes are 1-based (resolved, [isis-header-index-base.md](isis-header-index-base.md)); the loader infers this per file from an axis vote that abstains on exact-edge files, and the integer bin discards the sub-bin position of `time_zero` (up to 8 ns at 16 ns binning ⇒ 39° TF phase at 0.1 T) | `nexus.py:1027-1068`; survey of 1,245 files | Half-bin stamp errors on some files; no cross-check when `time_zero` and `t0_bin` disagree (2003-era files) | correctness |

## B. Recommended design

The user's policy is: *default to file values; always display the value our
automatic method finds; warn when they diverge significantly.* The design
below implements it and fixes F1–F13.

### R1. The policy is stored, never inferred

- `T0Policy` becomes an explicit field of every profile and per-run
  override, defaulting to `from_file`. `_t0_policy_from_payload` is retired
  for new payloads; for legacy payloads (v11 migration, pre-fix projects) it
  infers Manual **only** from an explicit `effective_detector_t0_bins`, and
  otherwise compares against `_file_common_t0` (F/B groups with exclusions),
  the same function `_apply_t0_policy` uses.
- One-time heal on project open: a Manual policy whose resolved delta is 0
  for every run in scope is rewritten to `from_file` and logged. (This
  covers the mislabelled profiles already saved.)

### R2. One resolver for the effective per-detector t0

- `core/transform/t0.py` (or `grouping.py`) gains
  `effective_detector_t0_bins(run, grouping) -> list[int]`: override if
  present and well-formed, else the histograms' file values. `common_t0_for_groups`
  and `apply_grouping_aligned` take that list; every consumer in
  comparison.md §6 is routed through it (reduction, grouped Fourier, MaxEnt,
  count-domain fits, `nexus_writer`, the plot mask, `combine.py`).
- `tools/harness.py structural` gains a rule: no call to
  `common_t0_for_groups(` / `apply_grouping_aligned(` outside
  `core/transform/{grouping,reduce}.py` without a `detector_t0_bins=`
  argument. This is the "repeated review comment → harness" rule from
  `AGENTS.md`.

### R3. Always-visible detected value, with a divergence verdict

- On every preview-run change or group edit, the dialog requests a
  `RunT0Search` for the preview run from a debounced worker (the existing
  `_seed_t0_spin_from_detection` path already caches the resolve result; the
  search is keyed on the histogram content digest so it runs once per run).
- The **t0 Bin** row shows, in every mode, a read-only line:
  `File: bin 1606 · Detected: bin 1604 (prompt peak, spread 18) · Δ −2`.
  In From-file mode the spin shows the file value; in Auto the detected
  value; in Manual the resolved value; the line is the same in all three.
- **Verdict** (colour/icon on the line, plus a summary in the Apply
  validation): see R4.

### R4. Divergence thresholds

Thresholds are in bins because both the file and the search are quantised
to bins; they are exposed as module constants for calibration.

| Condition | Verdict | Initial threshold |
|---|---|---|
| file t0 missing / zero on a file that should carry one, or outside `[0, n_bins)` | **error** (banner; Auto-detect suggested) | — |
| `|consensus − file_common| > tol_source` | **warn** | `tol = 2` bins continuous (prompt peak is good to a few tenths of ns — textbook §15.3; PSI headers are set from the same peak), `3` bins pulsed (pulse-centre vs pulse-peak conventions differ by ~½ the rising edge; WiMDA places t0 at the pulse *peak*, comparison.md D9) |
| any detector: `|est_i − file_t0_i| > tol_source` | **warn (per detector)** — catches one dead or mis-set channel that the median hides | same |
| `spread_bins > 4·tol_source` | **warn** — strategy probably wrong (e.g. pulsed estimator on a continuous file with no facility token) | — |
| pulsed: `first_good − t0 < 0` or `first_good` before the rising edge has reached ≥ 90 % of peak | **warn** on tgood | — |

The corpus sweep in verification-plan.md replaces the initial numbers with
the observed `|Δ|` distributions per instrument before the thresholds ship.

### R5. Manual as an offset

- `T0Policy(mode="manual", offset_bins=k)`; resolution applies `k` to every
  run (`delta = k`), which is exactly what the absolute value already does
  for the reference run and removes the per-run inconsistency (F9). The
  spin keeps showing the resolved absolute bin for the preview run
  (`file_common + k`); editing it edits `k`. A per-run **override** (released
  run) may still store an absolute bin, since it applies to one run.
- Migration: an existing absolute `value` becomes
  `offset = value − file_common_t0(reference run)`; the project loader logs it.

### R6. Loader normalisation

- All loaders emit the same payload keys: `t0_bin` = F/B group max,
  `detector_t0_bins`, `detector_first/last_good_bins` (NeXus: from the
  per-detector `time_zero`/`t0_bin` array or the scalar broadcast) — F6.
- A missing/zero t0 sets `t0_source = "missing"` in the payload (no
  `Histogram.t0_bin` mutation; still 0) so R3/R4 can raise the error
  verdict instead of trusting bin 0 — F5.
- NeXus dataset axis rebuilt from `t0_bin` like PSI/ROOT (F8); the file
  `corrected_time` is kept in metadata for diagnostics.
- ISIS header bins decoded as **1-based, always**: 0-based `t0_bin = attr − 1`,
  `first_good_bin = attr − 1`, `last_good_bin = attr − 1` inclusive (F17). The
  axis vote survives only as a cross-check; when it, or `floor(time_zero/res) + 1`,
  disagrees with the attribute the loader records `t0_source = "conflict"`
  and prefers `t0_bin` (the 2003 file with a stale `time_zero`).
- PSI/ROOT good window derived over the F/B group detectors (F13).

### R10. Carry the exact t0 (decision needed)

ISIS `time_zero` sits at arbitrary sub-bin positions and PSI MusrRoot stores
`Time Zero Bin` as a double for the same reason. Proposal: loaders record
`t0_time_us` per run (ISIS: `time_zero`; MusrRoot: value × width; PSI bin:
the float "real t0" at byte 792 when non-zero) alongside the integer bin, and
the time-axis stamp becomes `(k + 0.5)·w − t0_time_us` (bin centre minus
exact t0) instead of `(k − t0_bin)·w`. Integer bins remain for detector
alignment, the good window, and every existing payload key, so the change
is confined to `rebin.py`/`asymmetry.py` stamps plus a `t0_time_us`
fallback of `(t0_bin + 0.5)·w` when the file has no exact value (which
reproduces today's numbers exactly). Mantid already does this; musrfit
does on its asymmetry path. Cost: a half-bin change on files whose exact
t0 is not at a bin centre, i.e. a visible (and correct) phase shift in
existing TF fits. Alternative: keep integer bins and document the ≤ ½-bin
limit — cheaper, but leaves up to 39° of TF phase error at 0.1 T on ISIS.

### R7. Fix the count-fit promotion (F4, F12)

`new_bin = current − round(t0_us / w)`; shift `first_good_bin` by the same
delta and, when an override list exists, shift it too (reuse
`_apply_t0_policy`'s offset routine). Pin with the synthetic round-trip test
in verification-plan.md.

### R8. Writer and plot consistency (F7, F11)

`nexus_writer` writes either file values throughout (default) or effective
values throughout (explicit flag), never a mix; the plot mask maps bins via
the reduction's `common_t0`.

### R9. Docs

Fix `t0_search.rst` (side-by-side claim becomes true), document the
`effective_detector_t0_bins`, `t0_search_*`, `t0_method`,
`t0_reference_run`, `t0_source` keys in `project_files.rst`, state the
bin-centre convention in `data_reduction` docs and the `rebin.py`
docstring, and correct the `Histogram.good_bin_start` docstring.

## C. Options considered and rejected

| Option | Why not |
|---|---|
| Default to **Auto-detect** (musrfit `-g` style) | Contradicts the user's policy and the facilities' own practice: ISIS writes an "exact value for time zero" from the DAQ; PSI's header is per detector and finer than a bin (`Double_t`). File first, detection as a check, matches WiMDA's and Mantid's defaults. |
| Keep inferring Manual from values, just fix the comparison | Inference from a *value* can never distinguish "user typed the file value" from "file value"; explicit storage (R1) removes the class of bug. |
| Threshold in microseconds | The search and the header are both bin-quantised; a µs threshold would flip meaning between 0.1 ns (HAL/GPS) and 16 ns (ISIS) binning. Bins, with a per-source constant, is the stable unit. |
| Sub-bin t0 (fractional `t0_bin`) | Only the count-fit nuisance needs it and it already has a continuous `t0`; MusrRoot's `Double_t` is rounded by everyone except musrfit's asymmetry axis. Out of scope; note as a follow-on. |
| Mantid-style common shift only (drop per-detector alignment) | Loses the PSI/ROOT per-detector header, which WiMDA and musrfit both honour. |

## D. Phasing

1. **P0 correctness** — R1 (inference + heal), R2 (resolver + harness rule),
   R7 (promotion sign), F3 (MaxEnt axis). Core + tests only.
2. **P1 loaders** — R6.
3. **P2 GUI** — R3, R4, R5 (offset semantics + migration), R8.
4. **P3 docs** — R9, and the `t0_search.rst` screenshot scenario.

Each phase is a separate PR; the study is updated with the final decisions
and the calibrated thresholds.
