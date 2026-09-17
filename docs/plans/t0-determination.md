# t0 determination: file default, always-visible detected value, divergence warning

Status: planned 2026-09-17 on `study/t0-determination` (study commits
52e6007, c1cf15d). Implementation lands as **one PR** from a feature branch
off `main`, built in six phases by subagents with a lead review gate after
each. Study and evidence:
[porting/t0-determination](../porting/t0-determination/README.md).

## Problem

Verified in code and data on 2026-09-17 (line numbers are anchors and must
be re-checked before editing):

1. **Fresh drafts open in Manual.** `_t0_policy_from_payload`
   (`core/project/profiles.py:895-928`) compares the stored `t0_bin`
   (forward/backward group max) with the max over *all* `detector_t0_bins`,
   so any out-of-group detector with a later header t0 flips the mode.
   Reproduced on a 15-detector GPS run; both profiles of that project carry
   the mislabelled policy.
2. **The policy is honoured by reduction only.** `EFFECTIVE_DETECTOR_T0_KEY`
   is read in `core/transform/reduce.py:247-296` and
   `grouping.py::group_forward_backward`; grouped Fourier
   (`core/fourier/grouped.py:308-320`, `gui/mainwindow.py:8302-8304`),
   MaxEnt (`core/maxent/engine.py:583-590`, which also builds its axis from
   the *shifted* `grouping["t0_bin"]`) and count-domain fits
   (`core/fitting/grouped_time_domain.py:292,346-349`) align on file t0.
3. **Promote t₀ has the wrong sign.** Model `t_eval = time + t0`
   (`core/fitting/count_domain.py:744,960`); `promote.py:107-108` adds
   `round(t0/w)`. Numerically: +3-bin corruption → fitted +1.9 bins →
   promoted 103 → 105 (truth 100).
4. **ISIS header bins are 1-based** (`t0_bin`, `first_good_bin`,
   `last_good_bin`; 1,245-file survey, [evidence](../porting/t0-determination/isis-header-index-base.md)).
   `nexus.py:1027-1068` infers this per file from an axis vote that abstains
   on exact-edge files; the integer bin discards the sub-bin position of
   `time_zero` (≤ 8 ns at 16 ns binning, 39° of TF phase at 0.1 T).
5. **No file-vs-detected display or warning**; `t0_search.rst:15-17`
   claims one exists. Missing header t0 silently becomes bin 0 in every
   loader. NeXus payloads carry detector 0's t0 and no `detector_t0_bins`
   (`nexus.py:816-828`). `nexus_writer.py:129-133,188-196` mixes file and
   effective values. The plot good-window mask uses detector 0's axis
   (`plot_panel.py:6268-6272`).

## Settled design (decision log)

Decisions taken with Ben on 2026-09-17.

- **D1 File default, stored policy.** `T0Policy.mode` defaults to
  `from_file` and is an explicit stored field; it is never inferred from a
  value comparison for new payloads. Legacy payloads infer Manual only from
  an explicit `effective_detector_t0_bins`, else compare against the same
  `_file_common_t0` the resolver uses.
- **D2 Heal silently, log it.** On project open, a Manual policy whose
  resolved delta is 0 for every run in scope becomes `from_file`; one log
  line per healed profile; the file is not rewritten until saved.
- **D3 Manual is an offset.** `T0Policy(mode="manual", offset_bins=k)`;
  every run shifts by `k` relative to its own file t0. The spinbox shows the
  resolved absolute bin of the preview run; editing it edits `k`. Legacy
  `value` migrates to `offset = value − file_common_t0(reference run)` at
  project open (schema v20 → v21). A released-run override may still store an
  absolute bin.
- **D4 Exact t0 for the time axis.** Loaders record `t0_time_us` (exact,
  µs from acquisition start). Stamps become `(k + 0.5)·w − t0_time_us`;
  the fallback `t0_time_us = (t0_bin + 0.5)·w` reproduces today's
  `(k − t0_bin)·w` exactly. Integer bins remain for detector alignment,
  the good window and every existing payload key. Sources: ISIS
  `time_zero`; MusrRoot `(Time Zero Bin + 0.5)·w` with the double kept;
  PSI bin and everything else: fallback. Per-detector data: the run's
  `t0_time_us` is the mean exact t0 of the detectors whose integer t0 equals
  the common bin; other detectors' sub-bin residuals are dropped (≤ ½ bin,
  documented). A Manual offset or Auto-detect consensus moves `t0_time_us`
  by whole bins (`delta·w`).
- **D5 ISIS decode is deterministic.** 0-based `t0_bin = attr − 1`,
  `first_good_bin = attr − 1`, `last_good_bin = attr − 1` inclusive. The
  axis vote survives only as a cross-check.
- **D6 Conflict: `t0_bin` attribute wins** over a `time_zero` that maps to a
  different bin (`floor(time_zero/res) + 1 ≠ t0_bin`); `t0_source =
  "conflict"`, `t0_time_us` falls back to the bin centre, loader warning.
- **D7 Missing t0 → auto-detect + warning.** A file with no usable t0
  (absent or zero on a format that carries one) gets `t0_source =
  "missing"`; resolution runs the search for that run and uses the
  consensus; the grouping window and run info show a persistent warning.
- **D8 Divergence thresholds per source, from the survey.** Continuous
  2 bins, pulsed 3 bins, as module constants (`T0_TOLERANCE_BINS`). Checks:
  `|consensus − file_common| > tol` → warn; any detector
  `|est_i − file_i| > tol` → warn (names the detectors); `spread > 4·tol` →
  warn (strategy suspect); file t0 outside `[0, n_bins)` → error; missing →
  error until D7 resolves it. Verdicts are produced in core
  (`assess_t0`), rendered by the GUI.
- **D9 Warn only, never block.** Apply always proceeds with the chosen
  mode; error-level verdicts show a banner and clamp.
- **D10 One resolver, harness-enforced.** `effective_detector_t0_bins(run,
  grouping)` in `core/transform/t0.py` is the only source of per-detector
  alignment; a structural rule forbids `common_t0_for_groups(` /
  `apply_grouping_aligned(` outside `core/transform/{grouping,reduce}.py`
  without `detector_t0_bins=`.
- **D11 Always show the detected value.** In every mode the t0 row carries
  a read-only line `File: bin N · Detected: bin M (strategy, spread S) · Δ`
  with the verdict, computed off the GUI thread once per run digest.
- **D12 Promotion sign.** `new_bin = current − round(t0_us / w)`;
  `first_good_bin` and any override list shift by the same delta.

## Code map

| Area | Files |
|---|---|
| Policy, resolution, heal, migration | `core/project/profiles.py` (`T0Policy`, `_t0_policy_from_payload`, `_apply_t0_policy`, `_file_common_t0`, `_PER_RUN_FACT_KEYS`), `core/project/schema.py` (v21 migration), `gui/mainwindow.py::_restore_project_state_impl` (~16404) |
| Resolver, search, verdicts | `core/transform/t0.py` (`find_t0`, `find_t0_for_run`; new `effective_detector_t0_bins`, `assess_t0`, `T0_TOLERANCE_BINS`), `core/transform/grouping.py` (`_detector_t0`, `common_t0_for_groups`, `apply_grouping_aligned`, `detector_t0_overrides`) |
| Exact t0 stamps | `core/data/dataset.py` (`Histogram`), `core/transform/rebin.py:174-270` (`binned_fb_asymmetry`), `core/transform/asymmetry.py:173` (`slice_to_good_window`), `core/transform/reduce.py`, `core/representation/time.py`, `core/transform/integral.py:660`, `core/data/combine.py:245-251`, `simulate.py:419-437,1858` |
| Loaders | `core/io/nexus.py` (`_t0_bin_values_from_attr`, `_infer_v2_bin_index_offset`, `_build_histograms`, payload 816-828), `core/io/hdf4.py` (v1 adapter), `core/io/psi.py:234-236,1019-1043,1077-1082,1146-1155`, `core/io/root.py:578-583,605-656`, `core/io/periods.py:403-476`, `core/io/nexus_writer.py` |
| Consumers | `core/fourier/grouped.py:308-320`, `core/fourier/spectrum.py:449`, `core/maxent/engine.py:583-590`, `core/fitting/grouped_time_domain.py:292,346-349`, `core/transform/promote.py:77-124`, `gui/mainwindow.py:8302-8304`, `gui/panels/plot_panel.py:6268-6272` |
| GUI | `gui/windows/grouping/dialog.py` (t0 row 547-592, 653-662, 874-882; `_seed_t0_*` 2336-2480; `_current_t0_policy` 2356; `_on_find_t0` 4285; validation 3028-3046), `gui/tasks.py::TaskRunner`, run-info surfaces |
| Tests | `tests/core/test_grouping_profiles.py`, `tests/io/test_t0_search.py`, `tests/io/test_{psi,root,nexus,hdf4}_loader.py`, `tests/io/test_nexus_writer.py`, `tests/core/test_promote_calibrations.py`, `tests/core/test_count_domain_fits.py`, `tests/gui/test_grouping_dialog.py`, `tests/gui/test_grouping_dialog_perf.py`, `tests/project/test_grouping_profile_migration.py`, `tests/tools/test_harness.py`, `tests/porting/t0-determination/isis_t0_header_survey.py` |
| Docs | `docs/reference/data_reduction/t0_search.rst`, `docs/reference/detector_grouping.rst:439-471,795-828`, `docs/reference/loading_data.rst:55-67,110-124`, `docs/reference/project_files.rst:318-323,410-413`, `docs/reference/count_domain_fitting.rst:146-148,224-229`, `docs/explanation/glossary.rst:100-106`, `docs/ARCHITECTURE.md:492-528`, `CHANGELOG.md` |

## Phases

Phases run **sequentially on the feature branch** (no worktrees), each ending
in one commit and a lead review before the next starts. Every agent brief
carries the standing rules below.

**Standing rules for every agent**

- Lean, efficient code: no new abstractions beyond those named in the
  phase; no defensive guards — prevent bad states by construction and raise
  when a contract is broken (Ben's rule); no `try/except` around code that
  cannot fail; no speculative options or feature flags.
- Core stays Qt-free. Never run reductions or searches on the GUI thread;
  worker results marshal through GUI-thread slots (`docs/GUI_GUIDELINES.md`
  § responsiveness).
- Tests live beside the behaviour (`tests/<layer>/`), mirror the module
  name, and are focused: run `python tools/harness.py test -- <files>`
  while iterating, `--tier fast` after core changes, affected GUI files
  focused after GUI changes. Do **not** run `validate` — the lead does,
  once, before the PR.
- Re-check every line anchor before editing; the study's numbers are from
  2026-09-17.
- Report back with: files touched, tests added (names), the exact harness
  commands run and their results, and anything left open. No summaries of
  code that was not changed.

### Phase 1 — Policy semantics and the resolver (core) · **Opus**

Goal: D1, D2, D3, D10, D12 and the MaxEnt axis, in core only.

- `core/transform/t0.py`: add `effective_detector_t0_bins(run_histograms,
  grouping) -> list[int]` (override if present and same length, else file
  values). Route `core/transform/reduce.py` and `grouping.py::group_forward_backward`
  through it (they currently call `detector_t0_overrides` directly).
- `core/transform/grouping.py`: `common_t0_for_groups` /
  `apply_grouping_aligned` keep their `detector_t0_bins=` parameter; no
  other change.
- `core/project/profiles.py`: `T0Policy` gains `offset_bins: int | None`
  (manual); `to_dict` emits `offset_bins`; `from_dict` accepts legacy
  `value` into a `legacy_value` field that resolution rejects loudly
  (migration converts it — Phase 1 also owns the v21 migration in
  `schema.py`, converting `value` → `offset_bins` using the stored
  `source_run`'s file t0 when the project loader has the run, else deferring
  to the project-open heal in Phase 5). `_apply_t0_policy`: manual delta =
  `offset_bins`. `_t0_policy_from_payload`: Manual only from an explicit
  `effective_detector_t0_bins`, else compare against `_file_common_t0`
  (F/B groups with exclusions); used for legacy payloads only. Fix the
  `delta == 0` early return so `t0_search_*` provenance is written with the
  consensus.
- `core/transform/promote.py`: D12.
- `core/maxent/engine.py:583-590`: axis from the resolver's common t0, not
  `grouping["t0_bin"]`.
- `tools/harness.py structural` + `tests/tools/test_harness.py`: D10
  rule.
- Tests (`tests/core/test_grouping_profiles.py`,
  `tests/core/test_promote_calibrations.py`,
  `tests/core/test_count_domain_fits.py`,
  `tests/project/test_grouping_profile_migration.py`, MaxEnt): out-of-group
  detector ⇒ `from_file`; legacy override ⇒ manual; offset semantics across
  two runs with different file t0; promotion round-trip (+3 and −3 bins
  recover the truth within 1 bin, `first_good_bin` moves with it); MaxEnt
  axis equals reduction's under a manual policy; structural rule fails on a
  bare call.

Acceptance: `--tier fast` green; the GPS repro (load one run, build a draft
from its payload) yields `from_file`.

### Phase 2 — Loaders: 1-based decode, exact t0, missing/conflict states · **Opus**

Goal: D4 (loader half), D5, D6, D7 (loader half), F6, F13.

- `core/data/dataset.py`: `Histogram.t0_time_us: float | None = None`
  with property `t0_time_us_effective` = value or `(t0_bin + 0.5)·bin_width`;
  fix the `good_bin_start` docstring (absolute index).
- `core/io/nexus.py` (+ `hdf4.py` v1 path): decode attributes as 1-based;
  `t0_time_us = time_zero` per detector when consistent, else conflict rule
  (D6); `t0_source ∈ {"file", "conflict", "missing"}` in the payload; write
  `t0_bin` = F/B group max and `detector_t0_bins` (+ good-bin tables) like
  PSI/ROOT; rebuild the loader dataset axis from the run's `t0_time_us`
  (F8), keeping the file `corrected_time` in metadata. Delete the axis vote
  as a decision path; keep it only as the cross-check that sets
  `"conflict"`. `bin_index_base` stays 1 for ISIS (display).
- `core/io/root.py`: keep the double `Time Zero Bin` → `t0_time_us`
  (D4); missing key ⇒ `t0_source = "missing"`, not 0.
- `core/io/psi.py`: `t0_source = "missing"` when the header t0 is zero for
  every histogram; good window over F/B detectors (F13); `t0_time_us`
  fallback.
- `core/io/periods.py`, `nexus_writer.py`: carry `t0_time_us`; the writer
  writes either file values throughout (default) or effective values
  throughout (`effective=True`), never a mix (F7).
- `core/project/profiles.py::_copy_per_run_facts` / `_PER_RUN_FACT_KEYS`:
  add `t0_time_us`, `t0_source`; `_apply_t0_policy` shifts `t0_time_us` by
  `delta·w`; D7: `t0_source == "missing"` ⇒ run the search at resolve and
  record `t0_source = "detected"`.
- Tests per `verification-plan.md` P1 plus: attr N ⇒ `t0_bin == N − 1` on an
  exact-edge synthetic file; `last_good_bin == n_bins` ⇒ `good_bin_end ==
  n_bins − 1`; conflict ⇒ attribute wins; missing ⇒ `"missing"`; writer
  self-consistency both ways; gated reader-parity against
  `ASYMMETRY_MUSRFIT_DATA`. The survey script must still report `floor+1`
  on 100 % of unambiguous files (run it on the local corpus list and paste
  the summary lines into the report).

Acceptance: `--tier fast` green; loader outputs on the study's nine
representative files (listed in the lead's review notes) match the
evidence table.

### Phase 3 — Exact-t0 time stamps · **Opus**

Goal: D4 (reduction half).

- `core/transform/rebin.py::binned_fb_asymmetry` and
  `core/transform/asymmetry.py::slice_to_good_window`: stamps from the run's
  effective `t0_time_us` (`(k + 0.5)·w − t0_time_us`); merged-bin stamps
  stay the mean of member stamps. Thread `t0_time_us` through
  `reduce.py`, `representation/time.py`, `integral.py`, `combine.py`
  (cross-run: mean over the aligned detectors as in D4),
  `simulate.py`, `fourier/grouped.py` and `fitting/grouped_time_domain.py`
  (Phase 4 reroutes their alignment; Phase 3 only fixes their stamps), and
  the `docs`-facing docstrings (`rebin.py:196-201`: bin-centre, not
  left-edge).
- Tests: with `t0_time_us` unset, every existing numeric test is unchanged
  (this is the acceptance gate); with `t0_time_us = (t0_bin + 0.3)·w` the
  axis shifts by exactly `−0.2·w`; a musrfit packing-formula oracle
  (`test-data.md`) for `p ∈ {1, 5}`.

Acceptance: `--tier fast` green with zero changes to existing expected
values; the reduction cache key includes `t0_time_us`.

### Phase 4 — Consumers through the resolver · **Sonnet**

Goal: D10 applied everywhere; F11.

- Route `core/fourier/grouped.py`, `gui/mainwindow.py:8302-8304`,
  `core/fitting/grouped_time_domain.py`, `core/fourier/spectrum.py` (digest
  already includes the override — keep), `core/transform/deadtime.py:98`
  (calibration window from the effective t0), `gui/panels/plot_panel.py`
  good-window mask (common t0 from the resolver) through
  `effective_detector_t0_bins`. Mechanical; no new behaviour beyond the
  reroute.
- Tests: a Manual offset of +2 bins changes the grouped Fourier signal, the
  count-fit trace start and the plot mask identically to reduction
  (`tests/core/test_fourier_grouped.py`, `tests/core/test_count_domain_fits.py`,
  `tests/gui/test_plot_panel.py`).

Acceptance: `--tier fast` green; structural rule passes with no exemptions
added.

### Phase 5 — GUI: detected line, verdicts, Manual offset, heal · **Opus**

Goal: D2 (project-open half), D3 (UI), D7 (display), D8, D9, D11.

- `core/transform/t0.py`: `assess_t0(run, grouping, search) -> T0Assessment`
  (`level ∈ {ok, warn, error}`, `messages: list[str]`, `delta_bins`,
  per-detector outliers) and `T0_TOLERANCE_BINS`. Pure, tested in core
  (`tests/io/test_t0_search.py` or a new `tests/core/test_t0_assessment.py`).
- `gui/windows/grouping/dialog.py`: the t0 row gains the read-only
  file/detected/Δ line with the verdict in every mode; detection runs via
  `TaskRunner` keyed on the preview run's content digest (reuse the cached
  resolve in `_last_resolved_seed` when present; never scan on the GUI
  thread); Manual edits store `offset_bins`; Find t0 fills the offset
  (spin shows the absolute); Apply summary lists warnings, never blocks
  (D9). Run-info surface shows `t0_source` and the verdict for
  missing/detected/conflict runs.
- `gui/mainwindow.py::_restore_project_state_impl`: D2 heal + D3 legacy
  `value` conversion where the loader could not (log lines via the existing
  logger).
- Tests: `tests/gui/test_grouping_dialog.py` (line present and identical in
  all modes; verdict levels for each D8 condition; offset semantics across
  two preview runs; Find t0 fills offset), `tests/gui/test_grouping_dialog_perf.py`
  (one scan per run digest), `tests/gui/test_mainwindow_additional.py`
  (heal + log line; legacy value migrated).

Acceptance: focused GUI files green; the GPS project opens with both
profiles healed to From file and a log line each; `gui-smoke` green.

### Phase 6 — Docs, changelog, study close-out · **Sonnet**

- `t0_search.rst` (side-by-side line and verdicts, quoting UI strings
  verbatim), `detector_grouping.rst` (modes; Manual as offset; exact t0
  and the bin-centre convention; `t0_source`), `loading_data.rst` (ISIS
  1-based decode replaces the heuristic text), `project_files.rst`
  (`offset_bins`, `t0_time_us`, `t0_source`, `effective_detector_t0_bins`,
  `t0_search_*`, `t0_method`, `t0_reference_run`; v21),
  `count_domain_fitting.rst` (promotion sign statement), glossary,
  `ARCHITECTURE.md`, screenshot scenario for the grouping window's t0 row (`docs/screenshots/scenarios/grouping_window_profile_editor.py`),
  `CHANGELOG.md [Unreleased]`, and `docs/porting/t0-determination/README.md`
  status → implemented with the final decisions.
- `python tools/harness.py docs`.

### Lead gate (after every phase)

Read the diff, not the report. Check: no guards, no dead options, resolver
used rather than re-derived, tests assert numbers not just shapes, anchors
re-checked, harness commands actually run. Rerun the phase's focused tests
and `--tier fast`. Before the PR: `validate` once, `gui-smoke`, `docs`,
the survey script on the local corpus, and the GPS project re-check
(delta 0, profiles healed).

## Acceptance criteria (PR)

- A fresh PSI/ROOT/NeXus run opens the grouping window in From file; the t0
  row shows file, detected, Δ and a verdict in all three modes.
- Switching to Manual and typing +2 bins shifts the asymmetry, grouped
  Fourier, MaxEnt axis, count-fit traces and the plot mask by the same
  amount; the raw histograms are untouched.
- Promote t₀ on a run whose stored t0 is 3 bins late moves it to within 1
  bin of the truth.
- ISIS files decode 1-based deterministically; the survey script reports
  `floor+1` on 100 % of unambiguous files; a 2003-era conflict file loads
  with `t0_source = "conflict"` and the attribute's bin.
- With `t0_time_us` absent every existing numeric expectation is unchanged;
  with an ISIS `time_zero` off-centre the axis moves by the sub-bin amount.
- The GPS project opens with both profiles healed to From file and delta 0
  on every run; a v20 project with a Manual `value` opens with the
  equivalent `offset_bins`.
- `validate`, `gui-smoke`, `docs` green; changelog and docs updated in the
  PR.

## Risks

- Exact-t0 stamps change TF phases on some existing fits (correctly). Call
  it out in the changelog and release notes.
- Reduction/Fourier caches keyed on payload digests must include
  `t0_time_us` and `t0_source`, or stale results survive a policy change.
- The GUI detection worker must not race the resolve worker; keying both on
  the run digest and reusing the cached resolve is the intended guard by
  construction.
