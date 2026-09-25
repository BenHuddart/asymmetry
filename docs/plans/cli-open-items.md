# Agent CLI: the open corpus items

Branch `feat/cli-open-items`, one PR. It follows #334 (reduction options) and
#335 (ALC scans) and takes every item those PRs and the 2026-09-23 corpus audit
left open, except the pass-17a evaluation-reliability items (skill work, which
waits until the CLI unlocks the whole corpus — see
`docs/plans/agent-cli-skill.md`). CLI and core only: no skill text, rubrics or
evaluation waves; the generated `references/commands.md` is refreshed at the
end.

The evidence below comes from four read-only studies run on 2026-09-25 against
`main` at 8845d9a and the WiMDA muon school corpus
(`~/Documents/WiMDA muon school/`, never copied into the repo).

## The items

| # | Item | Unblocks | Phase | Agent |
|---|---|---|---|---|
| 1 | Encoded period run numbers in integral scans and alpha | benzene RG, any `--period` scan | 1 | Sonnet |
| 2 | `recipe --run` seeds a negative amplitude and no frequency | every TF recipe written by hand (LiFeAs) | 1 | Sonnet |
| 3 | Survey of a folder whose runs sit in sub-folders says nothing | benzene `data/` | 1 | Sonnet |
| 4 | Two instruments sharing run numbers in one folder | Basics | 2 | Opus |
| 5 | Background subtraction in integral scans | PSI integral scans; GUI integrals | 3 | Opus |
| 6 | `RFResonanceMuP` couplings seeded from the scan | RF scans of other radicals | 3 | Opus |
| 7 | Differential (red/green) ALC line shape | benzene liquid RG scan | 4 | Opus |
| 8 | HiFi longitudinal runs stamped TF | every HiFi ALC survey | 5 | Opus |
| 9 | PSI logged sample temperature; implausible logged T in Basics | EuO, LiFeAs, Basics | 5 | Opus |
| 10 | Co-adding runs and the correlation spectrum | benzene high TF | 6 | Opus |
| 11 | Fits across stored results (batch of groups; law through trend fits) | maleic Arrhenius, LLZ | 7 | Opus |
| 12 | Radical repolarisation | benzene repolarisation, corannulene | 7 | Sonnet (seeds only — D5) |
| — | Docs, changelog, command reference, corpus re-check | — | 8 | Sonnet, then lead |

## Decisions for the maintainer (proposed; settle before Phase 2)

- **D1 — two instruments in one folder.** Every command takes `--instrument
  NAME`, matched case-insensitively against the file prefix (`EMU` selects
  `EMU…` and `emu…`, which are one instrument across eras), and a work
  directory is bound to *(folder, instrument)* in its manifest. Without the
  flag a folder whose run numbers collide is refused with the instruments
  named; `survey` still lists every file but measures alpha per file, not per
  run number. Rejected: keying the work directory on `prefix+run` (touches
  every stored name and every `--runs` parser for one folder).
- **D2 — co-added runs.** `reduce --coadd` reduces the sum of the named runs
  (`combine_runs`, the GUI's path) and stores it under the *first* member's run
  number, the entry recording the members and every command naming it
  "co-added from 3678–3682". A work directory therefore holds either the co-add
  or its first member's own reduction; use `--workdir` for both. Rejected:
  string run keys (a work-directory schema change for one worksheet step).
- **D3 — PSI logged sample temperature.** The `.bin` header carries four
  unlabelled sensor means (float32 at offset 716) and deviations (offset 738).
  Record every sensor as `psi_sensor_temperatures`/`_deviations`; take
  `sample_temperature_logged` from sensor 1, which tracks the sample on every
  GPS and GPD run checked (sensor 0 sits on the setpoint — the control
  sensor), and only when its deviation is under 5 % of its mean and it is
  within a factor of two of the setpoint (EuO 2923 logs 37 ± 17 K and 134 ±
  34 K against a 5 K setpoint and gets none). This is a reading of the
  numbers, not of a label; the docs say so.
- **D4 — HiFi geometry from logged coils.** A new measured geometry source,
  `coils`: HiFi logs `Field_Main`, `Field_X`, `Field_Y`, `Field_Z` and Hall
  probes per run. Axial field (`Field_Main + Field_Z`) above 10× the transverse
  (`hypot(Field_X, Field_Y)`) is LF; transverse above 10× axial minus the
  ~7–12 G Z compensation is TF; otherwise unknown. These are the run's own
  readbacks, which the field-geometry study (`docs/porting/field-geometry/`)
  ranks above the file's stamp; it forbids inferring from *names*, which this
  does not. It ranks below measured precession.
- **D5 — radical repolarisation.** Not a CLI gap as the audit framed it. The
  benzene EMU curve still rises at 4 kG; an exact six-proton radical model
  saturates by ~1 kG (χ²ᵣ ≈ 9300), one free `MuRepolarisation` gives χ²ᵣ ≈ 440
  and two give χ²ᵣ ≈ 24 (A ≈ 943 and 5311 MHz — a muonium-like precursor or an
  instrumental baseline). This PR gives `MuRepolarisation` a data seed so
  multi-term fits start sensibly, and documents the composition. A general
  N-nucleus radical repolarisation model is a physics study
  (`docs/porting/radical-repolarisation/`) for a later PR.
- **D6 — fits across stored results.** Two derived series kinds that reuse
  `trend` unchanged: `fit-global --groups` writes one `kind: "global"` series
  per group and a `kind: "global-batch"` series whose trend rows are the
  groups' shared parameters; `trend --from-fits` writes a `kind: "fit-trend"`
  series whose rows are one stored trend fit's parameters per member series.
  `trend --model` then fits either (Arrhenius through k_Mu). Rows are keyed by
  member series, not run.

## Agent rules (embedded verbatim in every phase prompt)

- Work on `feat/cli-open-items` in the main checkout unless the lead gives you
  a worktree; in a worktree prefix every Python or harness call with
  `PYTHONPATH=<worktree>/src` (the editable install otherwise imports the main
  checkout).
- **Lean code rules** (`AGENTS.md`, binding): prevent bad states by
  construction — no `hasattr`/`getattr(..., None)` on our own attributes, no
  `try/except` around our own code, no latches, no `if x is None: return`
  where the design can make `None` impossible; validation only at file,
  project and user-input boundaries; one source of truth per fact; a private
  helper needs two callers or a domain name; generalise a scaffold rather than
  mirror it; comments state the invariant, longer narrative goes in
  `docs/plans/`; delete, don't deprecate.
- **If you cannot see how to prevent a bad state by construction, stop.** Put
  the question under "Questions for the lead" in your report and leave that
  part unimplemented.
- `asymmetry.core` must not import Qt, matplotlib or `asymmetry.gui`. CLI
  modules import core lazily inside `run()` (parser construction must stay
  cheap).
- No research data, run numbers, sample names or private paths in the repo or
  tests; tests use synthetic runs (`asymmetry.core.simulate`,
  `tests/core/conftest.py`). Corpus checks run from the session scratchpad.
- Reuse existing core functions named in your phase; do not reimplement
  reduction, grouping, seeding, integration or fitting.
- Tests only through `python tools/harness.py test -- <files>`; run
  `--tier fast` once before committing. Never run `validate` (the lead does).
- Docstrings yes; Sphinx docs and CHANGELOG no (Phase 8 owns them).
- Commit once at the end of the phase, conventional message ending with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (or your own model
  name). Do not push.
- Report: files touched, what was deleted, tests added, commands run with
  results, corpus checks run (numbers), and "Questions for the lead".

## Phases

Phases run in order unless marked parallel; each commits at its end. The lead
reads every diff before running anything, answers questions by resuming the
same agent, then runs the phase's tests and corpus check.

### Phase 1 — three small fixes (Sonnet)

**1a Encoded period run numbers.** `select_period`/`period_run` give a period
dataset `run_number = run*1000 + period` and keep `metadata["source_run_number"]`.
In `core/workflow/integral_scan.py::build_integral_scan` map every point and
exclusion through the source run number (one encoded→source dict built from the
resolved runs) and replace the green − red branch's `// 1000` with the same map;
leave `core/transform/integral.py::build_field_scan` alone (the GUI keys periods
apart by the encoded number). `core/workflow/reduction.py::estimate_alpha_for_run`
reports the source run number, and `_reduce_period`'s error message names it.
Test: `integral-scan --period red` on a synthetic two-period run lists source
run numbers; `alpha --period red` prints the source run.

**1b `recipe --run` seeding** (`core/fitting/seeding.py`). Layer 2
(`_record_scale_values`) takes `abs()` of `record_scale_estimate` for amplitude
roles and seeds a `phase` parameter 0 or π from its sign (the wizard's rule,
`fit_wizard.py` ~4833); layer 3 (`_applied_field_values`) also seeds base name
`frequency` from `field_gauss_to_frequency_mhz(field)` (core/fitting/spectral.py)
when it is below the dataset's Nyquist, marked run-bound (so `fit-series` and
`fit-global` re-seed it per run from each run's field). Test: a synthetic TF run
with phase 90° seeds a positive amplitude, phase π or 0 consistently, and
frequency = γB. Check that `tests/core/test_seeding.py` and the wizard seeding
tests still pass unchanged or are rewritten deliberately.

**1c Survey of sub-folders.** `survey` of a folder with no run files but
sub-folders that hold them lists those sub-folders with their run counts and
exits 1 (a user error: point at one of them), writing nothing. Uses
`scan_run_files` on each immediate sub-folder; no recursion deeper than one
level.

Gate: those tests, `--tier fast`; corpus: `recipe --run 3366` on LiFeAs
(`--pair Up/Down --background range`) seeds A_1 > 0 and frequency ≈ 5.42 MHz,
and the 400 G `fit-series` converges without `--initial`.

### Phase 2 — two instruments in one folder (Opus; needs D1)

Files: `cli/_runs.py` (`run_files`, `resolve_runs`, `resolve_run`,
`_duplicate_run_message`), `cli/_workdir.py` (declare `--instrument` beside
`--workdir`, pass it to `bind`), `core/workflow/workdir.py`
(`bound_folder`/`bind`/`write_manifest` record the instrument;
`WorkDirMismatchError` names it), `core/workflow/survey.py` (alphas keyed per
file, not per run number, at the `alphas` dict ~955/974), every command that
resolves runs (reduce, integral-scan, alpha, `_reduction.py` `--alpha-from`,
survey). `core/io/run_range.py::scan_run_files` stays as is (prefixes stay
case-sensitive in the file listing; matching is case-insensitive in the CLI).
Work-directory `SCHEMA` → 4 (manifest gains `instrument`).

Tests (synthetic folder with two prefixes sharing run numbers, as in
`tests/core/test_cli_commands.py::_write_run_file`): every command refuses
without `--instrument` naming both; with it, reduce/fit work and the manifest
records it; a second instrument in the same work directory is refused;
`survey` lists both and gives each file its own alpha. Corpus: Basics —
`survey`, `reduce --instrument EMU --runs 18850-18863`, and the t0 phase check
from the #334 notes (a 10-bin `--t0-offset` moves a 100 G phase by ~1.41 rad).
Also look at why the survey's `T log` reads ~100 K for the MUSR files and
~644 K for the lowercase `emu` files against 290/295 K setpoints (item 9's
second half — report findings; the fix lands in Phase 5).

### Phase 3 — integral scans: background and RF coupling seeds (Opus)

**3a Background in integrals.** Factor the windowed variance the alpha
estimator already computes (`core/transform/asymmetry.py::_ratio_alpha_error`,
~729–791: `S_F + n·k_F` plus `(n·σ_kF)²` from a `SubtractedBackground`) into one
helper, `windowed_count_variance(counts, window, background) -> (var_F, var_B)`,
used by both the alpha error and `integrate_asymmetry`. `_reduce_run_to_fb`
(`core/transform/integral.py`) takes the grouping's background flag like the
time-domain reduction (drop `use_background=False`) and returns the
`CorrectedGroupedCounts` it needs; `integrate_asymmetry` then forms the window
sums and calls `compute_asymmetry_with_count_errors(ΣF, ΣB, √var_F, √var_B, α,
β)`. Per mode: range/tail_fit/fixed use the shared constant-level term;
reference_run sums the per-bin variances (independent bins). For the
`differential` method the per-bin errors come from
`compute_asymmetry_with_count_errors` and the mean's error adds the correlated
level term — derive it in the docstring (physics argument, not narrative).
`integral-scan` gains `--background` (drop `background=False` from its
`add_reduction_arguments` call). Tests: a synthetic continuous-source scan with
a flat background — the subtracted integral recovers the undiluted asymmetry
and its error matches Monte-Carlo scatter over 200 seeds within 10 %; zero
background gives today's numbers bit for bit; the GUI's integral scan picks up
a grouping's background (one GUI-free core test).

**3b RF coupling seeds.** `suggest_model_seeds` gains `known:
Mapping[str, float]` (values the caller holds: `--initial` and `--fix`
merged before seeding, which `core/workflow/integral_scan.py::_parameters`
currently applies after). The `RFResonanceMuP` estimator, given `nu_RF` in
`known`, finds the two dips with `_lcr_peaks(x, y, 2)` and solves
`[f1(B1; A_mu, A_p) − ν, f2(B2; A_mu, A_p) − ν] = 0` with
`scipy.optimize.least_squares` from (515, 124), `x_scale=(100, 30)`, using
`core/fitting/muon_proton.py::rf_transition_freqs` at the fixed fields (never
through `rf_resonance_fields`, which returns nan where a crossing is missing).
Other callers (`fitting/field_scan.py::rf_resonance_seeds`, the multi-start in
`fit_parameter_model`, `gui/panels/model_fit_dialog.py`) pass what they hold.
Tests: synthetic RF scans at (540, 140) and (480, 90) MHz fit from their own
seeds. Corpus: the benzene DEVA scan fits from `--initial A_mu=470 --initial
A_p=90` (failed in #334).

### Phase 4 — differential ALC line shape (Opus)

A field-scan component `LorentzianLCRPair` (params `f`, `B0`, `Bwid`, `dB`):
`f·[L(B; B0, Bwid) − L(B; B0 + dB, Bwid)]` — the green − red signal of a
resonance whose red period sits `dB` below the recorded field, so the red dip
appears `dB` above. Registered like `LorentzianLCR` (`parameter_models.py`
component definition, `fwhm_factor=2.0`, category "Field scan"),
`component_docs.py` applicability and reference entries, and seeded through
`_LCR_COMPONENTS`/`_lcr_peaks` (the dip at `B0`; `dB` from `known` or the
component default). The evidence for `dB`: HiFi logs no channel for the RG coil
current, but `Field_Hall_Z` binned by `Beamlog_Period_Num` reads the red period
~45 Hall units lower (≈ 44.4 G after the 1.017 Hall-to-main slope) on every run
of the benzene RG scan. The NeXus loader records it per period as
`period_field_offset_gauss` (red − green, from the Hall-Z log binned by period
number, scaled by the run's Hall-to-main ratio), and `integral-scan
--period green-red` prints the scan's mean offset and a `Next:` line
`--fix dB=<value>` when the model has a `dB`. With `dB` free the fit is
degenerate at 20 G sampling against 14 G lines, which the docs say. Tests:
synthetic pair recovers B0 and w with `dB` fixed; the component doc test
covers it; the loader's per-period offset on a synthetic period log. Corpus:
benzene liquid 29809–29854 with `LorentzianLCRPair + LorentzianLCRPair +
Constant --fix dB_1=44.4 --fix dB_2=44.4`: B0 ≈ 28938.5 and 29536.3 G,
HWHM ≈ 14 G, χ²ᵣ ≈ 1.3.

### Phase 5 — metadata the files hold: HiFi geometry and logged temperatures (Opus; needs D3, D4; may run parallel to Phase 4 in a worktree)

**5a HiFi coil geometry.** `core/workflow/survey.py::resolve_row_geometry`
gains the `coils` source between measured precession and the file stamp,
reading `Field_Main`, `Field_Z`, `Field_X`, `Field_Y` active means from
`metadata["nexus_time_series"]` (`core/io/nexus.py::active_series_mean`) with
the D4 ratio rule; `ROW_GEOMETRY_SOURCES` and the survey's `geom` legend gain
it. Tests: synthetic rows for LF (19810/0/0/0), TF20 (0/0.01/19.96/11.99), a Z
sweep on a persistent main, and an ambiguous mix. Corpus: benzene liquid,
solid and solution, corannulene and TCNQ — HiFi scans report LF; the TF20
calibrations TF; no non-HiFi folder changes (re-run the 27-folder survey
comparison from #335's notes).

**5b PSI logged temperature** (`core/io/psi.py`, PSI-BIN header path ~241–276).
Read the four sensor means (float32 at 716) and deviations (738); record them
as `psi_sensor_temperatures`/`psi_sensor_temperature_deviations`; set
`sample_temperature_logged` by D3's rule, never overriding a `.mon` sidecar
log. Tests: a synthetic header round-trip (write the bytes with `struct`),
the deviation and factor-of-two gates. Corpus: EuO 2935 → 52.76 K, LiFeAs 3366
→ 1.739 K, EuO 2923 → none; `survey` shows `T log/K`, and `fit-series --order
sample_temperature_logged` works on EuO.

**5c Implausible NeXus logged temperature** (Basics MUSR ~100 K and lowercase
`emu` ~644 K against 290/295 K setpoints; Phase 2 reports the cause). Fix at
the loader boundary so a series that cannot be the sample thermometer
(`_is_sample_temperature_path` in `core/io/nexus.py`) is not reported, or
report that the reading is genuine and document it.

### Phase 6 — co-add and the correlation spectrum (Opus; needs D2)

**6a `reduce --coadd`.** With `--coadd`, `reduce` loads the named runs,
combines them with `core/data/combine.py::combine_runs` (via
`runs_with_dataset_metadata`, as the GUI's data browser does) and reduces the
combined run through `reduce_run` under the reduction options, storing it under
the first member's run number. `ReducedEntry` gains `members: list[int]` (empty
for a plain reduction; digest covers the members' file fingerprints); every
command that reads a reduced run prints "co-added from …" for it. A
`CombineError` (mismatched runs) is a user error naming the mismatch. Tests:
co-adding two synthetic runs matches `combine_runs` + `reduce_combined_run`;
the cache is invalidated when a member changes.

**6b `fourier --correlation`.** The correlation spectrum needs per-group
spectra from histograms (`core/fourier/spectrum.py::compute_average_group_spectrum`
with `GroupSpectrumConfig(display="correlation", correlation_order=...,
correlation_reference_field_gauss=...)`), not the reduced curve the CLI's
`fourier` transforms. `fourier --correlation` reloads the reduced entry's
source run(s) (the members for a co-add) and calls that function with the
entry's reduction settings; the stored spectrum's axis is the hyperfine
coupling (MHz) and the peak table reports couplings. Tests: a synthetic
two-line radical run gives a correlation peak at the sum of the two
frequencies. Corpus: co-add benzene 3678–3682 (GPD, 3000 G); FFT lines near
41 (diamagnetic), 208.6 and 305.6 MHz; correlation peak near 514 MHz.

### Phase 7 — fits across stored results, repolarisation seeds (Opus, then Sonnet; needs D5, D6)

**7a Batch of groups** (`cli/commands/fit_global.py`,
`core/workflow/global_fit.py`). `fit-global --groups "a,b,c;d,e,f;…"` (with
`--order` for the within-group axis as now and `--group-order NAME --group-x
i=VALUE,…` or `--group-order temperature` for the outer axis, reusing
`cli/_axis.py` parsing with series-index keys) fits each group, writes each as
a `kind: "global"` series `<name>-<i>`, and writes `<name>` as `kind:
"global-batch"`: `members`, the outer `order_key`, and a trend table whose rows
are each group's shared parameters and uncertainties. The trend table's row
key generalises from `run: int` to `key: str` (a run number or a member
series name); `trend --exclude` and `fit_trend` read the key. **7b Law through
trend fits.** `trend --from-fits <series>,… --param NAME [--fit PARAM]`
builds a `kind: "fit-trend"` series from each member's stored
`trend_fits[PARAM]` parameter NAME, with the x axis from `--x
series=VALUE,…` or each member's group temperature; `trend --model Arrhenius`
then fits it as any series. `trend_fits` is keyed `param:expression` so two
laws on one column no longer overwrite each other (schema migration of stored
series keys is in the work-directory schema bump, SCHEMA 4 from Phase 2).
Members' digests are recorded so a refitted member marks the derived series
stale. Tests: synthetic groups with a known shared parameter vs temperature;
a synthetic two-level chain (Linear per temperature → Arrhenius). Corpus:
maleic acid — nine groups (one run per concentration at each setpoint; note the
alpha step between 78280 and 78281), `trend --model Linear` per group for k_Mu,
then `trend --from-fits … --model Arrhenius` for E (report it beside the
worksheet's expectation); Al-LLZ LF triplets as a batch.

**7c `MuRepolarisation` seeds (Sonnet).** An estimator in
`_MODEL_SEED_ESTIMATORS`: `a_Dia` from the low-field plateau, `a_Mu` from the
rise, `A_hf` from the half-rise field B½ via `B0 = A_hf/(γe+γµ)`
(`isotropic_mu_b0_gauss`); for two components the second takes the second
half-rise (successive, like `_lcr_peaks`). Test: synthetic one- and two-term
curves seed within 20 % and fit. Corpus: benzene repolarisation with two terms
reaches χ²ᵣ ≈ 24 from seeds alone.

### Phase 8 — docs, changelog, reference, corpus re-check (Sonnet, then lead)

Sphinx: `docs/reference/agent_workflow.rst` (every new flag, verbatim from
`--help`; the D1/D2/D6 behaviours; the correlation spectrum; the pair model),
`docs/reference/loading_data.rst` (PSI sensors, HiFi per-period Hall offset,
coil geometry), the field-scan model reference page for `LorentzianLCRPair`
and the `MuRepolarisation` seeds (find it via `reference/index.rst`).
`CHANGELOG.md` `[Unreleased]` (Added/Fixed; merge into the existing sections).
`python tools/agent_eval/render_command_reference.py`. Record outcomes in
`docs/plans/agent-cli-skill.md` beside the #334/#335 sections. Lead: the
27-folder survey comparison, `validate`, PR.

## Gates

A phase is accepted when its tests and `--tier fast` are green, its corpus
check reproduces the numbers above (or the report explains a difference), the
diff adds no guard of the kinds the agent rules forbid, and it deletes what it
replaces (e.g. Phase 3 deletes `use_background=False` and
`add_reduction_arguments(..., background=False)`; Phase 1a deletes the `// 1000`
decode). The lead runs `validate` once, after Phase 8.

## Out of scope (recorded for later)

The general N-nucleus radical repolarisation model (D5); corannulene's
rise-plus-step baseline (fit in windows); the pass-17a evaluation-reliability
items (skill work); a model for anisotropic (solid) radical ALC D1 lines.
