# Implementation options & chosen design: fit-workflow-diagnostics

Seam anchors are from the worktree at study time; treat them as starting points.

## MINOS — the W13 shared helper

**Chosen: one helper in `engine.py`, called from all three minimiser drive sites.**

Today three sites build a `Minuit`, set limits, run migrad, and pack a `FitResult`:
- `FitEngine.fit` (`engine.py` ~169–182)
- `FitEngine.global_fit` (`engine.py` ~465–499)
- `count_domain._solve` (`count_domain.py` ~408–421)

They differ in migrad arguments (`global_fit` passes `ncall/iterate/use_simplex`,
strategy, tol; `count_domain` sets `errordef=1.0`), so the helper does **not** own
`Minuit` construction or limit-setting — it owns the post-construction drive:

```python
# engine.py
def drive_minuit(
    m,                                   # a constructed, limit-set Minuit
    *,
    method: str = "migrad",
    migrad_kwargs: dict | None = None,   # ncall/iterate/use_simplex per caller
    run_hesse: bool = True,              # explicit HESSE for covariance fidelity
    minos: bool = False,                 # opt-in
    minos_parameters: Sequence[str] | None = None,  # default: all free
) -> dict[str, tuple[float, float]] | None:
    if method == "simplex":
        m.simplex(**(migrad_kwargs or {}))
    else:
        m.migrad(**(migrad_kwargs or {}))
    if run_hesse and m.valid:
        m.hesse()                        # refine covariance before reporting σ/MINOS
    if not minos or not m.valid:
        return None
    try:
        m.minos(*(minos_parameters or ()))   # empty tuple = all free params
    except (RuntimeError, ValueError):
        return None                      # whole-scan failure → fall back to HESSE
    out: dict[str, tuple[float, float]] = {}
    for name in (minos_parameters or m.parameters):
        me = m.merrors.get(name)
        if me is not None and me.is_valid:
            out[name] = (float(me.lower), float(me.upper))  # lower<0, upper>0
    return out or None
```

Notes:
- **Explicit HESSE** (`m.hesse()`) is run before reporting whenever valid: migrad's
  EDM-time covariance is approximate; an explicit HESSE call is the directive's
  "explicit HESSE where it improves covariance fidelity," and MINOS itself benefits
  from an accurate starting covariance.
- **All free params at once** (Ben's decision): `minos()` with no args scans every
  free parameter. `minos_parameters` exists so count-domain can scan just `alpha`
  cheaply if we choose, but default is all-free for parity with the panel action.
- Per-parameter validity: only `is_valid` merrors are kept; a parameter whose scan
  failed silently keeps its HESSE σ in `uncertainties` and gains no asymmetric entry.

**FitResult additive field:** `minos_errors: dict[str, tuple[float, float]] | None
= None`. Populated by the three sites from `drive_minuit(...)`. `uncertainties`
(HESSE) is untouched.

**Threading MINOS through the API:** add `minos: bool = False` to `FitEngine.fit`,
`FitEngine.global_fit`, `fit_grouped_time_domain` / `fit_grouped_series`, and the
count-domain entry points (`fit_fb_alpha` etc.). GUI passes it from the opt-in
action.

**count-domain α (F7 boundary):** `fit_fb_alpha`'s √α-parameterised α gets a MINOS
interval shown in the panel with its α–amplitude correlation context (the covariance
is already extracted in `_result_from_minuit`). The promote path
(`fit_panel.promote_count_alpha` ~4019 → `promote_alpha_to_grouping`) is **untouched**
— it keeps feeding the scalar HESSE σ as `alpha_error`. MINOS on α is overlay-only.

## χ² quality wiring (W7)

**Chosen: assess in `fit_result_summary`, render from the `"quality"` key.**

`result_summary.fit_result_summary` (~14–37) is the convergence point for single,
grouped, global, and batch records (call sites: `mainwindow.py` ~6132/6355/6393/6422/6781).
Add:

```python
summary["quality"] = _quality_dict(fit_result)         # verdict/band/dof/confidence or None
summary["uncertainties_asymmetric"] = _minos_dict(fit_result)  # {name: [lo, hi]}
```

`_quality_dict` calls `assess_fit_quality(chi², dof, 0.95)`. **dof source:** add an
additive `dof: int = 0` field to `FitResult`, populated at the three core sites where
`ndata`/`nfree` are already known (`engine.fit` ~223, `global_fit` per-dataset ~596,
`_result_from_minuit` ~473). `fit_result_summary` falls back to inferring
`dof ≈ round(chi²/reduced_chi²)` when `dof` is unset (legacy/other callers), guarded
on `reduced > 0`. A confidence constant `FIT_QUALITY_CONFIDENCE = 0.95` lives beside
the helper.

**Surfaces consuming `"quality"`:**
- Fit panel result label (`_fit_success_html` ~552): append a coloured verdict chip;
  tooltip teaches the band ("χ²ᵣ=… ; good fits at ν=… fall in [low, high] at 95%; an
  overdone fit reproduces the data better than the errors allow → overestimated
  errors or an over-flexible model").
- Result summary text (mainwindow ~6126): a verdict line.
- Trend/series and model-fit dialog: the same chip from the same key.

Chip colours track the post-#53 palette tokens (`OK`/`WARN` already used by the log
panel tags); overdone uses a distinct muted/accent token so it reads as "suspicious"
not "bad".

## Chain-seeding (W5)

**Chosen: a seeding mode on the series/batch path; member N+1 seeded from member N's
fitted values, re-normalised to the grouped contract, ordered by the series key.**

Insertion point: the grouped-series driver currently builds **static** per-run seeds
(`fit_panel.py` ~4176: `initial_params = {run: build_grouped_initial_params(...)}`)
and fits members independently (`grouped_time_domain._fit_grouped_series_independent`
~854). Replace the independent loop's seed source with a chained one when the mode is
on:

```
order = sorted(members, key=series_order_key)   # field/temperature, not run id
seed = build_grouped_initial_params(order[0])   # member 1: table/average seed
for member in order:
    result = fit_grouped_time_domain(member, ..., initial_params=seed)
    record(result)
    if chain_mode and result.success:
        carried = extract_fitted(result)                 # rates/fields/fractions/phases
        seed = normalize_to_grouped_contract(model_names, carried)  # amp→1, bg→0
```

- **Order key:** the existing series order key (field/temperature). Documented that
  chaining is only meaningful in that order (the physical motivation — a scan
  crossing Tc — depends on adjacency in the control parameter, not acquisition
  order).
- **Interaction with seed-averaging:** the existing batch seed-averaging produces
  member 1's seed and remains the default mode. "Chain from previous run" is an
  *alternative* seeding mode (not a blend): member 1 uses the table/average seed,
  each subsequent member inherits the previous fit. Documented in the user guide
  alongside the warm-start disambiguation.
- **Failed member:** if member N fails, chaining falls back to the static/average
  seed for member N+1 (don't propagate a diverged seed) and the event is logged.

**Global-tab batch path:** the global fit wizard's batch execution
(`global_fit_wizard.py`) gets the same mode via a new `initial_seed_by_run` channel,
kept distinct from the existing `warm_start_source` (single-fit→global). Naming in
the UI and docs: **"chain from previous run"** vs the wizard's existing **"warm
start from single fits."**

## Abort (W6)

**Chosen: core `cancel_callback` + `FitCancelledError`; in-fit raise + between-member
checks; nothing recorded on abort.**

- New `class FitCancelledError(RuntimeError)` in `asymmetry.core` (a small
  `core/fitting/errors.py` or alongside the engine; exported from `asymmetry.core`).
- Add `cancel_callback: Callable[[], bool] | None = None` to `FitEngine.fit`,
  `global_fit`, `fit_grouped_series`, and the count-domain drivers — same name as the
  MaxEnt kwarg.
- **In-fit granularity:** the model wrapper / cost function checks
  `cancel_callback()` and raises `FitCancelledError`; migrad propagates it. The
  `Minuit` object is discarded; no `FitResult` is built. (Decision: this is the
  adopted finer granularity — verified safe because we never read an aborted Minuit.
  A check counter throttles the callback to every ~N evaluations to keep it cheap.)
- **Between-member granularity (minimum contract):** series/global loops check
  `cancel_callback()` before each member fit and raise `FitCancelledError` if set —
  guaranteeing a clean stop even for back-ends that swallow cost-function exceptions.
- On `FitCancelledError`, the loop records **no** partial `FitSeries`/`FitSlot`; the
  project state is unchanged and the next fit is unaffected.

**GUI:** the existing fit workers (single, grouped-series `GroupedSeriesFitWorker`
~4189, and the global/batch worker) gain a `cancel()` bool flag + `cancelled` signal,
wired exactly like `MaxEntWorker`/`_launch_maxent_worker` (~217/~5497). A **Stop**
button replaces the disabled Fit button while busy (enabled-while-busy like
`maxent_panel`). The `cancelled` slot resets the panel to idle and records nothing.

**Single-fit worker:** if single fits currently run synchronously on the GUI thread,
they stay synchronous for the fast HESSE path; when **MINOS is requested** (slow), the
fit runs on a worker so the Stop button and progress indicator apply. (Confirm the
current single-fit threading during implementation; the contract is "abortable when
it can run long.")

## FitLog (W14, reframed)

**Chosen: enrich the persisted record + a Qt-free formatter + on-demand export. No
background file, no schema break.**

- **Enrich the stored snapshot:** `FitSlot.result` and `FitSeries.results_by_run`
  entries already carry the `fit_result_summary` dict; they automatically gain the
  additive `"quality"` and `"uncertainties_asymmetric"` keys from items 1–2. Add
  light provenance to the summary: `"model_name"`, `"fit_range"` (t_min/t_max or
  freq range), `"timestamp"` (ISO-8601, set at record time in the GUI layer — core
  stays clock-free), `"npar"`, `"ndof"`, `"provenance"` (single/batch/global). All
  additive in the result dict → round-trips through `.asymp` with no schema version
  bump (the schema tolerates unknown keys).
- **`core/fitting/fit_log.py` / `FitLog`:** a Qt-free formatter turning one such
  record dict into a human-readable provenance block (WiMDA `.fit`-style: titled
  header with run/model/timestamp, aligned `param = value ± σ  (+hi −lo)` lines,
  `χ²ᵣ = … [verdict]  target [low, high]`). One source of truth for: the `LogPanel`
  entry (optionally enriched from the one-liner), the result-summary text, and the
  export. Pure function, fully unit-testable.
- **Route through `LogPanel`:** unchanged event path. The fit-completion sites that
  emit `self._log_panel.log("Fit completed: χ²ᵣ=…", tag="fit")` now include the
  verdict; the full record lives in `.asymp`.
- **On-demand export:** a single "Export fit report" action writes the formatted
  text for the current dataset (or all datasets' latest fits) — the closest WiMDA
  `.fit` analogue, generated from the structured records. No automatic writes.

## Final UX decisions (settled with Ben, 2026-06-12)

- **MINOS display:** inline in the parameter-table value cell — `value ± σ` (HESSE)
  normally, switching to `value +hi / −lo` when MINOS errors are present (extend the
  existing `_ValueUncertaintyDelegate`). Result-summary text gains the asymmetric
  rows. α shows its α–amplitude correlation context.
- **MINOS trigger:** a pre-fit toggle ("Asymmetric errors (MINOS)") near the Fit
  button; when on, the fit runs MINOS for all free params as part of the fit, on a
  worker so Stop + progress apply uniformly across single/grouped/global/count.
- **Abort:** in-fit (cost-function raise) **and** between-member checks.
- **Chain-seeding = Auto by default, order-key driven.** API exposes four modes:
  `per_run_seed`, `average_seed`, `chain_previous`, `auto`. `auto` chooses
  **chain_previous** when the members carry a usable temperature/field order key
  spanning a real range over **≥3 members** (an ordered scan); otherwise
  **average_seed**. Per-member divergence falls back to the average/table seed for
  the next member. Auto **logs its choice and reason** to the `LogPanel` (never
  silent). GUI override: a menu-bar **Batch seeding** submenu (radio: Auto [default]
  · Per-run seed · Average seed · Chain from previous run); the selection persists
  with the project (additive key). Rationale: chaining wins on ordered scans
  (especially Tc-crossings, per the textbook) and is harmful on unordered/repeat
  collections where a diverged member poisons the chain — so Auto gates on the order
  key. The try-then-measure alternative was rejected as too heavy (runs the batch
  up to twice).

## Rejected / deferred

- **External append-only log file** — rejected; `.asymp` already holds the
  latest-per-dataset snapshot WiMDA's overwrite logs represent (Ben, this session).
- **JSONL / dual-format export** — deferred; the structured copy already lives in
  `.asymp`, so an external machine-readable file is redundant for now.
- **Configurable R** — deferred; fixed 0.95, helper already supports configurability.
- **In-batch co-add / re-fit-coadded; `fgAll`→Poisson cost-factory unification** —
  follow-ons per the directive (run-arithmetic dependency / separate refactor).
- **Dedicated "Fit log" window** — rejected; existing surfaces suffice (Ben).
