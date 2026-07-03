# GUI Duplication Map (exploration report, 2026-07-03, main @ 3d2359a)

Produced by a read-only exploration pass; feeds PLAN.md. Line numbers are
landmarks, not gospel — re-grep before editing.

## Executive summary

The GUI contains significant duplication across ~64K lines in `src/asymmetry/gui/`:

1. **Axis limit controls** — two divergent `_FloatLimitField` implementations
2. **Fit parameter tables** — shared class, duplicated inline setup
3. **Two parallel fit workflows** — SingleFitTab vs GlobalFitTab (~1,500 lines near-duplicate)
4. **Two wizard windows** — copy-pasted thread lifecycle + result caching
5. **Matplotlib canvas wrappers** — 4 independent setups
6. **Progress/cancel UI** — duplicated in maxent + fit tabs
7. **Export (TSV/GLE/PNG)** — 3 overlapping implementations

Positive: `TaskRunner` in `gui/tasks.py` (306 lines) is good shared
worker-thread infrastructure; `gui/styles/widgets.py` (579 lines) is widely
reused styling helpers.

## 1. Axis-limit controls

| File | Lines | Status |
|------|-------|--------|
| `plot_panel.py` | 154–200 | Plain field; 3-decimal; width 76px |
| `fit_panel.py` | 619–733 | Near-identical but with `_clamp()` (652–660), Return/Enter handling (678–687), `_commit()` forcing (665–676) |
| `alc_panel.py` | imports plot_panel's | reuse |

The fit_panel version is more featured; plot_panel doesn't benefit. Axis-limit
toolbar assembly also duplicated: `plot_panel._create_limit_controls()`
(458–623: X/Y min/max + log toggle + decimation) vs `alc_panel` manual layout
(~332–340: X/Y only).

## 2. Fit parameter table & delegates

- `FitParameterTable` (`fit_panel.py` 1499–1755) IS shared by both tabs — good.
- `_CommitOnTabDelegate` (1364–1400), `_ValueUncertaintyDelegate` (1401–1498).
- But initialization (~80 lines: 16 rows, signals, styling) is duplicated
  inline: SingleFitTab ~1900–1980 vs GlobalFitTab ~3400–3500.
- `_RowHighlightDelegate` in `data_browser.py` (328+) is a separate impl for a
  different table.

## 3. SingleFitTab vs GlobalFitTab (fit_panel.py, 8,934 lines)

- SingleFitTab: lines 1987–3159 (~1,173 lines); single dataset,
  `FitEngine.run()`.
- GlobalFitTab: lines 3160–7600 (~4,441 lines); multi-dataset
  global/local/fixed classification, `fit_grouped_time_domain()` /
  `fit_grouped_series()`.
- Shared utilities & dispatchers: ~3,320 lines.

Duplicated between the tabs:

| Feature | Single | Global | Similarity |
|---|---|---|---|
| Model formula box (`_make_formula_box`) | 2055–2057 | 3291–3296 | ~99% |
| Edit Model dialog | 2059 | 3297 | same handler |
| Wizard plumbing (`_fit_wizard_window`, `_cached_wizard_recommendation`, `_cached_wizard_signature`, `_cached_wizard_log_text`) | 2026–2027 | 3281–3285 | identical |
| Fit-range spinbox pair + `_on_fit_range_committed` | 2127–2145 | 3341–3359 | duplicate |
| Parameter table setup | ~1900–1980 | ~3400–3500 | duplicated inline |
| Stop button + result label | 2163–2164 | 3545–3546 | duplicate |
| Fit-run loop | 2651–2750 | 4800–5150 | genuinely different algorithms |
| Progress callback setup | 2680+ | 5000+ | duplicated pattern |

Both tabs also independently validate parameter bounds, dataset compatibility,
missing data — no shared `_validate_fit_preconditions()`.

## 4. Two wizard windows

- `fit_wizard_window.py` (937 lines): single dataset; Fingerprint → Portfolio →
  Compare → Apply; `FitWizardWorker` (43–72); manual QThread (~96–99).
- `global_fit_wizard_window.py` (1,553 lines): series; Screening → Optimize →
  Review; `GlobalFitWizardWorker` (80–300+); manual QThread (~80–120).

Both duplicate: thread lifecycle, progress bar + label, tab-widget setup,
worker signal connections, error handling, and an identical result-cache
signature strategy (`_cached_log_text`, `_cached_signature`). No base class.

## 5. Matplotlib canvas wrappers

Four independent `Figure(tight_layout=True)` + `FigureCanvasQTAgg` setups:

| File | Lines | Toolbar |
|---|---|---|
| `plot_panel.py` | 258–290 | NavigationToolbar2QT |
| `fit_parameters_panel.py` | ~665 (lazy) | none |
| `global_parameter_fit_window.py` | 130–141 | none |
| `alc_panel.py` | 25–26 | — |

No factory, no base class, ad-hoc toolbar/import handling.

## 6. Progress/cancel UI

| Location | Lines | Pattern |
|---|---|---|
| SingleFitTab | 2163–2164 | Stop btn + `_result_label`, manual polling |
| GlobalFitTab | 3545–3546 | Stop btn + `_result_text`, manual polling |
| MaxEntPanel | 362–375 | Cancel btn + indeterminate QProgressBar |

`TaskRunner` workers already expose `cancel()`; the tabs could share a wired
control builder.

## 7. Export

| File | Types | Lines |
|---|---|---|
| `plot_panel.py` | PNG/SVG/TSV/GLE | 5755–6500+ (`_collect_export_payloads`, `_export_figure_size`) |
| `fit_parameters_panel.py` | TSV/GLE | 5156–5945 |
| `global_parameter_fit_window.py` | TSV/GLE | similar, cross-group |
| `alc_panel.py` | PNG | 1552–1560 (`export_current_plot`) |

TSV header/row/value formatting near-identical between fit_parameters_panel and
global_parameter_fit_window; both independently `subprocess.Popen` GLE.

## 8. Existing shared infrastructure

- ✅ `gui/tasks.py` `TaskWorker` (99–143) + `TaskRunner` (187–307): cooperative
  cancel, signal relay, orphan reaper. Used by fit_parameters_panel,
  global_fit_wizard_window, fit_panel (partially). NOT used by
  fit_wizard_window (own QThread).
- ✅ `gui/styles/widgets.py`: `make_section()`, `apply_param_table_style()`,
  `build_primary_button_qss()`, `error_html()` etc. Widely reused.
- ❌ `_format_param_label` duplicated: `fit_panel.py` and
  `fit_parameters_panel.py` (~137–138).
- ❌ No base classes for: plot-bearing panels, fit tabs, worker-dialog windows,
  export-capable panels.

## 9. Other

- Fit-range synchronization managed independently in 3 places: `fit_panel.py`
  `_on_fit_range_committed` (~2550), `plot_panel.py` drag handles (~5200),
  `alc_panel.py` spinbox↔plot (~150–170).
- Seeding helpers (`_seed_group_background_and_n0` 178–265,
  `_seed_group_phase_degrees` 310–368, `_seed_group_absolute_phases` 370–440)
  live at fit_panel module level; wizard/batch paths reimplement locally.

## File sizes (god-file risk)

| File | Lines |
|---|---|
| `gui/mainwindow.py` | 13,032 |
| `gui/panels/fit_panel.py` | 8,934 |
| `gui/panels/plot_panel.py` | 7,293 |
| `gui/panels/fit_parameters_panel.py` | 6,287 |
| `gui/panels/data_browser.py` | 4,366 |
| `gui/windows/global_parameter_fit_window.py` | 2,418 |
| `gui/windows/global_fit_wizard_window.py` | 1,553 |
| `gui/panels/alc_panel.py` | 1,569 |
| `gui/windows/fit_wizard_window.py` | 937 |
| `gui/panels/maxent_panel.py` | 779 |

Estimated consolidatable duplication: ~2,000–2,500 lines.
