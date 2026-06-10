# Project brief: workflow-visualisation

Umbrella: `wimda-parity-gap` · Wave C (alone — touches files Waves A/B edit)
· Size M (2 phases)

## Motivation

A basket of beamline-honed workflow conveniences that individually are too
small to be projects but collectively define the "WiMDA feels fast at the
instrument" experience: run stepping, plain ASCII export, diagnostic plot
views, a data-snapped cursor.

## WiMDA reference & scope

**Phase 1 — workflow.**
- Run-number stepping / filename-pattern walker (`WiMDA_Main.pas`
  `PrevRun/NextRun/GetPrefixSuffix`): next/previous run buttons + keyboard
  shortcuts in the Data Browser, deriving the pattern (prefix + zero-padded
  number + suffix) from the selected run; works across a directory without
  reloading the table.
- ASCII export with provenance (`SaveAsItemClick:1414`, `SaveRange.pas`,
  batch `Plot.pas:3082`): "Export data as text…" for the current curve(s) —
  data / data+fit / fit-only, optional x-range, commented header carrying
  run info, grouping, recipe provenance (the `.wim` idea, modernised);
  batch variant over a run selection.
- Events columns in the Data Browser (`LogbookUnit:595–696`): total good
  events (MEv) and events/frame — cheap, useful run-quality columns.
- B-from-log as the run's field value (analogue of the existing
  "Use temperature from log"; `Bfromaveinblog1`).
- Deadtime-file auto-discovery + stale-calibration-date warning
  (`WiMDA_Main:925–975`) — small grouping-dialog affordance.

**Phase 2 — plot diagnostics.**
- Raw-count and log-count display modes (`Plot.pas GetDataGroup`) — the
  standard t0/deadtime/background diagnostic views.
- F,B overlay view (`FBOverlay`) — α-calibration aid.
- Data-snapped cursor with readout (`Plot.pas:1159–1228, 2962–3006`):
  snap-to-point crosshair, x/y±err, S/N, 3-point parabolic peak readout,
  windowed average ± error over the visible range (the peak and
  window-average readouts are the genuinely-missed parts).
- Cosmetic basket if budget allows: error-bar toggle, marker styles.

**Optional late phase — live current-run monitoring** (`muondata.pas:1376`
"run 0" from ISIS DAE temp files, freshest of `macq*.tmp`/`auto_*.tmp`,
auto-refresh): contingent on beamline access for testing (decision
2026-06-10); design the loader hook now (a refreshable dataset whose
source can change underneath), implement when testable.

**Out** (recorded drops): printing, in-app GLE editor, click-to-set x-range,
cursor-point→fit-table batch (superseded by fit-based trending), laser/aux
`.mon` tlog ingestion (pre-NeXus legacy), histogram report as a standalone
tool (if wanted later: data-browser export columns).

## Current Asymmetry state

Browser-table navigation only; GLE `.dat` sidecars but no standalone ASCII
export; FB-asym + per-group views only; free cursor coordinates only.

## GUI/UX sketch

Stepping as ◀/▶ buttons + PgUp/PgDn in the browser; export under File →
Export; raw/log/FB-overlay join the existing view selector in the plot
workspace; cursor readout in the status bar with a toggle in plot controls.

## Conflicts & dependencies

Primary surfaces: `plot_panel.py`, `data_browser.py`, `mainwindow.py` — the
three highest-traffic GUI files, which is why this runs alone in Wave C
(after `run-arithmetic` and `rrf` finish their edits to two of them).

## Verification sketch

Pattern walker on corpus directories (incl. PSI naming) steps correctly at
boundaries; ASCII golden files round-trip (export → parse → values match
plotted data); raw/log views vs WiMDA screenshots for one run; windowed
average matches `integrate_curve` over the same range; events column vs
WiMDA logbook values for identical runs.
