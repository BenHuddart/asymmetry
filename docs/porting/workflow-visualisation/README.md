# Study: workflow-visualisation (WiMDA parity, Wave C)

Umbrella: [`wimda-parity-gap`](../wimda-parity-gap/README.md) · Wave C (alone —
it touches `plot_panel.py`, `data_browser.py` and `mainwindow.py`, the three
highest-traffic GUI files, so it is sequenced after the Wave A/B edits to two of
them). Project brief:
[`projects/workflow-visualisation.md`](../wimda-parity-gap/projects/workflow-visualisation.md).

## The governing test

This study does **not** ask "what does WiMDA do that Asymmetry doesn't?" — that
question over-fits to WiMDA's main-window habits. It asks: **does this WiMDA
convenience benefit the *established Asymmetry workflow*?** Asymmetry's workflow is
browser-centric — a persistent multi-run [Data Browser](../../../src/asymmetry/gui/panels/data_browser.py)
with representations, the time/frequency **view selector**, the **inspector deck**
(post-PR #53), and the **Options → Advanced** gate (post-PR #72, session ②).
WiMDA, by contrast, loads one run at a time into a single main window; many of its
"conveniences" exist only to compensate for *not* having a browser.

Every scope item therefore earns one of:

- **ADOPT** — a clear benefit; ship it, possibly reshaped to the Asymmetry idiom.
- **ADAPT** — the underlying benefit is real, but achieved through an
  Asymmetry-native mechanism rather than WiMDA's.
- **REJECT** — mimicry without benefit. An expected, acceptable outcome for some
  items: the browser already serves the need, or the native workflow is strictly
  better.

When the two idioms conflict, **the Asymmetry idiom wins** — that is the thesis of
this Wave-C session.

## Verdict summary

The full reasoning, with the WiMDA mechanism and the native alternative weighed
side by side, is the centrepiece of [`comparison.md`](comparison.md). In brief:

| # | Item | Verdict |
|---|---|---|
| 1 | Run stepping / filename-pattern walker | **REJECT** — the browser + native arrow-key navigation already steps + auto-loads |
| 2 | ASCII / data-only export with provenance | **ADAPT** — strengthen the existing GLE `.dat` path to a data-only text export; no second button |
| 3 | Events columns (good MEv, events/frame) | **ADOPT** — add good-range + per-frame columns and a user surface to add columns |
| 4 | B-from-log (field from the data log) | **ADOPT** — mirror the existing temperature-from-log machinery for field |
| 5 | Deadtime-file auto-discovery + staleness warning | **REJECT** — the estimate/calibrate/promote family + project persistence is fresher and native |
| 6 | Log-count display mode | **ADOPT** — a log-y diagnostic on the raw-counts view |
| 7 | F,B *balance* overlay (renamed — see below) | **ADAPT (borderline)** — overlay F and α·B on shared axes; the one eyeball the estimators can't give |
| 8 | Data-snapped cursor: S/N, parabolic peak, windowed average | **ADAPT** — build on the existing cursor signal + status bar; subset is Ben's call (workflow explained in `comparison.md` §8) |
| 9 | Cosmetic basket (error-bar toggle, markers, ticks, ns/bins units) | **REJECT here** — UI-polish-pass territory |
| 10 | Live current-run monitoring | **DESIGN-ONLY** — design the refreshable-loader hook; implement only with beamline access |

> **Naming note (from the Wave A closeout, §3 item 6):** "Overlay" now names the
> *multi-run* overlay feature already in `plot_panel.py`. Item 7 is therefore
> called the **F,B balance overlay** throughout this study to avoid the collision.

## Implementation scope (pending the checkpoint)

If Ben confirms the table, the implement-straight-through list is:

- **ADOPT:** events columns (§3), B-from-log (§4), log-count diagnostic (§6).
- **ADAPT:** data-only export (§2), cursor readouts (§8, subset TBD), F,B balance
  overlay (§7, borderline — may drop to REJECT on Ben's call).

REJECT items (§1, §5, §9) ship nothing; §10 ships a documented hook design only.

None of this is advanced/niche enough to hide — it is mainstream browser-and-plot
workflow and belongs on the **primary** surfaces. The **Options → Advanced** gate
(session ②) is reserved for genuine specialist toggles; nothing here qualifies.
Per the standing rules: **no `schema_version` bumps**; panel-state keys are
**additive only**; GPL sources are **oracle-only**.

## Documents

- [`comparison.md`](comparison.md) — the verdict table + per-item reasoning
  (WiMDA mechanism vs Asymmetry-native alternative). The heart of the study.
- [`implementation-options.md`](implementation-options.md) — for each ADOPT/ADAPT
  item, the concrete seam, the chosen shape, and the divergences from WiMDA.
- [`test-data.md`](test-data.md) — corpus runs (incl. PSI naming) and the
  `$WIMDA_SRC` oracle anchors for each verifiable item.
- [`verification-plan.md`](verification-plan.md) — the checks that gate each item.
