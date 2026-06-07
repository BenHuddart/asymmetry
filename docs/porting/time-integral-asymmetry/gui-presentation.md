# ALC / field-scan — how the reference programs present it to the user

This file studies the **UX** of ALC / integral-asymmetry analysis in WiMDA and
Mantid (the on-screen controls and workflow), to scope the Asymmetry GUI. It is
the presentation companion to [comparison.md](comparison.md) (which covers the
maths). The core observable is already implemented Qt-free
([README.md](README.md)); this is about how to surface it.

## WiMDA — minimal, reuses the general machinery

WiMDA treats an ALC scan as *just another fit table*. There is **no dedicated
ALC window**; the feature is a thin layer over existing UI.

- **Entry:** an ALC scan is recognised from the data (file with an `alc` prefix /
  fit-table metadata), or assembled via **Fit Table → Extras → "Combine ALC
  scans"** — a tiny modal: a *run-list* box ("300-310") and an *output name* box,
  with a **[Go]** button (`ALCscans.dfm`, `ALCscans.pas:37-92`). The Extras menu
  is hidden by default.
- **Display:** the scan appears in the **Fit Table** window as a text table —
  columns `B(T) | asym(%) | err | B(G)` — and is plotted in the model-plot window
  (x = field, y = integral asymmetry). The main window status shows the field
  range and the number of field points (`WiMDA_Main.pas:596,763`).
- **Operations (Fit Table → Tools):**
  - **"Set ALC scan threshold"** — a one-field dialog: the max field step (G,
    default 1000) used to decide adjacent points (`SetALCthresh.dfm`).
  - **"Make differential ALC"** — writes `dA/dB` in %/kG for adjacent points
    within the threshold (`FitTableUnit.pas:322-402`).
- **Fitting the curve:** there is **no ALC-specific baseline or peak UI**. The
  user fits the scan with WiMDA's ordinary model-fitting workflow (pick a model,
  fit), exactly as for any fit table.

**Takeaway:** ALC is a *data-prep + tabular/plot view + a `dA/dB` transform*, and
curve fitting reuses the normal fit machinery. Almost no new surface.

## Mantid — a dedicated three-step wizard

Mantid ships a separate **"ALC Analysis"** interface — a linear wizard with a
status line `Step N/3 — <name>` and Previous/Next buttons
(`ALCInterface.cpp:70`).

**Step 1 — Data loading** (`ALCDataLoadingView.ui`): builds the scan. Controls:
instrument; **Runs** (range/list, with "Auto add"); **Plot By Log** = a *log*
selector + **Function** (Mean / Min / Max / First / Last) for time-series logs;
**Dead Time Correction** (None / From Data File / From Custom File);
**Grouping** (Auto, or Custom forward/backward + **Alpha**); **Periods** (Red,
**Subtract** checkbox, Green); **Calculation → Type** = **Integral** /
**Differential**; **Time limits** (`From [µs]` / `Max [µs]` = the integration
window); a **Load** button; and a "Loaded data" plot (asymmetry vs log value).

**Step 2 — Baseline modelling** (`ALCBaselineModellingView.ui`): add a baseline
**Function** (Linear/Polynomial/…) via a function browser; define **Sections**
(a `Start X | End X` table, draggable on the plot) marking the non-resonant
regions; **Fit**. Two plot tabs: **"Baseline model"** (data + fitted baseline +
section markers) and **"Corrected data"** (baseline subtracted).

**Step 3 — Peak fitting** (`ALCPeakFittingView.ui`): add **Peaks** (Gaussian/
Lorentzian/…) via a function browser; **Plot guess**; **Fit**. Read peak
**position = resonance field** and **width** from the browser. (A graphical peak
picker exists but is disabled in current Mantid.)

**Results:** **Export results… / Import results…** at any step, to ADS
workspaces (`Muon ALC.rst`).

**Takeaway:** ALC is a *self-contained pipeline* — build the scan, then
**baseline-subtract**, then **peak-fit** — with its own UI separate from the
normal time-domain fitting.

## Side-by-side

| Aspect | WiMDA | Mantid |
| --- | --- | --- |
| Home | reuses Fit Table (no dedicated window) | dedicated 3-step "ALC Analysis" wizard |
| Build the scan | from file / "Combine ALC scans" | Step 1 form (runs, log, type, time limits, grouping…) |
| Integration window | run good-bin range (baked in) | explicit `From/Max [µs]` |
| Integral vs Differential | Integral only | **Type: Integral / Differential** |
| x-axis | field (from metadata) | **log selector + Function** (Mean/Min/Max/First/Last) |
| Grouping / α / periods / deadtime | run/global settings | explicit controls in Step 1 |
| Derivative `dA/dB` | **"Make differential ALC"** + threshold | — (not offered) |
| Baseline subtraction | — (assumed pre-done) | **Step 2** (sections + function fit) |
| Peak/resonance fit | ordinary model fit | **Step 3** (dedicated peak fit → position/width) |
| Export | ordinary save | Export/Import results |

## Functionality checklist for Asymmetry

To match "the same functionality", the GUI must let the user:

1. **Choose the integration window** — Asymmetry already has the fit-range
   control; in scan mode it sets `[t_min, t_max]` (core `integrate_run`). ✓ seam.
2. **Build the scan across a series** ordered by **field / temperature / run** —
   core `build_field_scan(order_key=…)` + the existing `FitSeries` membership. ✓.
3. **Pick the reduction** — Integral (default) vs Differential — core `method`. *(Mantid parity.)*
4. **See excluded runs with a reason** — core `FieldScan.excluded`. ✓.
5. **Use the user's grouping / α / period / deadtime** — pass the F-B
   representation's effective grouping via `grouping_ref`; period selected
   upstream. ✓ seam.
6. **View the `dA/dB` derivative** — core `differentiate_scan(max_gap=…)`. *(WiMDA parity.)*
7. **Fit the scan curve** — *(WiMDA: ordinary model fit; Mantid: baseline + peak
   fit.)* This is the open design question below.
8. *(Mantid-only, deferred)* **baseline subtraction** and **dedicated peak
   fitting** with position/width read-off — the `alc-avoided-level-crossing`
   follow-up candidate.

Items 1–6 are covered by the implemented core API; the GUI work is wiring +
presentation. Item 7/8 (how deep to take *curve fitting*) is the scope decision.

## Design implications for Asymmetry

Asymmetry's model is **per-run representations** (F-B asymmetry, groups, FFT,
MaxEnt) each with a recipe + a single/batch **fit** + a **parameter-trending**
panel for the series. A field scan is intrinsically a **series-level** object:
one curve from N runs, where the "fit" is of *A vs field*, not of a time
spectrum. So the two reference philosophies map onto Asymmetry as:

- **WiMDA-like (minimal, the proposed approach):** ALC is a *mode* of the F-B
  asymmetry representation. A toggle switches the fit workflow from "fit the time
  spectrum" to "integrate over the (re-used) fit range and build the scan across
  the series"; the scan renders + is fitted in the space the trending panel
  occupies. No new domain/representation; maximal reuse.
- **Mantid-like (dedicated):** a separate "Field scan / ALC" representation (a new
  domain in the workspace selector) carrying its own loading → baseline → peak
  steps. More discoverable, more surface, more code, and a second fitting path to
  maintain.

The proposed direction (toggle within F-B asymmetry; fit-range doubles as the
integration window; ALC curve fitting in the trending space) is the WiMDA-like,
minimal option. The open scoping questions — captured for the discussion with the
maintainer — are:

- **Entry point:** mode toggle on F-B asymmetry vs a separate representation.
- **First-pass depth:** ship build+display+`dA/dB`+(fit via existing machinery)
  now, and defer Mantid's baseline + dedicated peak-fit; or build the full
  pipeline up front.
- **Option exposure:** how prominently to surface the *Integral/Differential*
  reduction and the `dA/dB` derivative (note these are two *different* "differential"
  ideas — a reduction method vs a transform of the scan) without cluttering the UI.
- **Rendering:** where the scan curve lives (main plot, replacing the time
  spectrum, vs the trending area) and what "single" vs "batch" mean in scan mode
  (single = the selected run's integral read-out; batch = the whole scan).

> Decisions from that discussion will be recorded in
> [implementation-options.md](implementation-options.md) before GUI work starts.
