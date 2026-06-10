# Project brief: run-arithmetic

Umbrella: `wimda-parity-gap` · Wave B · Size M

## Motivation

Partly a **correctness fix**: Asymmetry's current co-add
(`data_browser.py:_coadd_datasets`) averages reduced asymmetry curves with
unweighted means — statistically wrong for low-count runs, and it discards
histograms (the combined dataset has `histograms=[]`, so grouping, deadtime,
background and count-domain fitting can't run on it). WiMDA co-adds at the
raw-count level. Co-*subtract* (laser-on/off photo-μSR, background runs)
doesn't exist at all.

## WiMDA reference

`muondata.pas:2418–2490`: bin-by-bin histogram addition (and subtraction,
`cosign=−1` via `Cosubmode1Click`) including R/G and per-period histograms;
frame/spill accumulation; event-weighted averaging of temperature, field and
laser energy; combined label "12345+12346". `BatchFit.pas` consumes the same
machinery for in-batch co-adding.

## Scope

- New Qt-free `core/data/combine.py`: `combine_runs(runs, sign=+1)` →
  first-class `Run` with summed histograms (period-aware), accumulated
  `good_frames`, event-weighted metadata averages, provenance metadata
  recording the constituent runs; validation (compatible bin width, detector
  count, t0 alignment policy — study decides whether to align-by-t0 like
  `apply_grouping_aligned` or require equality).
- Co-subtract with correct error propagation (variances add) and a guard
  against negative expected counts.
- Rewire the Data Browser data-group co-add path onto the new kernel
  (existing `.asymp` data-groups keep working — migration note in study).
- Expose "Subtract run…" as a browser action.

**Out**: in-batch co-add during sequential fitting (optional phase of
`fit-workflow-diagnostics`, which depends on this kernel); background-run
subtraction as a *correction* (that's `data-reduction-parity` Phase 2 — it
scales by frame ratio rather than combining datasets; the two should share
arithmetic helpers — coordinate in studies).

## Current Asymmetry state

Curve-mean co-add only (`data_browser.py:2187`); no subtraction; combined
runs lose histogram-level data.

## GUI/UX sketch

No new surfaces: the existing data-group create/co-add flow gains correct
behaviour invisibly; "Combine → Subtract reference run…" joins the browser
context menu; combined-run info dialog lists constituents and the weighting
applied.

## Physics-correctness notes

Sum counts, then reduce — never average reduced curves (errors and any
nonlinear correction must see total statistics). Metadata averaging weighted
by good events (WiMDA's choice, and the defensible one); record the spread
(e.g. T min/max across constituents) so users can spot inhomogeneous groups.

## Conflicts & dependencies

Primary surfaces: `core/data/` (+new module), `data_browser.py`. Wave B:
disjoint from 4/7/10/12. `workflow-visualisation` (Wave C) also edits
`data_browser.py` — sequenced after. Unblocks the optional batch phases of
`fit-workflow-diagnostics`.

## Verification sketch

Co-add of N synthetic Poisson runs is distributionally identical to one
N×-events run (pull tests); co-add then reduce equals WiMDA arithmetic on a
corpus pair; co-subtract of identical runs is zero with √2-scaled errors;
existing data-group corpus projects load and reproduce prior curves within
the documented correction.
