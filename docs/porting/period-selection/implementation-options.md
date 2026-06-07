# Period selection — implementation options

## Goal

Expose period selection in `asymmetry.core` (Qt-free), have the GUI call it,
and keep the loader's existing return contract working. Validate inputs at the
boundary.

## Option A — minimal, faithful (chosen)

Keep `load()`'s historical return shape (2-period → one combined `MuonDataset`,
3+ → `list`). Add a `core/io/periods.py` module:

- `select_period(data, period) -> MuonDataset` for any shape.
- `load(filepath, period=...)` convenience wrapper.
- `period_count` / `period_labels` / `resolve_period_index` helpers.
- Move the GUI's `_period_histograms_for_mode` body and the G−R/G+R arithmetic
  into `select_period_histograms` / `combine_period_asymmetry`; the GUI calls
  these. **Single implementation.**

To return *exactly* what the loader produced per period without re-deriving the
good-bin window, `_combine_two_period_datasets` also stores
`grouping["period_reduced"] = [(t,a,e)_red, (t,a,e)_green]` (in memory only —
`grouping` is whitelisted, not dumped verbatim, into `.asymp` projects, so this
does not change the on-disk schema). `select_period` returns those arrays plus
per-period provenance.

**Pros:** no behaviour change for existing callers; GUI/core provably agree
(shared functions); exact per-period arrays; small surface. **Cons:**
`period_reduced` is reconstructed only on a fresh `load()`, not after a project
round-trip (pre-existing limitation — `period_histograms` was already dropped on
save).

## Option B — always return a list / `dataset.periods`

Make `load()` always return a list for multi-period files and expose a
`.periods` collection on the dataset.

**Rejected:** invasive — the 2-period combined `MuonDataset` is relied on
throughout the GUI and project schema; changing the return type risks wide
breakage for a feature that is mostly about *selecting* one period.

## Option C — recompute in `select_period` from histograms

Re-run grouping + `compute_asymmetry` + good-bin windowing from
`period_histograms` on demand.

**Rejected as the default path:** it would re-implement the loader's good-bin
resolution (which depends on time-based parameters not all stored in
`grouping`), risking silent drift from the loader. Storing the loader's own
output (Option A) is exact by construction. The low-level
`select_period_histograms` still exposes the histograms for callers (the GUI)
that *do* recompute with user alpha/deadtime/background.

## Validation rules (boundary)

- Integer period numbers are 1-based; out-of-range → `ValueError`.
- `"red"`/`"green"` labels accepted only for 2-period runs; unknown label →
  `ValueError`.
- Non-int/str selectors (incl. `bool`) → `TypeError`.
