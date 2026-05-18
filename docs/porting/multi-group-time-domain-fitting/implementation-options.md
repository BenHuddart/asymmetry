# Multi-Group Time-Domain Fitting Implementation Options

## Decision Framing

The comparison shows that Asymmetry does not need a new simultaneous-fit
optimizer first. It already has one. What it needs is a richer time-domain data
contract and model wrapper so that simultaneous fits can operate on grouped
counts and later detector counts, not only asymmetry traces.

The main design choice is therefore where to add those missing semantics.

## Option 1: Extend the existing asymmetry-only global fit

Description:

- Keep `FitEngine.global_fit` exactly as the only fitting engine.
- Add more GUI around local and global parameter roles.
- Continue fitting only asymmetry datasets.

Pros:

- Lowest implementation risk.
- Reuses the existing Asymmetry code path without a new domain model.
- Good fit for Mantid-style group or run simultaneous asymmetry fitting.

Cons:

- Cannot represent WiMDA's grouped-count `N0`, background, and relative phase
	semantics.
- Cannot represent musrfit single-histogram count fits.
- Leaves detector-count fitting out of scope from the start.

When to choose it:

- Only if the target is strictly "simultaneous asymmetry fitting across already
	reduced groups" and count-domain parity is intentionally deferred.

## Option 2: Port a WiMDA-style grouped-count fitter directly

Description:

- Add a count-domain fitting path specialized for grouped time traces.
- Represent one fit as a stacked vector over all selected groups.
- Carry per-group `N0`, background, amplitude scale, and relative phase
	explicitly.

Pros:

- Closest feature match to WiMDA.
- Directly addresses the user's requested multi-group time-domain workflow.
- Keeps group-specific count semantics intact.

Cons:

- Risks copying WiMDA's overloaded transport mechanism too literally.
- Can diverge from Asymmetry's existing clean global/local role API.
- Makes later detector-level fitting harder if the design is too group-specific.

When to choose it:

- If WiMDA parity is the only goal and detector-level generalization is not a
	priority.

## Option 3: Port a Mantid-style split architecture

Description:

- Keep one generic multi-domain simultaneous-fit core.
- Put count normalization, TF asymmetry wrapping, and detector phase handling in
	separate adapters or algorithms.
- Use explicit ties for shared parameters and one domain per included dataset.

Pros:

- Cleanest architecture.
- Scales well from groups to detectors to runs.
- Aligns closely with Asymmetry's current global-fit engine shape.

Cons:

- By itself it does not tell Asymmetry what the grouped-count domain equation
	should be.
- Needs more design work up front to avoid an underpowered abstraction.
- Risks under-specifying WiMDA's per-group semantics if copied too generically.

When to choose it:

- If long-term extensibility matters more than immediate WiMDA parity.

## Option 4: Hybrid design using Asymmetry's current engine plus a new
## explicit time-domain fit-domain contract

Description:

- Keep Asymmetry's current simultaneous-fit engine and parameter-role system.
- Add one core fit-domain abstraction with explicit domain kinds such as:
	- `asymmetry`
	- `grouped_counts`
	- later `detector_counts`
- For count domains, add explicit per-domain parameters or metadata for:
	- `N0`
	- background
	- relative phase
	- `alpha` and `beta` where needed
	- lifetime strategy
- Build model adapters so one physical polarization function can be observed as
	either asymmetry or counts.

Pros:

- Preserves Asymmetry's strongest existing abstraction.
- Captures WiMDA's and musrfit's missing count semantics.
- Leaves room for Mantid-style detector phase tables later.
- Avoids baking UI or file-format behavior into the core engine.

Cons:

- More design work than Option 1.
- The first implementation slice still needs careful scoping.

When to choose it:

- This is the recommended direction.

## Recommended Default

Recommended for the implementation pass: Option 4.

Reasoning:

- Asymmetry already solved the hard part of parameter-role aware simultaneous
	fitting.
- WiMDA and musrfit show that count-domain fitting needs explicit `N0`,
	background, and lifetime semantics.
- Mantid shows that those semantics should be separate from the generic
	simultaneous-fit core whenever possible.
- The hybrid design is the smallest path that can eventually cover grouped
	counts, detector counts, and asymmetry fits without two unrelated fitting
	engines.

## Proposed First Implementation Slice

The smallest worthwhile implementation slice is:

1. Add a core grouped-count fit domain in `asymmetry.core`.
2. Reuse the existing simultaneous-fit engine with one domain per selected
	 group.
3. Support one shared physical polarization function plus local per-group:
	 - `N0`
	 - background
	 - relative phase
4. Keep lifetime fixed to the physical muon lifetime in the first slice.
5. Preserve current asymmetry-domain global fitting unchanged.
6. Defer detector-level raw histogram fitting to the next slice.

This first slice captures the most important WiMDA behavior while still fitting
Asymmetry's architecture.

## Feature Selection Guidance

Implement in the first pass:

1. Per-group `N0` and background parameters.
2. Per-group relative phase offsets.
3. Explicit global or local parameter roles for physical parameters.
4. Shared fit across many groups of one run and across many runs of one sweep.

Defer to later passes:

1. Detector-level raw count fitting.
2. Mantid-style quadrature reconstruction from detector phase tables.
3. musrfit-compatible alternate normalization modes.
4. WiMDA-specific UI details that are not required by the core data model.

## Explicitly Rejected For The First Slice

- Reproducing WiMDA's `x2` concatenation as the Asymmetry public API
- Adding a second, separate simultaneous-fit engine next to the existing one
- Making detector phase a hidden convention inside the fitting GUI