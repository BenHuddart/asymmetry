# Multi-Group Time-Domain Fitting Test Data

## Synthetic Cases For The First Slice

1. Two-group shared-physics count fit:

	 - domain A: `N0_A * exp(-t/tau_mu) * (1 + A_A * cos(2 * pi * f * t + phi_A)) + B_A`
	 - domain B: `N0_B * exp(-t/tau_mu) * (1 + A_B * cos(2 * pi * f * t + phi_B)) + B_B`
	 - shared parameters: `f`, common relaxation rate
	 - local parameters: `N0`, background, amplitude, relative phase
	 - expectation: one shared fit recovers the common physical parameters while
		 preserving different local count scales and phases

2. Forward and backward paired-count case with alpha ratio:

	 - two domains representing forward and backward grouped counts from one pair
	 - expectation: a shared polarization function with pair-efficiency metadata
		 reproduces the same asymmetry that a derived asymmetry fit would see

3. Multi-run field-sequence case:

	 - three or more runs with the same model family and field-dependent local
		 metadata
	 - expectation: one global fit shares selected physical parameters while
		 allowing field- or run-local parameters to vary

4. Relative-phase stress case:

	 - four domains with equal amplitudes and phases offset by roughly
		 `0`, `pi/2`, `pi`, `3pi/2`
	 - expectation: the fit only succeeds when phase is modeled as an explicit
		 per-domain quantity rather than being absorbed into amplitude or baseline

5. Lifetime-sensitivity count case:

	 - grouped counts with a visible exponential envelope and non-zero
		 background
	 - expectation: the approved count-domain model produces materially better
		 residuals than an asymmetry-only approximation

6. Raw-count versus derived-asymmetry equivalence case:

	 - generate forward and backward counts from one known asymmetry model, then
		 derive asymmetry from them
	 - expectation: when nuisance count parameters are handled correctly, the
		 physical parameters inferred from the count-domain and asymmetry-domain
		 fits agree within tolerance

## Reference Comparison Targets

- WiMDA `fgAll` fit of several groups from one run with different `GrpN0`,
	`GrpBG`, `GrpAmpl`, and `GrpPhas` values
- WiMDA multifit LF sequence with one shared function over several runs and
	`x2` carrying field or delay
- musrfit multi-RUN global fit where at least one physical parameter is global
	and `alpha`, background, or normalization remain local
- musrfit single-histogram fit with explicit `N0` and optional lifetime
- Mantid simultaneous fit across several group or pair workspaces using
	`MultiDomainFunction` ties
- Mantid TF asymmetry conversion with per-domain normalization values
- Mantid `PhaseQuadMuon` on a detector phase table fixture to validate the
	detector-phase metadata contract separately from the fit engine

These reference comparisons do not all need to be automated in the first code
slice. The early harness should start with deterministic synthetic fixtures and
only add reference-program comparisons where the data export path is stable.

## Data Contracts To Preserve

- group identity remains explicit per domain
- detector identity remains explicit when a domain is created from detector
	counts rather than grouped counts
- `N0`, background, `alpha`, `beta`, and lifetime provenance are stored
	explicitly, not inferred from array shape
- relative phase is represented separately from the intrinsic model phase
- grouped-count domains preserve the original counts and error model rather
	than converting to asymmetry too early

## Useful Fixture Shapes

- one single run with four detector groups and a known phase pattern
- one forward/backward pair with a non-unit alpha ratio
- one field sequence of three or more runs with one shared relaxation model
- one low-count case to exercise error propagation and normalization behavior