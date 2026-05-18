# Multi-Group Time-Domain Fitting Comparison

## Scope

This comparison covers the parts of each program that decide behavior for
simultaneous time-domain fitting:

- how the same model is applied across multiple groups, detectors, or runs
- how count normalization and `N0` are represented
- how relative phases between groups or detectors are represented
- how muon lifetime is treated in the count-domain model or preprocessing
- what the current Asymmetry baseline already provides and what is still
	missing

## Summary

All three reference programs support "one physics model across many domains",
but they differ sharply in where they draw the abstraction boundary.

- WiMDA fits grouped counts directly. It lifetime-corrects the observed data,
	stacks all included groups or runs into one vector, and uses one parameter
	array containing shared component parameters plus per-group `N0`,
	background, amplitude, and relative phase.
- musrfit global fitting is run-block centric. It sums chi square across RUN
	blocks and lets the `.msr` file decide which parameters are global or local.
	Its single-histogram path keeps explicit `N0` and lifetime in the count
	model; its asymmetry path instead fits derived asymmetry with `alpha` and
	optional `beta` corrections.
- Mantid splits the problem. Simultaneous fitting itself is a generic
	`MultiDomainFunction` with explicit parameter ties across selected
	workspaces. Count normalization, TF asymmetry wrapping, and detector phase
	quadrature reconstruction are handled by separate framework algorithms.
- Asymmetry is currently closest to Mantid's simultaneous-fit core and furthest
	from WiMDA's and musrfit's count-domain semantics. It can already share or
	localize parameters across asymmetry datasets, but it has no count-domain
	fit contract, no per-domain `N0`, and no time-domain detector phase table.

The design implication for Asymmetry is clear: do not port WiMDA's UI-level
concatenation literally, but do port its per-group parameter semantics. Use a
Mantid-style explicit multi-domain core plus a musrfit-style explicit global or
local parameter-role contract.

## Program Comparison

| Program | Owning simultaneous-fit surface | How one function is shared | Count or `N0` handling | Relative phase handling | Lifetime handling | Tests or validation surface |
| --- | --- | --- | --- | --- | --- | --- |
| WiMDA | `Analyse.pas`, `AsymFitFunction.pas`, `Multifit.pas` | One fit vector is assembled from all selected points; `x2` identifies group or multifit field or delay | Per-group `N0` and background parameters live in `GROUP_base + 4*(g-1)` blocks; data are scaled by `exp(t/tau_mu) / nbin` | Per-group relative phase is `GROUP_base + 4*(g-1) + 3`; shared component phase and group-relative phase are added inside the model | Count data are lifetime-corrected before fitting by multiplying with `exp(t/tau_mu)`; model background terms therefore carry `exp(t/tau_mu)` too | No focused automated tests found in the repo scan |
| musrfit | `PRunListCollection`, `PRunAsymmetry`, `PRunSingleHisto`, `.msr` parser in `PMsr2Data.cpp` | One MINUIT parameter vector is reused for every RUN block; `.msr` global flags decide which parameters are shared | Single-histogram fits use explicit `N0`; asymmetry fits work on forward and backward derived asymmetry with `alpha` and optional `beta` | No separate detector phase table in time-domain fitting; phase is just another model parameter and can be global or local through parameter mapping | Single-histogram fits use `N0 exp(-t/tau)(1+P(t)) + B`; asymmetry fits mostly cancel lifetime in the derived observable | Doxygen-rich code and some fit tests, but no focused multi-group characterization tests identified |
| Mantid | Muon GUI `GeneralFittingModel` and `TFAsymmetryFittingModel`; framework algorithms `ConvertFitFunctionForMuonTFAsymmetry`, `CalculateMuonAsymmetry`, `PhaseQuadMuon` | `MultiDomainFunction` creates one domain per workspace and explicit tie strings share selected parameters across domains | Per-domain normalization comes from a table or the `analysis_asymmetry_norm` run property; TF asymmetry uses one `N0` per domain | Detector phases are externalized into a phase table and processed by `PhaseQuadMuon`; simultaneous fitting itself only sees domain functions and ties | TF conversion wraps user function as `N0*(1+f) + ExpDecayMuon`; `PhaseQuadMuon` estimates detector `N0` values from exponential decay | Framework tests exist for TF conversion and phase-quad algorithms |
| Asymmetry | `FitEngine.global_fit`, `global_fit_wizard.py` | One objective is built from concatenated asymmetry datasets; global and local parameters are explicit lists | No count-domain `N0` contract; grouped counts are reduced before fitting and fits operate on asymmetry arrays | Phase can be global or local only as an ordinary model parameter; there is no detector or group phase-table contract in time-domain fitting | No built-in lifetime wrapper in the time-domain fit engine; lifetime handling exists separately in Fourier preprocessing | `tests/test_global_fit_wizard.py`, `tests/test_global_fit_wizard_window.py` |

## Detailed Notes

### WiMDA

Confirmed entry points:

- `src/Analyse.pas`
- `src/AsymFitFunction.pas`
- `src/Multifit.pas`
- `src/globals.pas`

#### How simultaneous fitting works

- `Multifit.pas` loads each selected run, regroups it, appends the grouped
	data into one shared array, and records one extra value per point in
	`groupx2`.
- In LF-sequence multifits that extra coordinate is the per-run field or delay
	value, not a detector identifier.
- In `Analyse.pas`, the `fgAll` fitting mode instead loops over all groups and
	appends each group's trace into one fit vector with `x2[n] := g`.
- `AsymFitFunction.pas` then interprets `x2` according to context:
	- as the active group index when fitting all groups of one run
	- as the active field or delay value for LF-sequence multifits

WiMDA therefore solves both "many groups of one run" and "many runs of one
sequence" by reusing one function evaluator over one stacked data vector.

#### How counts and `N0` are handled

- `Analyse.pas` constructs the fit data as lifetime-corrected counts:

	`Y[n] = groupd[g, i] * exp(t * lam_mu) / nbin[i]`

	with propagated Poisson errors scaled by the same factor.
- The group-parameter block stores four values per group:
	- `N0`
	- background
	- asymmetry amplitude scale
	- relative phase
- In `AsymFitFunction.pas`, the single-pulse count model for `fgAll` is:

	`c = (1 + a * af) * N0_g + exp(t / tau_mu) * BG_g`

	where `af` is the per-group amplitude factor and `a` is the shared physics
	function value.

This is one of the most important behaviors to port: WiMDA does not throw away
group-specific count normalization when it goes into simultaneous fitting.

#### How relative phase is handled

- When fitting all groups, `AsymFitFunction.pas` reads the per-group relative
	phase from `GROUP_base + 4*(group-1) + 3`.
- That relative phase is added to the component phase of oscillatory terms.
- WiMDA therefore distinguishes:
	- intrinsic model phase, shared with the component parameters
	- group-relative phase, specific to one group

#### How lifetime enters the model

- `globals.pas` defines `tau_mu` and `lam_mu = 1 / tau_mu`.
- `Analyse.pas` lifetime-corrects the observed counts before fitting.
- Because the data are moved to a lifetime-corrected scale, the model's
	background terms carry `exp(t * lam_mu)` explicitly.

#### Porting implications from WiMDA

Carry forward:

1. Per-group `N0`, background, amplitude, and relative phase must survive into
	 the simultaneous-fit contract.
2. One physical polarization function should be shareable across many groups.
3. Group identity must remain explicit per domain.

Do not port literally:

1. The stacked `x2` transport hack should become explicit domain metadata in
	 Asymmetry rather than a second overloaded axis.
2. UI decisions should not own the count model.

### musrfit

Confirmed entry points:

- `src/include/PRunListCollection.h`
- `src/classes/PRunListCollection.cpp`
- `src/classes/PRunAsymmetry.cpp`
- `src/classes/PRunSingleHisto.cpp`
- `src/classes/PMsr2Data.cpp`

#### How simultaneous fitting works

- `PRunListCollection` owns the global-fit objective. Its `GetAsymmetryChisq`
	and sibling methods sum the chi-square contributions from all stored runs.
- Each RUN block is turned into a `PRunBase`-derived object, and all of them
	receive the same MINUIT parameter vector during fitting.
- `PMsr2Data.cpp` scans RUN blocks and marks norm, background, `alpha`, `beta`,
	lifetime, and map parameters as global or local based on the parameter list.

This is musrfit's core strength for porting: shared versus local behavior is an
explicit property of parameters, not a side effect of how the data are loaded.

#### How counts and `N0` are handled

- musrfit has two relevant time-domain paths.
- `PRunSingleHisto.cpp` fits counts directly with:

	`N_theo(t) = N0 * exp(-t / tau) * (1 + P(t)) + B`

- In that path:
	- `N0` can be a fit parameter or a user-defined function
	- `tau` can be a fitted lifetime parameter or default to the physical muon
		lifetime
	- background can be fitted, fixed, or estimated from a range
	- normalization to `1/ns` is explicit and the objective applies a scaling
		correction when that mode is active
- `PRunAsymmetry.cpp` instead derives asymmetry from forward and backward counts
	using `alpha` and optional `beta`, so it no longer has a separate `N0`
	parameter in the observable itself.

#### How phase is handled

- musrfit does not use a dedicated detector phase-table contract in time-domain
	fitting.
- If the selected theory function has a phase parameter, that phase is just an
	ordinary model parameter.
- Whether phase is global or local is decided by parameter numbering and the
	`.msr` global flag machinery, exactly like any other fit parameter.

This is simpler than WiMDA and Mantid, but it means detector-relative phase is
not a first-class time-domain data object.

#### How lifetime enters the model

- In `PRunSingleHisto.cpp`, lifetime is part of the explicit count model.
- In `PRunAsymmetry.cpp`, the asymmetry observable mostly removes the shared
	muon decay envelope by construction.
- musrfit therefore supports both strategies that Asymmetry may want later:
	- fit counts with explicit lifetime in the model
	- fit asymmetry with no separate lifetime envelope term

#### Porting implications from musrfit

Carry forward:

1. Shared versus local parameter roles should remain explicit and orthogonal to
	 the UI.
2. Count-domain fits should be able to keep `N0`, background, and lifetime as
	 first-class parameters.
3. Derived asymmetry domains should continue to work as a separate observable
	 mode.

Defer unless requested:

1. Full `.msr`-style syntax and run-block text configuration.
2. Every musrfit normalization mode and background-estimation detail.

### Mantid

Confirmed entry points:

- `qt/python/mantidqtinterfaces/mantidqtinterfaces/Muon/GUI/Common/fitting_widgets/general_fitting/general_fitting_model.py`
- `qt/python/mantidqtinterfaces/mantidqtinterfaces/Muon/GUI/Common/contexts/fitting_contexts/general_fitting_context.py`
- `qt/python/mantidqtinterfaces/mantidqtinterfaces/Muon/GUI/Common/fitting_widgets/tf_asymmetry_fitting/tf_asymmetry_fitting_model.py`
- `Framework/Muon/src/ConvertFitFunctionForMuonTFAsymmetry.cpp`
- `Framework/Muon/src/CalculateMuonAsymmetry.cpp`
- `Framework/Muon/src/PhaseQuadMuon.cpp`

#### How simultaneous fitting works

- The Muon GUI fitting model creates one `MultiDomainFunction` with one domain
	per selected workspace.
- `GeneralFittingContext` stores both the multi-domain function and the list of
	global parameters.
- `GeneralFittingModel` adds explicit tie strings such as
	`f0.frequency=f1.frequency=f2.frequency` for each selected global parameter.
- The GUI stores whether simultaneous fitting is by `Run` or by `Group/Pair`.

Mantid's simultaneous-fit core is therefore very close to the shape that
Asymmetry should preserve: one domain per dataset and explicit global ties.

#### How counts and `N0` are handled

- Mantid handles count normalization outside the generic simultaneous-fit core.
- `CalculateMuonAsymmetry.cpp` normalizes a workspace by `N0` and subtracts 1:

	`normalized = unnormalized / N0 - 1`

- `ConvertFitFunctionForMuonTFAsymmetry.cpp` retrieves one normalization value
	per domain from either a normalization table or the
	`analysis_asymmetry_norm` run property.
- It then wraps the user function into a TF asymmetry composite:

	`N0 * (1 + f) + ExpDecayMuon`

- In simultaneous TF mode, `TFAsymmetryFittingModel` keeps one normalization
	parameter per domain.

This separation is important: Mantid does not force every simultaneous fit to
know about detector counts, but it has a stable algorithmic seam when counts
and normalization do matter.

#### How relative phase is handled

- Detector-relative phase handling lives in `PhaseQuadMuon.cpp`, not in the
	simultaneous-fitting model.
- `PhaseQuadMuon` expects one row per detector in a phase table containing
	detector id, asymmetry amplitude, and phase.
- It estimates one `N0` value per detector from an exponential-decay fit,
	subtracts the exponential baseline, projects the detector residuals into real
	and imaginary quadratures using the phase table, and then restores the common
	decay envelope.

Mantid therefore treats detector phase as detector metadata plus a preprocessing
or reconstruction step, not as a hidden fit-browser convention.

#### How lifetime enters the model

- `ConvertFitFunctionForMuonTFAsymmetry.cpp` appends `ExpDecayMuon` with fixed
	lifetime to each domain's composite function.
- `PhaseQuadMuon.cpp` estimates detector `N0` using the fixed physical muon
	lifetime and removes plus restores the exponential decay during quadrature
	reconstruction.

#### Porting implications from Mantid

Carry forward:

1. One explicit domain object per dataset or group.
2. Explicit global ties rather than hidden parameter coupling.
3. A separate normalization and detector-phase seam, not hard-coded GUI state.

Do not port literally:

1. Mantid's ADS and workspace naming machinery.
2. Algorithm-wrapping conventions that depend on Mantid's framework base
	 classes.

### Asymmetry

Confirmed entry points:

- `src/asymmetry/core/fitting/engine.py`
- `src/asymmetry/core/fitting/global_fit_wizard.py`
- `src/asymmetry/core/fitting/models.py`
- `docs/user_guide/global_fit_wizard.rst`

#### Current simultaneous-fit behavior

- `FitEngine.global_fit` already builds one least-squares objective over many
	datasets.
- The API already separates parameter roles into:
	- global parameters shared across every dataset
	- local parameters that vary per dataset
	- fixed parameters
- The global-fit wizard layers model selection and role recommendations on top
	of that engine.

#### Current counts, phase, and lifetime behavior

- The fitting engine operates on asymmetry arrays and their errors.
- There is no explicit `N0` parameter, count-domain background term, or muon
	lifetime wrapper inside the time-domain fitting engine.
- Phase is only whatever phase parameter exists in the chosen model, and it can
	be global or local, but there is no detector or group phase-table contract.

#### What this means for porting

Asymmetry already has the right outer skeleton for simultaneous fitting. The
main missing pieces are the lower-level domain semantics that the reference
programs expose:

1. grouped-count domains instead of asymmetry only
2. explicit per-domain `N0` and background
3. explicit lifetime strategy for count fits
4. explicit group or detector phase offsets

## Cross-Program Conclusions

The best synthesis for Asymmetry is:

1. Keep Asymmetry's current multi-dataset engine shape.
2. Add a count-domain model layer inspired by WiMDA and musrfit.
3. Use Mantid-style explicit domains and explicit ties instead of hidden
	 concatenation rules.
4. Treat detector phase tables and detector normalization as explicit data,
	 not just GUI choices.

## Features Worth Bringing Into Asymmetry

Bring in first:

1. One core simultaneous-fit objective over many domains, not only asymmetry
	 datasets.
2. Per-group `N0`, background, amplitude scale, and relative phase parameters.
3. Explicit global or local parameter roles for every physical parameter.
4. A stable count-domain equation equivalent to either WiMDA's lifetime-corrected
	 count fit or musrfit's explicit `N0 exp(-t/tau)(1+P(t)) + B` form.
5. A separate detector-phase metadata seam.

Bring in later:

1. Detector-level phase quadrature reconstruction similar to Mantid
	 `PhaseQuadMuon`.
2. musrfit-style alternate normalization and background-estimation modes.
3. UI polish for WiMDA-style per-group editing workflows.

Do not bring in as-is:

1. WiMDA's overloaded `x2` transport mechanism.
2. Mantid's framework-specific workspace plumbing.
3. musrfit's `.msr` file syntax as the main Asymmetry user interface.