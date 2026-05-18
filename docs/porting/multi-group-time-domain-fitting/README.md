# Multi-Group Time-Domain Fitting Study

Status: study + initial implementation

This study records how WiMDA, musrfit, Mantid, and the current Asymmetry
codebase handle simultaneous time-domain fitting across multiple detector
groups.

The first-pass goal is to compare implementation choices before any Asymmetry
port work begins.

## Implementation Status

An initial implementation slice is now present in Asymmetry.

Implemented so far:

- a core grouped time-domain builder that derives lifetime-corrected grouped
  count traces from one active dataset and its saved grouping payload
- a core grouped fitting adapter that reuses Asymmetry's simultaneous-fit
  engine with explicit per-group nuisance parameters
- GUI wiring in the fit panel for a grouped mode inside the existing Global tab
- stacked time-domain subplots showing grouped lifetime-corrected traces for
  the active dataset

Still intentionally deferred:

- grouped fitting across multiple runs
- detector-level fitting rather than grouped traces
- detector phase tables and quadrature reconstruction workflows
- grouped-wizard automation comparable to the existing global-fit wizard

## Main Result

The reference programs solve the same product problem with three different
architectures:

- WiMDA uses one monolithic count-domain fitter. It stacks group or run data
  into one fit vector, uses a second coordinate to identify either the active
  group or the multifit field or delay value, and keeps per-group `N0`,
  background, amplitude, and relative phase inside one shared parameter array.
- musrfit uses a run-block based global-fit engine. Multiple RUN blocks share
  one MINUIT parameter vector, and the `.msr` file decides which parameters are
  global or local. Single-histogram fits keep explicit `N0` and lifetime in
  the model; asymmetry fits instead work on forward/backward derived asymmetry
  with `alpha` and optional `beta` corrections.
- Mantid splits the problem across layers. Simultaneous fitting in the Muon
  GUI is done with `MultiDomainFunction` plus explicit global ties across
  selected runs or group/pair workspaces. Detector-count normalization and
  detector-phase handling live in separate muon algorithms such as
  `ConvertFitFunctionForMuonTFAsymmetry`, `CalculateMuonAsymmetry`, and
  `PhaseQuadMuon`.
- Asymmetry already has the right high-level simultaneous-fit abstraction for
  asymmetry data: one objective over multiple datasets with explicit global and
  local parameter roles. What it does not yet have is the lower-level domain
  model needed for grouped-count fitting, per-group or per-detector `N0` and
  background terms, explicit lifetime wrappers, or detector phase contracts.

## Recommended Direction

Recommended for the future implementation pass: keep Asymmetry's current
global/local parameter engine, but add a lower-level multi-domain time-domain
fit contract that can represent either:

- asymmetry domains, which preserve the current Asymmetry behavior
- grouped-count domains, which add WiMDA and musrfit style `N0`, background,
  lifetime, and relative-phase handling

The core design to carry forward is:

1. Use Mantid and Asymmetry's explicit multi-domain separation rather than
   WiMDA's concatenated `x2` data hack as the long-term API.
2. Port WiMDA's per-group fit semantics: each group can carry its own `N0`,
   background, amplitude scaling, and relative phase while still sharing one
   physical polarization function.
3. Port musrfit's global or local parameter discipline: shared physical
   parameters must be an explicit role choice, not an incidental UI behavior.
4. Keep detector phase tables and detector normalization provenance explicit at
   the data boundary, not implicit inside GUI state.

No implementation is chosen or started in this study pass. The recommendation
above is the intended starting point for a later implementation pass.

## Scope

- simultaneous fitting of multiple groups or detectors with one model
- detector and group count handling
- relative detector phases
- muon lifetime correction or normalization assumptions in the fitted model
- candidate seams for a future Asymmetry implementation pass

## Study Files

- `comparison.md`: implementation comparison across the reference programs
- `implementation-options.md`: candidate ways to port the feature into
  Asymmetry
- `test-data.md`: proposed comparison datasets and golden outputs
- `verification-plan.md`: validation plan for a later implementation pass

## Current Asymmetry Baseline

- Core simultaneous-fit owner: `src/asymmetry/core/fitting/engine.py`
- Fit-candidate analysis: `src/asymmetry/core/fitting/global_fit_wizard.py`
- Built-in asymmetry models: `src/asymmetry/core/fitting/models.py`
- Current docs and tests:
  - `docs/user_guide/global_fit_wizard.rst`
  - `tests/test_global_fit_wizard.py`
  - `tests/test_global_fit_wizard_window.py`

Current Asymmetry behavior supports:

- simultaneous fitting of multiple asymmetry datasets
- explicit `global`, `local`, and `fixed` parameter roles
- shared model selection across a run series
- model parameters such as phase being global or local by user choice

Current Asymmetry behavior does not yet support:

- per-group or per-detector `N0` parameters
- explicit count-domain background terms inside the fit engine
- detector phase tables or detector quadrature reconstruction in the time
  domain
- a first-class contract for fitting one physical function across several
  detectors of the same run while preserving detector provenance

## Candidate Port Seams

1. Data-domain seam: define a stable fit-domain object for asymmetry,
   grouped-count, and later detector-count inputs.
2. Parameter-role seam: reuse the existing Asymmetry global/local role system
   instead of inventing a second tying mechanism.
3. Normalization seam: make `N0`, background, `alpha`, `beta`, and lifetime
   explicit per-domain metadata or parameters instead of inferring them from
   grouped arrays.
4. Phase seam: carry group or detector phase offsets in a dedicated contract,
   separate from the intrinsic phase parameter of the physical model.
5. GUI seam: let future fitting panels choose which domains to include and
   which parameters are shared, but keep the actual objective and model math in
   `asymmetry.core`.

## Open Questions

- Should the first implementation slice target grouped counts only, or grouped
  counts plus detector-level raw histograms?
- Should Asymmetry expose musrfit-style optional fitted lifetime in the first
  slice, or keep lifetime fixed to the physical constant for count fits?
- Is WiMDA-style forward/backward alpha-ratio handling part of the first slice,
  or should it wait until pair-specific fitting is introduced?