# Architecture

Asymmetry is a Python analysis library, a command-line interface, and a PySide6
application for μSR. Scientific requests and results belong to the core;
front ends select inputs, present results, and manage interaction. Plans and
unresolved decisions belong in GitHub issues. This page describes the shipped
ownership and compatibility boundaries.

## Layers and ownership

Dependencies point from the front ends into `asymmetry.core`. Core code does not
import Qt, matplotlib, or `asymmetry.gui`. Optional file readers are loaded at
the format boundary; drawing and figure export belong outside core.

| Owner | Responsibility | Detailed contract |
| --- | --- | --- |
| `core.data` and `core.io` | Acquired histograms, observations, source identity, format decoding, and selected-period loading | [Loading data](reference/loading_data.rst) |
| `core.instruments` | Frozen instrument facts, recognition, acquisition maps, source assessment, geometry, and readout eligibility | [Instrument definitions](reference/instruments/index.rst), [authoring](contributing.rst#instrument-definitions) |
| `core.project` | Grouping policies, versioned persistence, detached project preparation, and source replay | [Grouping](reference/detector_grouping.rst), [projects](reference/project_files.rst) |
| `core.transform`, `core.fitting`, `core.fourier`, `core.maxent`, and `core.negmu` | Numerical preparation, likelihoods, models, calibration, and analysis | [Feature reference](reference/index.rst), [numerical accuracy](explanation/numerical_accuracy.rst) |
| `core.representation` | Analysis recipes, single-fit collections, series, studies, and project ownership | [Fitting](reference/fitting.rst), [projects](reference/project_files.rst) |
| `core.workflow` and `cli` | Scriptable operations, work-directory persistence, command parsing, and output | [Python API](api/index.rst), [CLI](reference/agent_workflow.rst) |
| `gui` | Widgets, editable drafts, queued work, plotting, and application orchestration | [GUI guidelines](GUI_GUIDELINES.md) |
| `tools` and `tests` | Generated review material, structural checks, and executable contracts | [Harness](HARNESS.md), [test placement](../tests/README.md) |

These owners are not a claim that all orchestration has moved out of the GUI.
`MainWindow` still coordinates application state, selection, and publication of
results. Compatibility imports such as `core.instrument`, `gui.panels.fit_panel`,
and `gui.windows.grouping_dialog` remain live. New work belongs in the owning
modules, not in another parallel implementation behind those facades.

## From acquisition to representation

The principal flow is:

```text
file or acquired arrays
  → decode and capture source facts
  → resolve identity, acquisition, source, and replay authority
  → prepare an explicit scientific request
  → compute with the selected observations and policies
  → publish a result, its provenance, and its representation
```

A `Histogram` retains detector counts, timing, good support, and any supplied
variance. A `Run` holds the acquired histograms, metadata, grouping request, and
source/instrument decisions. A `MuonDataset` carries a curve with its run and,
where available, captured histogram observation and effective reduction recipe.
A reduced curve is not a replacement for the raw acquisition.

File, project, and user-input boundaries interpret external values and produce
canonical records. Numerical preparation resolves actual selected channels,
periods, alignment, bin membership, correction support, and observation model.
Consumers use those resolved domains; they do not independently reconstruct
windows or infer physical conventions from an instrument label. Raw-array APIs
without acquisition context retain their explicitly documented input contracts.

Grouping profiles hold shareable authored policies. Resolving a profile against
a run combines those policies with that run's acquired facts; a manual timing
offset is not another file time zero. Effective recipes describe the settings
actually executed, including corrections and reference choices. Counts-first
reduction and covariance/likelihood handling remain owned by the numerical
preparation and fitting layers, not by widgets or command handlers.

A representation owns its recipe and analysis state. A run's exploratory single
fits are distinct from results owned by a `FitSeries`; a batch or global fit does
not overwrite the run's single-fit collection. Data groups own run-series
membership, while fitted results retain the membership actually analysed.
Changing live members can make a series stale without deleting its historical
results. Joint fits and parameter-fit studies retain their own identities and
input snapshots. Display selection and browser highlighting do not redefine
those scientific owners.

## Instruments, views, and readouts

Each bundled instrument has one frozen `InstrumentProfile` in a facility-qualified
module. Its ordered sections declare identity, source, recognition, detectors,
presets, quirks, defaults, sources, and review. The catalogue discovers profiles;
recognition combines facility/name/label evidence with acquired-map compatibility.
A histogram count alone does not establish an instrument. Unknown identity or
unavailable geometry uses a preset-free generic layout rather than guessed facts.

Physical elements, acquired channels, and drawing occurrences are separate.
Acquisition channel keys and recorded ordinals define scientific membership.
Legacy layout keys can distinguish file configurations under one display name.
Composed directional rows, circular/grid arrangements, and explicit placements
expand through shared geometry; moving a drawing does not change a preset.
Read-only context repeats an existing channel without creating another editable
one. Unlocated auxiliaries remain selectable outside spatial views.

Numbering slots are qualified by profile, acquisition, and channel, independently
of panel order. A slot can have unknown or multiple physical associations.
Beam direction and view bases are explicit declarations in a common arrangement
frame; detector placements remain view-local. Unknown axial viewing sense stays
unknown. A beam cue does not establish a run's field or spin direction.

Readout sets are applied choices among real acquired channels. Known combined
constituents and conditional contributors describe shared events, not exact
exported-bin equations or a covariance model. One core eligibility check covers
the effective requested selections after exclusions, including each original
acquisition and its own reference before period pooling changes coordinates.
Joint analyses reject declared overlaps; separate diagnostic traces do not become
a joint request merely because they are prepared together. Unresolved conditional
contributions are not assigned a port by inspecting count values.

Presets state explicit memberships, ordered asymmetry legs, and nominal balances.
Changing a view or source profile does not silently rewrite saved groups or
transfer fitted calibration to different channels. Fresh defaults and saved
requests have separate preparation paths. Evidence status, maintainer acceptance,
and an instrument-scientist review are different facts; reviewer names require
consent. Generated review pages and diagrams derive from these declarations
outside pure core.

## Provenance, execution, and cache identity

Keep three authorities distinct:

- The source observation records what was acquired, with content identity,
  instrument evidence/choice, source assessment, and acquired-period selection.
- The authored request records what analysis was requested, including grouping,
  calibration policies, timing, corrections, and references.
- The result records what was executed: effective recipe, observation/likelihood
  attribution, numerical policies, and computation revision.

A filename or run number alone does not establish identical acquisition content.
A saved source assessment, including captured unknown, is not permission to infer
a fresh profile fallback on reopening. Background references carry their own
source, timing, exposure, and instrument authority rather than inheriting the
sample's. Contrary acquired evidence requires an explicit identity decision.

Built-in loaders accept source-bound `ReplayState` before quick-look reduction.
The registry also supports legacy `BaseLoader.load(filepath)` implementations;
a bare `**kwargs` does not declare the replay protocol. The compatibility adapter
preserves a decoded owner's observations and restores captures only under the
supported identity/source conditions. Opaque decoders cannot be made replay-aware
by guessing fresh defaults. See [loading](reference/loading_data.rst) and
[grouping](reference/detector_grouping.rst) for availability and repair behaviour.

A reusable preparation or cache entry must match everything its consumer reads:
actual observations, effective selections, timing/correction policies, calibration,
source/reference identity, and canonical instrument context. Original constituent
authorities remain relevant after pooling. Matching an event or object address
alone is not a scientific freshness guarantee.

`core.computation.COMPUTATION_REVISION` is independent of package versions,
storage schemas, and method-specific policy identifiers. Change it when numerical
interpretation or execution eligibility changes: reduction, acquired timing/count
interpretation, calibration, uncertainty, likelihoods, or fitted semantics.
Documentation and storage-only changes do not require a bump. The revision is a
compatibility promise, not a correctness certificate.

Capture execution attribution when computing. Codecs preserve it; absent legacy
values stay unknown. Known matching revisions permit reuse only when the other
input and policy checks also match. Saving, reopening, relocating, or exporting
never upgrades a historical calculation. Successful explicit computation can
publish a replacement representation snapshot; passive restoration preserves
recorded snapshots even when a transient display must be regenerated.

## Project and released-format compatibility

`core.project.schema` dispatches versioned project migrations. Small converters
own released parameter/model and correction shapes; standalone wizard results
have their own versioned persistence codec. Keep converters and fixtures needed
to load every released project, including alpha releases. A current storage
version does not establish current scientific execution.

`prepare_project` validates canonical structure and constructs detached
`PreparedProject` domain objects before active application state is cleared.
Structural failures identify the serialized field path. Missing acquisitions,
unavailable user models/plugins, and historical membership/constraint archives
remain inspectable unavailable states, not repaired current results. Source
verification and explicit replacement decisions are separate from structural
validation. Canonical project JSON contains no live runtime objects.

The GUI prepares projects off-thread. Cancellation before application leaves the
active session intact; cancellation during source/item application retains
completed items and the explicit incomplete-project save guard. Editable forms,
source decisions, fit identities, and historical results survive persistence;
runtime widgets do not implement a second migration path.

## Front ends and long work

The CLI parses input, calls `core.workflow`, and renders results or actionable
errors. Work-directory persistence is a separate session surface from `.asymp`
projects. Shared numerical operations remain callable without Qt.

The GUI owns drafts and interaction. Applying an accepted request publishes its
results through existing domain owners; cancelling an editor does not mutate the
source analysis. Shared widgets, tokens, fonts, metrics, canvas construction,
and axis policies are described in [GUI guidelines](GUI_GUIDELINES.md).
Presentation density and axis limits do not change full-resolution analysis or
export inputs.

Use `TaskRunner` and cooperative cancellation for loading, fitting, transforms,
and reconstruction. Queued QObject-bound receivers publish results on the GUI
thread; request identity/generation checks discard superseded results. Owners
retain workers and shut them down on close/reject. Worker-object destruction and
screen-wrapper lifetime have shared owners, not per-window workarounds. See the
GUI guidelines for the exact lifetime and responsiveness contracts.

Boundary changes need focused tests at the owning layer, followed by the
[harness gates](HARNESS.md). Scientific claims cite primary sources and use
independent anchors; an old comparison transcript or a passing roundtrip alone
is not numerical certification. [Contributing](contributing.rst) describes the
instrument-authoring and review workflow.
