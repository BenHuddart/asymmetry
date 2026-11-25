# Harness Engineering for Asymmetry

This project applies the harness-engineering pattern described in OpenAI's
"Harness engineering: leveraging Codex in an agent-first world":
https://openai.com/index/harness-engineering/

The practical goal is simple: make the repository itself legible enough that an
agent can understand the product, choose the right files, run the right checks,
and improve the harness whenever it gets stuck.

## What Counts as Harness

Harness code is any repo-local artifact that improves agent reliability:

- `AGENTS.md`, the one binding guide for every agent (`CLAUDE.md` imports it and
  `.github/copilot-instructions.md` points to it, so no rule lives in a
  vendor-specific file; the harness caps its length and checks the import)
- architecture docs that describe boundaries and intended dependencies
- executable checks in `tools/harness.py`
- CI jobs that run the same commands agents run locally
- tests, examples, and smoke checks that turn product intent into feedback

Plans and decisions live in GitHub issues and PR bodies, not here.

## Validation Ladder

Run the smallest useful check first. Move upward when a change crosses a
boundary, touches shared behavior, or changes user-visible workflows.

| Command | Purpose |
| --- | --- |
| `python tools/harness.py structural` | Fast repository-shape checks, including the core/GUI dependency boundary. |
| `python tools/harness.py lint` | Ruff format check and lint check across `src`, `tests`, and `tools`. |
| `python tools/harness.py lint-all` | Alias for the full Ruff baseline. |
| `python tools/harness.py test -- tests/<layer>/test_name.py` | Focused pytest run while iterating on a specific behavior. |
| `python tools/harness.py test --tier fast` | Inner-loop suite: non-GUI tests only (~25s). |
| `python tools/harness.py test` | Standard tier (default): everything except `slow`/`integration` (~2 min locally). |
| `python tools/harness.py gui-smoke` | Headless GUI startup check using the app's `--smoke-test` path. |
| `python3 tools/harness.py docs --mode source` | Fresh offline source documentation gate. |
| `python3 tools/harness.py docs --mode publish` | Same defect gate plus ordinary screenshot completeness. |
| `python tools/harness.py validate` | Structural checks, lint, and the standard-tier test suite. |

### Documentation modes

`tools/docs_build.py` owns the fresh HTML invocation, subprocess exit handling,
and 600,000-byte page budget; the harness delegates to it. `tools/docs_policy.py`
owns the Sphinx mode, strict authored-reference checks, exact external-reference
limitations, and static screenshot contract used by both local and CI builds.
Sphinx is a documentation dependency, not an application dependency.

Use `python3 tools/harness.py docs --mode source` for a public checkout with
optional data locators unset. It runs no capture and requests no inventories.
Only exact registered missing generated images become visible notices; static,
logo, download, unknown-image, local/API-reference, markup, and unknown/untyped
warning defects still fail. The contributor default is `source`; unknown modes
are rejected before generation. `--screenshots` is an explicit capture request.

Use `python3 tools/harness.py docs --mode publish` with prepared ordinary GUI
assets. The existing registry/reference/completeness checks remain authoritative.
Corpus figures require existing PNGs, which may be the explicitly permitted
labelled placeholders produced by `capture_corpus --stubs-missing`; these do not
establish acquired-data capture. The hosted capture/fallback disclosure, timeout,
permissions, concurrency, retention, and deploy dependencies remain in force.

Every gate rereads all sources (`-E -a`), including repeated builds and mode
switches sharing output/doctree paths. It never relies on cached warnings or
subtracts parsed warnings from a failed Sphinx exit. Native `-W --keep-going`
rejects all remaining diagnostics, including multiline and untyped warnings;
config/import/extension/process failures retain their exits. Page budgets apply
after Sphinx succeeds. Failed builds cannot reach Pages artifact upload.

The exact fifteen external-object exceptions and annotation-rendering limits
are documented in `docs/README.md`. There is no category-wide warning exemption.
`show_warning_types` improves display on Sphinx 7.3+; it is not a gate dependency.
A successful limited source build does not certify publishing readiness or
external-reference completeness. Direct Sphinx and non-HTML make targets are
non-gating development commands. `make html` and `make publish` delegate to the
same harness gate.

### Module size and owned attributes

`structural` and `validate` enforce the maintained Python scope: regular `.py`
files at the repository root and below `src`, `tests`, `tools`, `docs`, `examples`,
`packaging`, `scripts`, `.github`, and `.claude/hooks`, including new local files.
Other agent configuration and `.claude/worktrees` are excluded. Other hidden directories,
`__pycache__`, build/generated output (`build`, `dist`, `_build`, `_generated`),
virtual environments (`venv`, `env`, or directories containing `pyvenv.cfg`),
`site-packages`, `node_modules`, `vendor`, `vendored`, and egg metadata are excluded.
File and directory symlinks are never followed, including dependency checkouts.
Other roots are outside maintained scope; tracked repository tooling under the
explicit hidden roots remains covered.

Physical lines are LF-delimited byte lines, including blank, comment, and
docstring lines and a final nonempty unterminated line. CRLF counts once;
a bare CR does not start a new physical line. Empty files have zero lines.
**999 lines passes for a new module; 1000 and 1001 fail.** This implements
AGENTS.md's “under 1,000” rule. Existing files of 1000 or more lines have exact
frozen ceilings in `tools/baselines/maintenance.json`. Resolved debt requires
lowering its ceiling to the actual count in the same change, or removing the
entry below 1000 lines or on deletion. Stale, malformed, duplicate, and
out-of-scope entries fail. Checks never edit files or regenerate allowances.

The same baseline records proven built-in `hasattr(receiver, "literal")` and
`getattr(receiver, "literal", None)` occurrences. Proof uses top-level project
class definitions, their assigned/annotated instance and class members, slots,
properties and methods, plus resolvable project inheritance. Direct instance
receivers (including a renamed first method argument), annotated arguments,
simple instance aliases, construction, and proven child declarations can establish
ownership. Project imports and relative re-exports are resolved statically,
without importing application, scientific, or Qt code. A child receiver alone
establishes no ownership of its members. Conflicting/unresolved child assignments
lose the child type. Built-in imports, qualified builtins and simple aliases are
recognised; parameter/local shadowing and reassignment invalidate bindings.
Methods use module globals, while bases bind at class definition. Closures lose
captured names that the enclosing function may rebind.
Compound statements conservatively discard rebound aliases at control-flow joins.

Qt/C-extension inherited APIs, unresolved plugin objects, Protocol interfaces
(including all classes derived from `typing.Protocol`), external
children, dynamic attribute names, and defaults other than literal `None` remain
outside this rule. These are receiver/member boundaries, with no file wildcard
exceptions. Protocol-derived classes remain unresolved even when a concrete
subclass declares additional own state. Static proof is deliberately incomplete: factories, arbitrary
container elements, dynamically assigned members, nested/local class definitions,
comprehensions/lambdas, descriptor return types, and general type inference are
unresolved. An unresolved probe is not certified safe. This is no general
`getattr` or `try/except` prohibition; manual ownership review still applies.

Each allowance identifies the path, lexical class/function scope, proven owner,
and complete normalized call AST (without line/column positions, preserving empty
fields consistently across Python versions), with its exact
multiplicity. Unrelated line shifts preserve an occurrence. A different literal,
receiver spelling, call, or scope cannot spend its allowance. Moving an identical
call within the same scope is indistinguishable; arbitrary monkeypatching and
interprocedural mutation are outside the proof. Remove resolved identities or
lower their multiplicity; do not replace one with a different probe of equal count.
Prefer direct access to construction-guaranteed state, or model optional state
explicitly. Review lifecycle assumptions before changing application code.

### Defined-base maintenance gate

A snapshot ceiling cannot establish touched-file shrinkage or prevent baseline
restoration. For a branch, choose the immutable reviewed base explicitly:

```sh
python3 tools/harness.py structural --base <commit-sha>
python3 tools/harness.py validate --base <commit-sha> -- -n 4 --dist loadfile
```

`ASYMMETRY_STRUCTURAL_BASE` supplies the same base to both commands; `--base`
takes precedence. The gate checks final worktree contents, including committed,
staged, unstaged, and new local files. Index and worktree Git changes (including executable-mode edits) also mark files touched,
even if unstaged edits restore their base bytes. CRLF conversion in a clean
`core.autocrlf` checkout is not a touch; Git still identifies actual line-ending
and mode edits. Intermediate index contents
are not a separate candidate: run the gate again after staging the final tree.
A touched oversized base file must shrink, including comment-only and same-size
edits. Unchanged debt may remain. Allowances cannot be added, raised, or restored
against the base, even by editing or deleting/recreating the baseline.

Paths are identities: deletion removes debt; renames and copies are new paths
and receive no allowance from the old path. Reduce a renamed/copied oversized
file below 1000 lines and remove its owned probes. This conservative rule avoids
rename-similarity heuristics and transferable looser ceilings.

Initial bootstrap is limited to the single reviewed pre-rollout commit
`606992f5a4f6de2a3781f60c3e5444ffbaae6069`. Only its actual source debt can
supply initial allowances; candidate additions/growth cannot. Other bases without
a baseline fail. After rollout, use a reviewed base containing the ratchet;
selecting an older base does not prove monotonicity since a newer one.

Without a base, the command prints **snapshot only**: it checks exact current
debt but cannot prove touched-file shrinkage, baseline monotonicity, or
non-resurrection. `--snapshot` explicitly chooses this limited mode and ignores
the environment base. Missing requested commits and Git errors fail, including
shallow clones lacking the base. History-aware checks require Git 2.45 or newer
to disable automatic promisor lazy fetching; local replacement refs are also
disabled so the named immutable object owns the comparison. Missing promised
objects fail with an actionable base-object error. Local commands never fetch, checkout, or reset;
make the required commit available yourself. Source bytes and NUL-delimited
index paths are read with structured Git arguments, once per tree rather than
per occurrence.

The always-run read-only CI checks job obtains full checkout history and fetches
the validated immutable event base before `structural --ci`. PRs compare the
checked candidate (including GitHub's merge checkout) against the event's
`pull_request.base.sha`; pushes compare against `before`; manual dispatch requires
an explicit 40-character `structural_base` input. Missing/zero/unsupported bases
fail. Neither HEAD versus itself, an inferred parent, nor a moving remote branch
is used as the CI base. The gate itself does no network operations.

### Grouped-count boundaries

Structural checks reject `min` calls that truncate multiple array lengths in
`core/transform`. Detector alignment computes its terminal crop explicitly.
Functions consuming `binned_fb_asymmetry` or `slice_to_good_window` read the
validated `BinWindow` rather than parsing window keys from grouping again.
Effective membership calls also supply the acquired histogram count so unknown
exclusions cannot bypass validation. All three violation counts are zero;
metadata displays and non-F/B analyses retain
their own window policies.

### Physical-unit authority

Unit comparisons in raw-file loaders belong to the shared physical-unit table.
Structural validation freezes the three NeXus normalizer comparisons and rejects
new dispatch elsewhere. The constant authority also covers the Celsius offset.

### Physical-constant authority

Structural validation scans Python assignments and numeric literals in the
application, screenshot scenarios, and tools. Physical constants come from
`core/utils/constants.py`; teaching examples derive their rounding from that
authority. Tests may type independent reference values. Source-reproducing
paper and guide arithmetic has a baseline with a frozen count per file;
removing a duplicate requires lowering its count or removing its entry.

### Test tiers

`test` and `validate` accept `--tier {fast,standard,full}`:

- **`fast`** — pure-Python tests only (no Qt, no file I/O, nothing `slow`).
  Over half the suite; runs in ~25s with xdist. Use this for the **inner dev
  loop**.
- **`standard`** (default) — everything except `slow` and `integration`. This is
  the pre-push gate and what CI runs. The GUI tests it adds each construct
  widgets (often a full `MainWindow`), so it takes ~2 min locally — run it once
  per task, not per iteration.
- **`full`** — every test, including `slow`/`integration`. The reusable `Full test
  suite` workflow runs inside the release workflow on the exact tagged commit
  and gates publication alongside both platform builds. Manual dispatch can
  validate other refs without publishing. It runs with reduced parallelism (`-n 2`
  and a raised per-test timeout) rather than `-n auto`: the exhaustive
  global-wizard harness test fans its own fits across a `cpu_count - 2` process
  pool, so a worker-per-core layout oversubscribes the runner and stalls
  neighbour fits past the timeout (historically crashing the release-tag run).
  The full-tier workflow logs each test's identity and stops on the first worker
  crash, retaining the failing test in the log instead of restarting workers.
  All tiers also pin the numeric backends to one thread each (`*_NUM_THREADS=1`)
  so xdist's process-per-core parallelism isn't multiplied by per-process BLAS
  threads.

Standard CI configures pytest's native stack watchdog at 90 seconds; the full
suite uses 600 seconds. Pytest resets this timer for each test and owns its
diagnostic file descriptor through capture and teardown. These stack dumps do
not change the test's failure deadline or establish why a worker terminates.

Standard shards set `ASYMMETRY_PYTEST_DIAGNOSTICS` to retain each worker's
terminal output, including pytest-timeout's report before a hard worker exit.
The runner log retains native-fault and watchdog output from stderr. A
`controller.jsonl` file records worker identities and PIDs at creation, whether
xdist reports an abnormal disconnect, and observed process return codes after
normal execnet teardown. Replacement workers receive their own records; an
abnormal disconnect remains a test failure under the existing restart policy.
On POSIX, a negative return code also records the terminating signal number.
An observed signal can reflect execnet's teardown cleanup; it does not by itself
identify the cause of an earlier disconnect. Interpret it with the disconnect
record and retained output.
The opt-in hook reads execnet's cached subprocess status; it never calls process
`poll` or `wait`, reaps, kills, or restarts a process itself. Cached-status
observation is bounded to 0.25 seconds for the whole session after teardown. Unsupported
transports/interfaces and exit codes still absent at that deadline are recorded
as unavailable, with no inferred signal. Creation and disconnect records survive
even if the controller cannot reach its final session hook.

The private local-process boundary is isolated in `tests/_worker_diagnostics.py`
and exercised with pytest-xdist 3.8.0 and execnet 2.1.2: direct local popen
workers expose `gateway._io` as `Popen2IOMaster` with a standard-library `Popen`.
Remote, proxied, or unrecognised IO objects have no local-process evidence.
Without the environment variable, local test output and process handling are
unchanged. A recorded timeout report identifies an expired pytest deadline.
Missing output does not identify a SIGKILL or an out-of-memory kill.

Standard shards retain `oom-before.json` and `oom-after.json`, snapshots of
Linux's runner-wide `/proc/vmstat` `oom_kill` counter. An EXIT trap records the
after snapshot on shell completion, including a failed test command, preserving
its exit status. A forcibly terminated shell or job can prevent that snapshot.
Unreadable or missing counters are explicitly unavailable. These files, worker
logs, controller records, and the runner log are uploaded for seven days even
when tests fail. The full-suite workflow separately logs the same counter and
available memory before and after tests, including test failures.

A counter increase establishes an out-of-memory kill somewhere on the runner;
it does not identify the victim process. Even an observed SIGKILL alongside a
counter increase cannot attribute that OOM kill to the worker. An unchanged
counter leaves other causes of worker termination unresolved.

Tests inherit a type marker automatically: anything not explicitly marked `gui`
or `io` is treated as `unit` (see `tests/conftest.py`). Naming explicit targets
(`-- tests/<layer>/test_x.py` or a `::node-id`) runs exactly those, bypassing
the tier filter.

`tests/` mirrors the `src/asymmetry/` layer boundaries — `tests/core/`,
`tests/gui/`, `tests/io/`, `tests/project/`, `tests/tools/`, plus
`tests/integration/` for genuinely cross-cutting end-to-end tests, and the
already-organized `tests/negmu/`, `tests/docs/`, `tests/porting/`. See
`tests/README.md` for the placement rules. Tiering and sharding are driven by
pytest markers and node-id hashes, not by directory, so this layout is
transparent to `--tier`/`--subset`/`--shard` and to CI's path filters (which
match `tests/**` recursively).

CI splits the standard tier across six runners with `--shard {1..6}/6`, each
running `-n auto --dist worksteal` on its 4 vCPUs. `--shard K/N` (a
`conftest.py` option) keeps a stable 1-of-N slice partitioned by a hash of each
test's node id, so the independent shard processes cover every test exactly once
with no overlap, regardless of `pytest-randomly` ordering, and together cover
exactly the standard tier. Each shard mixes GUI and non-GUI tests; the hash
keeps per-shard test time within about 25% of even, which is less than the
runner-to-runner variation, so the split needs no recorded durations. Every
shard pays a fixed cost of roughly 1.5 min (checkout, uv install, and test
collection, most of which is pytest rewriting asserts in ~430 test modules on
each xdist worker), so adding shards stops paying off as that fixed cost
dominates. `--subset gui`/`--subset non-gui` still select by marker for local
runs.

The harness sets `QT_QPA_PLATFORM=offscreen` by default for command runs so GUI
tests and smoke checks can execute in CI and local agent sessions. The current
CI validation uses the passing baseline: structural checks, Ruff, and pytest.

When a local `.venv` exists, `tools/harness.py` re-executes itself with that
interpreter. This keeps local agent runs on the project dependency set instead
of the ambient Python environment, which matters for fitting: the project pins
numpy 2.2.x and a working iminuit, and numpy ≥ 2.3 breaks fitting.

A per-test timeout (`--timeout=120`) makes a hung test fail rather than stall
the run.

### Workflow geometry regression

The unresolved-survey geometry regression in `tests/core/test_workflow_screen.py`
uses the real loader, reduction and screening path with `DynamicLorentzianKT`
explicitly excluded. It checks that an unresolved survey clears stale file
geometry and preserves the fitted time window. Finite-field dynamic-KT numerical
accuracy remains covered by `tests/core/test_dynamic_kt_accuracy.py`; field
seeding, free-to-pinned field behavior and fit recovery remain covered by
`tests/core/test_wizard_applied_field.py`. This scope keeps geometry propagation
independent of the expensive finite-field model's optimizer runtime.

### Optional numerical references

Default validation runs analytic checks and regressions against recorded native
outputs using owned synthetic input. It does not certify fresh reference execution.
Explicit integration targets enable bounded native comparisons; tests/README.md
specifies executable/source locators, pinned identities, and reproduction commands.
Unconfigured or uninstalled optional references skip specifically. A selected
reference with an incompatible identity, broken command/import/parser, timeout,
empty output, or numerical disagreement fails. Source-module execution is labelled
separately from installed algorithms, and neither is inferred from a manual alone.

## Agent Workflow

1. Read `AGENTS.md`, then open only the docs relevant to the task.
2. Identify the affected boundary: loader, transform, fitting, project schema,
   GUI panel, packaging, or documentation.
3. Write or update the smallest test or structural check that captures the
   desired behavior.
4. Implement the change in the layer that owns the behavior.
5. Run focused checks while iterating, then `python tools/harness.py validate`
   before handing work back when feasible.
6. If an issue repeats, add a harness rule or documentation note instead of
   relying on future memory.

## Boundaries Worth Encoding

- `asymmetry.core` remains independent of PySide6, matplotlib, and
  `asymmetry.gui`.
- GUI modules may depend on core modules, but they should not duplicate core
  analysis algorithms.
- File loaders should convert external formats into `MuonDataset`, `Run`, and
  `Histogram` objects with provenance attached.
- Project persistence should evolve through explicit schema handling in
  `asymmetry.core.project.schema`.
- Optional dependencies should be imported at the boundary that needs them and
  guarded with clear errors or skipped tests.

## When to Expand the Harness

Add a new harness check when:

- a bug would have been caught by a mechanical repository-shape rule
- an architectural boundary has become important enough to enforce
- a manual QA step can be made repeatable with a script, test, or smoke check
- a reviewer has to explain the same project convention more than once

Keep rules high signal. A harness that fails noisily for incidental style drift
will stop being trusted; a harness that encodes real project invariants will
compound.

## Retired planning locations

Plans and review decisions belong in GitHub issues. The structural check rejects
new files under `docs/plans/`, `docs/porting/`, `docs/studies/`, `docs/audit/`,
`docs/investigations/`, and at `docs/PLANS.md`. The retirement baseline at
`tools/baselines/retired_docs.txt` stays empty: any recreated file fails the
check. Hidden files and `__pycache__` directories are ignored. Lasting content
belongs in the owning Sphinx page, a regression test, or an invariant comment.
Local Markdown/RST links into retired locations are also rejected; historical
external links and prohibition/test fixtures are not active local references.

## Test user-state isolation

The root `qapp` fixture owns QApplication construction; structural checks reject
local `qapp` fixtures and direct application construction elsewhere in tests.
The root fixtures load Qt and run window cleanup only for tests marked `gui`.
A core-only run imports no Qt through fixtures and can run without PySide6.
GUI settings (including system fallbacks), app data, and cached fit times use
per-test temporary locations; plugin discovery is isolated for every test.
The temporary roots are independent between xdist workers.

`tests/tools/test_test_isolation.py` checks a core harness subprocess with Qt
imports blocked. On Linux it also runs a GUI case with its subprocess home
pointing at a regular file, which makes real user-config paths unwritable even
for a privileged test process. Reproduce both checks with
`python tools/harness.py test -- tests/tools/test_test_isolation.py`.

## Installed distribution smoke

CI always runs separate wheel jobs on Python 3.11, 3.12, and 3.13, with
constrained and unconstrained dependency resolution. Their failure,
cancellation, or skip fails `CI success`, including changes to metadata, resources,
entrypoints, build tooling, and workflows. Editable test shards and their optional
dependencies remain separate from the target environments.

Run it with an outer virtual environment containing `build` and `pip`:

```bash
python3 tools/distribution_smoke.py --profile core --profile gui --profile agent --profile all
python3 tools/distribution_smoke.py --profile core --profile gui --profile agent --profile all --resolution unconstrained
```

The output directory must be new and outside the checkout. `--python PATH` may
be repeated to use the same artifact with several interpreters. The driver builds
an sdist in backend isolation, unpacks it, and builds the wheel from that unpacked
source in a second isolated build. It installs the wheel and selected extras
into fresh venvs. The default `core` profile installs only the scientific stack;
`gui` requests `gui,hdf5,root`, `agent` requests `agent`, and `all` requests `all`.
The default resolution uses `constraints.txt`; `--resolution unconstrained`
uses ordinary index resolution within the distribution's declared bounds.
Build tooling stays in the outer environment. Successful target venvs are removed
after their evidence is recorded; failed targets remain available for diagnosis.

Child probes run outside the checkout with interpreter path injection removed,
user sites disabled, and temporary home/config/data/plugin locations. They check
artifact bytes and metadata, installed module origins, public workflow/fitting/data
imports, an analytic exponential value, console/module/subcommand help, bundled
skill resources, and `pip check`. Unrequested optional dependencies, development
tools, and CuPy must be absent. Requested readers load synthetic HDF5 NeXus and
ROOT counts; HDF4 exercises its native runtime with a write/read control.
Plotting writes a headless PNG, GLE generates a script without invoking the
external compiler, and GUI profiles run the installed `asymmetry-gui --smoke-test`
with isolated settings. Logs and the JSON manifest retain artifact hashes,
successful target dependencies, interpreter identities, and module origins.
Failed operations retain their logs even when no successful target record exists.

These Linux installation checks do not certify minimum dependency versions,
Windows/macOS optional-runtime installation, the external GLE compiler, or CUDA.
Native desktop artifact checks and release dependency reproduction have separate
owners. HDF4 checks on other platforms require their documented native runtime.

## Desktop release environment checks

`tests/tools/test_release_environment.py` exercises dependency/interpreter drift,
selected-extra completeness, immutable wheel origins, strict hashed input parsing,
compiler-engine identity, replay equality, allowlisted download observations, and
waited packaged-smoke success/failure with Python path injection removed.
Run it with the existing native workflow and installer tests:

```bash
python3 tools/harness.py test -- tests/tools/test_release_environment.py tests/tools/test_release_workflows.py tests/tools/test_windows_packaging.py -n 4 --dist loadfile
```

These deterministic checks use synthetic metadata without network or native GUI
execution. Native release rehearsals separately install the reviewed platform
inputs twice, check observed versions/tools, build each supported desktop artifact,
and execute existing packaged smoke checks. RELEASING.md describes input updates
and replay commands. The Linux core-wheel gate remains required and independent.

## Desktop artifact lineage checks

`tests/tools/test_hdf4_archives.py` substitutes tiny archives at the real fetcher
entry point: correct/cache bytes, early/later mismatches before extraction or
runtime mutation, and nonregular/ambiguous DLL members. Ordinary tests use no
network. `tests/tools/test_release_artifacts.py` exercises actual record
generation/verification, stale and swapped source/version/target/input identities,
changed components/notices, nested private metadata, framework links, and corrupt
final downloads. Synthetic native-runner substitutions reject swapped final
payloads, missing identity/notices, changed assembly resources, and false Python
module matches from embedded script names. Real CArchive bootstrap modules and
PYZ modules retain separate container identities. Workflow tests ensure manual dispatch exercises download
verification and publication rechecks its separate downloaded bytes.

```bash
python3 tools/harness.py test -- tests/tools/test_hdf4_archives.py tests/tools/test_release_artifacts.py tests/tools/test_release_environment.py tests/tools/test_release_workflows.py tests/tools/test_windows_packaging.py -n 4 --dist loadfile
```

Native rehearsals additionally inspect actual PYZ/collected components, staged
DLL lineage, retained notices, assembled DMG/EXE payloads and extra resources,
final bytes, and transported public records.
These checks supplement the existing replay/compiler and waited outside-source
startup checks; synthetic tests do not substitute for either native platform.

## Desktop scientific scenario checks

`tests/tools/test_scientific_acceptance.py` invokes the actual desktop diagnostic
entry with native HDF4 and ROOT files written through the declared dependencies.
It checks the nonzero analytic ratio, percent-scale constant fit and known-error
weighted-mean anchor, parsed text export, verified project source replay, settings
and app-data isolation, and bypassed plugin discovery. Stage failures include
wrong identity, missing/corrupt inputs, missing HDF4 reader, altered reduction,
failed fit, empty export, and failed project reload. A disposable source copy
checks actual omission of the used logo. Process failures, timeout, missing or
stale reports, and reuse of an invocation directory cannot pass the parent runner.

```bash
python3 tools/harness.py test -- tests/tools/test_scientific_acceptance.py tests/tools/test_native_acceptance.py tests/io/test_hdf4_loader.py tests/io/test_root_loader.py tests/project/test_project_preparation.py tests/tools/test_release_environment.py tests/tools/test_release_artifacts.py tests/tools/test_release_workflows.py -n 4 --dist loadfile
```

The fixture writer is build/test tooling; the frozen application imports neither
it nor tests. Reports contain original fixture hashes and explicit settings and
tolerances. Native delivery jobs start without a checkout or build environment and
use an isolated standard-library driver. Source verification binds transported
tools, inputs, identity, and the source comparison report to the exact candidate.
Both final installed Windows x64 and read-only mounted macOS arm64 launchers must
complete the same journey, with imports resolved inside their delivered payloads.

`tests/tools/test_native_acceptance.py` checks packet identity and tool tampering,
fixture transport, checkout/build-environment rejection, and shared cleanup after
scientific or native failures, including an attached image without a mounted
volume. Fresh matching omission reports and an intact-copy control reject unrelated
startup errors; only native-library omissions allow component-specific loader
diagnostics without an app report. The workflow checks require both native outcomes
on dispatch and before publication. Disposable native payload copies omit the used
logo, HDF4 library, and native reader backend; actual nonzero scientific or loader
failures are required, and a timeout cannot satisfy an omission. Original media,
payloads, and cleanup remain checked through the existing artifact owner. The
separate one-day scientific report artifacts contain explicit JSON reports only.
See RELEASING.md for diagnostic commands, finite budgets, and limits. The Linux
core-wheel check remains independently required.
