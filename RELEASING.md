# Releasing

Releases are agent-proposed, human-confirmed, and machine-executed. The
mechanical work — version bump, changelog roll, tag, platform builds, GitHub
release, docs deploy — is fully automated; the only human steps are saying
"yes" and (rarely) rotating a token.

## How a release happens

1. Someone (usually an agent — see the guidelines below) proposes a release
   and the maintainer confirms the bump kind.
2. The **Cut release** workflow is dispatched:

   ```bash
   gh workflow run cut-release.yml -f bump=minor      # or patch / major
   gh workflow run cut-release.yml -f version=0.7.0   # explicit override
   ```

3. The workflow refuses to run unless the `CI success` check is green on
   `main` HEAD. This pre-flight check is an early refusal; publication is gated
   separately on the exact tagged commit. It then runs `tools/release_prep.py`, which bumps
   `pyproject.toml`, promotes `[Unreleased]` in `CHANGELOG.md` to
   `## [X.Y.Z] - <date>`, and prints the promoted section into the run
   summary for a quick eyeball. Finally it commits `release: X.Y.Z` to
   `main` and pushes the `vX.Y.Z` tag atomically.
4. The tag push triggers the existing pipelines untouched: `release.yml`
   builds and smoke-tests the macOS DMG and Windows installer and publishes
   the GitHub release; `docs-pages.yml` deploys the documentation site.
   `release.yml` checks that the tag matches `pyproject.toml` and the changelog
   was rolled. Its macOS and Windows builds run alongside the reusable full
   Linux test suite, all checking out the triggering commit. Publication waits
   for successful validation and both artifact builds. A failed, cancelled,
   skipped, or pending validation cannot publish, and validation on another
   commit does not satisfy this dependency.

There is no release PR and no manual tagging. Version numbers live in exactly
one place (`pyproject.toml`); `asymmetry.__version__` reads it via package
metadata, and the release workflows derive theirs from the tag, with the
guard keeping the two honest.

## Non-publishing rehearsals and retries

Run validation and both artifact builds on a branch without creating a release:

```bash
gh workflow run release.yml --ref <branch>
```

The publishing job runs only for a tag push; a manual dispatch skips it. The
rehearsal reads the version from `pyproject.toml` for artifact names. Release
runs do not cancel each other.

For a failed tag run, use **Re-run failed jobs** on that run to retain its event
and commit. Build artifacts expire after one day; after that, use **Re-run all
jobs** so both builds upload fresh artifacts. GitHub permits re-runs for 30 days.
For an older run or a failure needing code changes, fix `main` and propose the
next patch release. Never move or delete the failed tag. Its version heading
remains in the changelog even though no GitHub release was published.

## Reviewed desktop inputs

`packaging/release/tools.json` selects CPython (including patch version), the
GLE 1.7.0 source commit, supported target identities, and Inno Setup. The two
platform requirements files install the complete selected desktop dependencies
and backend/packaging tools as exact, SHA256-checked wheels. They are release
inputs; `constraints.txt` and the library's flexible source-install ranges retain
their separate role. Linux core-wheel CI is also a separate required check.

Release, manual release rehearsals, and both preview workflows use the same
`release-environment` action. Each creates two fresh environments outside the
checkout, bootstraps the pinned pip wheel, installs every reviewed wheel with
`--no-deps --only-binary=:all: --require-hashes`, and checks the observed complete
distribution set. The selected pip supports these flags and installation reports;
the actual replay exercises that contract. The interpreter's bundled ensurepip
only installs the bootstrap pip; it does not resolve the release dependencies.

GLE is fetched at its full reviewed commit. GLE and Asymmetry wheels are built
with `--no-deps --no-build-isolation` in each dedicated environment, with index
access disabled. All backend tools are already installed from reviewed wheel
inputs. This is controlled build isolation through fresh environments, rather
than borrowing a development environment. The local/VCS sources are outside pip's
hash-mode install: their immutable source identity and resulting wheel hashes are
recorded separately. No editable install or hidden backend resolution is used.
The installed wheels must match those hashes. `pip check` and explicit evaluation
of the selected desktop extras check requirement completeness.

Run a replay on its actual native platform with the Python from `tools.json`:

```bash
python tools/release_environment.py --target macos-arm64 --output <new-external-directory>
# Windows x64 uses --target windows-x64 with its own native interpreter.
```

The output must be new and outside the source checkout. Run from a clean committed
candidate. Activate `replay-1/environment` for local packaging; CI adds its scripts
directory to subsequent steps automatically. Packaged smoke runs outside source
through the shared Python process runner, which clears Python path injection,
waits for the frozen process, and rejects a nonzero exit status on either platform. Before freezing, run with that
verified environment's Python:

```bash
python tools/release_environment.py --verify <replay-output-directory>
```

Verification rejects changed source commit, tracked source edits, untracked source/resources (including ignored native
extensions), stale input
hashes, changed distributions, wheel origins, interpreter identity, or selected
extra requirements. The two replays must have identical normalised versions and
GLE commit. Wheel and native installer bytes can differ between builds.

Windows installs the selected Chocolatey Inno package, then invokes the compiler
by its explicit installation path. `--inno` first compiles a small output-disabled
probe and checks the actual compiler-engine version before compiling the installer.
It records the compiler executable and engine DLL hashes. The platform replay
records runner image, OS, and, on macOS, Xcode/SDK identities. The hosted OS image,
system frameworks/tools, fonts, and native timestamps are observed rather than
fully pinned. Neither previews nor releases promise byte-identical DMGs or EXEs;
the macOS preview uses the same macOS 15 arm64 runner selection as releases.

### Intentional input updates

1. Start a reviewed change to the relevant platform requirements and/or
   `tools.json`. Select versions for compatibility with the source ranges and
   supported native Python; do not update unrelated library ranges. GLE's tag,
   full commit, version, and backend metadata must agree.
2. On each affected native target, use the pinned pip to download proposed wheels
   into a new temporary directory. For example,
   `python -m pip download --only-binary=:all: --no-deps PACKAGE==VERSION -d <directory>`
   followed by `python -m pip hash <wheel>`. Inspect wheel metadata and all changed
   transitive requirements; update every affected exact pin and SHA256 explicitly.
   Review backend requirements of both source projects as well as runtime extras.
   Retain the selected wheel filenames and upstream identities for review.
3. Run the two fresh native replays and inspect their `observed.json` records.
   A failure is a reason to investigate and review inputs, never to regenerate or
   relax pins automatically. In particular, cross-platform resolution does not
   replace native installation and requirement checks.
4. Run focused packaging/workflow tests through the harness, structural/lint,
   standard validation, and source docs. Independently review the complete diff
   before pushing the candidate for a non-publishing `release.yml` rehearsal.
   Inspect both native build/smoke logs and the actual downloaded artifacts, then
   obtain final independent clearance before opening the PR.

Replay observations are allowlisted JSON; raw pip reports stay in the temporary
build directory. CI transports the two observation files and the Windows compiler
record separately from the binaries, with the existing bounded retention. Save
required evidence before it expires. These records describe the build environment;
they do not claim to inventory everything PyInstaller bundles or to certify the
delivered scientific workflow.

See pip's [repeatable installs](https://pip.pypa.io/en/stable/topics/repeatable-installs/),
[secure installs](https://pip.pypa.io/en/stable/topics/secure-installs/), and
[VCS support](https://pip.pypa.io/en/stable/topics/vcs-support/) contracts. Exact
constraints alone are neither an installation nor proof of a complete replay.

## Bundle and final artifact records

`packaging/release/windows-archives.json` owns the exact existing conda-forge
win-64 archive URLs and SHA256 values. Obtain proposed hashes from the channel's
[repodata](https://conda.anaconda.org/conda-forge/win-64/repodata.json.bz2), match
package version/build/filename, and independently hash downloads before reviewing
an input update. Never accept a fresh download's hash as its own expected pin.
These input bytes participate in replay identity checks and use LF on Windows.

`packaging/windows/fetch_hdf4_dlls.py <target-directory>` verifies every archive,
including reused `_download` cache bytes, before opening any archive or changing
runtime DLLs. A mismatch names the archive and expected/actual SHA256. Resolve
its origin before replacing a damaged cache entry; the fetcher never silently
upgrades inputs. Only regular direct `Library/bin/*.dll` members with unique
case-insensitive destinations are staged. Its `archives.json` records archive
identities and DLL member/content hashes. The same operation supports local
Windows setup, previews, and releases.

`tools/release_artifacts.py` owns the public records. Native workflows first run
`prepare` with the verified replay, target, and triggering commit; a minimal
`build-identity.json` is embedded before freezing. The spec exports only names
from PyInstaller's documented Analysis lists. `collect` inspects the actual
collected tree and recursively lists the executable's PYZ with the pinned
archive viewer. It records Python module identities separately from native
libraries, resources, metadata, archive members, and notices. macOS framework
links remain relative links bounded to the bundle, rather than duplicate files
with invented content hashes. Windows DLL hashes must match the verified staging
record. Third-party notices must match the source notices.

The target-specific `*-build-components.json` includes packaged version, exact
commit, Python/target, reviewed input hashes, resolved environment distributions,
source/index wheel identities, immutable GLE source, and observed native tools.
The installed environment includes build dependencies; its distribution list
alone is not evidence that those packages are bundled. Analysis module names and
the collected file inventory describe inclusion. System APIs, drivers, and system
frameworks remain external. This is a bounded component record, not a complete
licence or system-library provenance scanner. Public fields are allowlisted; raw
pip reports, local origins, environment/configuration, and host TOCs stay private.

After DMG/EXE assembly, `finalize` rechecks bundle bytes and writes
`<target>-artifact.json` with the actual artifact basename, size, and SHA256,
linked by hash to the build/component record. Windows records also include the
verified Inno engine and compiler hashes. `<target>-SHA256SUMS` is derived from
that final record. The embedded identity has no installer hash; the final
sidecar has no self-hash, avoiding circular digests. Preview artifact labels may
include the preview suffix while the embedded package version stays unchanged.

Recheck downloaded release/rehearsal assets against the triggering checkout:

```bash
python3 -I tools/release_artifacts.py verify --target windows-x64 --commit <expected-commit> --directory <download-directory>
python3 -I tools/release_artifacts.py verify --target macos-arm64 --commit <expected-commit> --directory <download-directory>
```

The expected version/target/Python/input hashes come from that checkout, not the
manifest's assertions. Missing, swapped, stale, or corrupted records/binaries
fail. A read-only job runs this verifier for both dispatch and tag events;
tag publication waits for it, full validation, and both builds. The publishing
job rechecks its separate download with the same verifier, then selects only the
two binaries and their six public sidecars. Replay observations remain separate
Actions evidence. Retention and non-publishing dispatch behaviour are unchanged.
Save all required native records/logs/artifacts before expiry.

Checksums establish byte identity and traceability, not signatures or byte-identical
rebuilds. Startup/import smoke and component inventories do not certify delivered
load/reduce/fit/export/project workflows. See the pinned
[PyInstaller spec and archive documentation](https://pyinstaller.org/en/v6.16.0/spec-files.html).

## Guidelines for agents

Agents maintain the changelog and propose releases; they never cut one
unprompted.

- **Keep `[Unreleased]` current.** Every PR with user-visible changes adds
  its entries under `## [Unreleased]` in Keep-a-Changelog sections
  (`### Added` / `### Changed` / `### Fixed` / `### Removed`). Never edit the
  notes of an already-released version, and never remove or duplicate the
  `## [Unreleased]` heading — `tools/release_prep.py` and the release guard
  depend on this structure.
- **When to propose a release.** At the end of a task whose PR has merged,
  check `[Unreleased]`. Propose a release when it holds at least one
  user-visible entry **and** no feature series is mid-flight (do not propose
  between parts of a themed PR run; propose when the series completes).
  Internal-only work — CI, tests, refactors, docs infrastructure — does not
  warrant a release on its own.
- **How to propose.** State the suggested version, the bump kind, and a
  one-line rationale drawn from `[Unreleased]`, then ask for confirmation.
  For example: "`[Unreleased]` now carries the grouping-editor rebuild and
  three fixes — suggest releasing **0.6.0** (minor). Cut it?"
- **Bump rules** (pre-1.0 semver):
  - `minor` — any new user-visible feature, model, panel, or behaviour
    change, and any breaking change to the `.asymp` schema or core API
    (call the break out in the proposal).
  - `patch` — only fixes and small polish; nothing new.
  - `major` — reserved for 1.0.
- **Only after explicit confirmation**, dispatch the workflow with
  `gh workflow run cut-release.yml -f bump=<kind>`. Do not push version
  bumps, changelog rolls, or tags by hand — the workflow is the single path,
  so every release gets the same validation.
- **If the dispatch fails**, read the run log: a missing/expired
  `RELEASE_TOKEN` needs the maintainer to rotate the token (see below); a
  red or pending `CI success` means wait or fix `main` first; a push race
  (someone merged mid-run) is fixed by simply re-running the workflow.

## Permanent project compatibility

A project or wizard cache written by any release remains loadable, including
alpha releases. Keep versioned persistence converters and their released-shape
regression fixtures permanently. A stable format at v1.0 does not remove that
obligation.

Remove obsolete runtime compatibility branches once their conversion is owned
by the persistence boundary and every producer emits the canonical form. Runtime
fitting and editors consume canonical parameters; they do not repair stored
legacy names. A format-only move must preserve numerical meaning, unavailable
models, draft forms, and captured execution history without stamping it current.

## Operational notes

- **`RELEASE_TOKEN`** is a fine-grained PAT owned by the repo admin, scoped
  to this repository only, with **Contents: read and write**. It exists
  because `main` is protected (PRs + `CI success` required) and the admin
  role is the ruleset's bypass actor: a push authenticated by this token may
  land the release commit directly, and — unlike the default
  `GITHUB_TOKEN` — its tag push triggers the downstream tag workflows.
  Rotate it at Settings → Developer settings → Fine-grained tokens, then
  update the `RELEASE_TOKEN` Actions secret.
- **Release cadence** is a judgment call, not a schedule: with builds and
  publishing automated, releasing whenever a coherent user-visible unit has
  landed is cheap and keeps the newest download fresh. GitHub imposes no
  limit on the number of releases (limits are per-release: 1,000 assets,
  2 GiB per file — far above the two installers shipped here).
- **Fixing a bad release**: if the guard in `release.yml` fails after a
  manual tag, delete the tag, fix the tree, and re-tag — or just use the
  Cut release workflow, which cannot produce that mismatch.
