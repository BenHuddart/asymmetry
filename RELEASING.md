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
   `main` HEAD. It then runs `tools/release_prep.py`, which bumps
   `pyproject.toml`, promotes `[Unreleased]` in `CHANGELOG.md` to
   `## [X.Y.Z] - <date>`, and prints the promoted section into the run
   summary for a quick eyeball. Finally it commits `release: X.Y.Z` to
   `main` and pushes the `vX.Y.Z` tag atomically.
4. The tag push triggers the existing pipelines untouched: `release.yml`
   builds and smoke-tests the macOS DMG and Windows installer and publishes
   the GitHub release; `docs-pages.yml` deploys the documentation site.
   `release.yml` starts with a guard that fails fast if the tag does not
   match `pyproject.toml` or the changelog was not rolled.

There is no release PR and no manual tagging. Version numbers live in exactly
one place (`pyproject.toml`); `asymmetry.__version__` reads it via package
metadata, and the release workflows derive theirs from the tag, with the
guard keeping the two honest.

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

## Delete at v1.0

While the project is in alpha, a saved project / wizard cache written under an
older parameter-naming scheme is migrated on load. Those migrations are
**alpha-only**: at v1.0 the file format stops moving and each one is deleted
whole — module, call sites and tests — rather than carried forward. Nothing
outside these lists depends on them.

**Product amplitudes** (one scale per product, from the expression tree):

- Module: `src/asymmetry/core/fitting/legacy_product_amplitudes.py`.
- Test file: `tests/core/test_legacy_product_amplitudes.py`.
- Call sites:
  - `src/asymmetry/core/representation/base.py` — `FitSlot.from_dict`
    (`fold_legacy_product_amplitude_entries`).
  - `src/asymmetry/core/fitting/fit_wizard.py` —
    `_migrate_fit_result_fractions` (`fold_legacy_product_amplitude_set`).
  - `src/asymmetry/core/fitting/global_fit_wizard.py` —
    `_deserialize_global_candidate_assessment`
    (`fold_legacy_product_amplitude_set`) and
    `_migrate_global_param_name_tuple` (`fold_legacy_product_amplitude_names`).
  - `src/asymmetry/gui/panels/fit/single_tab.py` — `restore_state`
    (`fold_legacy_product_amplitude_state`).
  - `src/asymmetry/gui/panels/fit/global_tab.py` — `restore_state`
    (`fold_legacy_product_amplitude_state`).

**Fraction groups** (positional `fraction_<k>` → the n-1 `f_<Component>`
scheme):

- Functions in `src/asymmetry/core/fitting/composite.py`:
  `_legacy_fraction_numbering`, `_legacy_fraction_rename_map`, `_coerce_float`,
  `has_legacy_fraction_values`, `migrate_legacy_fraction_values`,
  `migrate_legacy_fraction_parameter_entries`,
  `migrate_legacy_fraction_parameter_set`, `migrate_legacy_fraction_state`.
- Tests: the `test_migrate_legacy_fraction_*` tests in
  `tests/core/test_composite_model.py`, and
  `test_fit_slot_migrates_legacy_fraction_parameters` /
  `test_fit_slot_from_dict_survives_malformed_legacy_fraction_value` in
  `tests/core/test_representation_model.py`.
- Call sites: the same five as above —
  `representation/base.py::FitSlot.from_dict`,
  `fit_wizard.py::_migrate_fit_result_fractions`,
  `global_fit_wizard.py::_deserialize_global_candidate_assessment` and
  `_migrate_global_param_name_tuple`, and `restore_state` in both
  `gui/panels/fit/single_tab.py` and `gui/panels/fit/global_tab.py`.

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
