# Contributing to Asymmetry

Read [AGENTS.md](AGENTS.md) before changing code or documentation. It defines the
scientific, compatibility, layer, and maintenance rules. The
[contributor guide](docs/contributing.rst) covers instrument authoring and review;
[architecture](docs/ARCHITECTURE.md) identifies current owners and data flow.

## Development and checks

Create a `feature/*` or `fix/*` branch from `main`. Install the development
requirements using the repository constraints:

```bash
pip install -c constraints.txt -e ".[all,dev]"
python tools/harness.py structural
python tools/harness.py lint
```

Run focused tests through the harness, then standard validation once the candidate
is stable. Test placement and optional dependencies are described in
[tests/README.md](tests/README.md) and [HARNESS.md](docs/HARNESS.md).

```bash
python tools/harness.py test -- tests/core/test_instrument_profiles.py
python tools/harness.py validate
python tools/harness.py docs --mode source
```

The harness uses the checkout's `.venv` when present. GUI tests use the shared
`qapp` fixture and isolated settings; follow the existing owning-layer patterns.
Do not substitute a bare pytest invocation for the harness. Enable the repository
pre-commit hook with `git config core.hooksPath .githooks` to check formatting and
lint on commits.

## Documentation and scientific evidence

The logo master is `src/asymmetry/resources/logo.svg`. Regenerate the application
PNGs and documentation artwork with `python tools/branding.py`, or verify them
with `python tools/branding.py --check`. Use `--platform windows` or
`--platform macos` to also build the installer ICO or ICNS and macOS iconset.
The SVG stores the small-size optical adjustments alongside its paths; the
generator applies them through 32 pixels. Edit the master rather than generated
artwork. Qt SVG and Pillow are included in the contributor dependencies.

Update the owning documentation, tests, and changelog with a behaviour change.
Use [docs/README.md](docs/README.md) for source/publishing gates and screenshots,
[STYLE.md](docs/STYLE.md) for prose, and
[GUI_GUIDELINES.md](docs/GUI_GUIDELINES.md) for widgets, workers, and plotting.
Instrument definitions are frozen data with scoped evidence and generated review
pages; the [worked authoring example](docs/contributing.rst#work-through-the-gps-declaration)
uses the bundled profile rather than a second authoritative definition.

Cite primary scientific sources and distinguish analytic tests, recorded native
outputs, and fresh comparisons. Preserve released projects through tested,
versioned converters. Unavailable inputs and historical results stay inspectable;
a successful serialization roundtrip does not establish numerical correctness.

## Issues and pull requests

Use the [issue forms](https://github.com/BenHuddart/asymmetry/issues/new/choose)
for work larger than a small fix. **Implementation** records the problem,
evidence, scope, decisions, compatibility, acceptance criteria, validation, and
documentation. Its **Blocked by** list is authoritative: prerequisites must be
closed as completed before dependent work starts. Plans and investigations belong
in issues, not new repository planning documents.

Use the [issue status and monitoring convention](docs/contributing.rst#issue-status-and-monitoring)
to separate actionable work, monitored uncertainties and deferred proposals.

**Instrument review** records the exact profile revision, checked sections,
evidence, and credit consent. A reviewer can choose **Do not name me**; the issue
and GitHub account remain public, and a maintainer can file on their behalf.
Maintainer acceptance of a diagram is separate from instrument-scientist review.

For CLI/API issue filing, use the form's field labels as third-level headings in
order. Before pushing, inspect the complete diff and run the required gates.
Use [.github/PULL_REQUEST_TEMPLATE.md](.github/PULL_REQUEST_TEMPLATE.md), an
imperative title, and the issue link. A first PR in a multi-PR issue uses
`Refs #n`; only the final PR satisfying the whole issue uses `Closes #n`.
