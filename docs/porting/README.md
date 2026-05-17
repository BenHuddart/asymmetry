# Porting Workflow

This directory defines the required study-first workflow for porting features
into Asymmetry from reference programs such as WiMDA, musrfit, and Mantid.

The rule is simple: do not start implementing a ported feature until the study
artifacts exist in stable paths under `docs/porting/<feature-slug>/`.

## Two-Pass Workflow

### 1. Study pass

Compare the feature across the relevant reference programs without changing
Asymmetry behavior yet.

Record, at minimum:

- entry points
- data flow
- dependencies
- edge cases
- test coverage
- implementation differences
- candidate port seams
- comparison data and caveats

Required files for every feature study:

- `docs/porting/<feature-slug>/README.md`
- `docs/porting/<feature-slug>/comparison.md`
- `docs/porting/<feature-slug>/implementation-options.md`
- `docs/porting/<feature-slug>/test-data.md`
- `docs/porting/<feature-slug>/verification-plan.md`
- `docs/porting/index.json`

Optional, lightweight study scaffolding when useful:

- `tests/porting/<feature-slug>/`
- `src/porting/<feature-slug>/`

Allowed study-pass content:

- documentation
- manifests and indexes
- golden-file placeholders
- comparison harness scaffolding
- minimal interfaces or adapters that define later implementation seams

Do not do a full feature port in the study pass.

### 2. Implementation pass

Implement only after the study exists and the implementation approach is chosen.

During implementation:

- use the study docs as the source of truth
- add verification tests against the original reference program behavior
- update the study with the final decision
- record comparison results and verification outcomes

## Naming And Stability

- Use kebab-case slugs, for example `background-correction`.
- Keep study file names stable so agents can navigate them mechanically.
- Prefer machine-readable paths and explicit JSON fields over free-form notes.
- Record uncertainty explicitly rather than hiding it in prose.

## Porting Index Schema

`docs/porting/index.json` is the machine-readable catalog of all studies.

Top-level shape:

```json
{
  "version": 1,
  "studies": []
}
```

Each study entry must include:

```json
{
  "slug": "feature-slug",
  "feature_name": "Feature Name",
  "status": "study",
  "path": "docs/porting/feature-slug",
  "references": ["WiMDA", "musrfit", "Mantid"],
  "docs": {
    "readme": "docs/porting/feature-slug/README.md",
    "comparison": "docs/porting/feature-slug/comparison.md",
    "implementation_options": "docs/porting/feature-slug/implementation-options.md",
    "test_data": "docs/porting/feature-slug/test-data.md",
    "verification_plan": "docs/porting/feature-slug/verification-plan.md"
  },
  "tests_path": null,
  "src_path": null,
  "updated": "YYYY-MM-DD"
}
```

`tests_path`, `src_path`, and `updated` are recommended but optional for the
structural harness. Use them when the study creates additional scaffolding.

## Authoring Checklist

Before handing off a study pass, confirm that:

1. the feature has a slugged folder under `docs/porting/`
2. all five required markdown files exist
3. `docs/porting/index.json` has a matching study entry
4. implementation differences and unresolved questions are recorded explicitly
5. any optional scaffolding in `tests/porting/` or `src/porting/` is lightweight

`python tools/harness.py structural` enforces the required repository layout.