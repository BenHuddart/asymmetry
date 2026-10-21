# GPT-6 Luna corpus improvement loop

Status: **paused 2026-10-21 05:56 UTC; all-24 objective not achieved.**
Branch `codex/luna6-corpus-improvements`
([#368](https://github.com/BenHuddart/asymmetry/pull/368)). Host Codex CLI /
Linux, model `gpt-6-luna`.

The loop ran from the 16-case expansion baseline at `9f9dcd6` (19 October) to
a hard stop at 06:00 UTC on 21 October. Each round made general CLI, skill and
evaluation-harness repairs against the failures in the last scored wave,
validated and committed them, then ran a new wave. It ran twelve targeted
rounds and nine fresh full 24-case confirmation gates.

## Records

- [`luna6-corpus-waves.json`](luna6-corpus-waves.json): one entry per wave,
  covering commit, concurrency, Must/Should and science/copy/combined/boundary
  totals, and the cases that failed science or exact copying.
- [`luna6-corpus-confirmation9.json`](luna6-corpus-confirmation9.json): the
  full per-case record of the latest fresh gate.
- Every other per-case record, and the unabridged execution log this page
  replaces, are archived in the private corpus repository under
  [`eval-records/luna6-2026-10/`](https://github.com/BenHuddart/wimda-muon-school-corpus/tree/main/eval-records/luna6-2026-10).
  Raw attempts stay on the evaluating host under
  `~/.cache/asymmetry-luna6-overnight-20261020/`, because they carry corpus
  copies.

## Protocol

- Rubrics, the `two_turn_image_review_v2` review protocol and the scoring
  policy stayed fixed throughout. Each case's rubric hash is recorded.
- The CLI, skill, runner and rubrics were frozen from a wave's launch until
  all of its cases finished, including early failures.
- Only first attempts were scored, by hand, against the agent's actual final
  reply. Historical passes are never pooled.
- Number and unit provenance counts only direct CLI and stored outputs. Reads
  of audits, drafts and logs do not count.
- From round 14 onwards, exact copying compares raw bytes, including terminal
  whitespace, with no normalisation and no substitution of a saved draft.
  Earlier gates normalised the final newline, so confirmation 4's drop to
  3/24 copy passes reflects the stricter check, not a regression.
- An AFM magnetic-physics scope check runs beside the literal rubric, because
  the Musts alone let a superconducting misreading of the AFM high-TF case
  pass.
- The corpus is found only through `ASYMMETRY_CORPUS_ROOT`. No rule dispatches
  on a run number, an expected answer or a corpus case name.

## Results

The latest fresh gate is **confirmation 9** at `3558f52`:

| Check | Result |
| --- | --- |
| Musts | 145/147 |
| Science, including quantity/unit provenance | 21/24 |
| Strict exact copy | 23/24 |
| Science and copy combined | 20/24 |
| Boundary | 24/24 |
| Shoulds | 54/104 |
| Independent AFM scope | Pass |

Four cases still fail:

- **EuO (PSI):** the dimensionless `CriticalDivergence` exponent's uncertainty
  is labelled µs⁻¹. The scalar unit checker missed it.
- **Copper diffusion:** the actual coldest-versus-next ZF ν comparison is
  missing.
- **Basics:** railed calibration-logger temperatures (943/~640/~550 K) are
  read as physical thermal confounds.
- **Benzene ALC:** science passes, but the reply was rewritten after
  preparation, so the raw-byte copy check fails.

Combined passes across the confirmations went 22, 14, 15, 2, 11, 14, 18, 19,
20. Confirmation 1 was scored with the older, normalising copy check and
three cases at a time, so it is not comparable with the later twelve-case,
strict-copy gates.

Round 20 (`d440092`) adds explicit dimensionless exponent metadata, unit
checks on values and uncertainties, flagged adjacent-endpoint diagnostics,
extreme-logger context checks and guidance to prepare again after every edit.
Replaying the recorded outputs catches the EuO, Copper and Basics patterns.
**Round 20 has had no fresh agent gate, and a replay is not a gate pass.**

## Findings that carry forward

- Units belong to the quantity, not the line. Check an uncertainty's unit
  against its own parameter, and mark dimensionless law parameters explicitly.
- Interpret a temperature scan's endpoints from the actual adjacent
  measurements, keeping flagged points visible, before reading the whole scan.
- Logger coordinates far from the experiment's context are faulty metadata,
  not physics. Warn on them generically; never dispatch on a material.
- The prepared reply is immutable. Any edit after `audit --prepare-reply`
  must be prepared again.
- Apparent plot/table mismatches in review images should be reproduced from
  stored rows before any regrade. Regenerating three questioned trend PNGs
  gave byte-identical images, so the scorer had misread them.

## Resources

Confirmation 9 ran twelve concurrent cases in 49.5 minutes. At least 20.2 GiB
of memory stayed available and memory PSI stayed at 0, with one wizard worker
by default and BLAS/OpenMP/MKL threads pinned to 1. Keep twelve. This is not a
controlled speed comparison. Never run broad test suites while a wave is live,
and keep raw outputs in a persistent cache rather than `/tmp`, which lost
evidence across a restart.

## Validation at `d440092`

Focused: 350 passed. Fast tier: 5,972 passed, 10 skipped, 1 xfailed. Lint,
structural, packaged-skill and generated-reference checks pass, and the docs
build with 80 existing warnings. The local standard tier has 9,988 passed and
4 failed: the four known GUI sizing tests in `test_fit_parameter_table.py` and
`test_grouping_dialog.py`. CI is green.
