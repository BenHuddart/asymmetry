# GPT-6 Luna corpus improvement loop

Continue unattended on `codex/luna6-corpus-improvements` until a fresh full
24-case gate passes on Codex / `gpt-6-luna`. Keep rubrics fixed and retain every
first attempt. General skill guidance and reusable CLI behavior are the allowed
improvements; no run-number, expected-answer or corpus-name dispatch rules.

The baseline expansion is committed as `95edcb9`. Its detailed record is
[agent-cli-skill-luna6-expansion-2026-10-19.json](agent-cli-skill-luna6-expansion-2026-10-19.json):
7/16 scientific passes, 86/100 Musts, 11/16 exact-copy checks and 3/16 combined.
The previous eight-case gate is historical, not a contemporaneous full wave.
The earlier raw `/tmp` runs are no longer present on this host;
their committed score/evidence record remains the baseline. New waves keep
their own fresh reports and transcripts and do not reuse those missing runs.
The high-TF AFM literal pass has an incorrect superconducting interpretation;
that warning must be resolved as well as the literal rubric failures.

## Operating rules

- Commit each round of improvements before its evaluation wave; keep the CLI,
  skill and rubrics frozen while the wave runs. Score the completed reports
  before starting the next round, recording Must/Should evidence, number and
  boundary checks, model/host, commit, skill/rubric hashes, and first attempts.
- Run up to three eval cases concurrently with all BLAS thread counts pinned
  to one. Check available memory and memory PSI before launching; reduce load
  if pressure rises. Never overlap broad tests with live evaluations. Cap
  pytest at two workers. Inspect live processes before resuming work to avoid
  launching duplicates after an interruption.
- Keep corpus copies and raw results in `/tmp`, outside the repository. Obtain
  the original corpus only through `ASYMMETRY_CORPUS_ROOT`.
- The baseline full validation has four GUI sizing failures in unchanged GUI
  code; record these honestly. Do not expand this task into GUI redesign.
- Quiet heartbeat `gpt-6-luna-corpus-improvements` wakes this chat every thirty
  minutes. Inspect this plan and current processes, continue real work, and
  notify only on meaningful scored rounds, completion or required input.
  Disable it once the objective is achieved.

## Round one

Implementation complete; awaiting evaluation:

- Skill checkpoints distinguish integral polarisation recovery, rate
  decoupling, resonances, magnetic broadening, superconducting Gaussian widths,
  calibration samples, temperature endpoints and KT control parameters.
- `trend` prints individual reliable cold/next/warm rate or width values and
  names the KT width/dynamics comparison. Exponential TF damping is directed
  to a Gaussian recipe rather than silently used in a gap law.
- `integral-scan --bound NAME=LO:HI` exposes validated, stored search intervals
  for broad or narrow lines; bound-pinned widths prompt a window/baseline check.
  RF contrast guidance preserves its LF experimental context.
- `survey` distinguishes the field-scan observables before suggesting a law.
- Final-report guidance explicitly forbids character changes after audit.

Validation: focused checks 240 passed in 7.68 s; fast tier 5,890 passed,
10 skipped, one xfailed in 104.67 s. Structural, Ruff and skill validation
passed. Sphinx built with 85 existing warnings. The standard gate, using two
workers and scratch XDG directories, finished with 9,903 passed, 25 skipped,
one xfailed and the same four GUI sizing failures as the baseline in 450.85 s:
`test_resting_columns_fit_a_narrow_inspector_dock`,
`test_narrow_columns_keep_their_character_widths`,
`test_both_columns_fit_without_scroll_at_default_size`, and
`test_default_width_fits_the_t0_line_on_a_fifteen_detector_run`.
No GUI source was changed; the standard gate is not green. Logs are
`/tmp/asymmetry-luna6-round1-{focused,fast,docs,validation}.log`.

Run the thirteen cases with
baseline science or exact-copy failures (including the AFM scope warning):

`plateau-redfield`, `sn-critical-field`, `euo-psi`, `maleic-mu-kinetics`,
`copper-diffusion`, `afm-high-tf-mdu`, `lifeas-psi`, `trsb-re6zr`, `basics`,
`corannulene-alc`, `benzene-rf`, `benzene-repolarisation`, `benzene-alc`.

Round one is committed as `ea920f9`. Its 13-case wave is running in
`/tmp/asymmetry-luna6-improvements/round1` with Codex / `gpt-6-luna`, three
concurrent cases and BLAS threads pinned to one. The first three cases entered
model execution successfully. `wave-metadata.json` records the commit, skill
and rubric hashes; console log `/tmp/asymmetry-luna6-round1-wave.log`.
The supervising exec session is `98755`; inspect host processes and per-case
`cost.json`/transcripts before relaunching after any interruption. No round-one
reports were scored at launch. Keep CLI/skill frozen until the wave completes.

First checkpoint: three reports scored in
[luna6-corpus-round1.json](luna6-corpus-round1.json), with 15/15 Musts.
Plateau and EuO also pass exact-copy; Sn deletes one plot-file entry after
audit, so its exact-copy check fails. EuO repairs its baseline missing
paramagnetic rate: it used pre-t0 background correction and a separate
warm-ZF exponential series, reporting roughly constant relaxation without
inventing a critical divergence. The wave remains active;
these are partial results, not a wave verdict. Scoring records include the
frozen hashes, final-summary hashes, actual command evidence and Should gaps.
Tests/logs use `/tmp/asymmetry-luna6-round1-*`. Once targeted cases pass,
run all 24 from fresh projects and score the entire wave; do not pool latest
passing attempts and call that a fresh gate.
