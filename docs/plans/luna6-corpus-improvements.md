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
  minutes initially, now every ten minutes to shorten idle gaps. Inspect this
  plan and current processes, continue real work, and
  notify only on meaningful scored rounds, completion or required input.
  Disable it once the objective is achieved.

## Round one

Implementation and evaluation complete:

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

Round one is committed as `ea920f9`. Its 13-case wave completed in
`/tmp/asymmetry-luna6-improvements/round1` with Codex / `gpt-6-luna`, three
concurrent cases and BLAS threads pinned to one. The first three cases entered
model execution successfully. `wave-metadata.json` records the commit, skill
and rubric hashes; console log `/tmp/asymmetry-luna6-round1-wave.log`.
The supervising exec session is `98755`; inspect host processes and per-case
`cost.json`/transcripts before relaunching after any interruption. No round-one
reports were scored at launch. Keep CLI/skill frozen until the wave completes.

Final score: thirteen reports scored in
[luna6-corpus-round1.json](luna6-corpus-round1.json), with 73/83 Musts,
6/13 scientific passes, 12/13 exact-copy passes and 5/13 combined passes.
Sn deletes one plot-file entry after audit, so its exact-copy check fails.
EuO repairs its baseline missing
paramagnetic rate: it used pre-t0 background correction and a separate
warm-ZF exponential series, reporting roughly constant relaxation without
inventing a critical divergence. AFM passes and resolves the earlier physics
scope warning: it explicitly uses magnetic TF scope and reports two lines,
rate changes and limits without a superconducting gap law.
Copper fails the TF envelope-change Must: early-window fits give Gaussian
at the EMU cold point and indistinguishable envelopes at the warm points,
while the report omits that comparison. Its cold/next ZF comparison and
exact-copy reporting now pass. Maleic acid regresses on the converged-slope
Must: it prints the fitted slope but then explicitly withdraws it as a
reaction-rate result. LiFeAs still fails the Gaussian-width Must: the new
sample-class caution led it to decline superconducting-model comparison and
retain exponential damping. Its final reply does pass exact-copy. It also
reads the personal skill path, violating the stated boundary. That path is a
symlink to the packaged source and has the frozen staged hash, so it introduces
no different skill content; the boundary violation is still recorded separately.
Benzene RF and repolarisation now pass: LF RF context and the actual integral
recovery observable are both repaired. Corannulene recovers the cold broad
resonance with a wider local polynomial window, but fails to interpret the
hot sharpening as molecular reorientation. TRSB now compares Delta and nu,
but a coupled-fit reversal leads it to reject a candidate TRSB headline.
Basics repairs foil saturation at 13–15, but still treats the Ag calibration
as ZF with a temperature axis and replaces the Ag TF frequency/phase exercise
with integrated asymmetry. Benzene ALC still fails by withholding both liquid
ring-proton fits and all solution fitted centres. Twenty width overrides were
used; one solution centre is stable across nearby windows with chi2_red 1.642,
but the two-width coverage rule makes the CLI call it range-limited. Distinguish
a qualified fitted centre from a demonstrated isolated resonance and baseline.
Scoring records include the
frozen hashes, final-summary hashes, actual command evidence and Should gaps.
Tests/logs use `/tmp/asymmetry-luna6-round1-*`. Once targeted cases pass,
run all 24 from fresh projects and score the entire wave; do not pool latest
passing attempts and call that a fresh gate.

Next-round candidates (do not edit CLI/skill while this wave runs): require
LF geometry and appropriate sample physics before recommending an integral
MuRepolarisation model; two-field TF/control pairs are not LF recovery scans.
Both copper and maleic acid spent effort on physically mismatched recovery
fits after round-one guidance. Make envelope comparison explicit even when
one shape wins throughout or warm points cannot distinguish shapes; resolve
window/baseline dependence rather than forcing the expected change. Distinguish
a converged empirical slope with uncertainty and model limitations from a
validated intrinsic kinetic constant, preserving the former as a fitted result.
Allow provisional physical-model comparisons with clearly conditional
interpretation when material context is incomplete; distinguish testing a
superconducting width model from asserting that broadening proves a vortex
lattice. Review why host skills remain discoverable despite the runner's
skip-host-skill-discovery option, before a fresh confirmation wave.
All thirteen first attempts have now been scored; round two may proceed.

Additional round-two candidates: choose the observable before satisfying scan
coverage (TF field scans require frequency/phase or amplitude, not the integral
of an oscillation). For known nonmagnetic calibration samples, test an
unrecorded weak TF rather than interpreting a header-ZF line with a powder
magnetic model. When a small KT onset reverses under nuisance sharing, compare
the static KT times exponential baseline and dynamic model with plausible
sharing choices before declaring it absent. ALC width comparisons should
include the conditional molecular-motion interpretation as well as fit values.

Isolation research: official Codex documentation describes per-skill disable
overrides (`skills.config`), whereas the current skip-host option did not
prevent the LiFeAs read. Sources checked:
https://learn.chatgpt.com/docs/config-file/config-reference and
https://learn.chatgpt.com/docs/build-skills. Test any runner isolation change
separately after this frozen wave; do not silently change its protocol.

## Round two

Implementation complete after all round-one results were scored:

- Survey guidance checks TF/LF geometry before suggesting recovery models;
  TF frequency/phase/amplitude observables are named explicitly.
- Envelope summaries retain indeterminate warm points. Conditional Gaussian
  and gap-model comparisons no longer require a firm material identification.
- Stable non-bound line centres may be reported as conditional fit estimates
  after window/background checks. Two-width coverage limits baseline certainty;
  it is no longer described as proof that no dip exists. Poor fits retain their
  mismatch caveats. Resolved-line gates and bound-width rejection are unchanged.
- Molecular ALC comparisons include a conditional reorientation/anisotropy
  interpretation; concentration slopes may be apparent fitted rate coefficients.
- Skill guidance addresses nonmagnetic calibration headers, unrelated sensor
  drift, static/dynamic KT and nuisance-sharing comparisons, and final copying.
- The runner explicitly names the project skill and disables personal copies
  through per-skill config on both turns. The scientific prompt, two-turn image
  review and rubrics are unchanged; record the runner hash with new waves.

Validation so far: 268 focused tests passed in 7.39 s, structural and lint
passed (one import-order issue repaired), skill quick validation passed.
The GPT-6 Luna isolation smoke at `/tmp/asymmetry-luna6-round2-isolation-smoke`
completed exit0 and read the project copy. It is not a scientific score.
Fast tier passed: 5,892 passed, 10 skipped, one xfailed in 102.92 s. Docs built
with 85 existing warnings. Standard validation first stopped at a leftover
import-spacing lint failure before tests; repaired and restarted. The completed
gate has 9,905 passed, 25 skipped, one xfailed and the same four known GUI sizing
failures as round one in 453.56 s, using two workers and scratch XDG paths.
No new failures appeared; the standard gate remains non-green for those GUI
failures. Log: `/tmp/asymmetry-luna6-round2-validation.log`.
Other logs: `/tmp/asymmetry-luna6-round2-{focused,fast,docs}.log`. No corpus
evals overlapped validation; round one had finished exit0 before it began.

Round two is committed as `a5047ae`. Its nine targeted cases are running with
Luna and three concurrent cases: `sn-critical-field`, `maleic-mu-kinetics`,
`copper-diffusion`, `lifeas-psi`, `trsb-re6zr`, `basics`, `corannulene-alc`,
`benzene-alc`, and `afm-high-tf-mdu` (physics-scope regression check).
Result root `/tmp/asymmetry-luna6-improvements/round2` contains first attempts
and `wave-metadata.json` with frozen commit and skill/runner/rubric hashes.
Exec session `19319`, console `/tmp/asymmetry-luna6-round2-wave.log`; inspect
processes and outputs before relaunching. LifeAs, TRSB and Basics started
successfully. First round-two report scored in
[luna6-corpus-round2.json](luna6-corpus-round2.json): Basics fails with 5/9
Musts, while exact-copy and boundary checks pass. Freeze CLI/skill until all
nine finish; score each final report against the unchanged rubrics.

Round-three candidate confirmed from Basics evidence: `_measured` censors
entire trend rows when only a nuisance damping parameter has `bound_pinned`.
(`large_rel_err` alone is already retained with its error bar.) At 13–15 foils
the amplitude errors are small, yet those
rows disappear from the step analysis and it incorrectly reports an
intermediate 9–11-foil plateau. Preserve parameter-specific reliability while
retaining whole-fit exclusions for actual failure/reseed/amplitude pathology.
Also print uncertainty-aware cold-pair comparisons: current endpoint wording
lets the agent call statistically compatible rates different. The Ag
frequency/phase observable is repaired, but MUSR calibration weak-TF context
and irrelevant sensor-temperature treatment remain wrong. The quartz line
amplitude series also failed; inspect frequency seeding before changing it.
