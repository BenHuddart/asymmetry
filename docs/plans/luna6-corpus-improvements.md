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

Three round-two reports are now scored: 19/24 Musts, one scientific/combined
pass, two exact-copy passes and three boundary passes. LiFeAs repairs the
Gaussian-width requirement (8/8 Musts), with conditional gap-model fits and
separate campaigns. Its report quotes earlier real fits before a later t0
revision while describing final reduction settings; retain this provenance
limitation. Inspection confirms a further general plotting problem: nuisance
flags remove the warm sigma points from axis scaling, leaving them clipped
below the frame. The agent reads that clipping as poor model agreement.
Parameter-specific reliability must also govern plotting and its limits.
TRSB improves to 6/7 Musts: both KT forms recover the cold width increase and
the control rate lacks an onset, but the report stops at additional static
broadening rather than identifying candidate spontaneous fields/TRSB at Tc.
Its exact-copy check fails because the reply adds "sample temperatures" after
"logged" in one table row. Both cases read only the staged skill with frozen
hash `5a03b792caf69290af6284bf80974e8ac4d35b3f123c670a466b578adb936af3`;
round-two isolation is working so far. All nine first attempts remain frozen;
benzene ALC, copper and maleic acid are running, with three more queued.
Host process inspection confirms one wave and three model subprocesses;
about 22 GiB is available with no swap/PSI pressure. Do not relaunch live runs.

Benzene ALC is also scored: 5/7 Musts, exact-copy and boundary pass. All
seventeen solution centres are now reported after width-bound expansion,
repairing the previous withholding of every centre. However sample identities
(aqueous, hexane, 20mM aqueous) disappear, and the report applies the
cross-sample field-reference warning to valid within-sample temperature trends.
The upper ordinary-scan ring line is fitted at 29538.289717 G, non-bound and
consistent with the broad fit, but omitted in favour of discussing a failed
high-field differential fit. Keep ordinary and differential evidence distinct;
large chi2 requires a mismatch caveat and stability check rather than erasing
all quantitative results. One differential fit's quoted chi2 belongs to an
earlier variant (3.076 versus final LF-only 4.410); a general provenance check
must bind each reported estimate to its own calibration/model/run subset.
Four reports are scored so far: 24/31 Musts, one scientific/combined pass,
three exact-copy passes, four boundary passes. Corannulene has started in the
freed slot; copper and maleic acid remain live. No source changes yet.

Copper is scored and now passes all six Musts, exact-copy and boundary checks.
It uses the full retained TF window and explicitly reports Gaussian cold versus
exponential warm envelopes, repairing the round-one omission. Its EMU ZF nu
cold-end upturn remains clear. Literal pass does not imply complete analysis:
ARGUS TF was mechanically fitted with the ZF KT recipe then abandoned, and the
LF scans remain weakly modelled. General model selection must match geometry
for every scan, rather than reusing one era's ZF recipe indiscriminately.
Five of nine reports scored: 30/37 Musts, two scientific/combined passes,
four exact-copy passes, five boundary passes. Three live cases are maleic acid,
corannulene and Sn; AFM is still queued. Host has 22 GiB available, zero swap
and zero memory PSI. Source remains frozen at round-two implementation.

Maleic acid is scored: 4/5 Musts, boundary pass, exact-copy fail. The final
report again withdraws the converged concentration slope as a rate constant
rather than retaining an apparent fitted coefficient with units and caveats.
The new CLI advice was printed explicitly, so merely repeating it in another
NOTE is unlikely to solve this. Consider making the result's quantity and units
part of the structured/primary fit output and reconciling the final report's
conclusions with its own retained estimates. Geometry is now appropriate:
no physically mismatched integral recovery fit was attempted. The copy failure
removes "in 78294" from one table cell and a final period from the last plot
entry after audit. Nuisance-bound flags also clip warm Mu-rate points despite
only the diamagnetic damping being pinned, reinforcing the shared reliability
fix. Six reports scored: 34/42 Musts, two scientific/combined passes,
four exact-copy passes and six boundary passes. Corannulene, Sn and AFM are
live; about 21 GiB available, no swap or memory pressure. Keep source frozen.

Corannulene and Sn are scored, leaving only AFM live. Sn retains all five
Musts and now passes exact-copy. Corannulene regresses to 5/10 Musts: hot
local centres remain good, but cold local Linear-background fits pin their
widths and no broader-bound/curved-baseline investigation follows. The cold
centre, comparative width/field and molecular-reorientation interpretation are
therefore absent. Dense ALC scans also remain labelled at nominal temperature
without an explicit logged offset, and the non-polynomial whole-background
reason is missing. Bound-pinned local fits should trigger a general sequence
of bounds, window and baseline checks before concluding no stable centre.
Eight reports scored: 44/57 Musts, three scientific/combined passes, six
exact-copy passes and eight boundary passes. One host wave and one AFM model
process remain; about 20 GiB available with no swap or memory pressure.
No CLI/skill changes until AFM completes and is scored.

Round two is complete (supervising session 19319 exited 0; host inspection
shows no remaining eval/model processes). Final result: 49/62 Musts,
4/9 scientific passes, 7/9 exact-copy passes, 4/9 combined and 9/9 boundary.
Same nine cases in round one: 52/62 Musts, 2/9 scientific, 8/9 copy and 1/9
combined. Scientific case count improved but Must coverage regressed, mainly
corannulene and Basics; this is not an all-pass gate. AFM retains all five
Musts and the corrected magnetic scope without a superconducting-gap fit.
All first attempts, host/model and frozen hashes are in the completed score
record. The CLI/skill freeze is now lifted for round three.

## Round three

Implement general repairs before launching the next Luna wave:
parameter-specific pinned/large-error diagnostics for summaries and plots;
uncertainty-aware endpoint differences; primary fitted quantity/units for
concentration slopes; a report reconciliation step preserving qualified
estimates, sample identities, calibration/model versions and final audited
text. Investigate broader local bounds/curved baselines for broad ALC dips
without corpus-specific branches. Keep the rubrics and two-turn protocol fixed.
Run focused and fast checks with at most two workers on this now-quiet host,
then the required standard gate (known four GUI sizing failures remain).
Commit all round-three changes before starting its evaluation wave.

Round-three implementation is in the working tree (not committed yet).
The shared per-parameter quality reader retains precise measurements when a
nuisance term is pinned/uncertain; summaries, law diagnostics and plot framing
use it. Trend rows now retain bound names under workdir schema 5, deliberately
requiring old caches/series to be regenerated. Cold endpoint comparisons print
the difference, combined error and consistency verdict. Linear rate fits store
supplied-axis units and prominently report a determined apparent coefficient.
Survey prints the unrecorded weak-TF alternative for a zero-field header with
a line, and local broad-dip guidance compares curved baselines as well as bounds.
The skill's blanket "flagged runs are never results" instruction was a direct
contradiction and is replaced by per-parameter interpretation. Report guidance
now reconciles versions, sample identity, qualified estimates and final copying.

Focused checks: 339 passed in 13.74 s (before the final move of the coefficient
label into the converged-result branch; rerun that focused check before gate).
Structural passed. Lint initially requested formatting for three files; fixed,
full lint passed. Skill validation passed. Docs built with the same 85 warnings;
a docs prose update followed. Fast tier is running in exec session 58290 with
exactly two workers, log /tmp/asymmetry-luna6-round3-fast.log. A prior fast
invocation was rejected by argparse because -n needed a -- separator; no tests
ran in that rejected invocation. No eval wave is live. Finish fast, then standard
validation with two workers/scratch XDG paths, record the known GUI failures
honestly, commit and launch a fresh targeted Luna wave. Inspect actual state
before resuming; do not duplicate tests or evaluations.

Fast tier completed: 5,897 passed, 10 skipped, one xfailed in 104.99 s,
with two workers. The moved coefficient check passed separately; docs rebuilt
with 85 warnings and lint is clean. Standard validation is now running in
exec session 41829, log /tmp/asymmetry-luna6-round3-validation.log, two workers,
BLAS pinned to one and scratch /tmp/asymmetry-luna6-round3-xdg directories.
Do not launch another test pool or any eval until it finishes. Next wave is
prepared but NOT launched: /tmp/asymmetry-luna6-round3-wave.sh, six cases
(basics, TRSB, corannulene, maleic acid, benzene ALC and LiFeAs plotting
regression). It requires committed source and a fresh round3 output directory.
Record gate results and commit the whole improvement round before running it.

Round-three validation completed: 339 focused passes in 13.74 s, the final
coefficient-result branch check passed separately (1.34 s), and fast tier
5,897 passed / 10 skipped / one xfailed in 104.99 s. Structural, full Ruff
and skill validation passed; docs rebuilt with the same 85 warnings.
Standard gate finished with 9,910 passed, 25 skipped, one xfailed and exactly
the four known GUI sizing failures in 453.53 s. No new failures appeared;
validation is not green. No GUI source was changed. Logs are
/tmp/asymmetry-luna6-round3-{focused,coefficient,fast,docs,validation}.log.
Scratch /tmp currently has about 12 GiB free; memory has no PSI/swap pressure.
Commit the source now, then start the prepared six-case wave with three
concurrent Luna cases and one BLAS thread. Source/skill must remain frozen
until that wave finishes. Keep all first attempts and score each final reply.

Round three is committed as `d70581e` and its six-case wave is now LIVE:
`/tmp/asymmetry-luna6-improvements/round3`, supervising exec session `90769`,
console `/tmp/asymmetry-luna6-round3-wave.log`. Basics, TRSB and corannulene
started successfully and read the staged project skill. There is exactly one
wave process and three model processes; no tests are running. Prelaunch memory
was 21 GiB available with no swap/PSI pressure. The frozen skill hash is
`584f7f167f99f5acf97434e2bf7c224226f3bd7adf3a1faf5171b549196de134`.
Runner, rubrics, two-turn review protocol and host-skill isolation are unchanged;
`wave-metadata.json` records all hashes and the committed source. All six are
fresh first attempts, not retries of earlier output directories. Keep CLI and
skill frozen until all finish; score completed final replies while others run.
Next: score in docs/plans/luna6-corpus-round3.json, investigate failures, make
and commit the next general improvement round before another wave. A fresh
all-24 gate still follows only after targeted failures pass; do not pool passes.

First round-three report scored: corannulene improves from 5/10 to 8/10 Musts,
with exact-copy and boundary pass. Local Cubic backgrounds recover the cold
15401.6 G centre/1137.8 G width, and the report now gives logged ALC temperatures
and conditional thermal-motion averaging of anisotropic hyperfine interactions.
Remaining gaps: no explicit absence of a narrow cold 7 kG line (it instead
emphasizes an unconstrained broad trough), and no explicit explanation that one
polynomial cannot describe the whole-scan background. General next-round work
should distinguish broad baseline curvature/undetermined width from a measured
narrow resonance, and make local-versus-whole-background limitations visible.
Hot fits quote real but unscaled errors despite printed scaled alternatives;
recorded as a limitation. Source remains frozen; Basics, TRSB and maleic acid
are live, with benzene ALC and LiFeAs pending. One wave/three model processes,
20 GiB available, no swap or PSI pressure. Score completed final replies only.

Basics and TRSB are scored. TRSB now passes all seven Musts, exact-copy and
boundary checks: it explicitly identifies candidate TRSB from the small ZF
width onset, stable control rate and LF decoupling, retaining model/sharing
sensitivity rather than claiming proof. Basics improves from 5/9 to 8/9 Musts:
MUSR unrecorded weak-TF context/sensor logging, usable Mu response amplitudes,
alpha blocks and Ag t0 test are all repaired. The foil CLI now correctly prints
saturation over 13–15 beginning at 13, but the final report omits the numeric
plateau coordinate despite tabulating those rows; this remains a report
completeness failure. Its final reply duplicates the sentence excluding run
18894, so exact-copy fails. All three completed reports obey the boundary.
Partial totals: 23/26 Musts, 1/3 scientific/combined passes, 2/3 copy and
3/3 boundary. Maleic acid, benzene ALC and LiFeAs are live. About 19 GiB
available, zero swap and PSI; do not change frozen CLI/skill until they finish.

Maleic acid and benzene ALC are scored and both now pass every scientific
Must (5/5 and 7/7). Maleic retains the apparent concentration coefficient with
its proper relative units and limitations, rather than withdrawing it; it
also computes slopes from further composition groups without inventing an
Arrhenius activation energy. Benzene retains both ordinary ring-proton fitted
centres separately from the failed differential control and preserves the
three solution identities/within-sample field trends. Maleic exact-copy passes;
benzene fails only by moving one closing backtick in a scan label after audit.
All five completed reports satisfy the tool boundary. Partial totals: 35/38
Musts, 3/5 scientific passes, 3/5 exact-copy, 2/5 combined and 5/5 boundary.
LiFeAs is the only live case; no source changes until it finishes. Memory
20 GiB available, no swap/PSI pressure. Preserve scientific pass versus
exact-copy failure as separate dimensions; never substitute the audited file
for the actual final reply to force a copy pass.

Round three is complete; session 90769 exited 0 and host inspection shows no
remaining eval/model/test processes. LiFeAs retains all eight Musts plus
exact-copy/boundary, with a measured Gaussian-width curve whose warm level
begins at logged 16.8973 K; it still declines an SC gap comparison (Should).
Final six-case score: 43/46 Musts, 4/6 scientific passes, 4/6 exact-copy,
3/6 combined and 6/6 boundary. Same six round-two cases: 33/46 Musts,
1/6 scientific, 4/6 exact-copy and 1/6 combined. TRSB, maleic and benzene ALC
science gaps are repaired. Remaining scientific gaps are report completeness
in Basics (saturation coordinate) and corannulene (narrow-line comparison and
whole-background explanation); Basics and benzene also fail literal copying.

## Round four

The CLI's measured outputs now contain the missing main estimates. Make a
focused, general skill/documentation improvement: preserve numeric coordinates
for reported plateaus/onsets in prose; distinguish no resolved narrow line
from unconstrained broad curvature; explain local-baseline limitations; keep
long reports concise enough to copy without duplicating or reformatting text.
Reconcile remaining blanket flag wording in the summary template with the new
per-parameter contract. This is documentation-only, so validate skill and
Sphinx docs rather than rerunning the already completed standard/core gates
(AGENTS.md explicitly exempts doc/comment-only follow-up edits). Commit before
a fresh targeted wave covering Basics, corannulene and benzene exact-copy;
then fresh all24 only if those pass. Keep rubrics/protocol unchanged.

Round-four changes are skill/owning-documentation only. The workflow contract
and summary template now retain numeric plateau/onset coordinates explicitly,
compare resolved narrow components separately from unconstrained broad troughs,
and explain whole-scan versus local baseline suitability. Long exploratory
tables stay in the CLI products; decisive values and all provenance/windows
remain in the report. Final copying forbids Markdown cleanup or merging drafts
and repeated sentences. Remaining summary-template blanket flag wording is
reconciled with per-parameter reliability. No CLI, runner or rubric changed.
Skill validation passed; Sphinx built with the same 85 warnings. Round-three
core/standard validation remains the applicable gate under AGENTS.md's
explicit documentation-only follow-up exception. Commit before fresh three-case
wave (Basics, corannulene, benzene ALC); score all before the next change/gate.

Round four is committed as `2741b45`; its three-case Luna wave is LIVE under
`/tmp/asymmetry-luna6-improvements/round4`, supervising exec session `64476`,
console `/tmp/asymmetry-luna6-round4-wave.log`. All three cases (Basics,
corannulene, benzene ALC) started and read the staged project skill. Source and
skill are frozen until all finish; one wave/three model processes, no tests.
Prelaunch memory was 20 GiB available, no swap/PSI; /tmp had 10 GiB free.
Frozen skill hash `95486cb2e648256bfeb8f410c4146e7fa4a3c981d47dd35e834639c12c87ecf8`;
runner/protocol/rubrics unchanged and hashes recorded in wave-metadata.json.
Score actual final replies in docs/plans/luna6-corpus-round4.json. Helper
/tmp/score-luna6-round4.py records manual decisions; it is not a judge.
If these three pass all scientific, exact-copy and boundary checks, launch a
fresh full 24-case confirmation wave without pooling earlier results. Include
the AFM magnetic-scope check. Otherwise make general improvements, commit and
evaluate again. No full gate has run yet and the objective is not complete.

First two round-four reports scored: Basics now passes 9/9 Musts, exact-copy
and boundary, explicitly retaining the 13–15 saturation range/from-13 start.
Benzene exact-copy also passes, but science regresses to 6/7 Musts because the
second solid line is missed. Its local window 21450–21750 G puts the real broad
feature near an edge; the fit locks a spurious narrow B0 at 21743.269249 G
(at bound). It correctly withholds that failed identification but stops there.
Current CLI prominently handles width bounds, not bound-centre/window-edge
recovery when chi2 is modest. A general next step should widen/shift nearby
windows to include both flanks and compare local baselines before concluding
there is no line. No corpus-specific window/centre dispatch is allowed.
Partial scores: 15/16 Musts, 1/2 scientific and combined, 2/2 copy/boundary.
Corannulene remains live; keep source frozen until its final reply is scored.
About 19 GiB available, no swap or PSI pressure. Full24 is still not launched.

Round four is complete; supervising session 64476 exited 0 and actual host
inspection shows no remaining eval/model/test processes. Final score: 24/26
Musts, 1/3 scientific passes, 3/3 exact-copy, 1/3 combined and 3/3 boundary.
Corannulene improves to 9/10 Musts, explicitly resolving the cold narrow-line
comparison. It still omits why one polynomial cannot follow the whole-scan
background. Its local model definition incorrectly satisfies the audit's broad
explanation regex; repair that general false positive. Both failures are now
fully recorded in luna6-corpus-round4.json. Source freeze is lifted. Next:
general bound-centre window-recovery advice and a genuine background-limitation
audit check, focused tests/fast/docs, commit, then two targeted fresh cases.
No full24 gate yet. Host has 19 GiB available, zero swap use/PSI and 9 GiB free
in /tmp. Standard validation retains the four documented GUI sizing failures.

## Round five

General CLI and skill repairs: a centre at a search bound now prints an explicit
window-recovery sequence even when chi-squared is modest (inspect full scan,
shift/widen around the feature, include both flanks, compare baselines/seeds).
The existing background-explanation audit now requires a limitation rather
than accepting a sentence that merely defines a local polynomial. Regression
cases cover that false positive and legitimate alternative explanations.
Skill and owning Sphinx page explain both behaviors; no rubric or runner edits.
Focused checks: 296 passed in 7.05 s. An initial edit placed the centre note in
the wrong helper; focused failures caught it, the note was moved into
_poor_fit_note and the complete focused pair passed. Full lint, structural and
skill validation pass; docs built with 85 existing warnings. Fast tier is in
progress with two workers; do not launch an eval until it finishes. No repeat
of the task's completed standard gate for this small CLI-output/audit repair;
its four existing GUI sizing failures remain unresolved and explicitly recorded.

Round-five fast tier completed: 5,902 passed, 10 skipped and one xfailed in
104.21 s with two workers. Focused, fast, lint, structural, skill and docs checks
are complete; no new failures. Standard validation remains the round-three
record with four known GUI sizing failures. Commit this improvement round,
then launch fresh corannulene-alc and benzene-alc cases using gpt-6-luna and
fixed rubrics/two-turn review. Prelaunch memory 18 GiB available, zero swap
use/PSI; /tmp has 8.9 GiB free. No evaluations overlapped the tests.

Round five is committed as `31bff2a`; the two-case Luna wave is LIVE under
`/tmp/asymmetry-luna6-improvements/round5`, exec session `49577`, console
`/tmp/asymmetry-luna6-round5-wave.log`. Host inspection confirms one wave, two
model processes and no tests. Both case runners started successfully. Freeze
CLI/skill until both finish and are scored. Metadata records frozen skill
`0f9857b9c93ef7ed0b352cef0cbfe8d768f98e623883006081b9f2b038061d92`,
runner/rubric hashes and first-attempt source commit. Record scores in
`docs/plans/luna6-corpus-round5.json`; `/tmp/score-luna6-round5.py` is a manual
record helper, not a judge. If both pass science, exact-copy and boundary,
launch a fresh `--set all` 24-case confirmation wave with the same committed
source; score every case, including AFM physics scope. Otherwise improve again
and commit before the next targeted wave. No full confirmation has run yet.

First round-five report scored: corannulene regresses to 4/10 Musts while
exact-copy and boundary pass. It applies MuRepolarisation to all LF scans and
stops after the failed whole-scan fits, never investigating local ALC lines.
The new centre-bound/background guidance is therefore never reached. The
report's numbers verify against actual non-audit CLI outputs, but the main
resonance science is absent. Next general candidate: expose non-monotonic
local-dip investigation independently of the initially requested recovery model;
a failed global model must not be interpreted as absence of another observable.
Benzene remains live in its review turn; keep CLI/skill frozen. One wave/one
model, no tests, 18 GiB available, zero swap use/PSI, 8 GiB free /tmp. Do not
launch a confirmation gate from these results.

Round five completed; host inspection shows no remaining eval/model/test
processes. Benzene passes all 7/7 Musts and boundary, including the
two converged broad solid lines at 19478.2/21474.6 G with stated mismatch.
Final two-case totals: 11/17 Musts, 1/2 scientific, 1/2 exact-copy,
0/2 combined and 2/2 boundary. Benzene deletes `plots/` from the solution plot
directory in its final reply, so literal copying fails. Corannulene's 4/10 regression prevents a full24 launch. Freeze
lifted only after both first attempts scored. Next round should make local dip
investigation available even when the initial whole-scan model is recovery;
the current core only finds next-dip windows for existing LCR models. This is
an observable/model-selection gap, not a reason for corpus-specific rules.

## Round six

Implementing a model-independent shape diagnostic in the pure core. Significant
interior minima (five combined errors of prominence) produce exploratory local
windows. These are measured minima, not fitted resonance fields or widths;
quarter-prominence widths keep windows near trough bottoms. Monotonic recovery
and sub-error excursions produce no candidates. The CLI prints LOCAL DIP
CANDIDATES when no resonance model was selected, even if a recovery fit failed,
and asks for nearby-window/Linear/Cubic baseline investigation before concluding
no line is resolved. Existing resonance and RF-hyperfine models keep their own
line diagnostics. No rubric/runner/protocol changes or corpus-name dispatch.

Focused checks: 320 passed in 8.38 s; structural/lint/skill checks passed.
Fast tier (two workers) and docs are running; no evaluation overlaps them.
An exploratory development probe on stored scan points outside the repository
confirmed that the new diagnostic exposes local structure in both extended
scans while the short monotonic recovery produces none. This is development
evidence, not a new eval score and not data supplied to a Luna agent. The task's
standard gate remains 9910 passes with four known GUI sizing failures; do not
call it green or repeat it as an inner-loop test. Commit round six before fresh
corannulene plus benzene exact-copy regression cases, then full24 only if both
pass all required dimensions. Raw evidence remains outside the repository.

Round-six validation completed: 320 focused passes (8.38 s), 5905 fast passes,
10 skipped/one xfailed (104.13 s), lint/structural/skill pass and docs built with
85 existing warnings. CLI smoke on stored high-field data without a requested
resonance model prints LOCAL DIP CANDIDATES and local-window commands; output
/tmp/asymmetry-luna6-round6-shape-smoke.json. This is development validation,
not an eval score. No broad tests overlapped evals. Standard gate remains the
recorded four existing GUI failures. Host is quiet, 18 GiB available, zero swap
use/PSI and 8 GiB scratch free. Commit then run the two prepared targeted cases.

Round six committed as `a24043a`, wave LIVE in
`/tmp/asymmetry-luna6-improvements/round6`, exec session `63872`, console
`/tmp/asymmetry-luna6-round6-wave.log`. Two cases: corannulene-alc and
benzene-alc (science/copy regression). Both started successfully; host shows
one wave/two model processes and no tests. Freeze CLI/skill until both finish
and are scored. wave-metadata.json records commit, skill/runner/rubric hashes,
model/host and unchanged two-turn protocol. Scores go in
`docs/plans/luna6-corpus-round6.json`; `/tmp/score-luna6-round6.py` is a manual
record helper. Inspect live processes and costs before any relaunch. If both
pass science/copy/boundary, launch fresh full24 with `--set all` and the same
source; include AFM magnetic scope and never pool historical passes. Otherwise
make general improvements and commit before the next wave. Do not repeat broad
tests while the eval wave runs. Standard validation still has four known GUI
failures; the objective is not complete and no full24 gate has run yet.

First round-six report scored: benzene 6/7 Musts, exact-copy and boundary pass.
Both solid lines and ordinary liquid lines remain fitted, but all seventeen
solution widths remain at default bounds and no representative recovery is
attempted. The report explicitly withholds every solution resonance pending
local-window/baseline work; the solution Must therefore fails. It also extends
cross-campaign field-reference caution to within-sample temperature trends and
misreads one same-feature cross-fit comparison as comparing the two distinct
ordinary lines. These are recorded science limitations, not number inventions;
all numbers verify against actual non-audit CLI output. Corannulene is still
live in review; do not change CLI/skill until it finishes and is scored. Host
has 17 GiB available, zero swap use/PSI, 7.2 GiB scratch free. No full24 launch.

Round six complete; no eval/model/test processes remain. Final 14/17 Musts,
0/2 scientific/combined, 1/2 exact-copy and 2/2 boundary. Corannulene recovers
8/10 Musts: the new shape diagnostic triggers local ALC fits and the whole-
background explanation. But it reuses a narrow hot window for the broad cold
feature, returning centres below the required range and unresolved widths;
thermal anisotropy-averaging headline remains missing. Copy fails by changing
14,828.1 G to 14.83 kG after audit. The actual reply is scored. Number checks
now explicitly exclude draft displays as well as audit output. Frozen wave
fully scored before edits. Next: distinguish fit-search width bounds from
resolved-line flank coverage; offer concrete wider-window/Cubic investigation
for truncated lines and independent windows for different conditions. The
span/4 automatic width ceiling is a geometric resolution criterion currently
used as a hard fitting prior, truncating broad but potentially fitted widths.
Test any change on synthetic broad lines and keep resolved-line criteria fixed.
No full24 launch until targeted scientific/copy/boundary failures pass.

## Round seven

General repair in progress: automatic LCR width searches extend to half the
fitted span, while the unchanged resolved-line gate still requires two widths
on both flanks, significant amplitude and a non-bound converged fit. Search
bounds no longer impose that geometric resolution criterion as a hard prior.
Explicit positive bounds remain supported. The CLI's insufficient-coverage
note prints a tripled-window Cubic check and explains that different conditions
need independent windows when one feature broadens. Skill and owning docs
retain conditional estimates without calling partial coverage a resolved line.

Focused checks: 321 passed in 8.39 s; lint/structural/skill pass. Fast tier
is running with two workers; docs built with 85 existing warnings. Development
probes on stored points (outside repo, not supplied to eval agents) recover a
non-bound broad cold fit in wider windows and a non-bound solution width with
the expanded default. Probes are diagnostic development evidence, not scores;
raw script /tmp/probe-luna6-round7.py. Synthetic tests verify broad-width
recovery and continued rejection as an isolated resolved line when flanks are
missing. No eval is live and no broad tests overlap one. Standard remains the
recorded four pre-existing GUI failures; do not call it green. Finish fast,
commit, then run fresh two-case targeted wave before any full24 confirmation.

Round-seven validation complete: 321 focused passes (8.39 s), 5906 fast passes,
10 skipped/one xfailed (104.49 s), lint/structural/skill pass and docs 85 existing
warnings. Host inspection confirms no eval/model/test processes before launch;
17 GiB available, zero swap use/PSI and 7.2 GiB free /tmp. No broad validation
repeat; standard remains the recorded four known GUI failures. Commit these
changes, then run both targeted cases with fixed rubrics/two-turn protocol.
