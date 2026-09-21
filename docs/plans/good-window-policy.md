# Good window as a profile policy: t_good offset and last good bin that persist

Status: planned 2026-09-21 on `feat/good-window-policy` (off `main` at
2238c04). Implementation lands as **one PR**, built in three phases by
subagents with a lead review gate after each. Follow-up to
[t0-determination](t0-determination.md), whose "Deferred (Ben, 2026-09-17)"
note is the origin of this work.

## Problem

Verified in code and by a scratch reproduction on 2026-09-21 (line numbers
are anchors; re-check before editing):

1. **The t_good offset has no home.** The grouping window shows one editable
   "t_good Offset" spinbox with no mode. Apply writes the value onto
   `run.grouping` for every run in scope
   (`gui/mainwindow.py:5289-5306, 5743`). `profile_from_payload`
   (`core/project/profiles.py:992`) discards it as a per-run fact, and a run
   that follows a profile persists only the profile's name (schema v17,
   `mainwindow.py:4805-4830`). The edit therefore has nowhere to live across
   a save.
2. **The resolver overwrites it on every resolve.** `resolve_effective_grouping`
   copies the run's `t_good_offset` (`_copy_per_run_facts`) and then
   `_rebase_on_profile_groups` (`profiles.py:1374-1422`) recomputes
   `first_good_bin`, `last_good_bin` and `t_good_offset` from the per-detector
   good-bin tables. Every real loader (PSI, MusrRoot, NeXus) emits those
   tables. Reproduction: user offset 25 on a run with tables resolves to the
   file's 10 in every t0 mode; without tables it resolves to 25. Whether the
   edit survives depends on the loader.
3. **It is lost in-session too, not only on reopen.** Project open
   (`mainwindow.py:17137`), profile reassignment/reattach (`:4640`, `:4733`)
   and reopening the grouping window (the dialog seeds its controls from
   `payload_from_profile_for_preview`, i.e. the resolver — `dialog.py:1621`)
   all show or apply the file value while the plot still uses the edited one.
4. **`last_good_bin` has the same lifecycle** and must be treated together
   with the offset or the docs paragraph at `detector_grouping.rst:487`
   ("per-run facts … unaffected by the t0 mode") stays half-true.

## Settled design (decision log)

Decisions taken by the lead on 2026-09-21 from the recommendation Ben
accepted ("implement these changes").

- **D1 — the good window is a profile policy, not a per-run fact.** "Start N
  bins after t0" is an analysis choice the user wants applied uniformly to a
  series, exactly as Manual t0 is an offset stored on the profile (t0 D3).
  WiMDA's `.mgp` grouping record stores `toff` and `tgoodend` alongside
  `tzero`, and its FileValues checkbox governs t0 and tgood together. A
  per-run persistence fragment for followers was rejected: it would
  reintroduce the divergence the v17 profile system removed.
- **D2 — `GoodWindowPolicy(mode, first_offset_bins, last_offset_bins)`**, a
  dataclass beside `T0Policy`. `mode` is `"from_file"` (default) or
  `"manual"`. Both offsets are **signed bins measured from the run's
  effective common t0** (the t0 the t0 policy resolved to), so a Manual or
  Auto-detect t0 shift carries the window with it and one profile gives
  every run the same window relative to its own t0 (WiMDA's `toff` /
  `tgoodend − max(tzero)` rule). The Last Good Bin spin keeps showing an
  absolute bin for the preview run; the conversion to and from an offset
  uses the preview run's effective t0, exactly as the Manual t0 spin shows an
  absolute bin but stores an offset.
- **D3 — one selector governs both ends.** A "From file / Manual" mode combo
  on the t_good row (mirroring the t0 row's combo) gates both the t_good
  Offset and the Last Good Bin spinboxes. Two independent toggles were
  considered and rejected for this PR: the t0 row pattern is already learned,
  and WiMDA's FileValues is one switch. In From file both spins are read-only
  and show the preview run's file-derived values; in Manual both are editable
  and seeded from the file values the moment the mode is switched.
- **D4 — from_file resolves bit-identically to today.** Resolution stores
  nothing for the default and `_rebase_on_profile_groups` stays the file
  base. A profile dict emits `good_window_policy` only when the mode is not
  `from_file` (the `t0_policy` precedent), so existing projects round-trip
  unchanged and **no schema bump is needed**. A released run's override
  payload keeps its absolute window keys and is untouched.
- **D5 — never inferred.** `profile_from_payload` reads an explicit
  `good_window_policy` dict from the payload when present and otherwise
  returns the default. Nothing reconstructs a policy from a payload's
  absolute window values (the t0 D1 lesson: inference flipped fresh drafts
  into Manual).
- **D6 — Apply derives the window per run from the policy.** The flat dialog
  payload carries the preview run's absolute window (consumers and tests rely
  on those keys) **plus** the `good_window_policy` dict. In
  `_apply_grouping_settings_to_dataset`, a payload that carries the policy
  derives each run's window from *that run's* t0 and file window through the
  same core helper the resolver uses; a payload without it (override edits,
  legacy callers, existing tests) keeps today's absolute-value behaviour.
- **D7 — clamping is the resolver's existing rule.** The window is clamped to
  the aligned group-sum length (`good_window_for_groups`' `n_bins`), and
  `last_good ≥ first_good`. A file's own window that starts before t0 keeps
  its negative offset (header truth), as today.
- **D8 — provenance line under the row (t0 D11 pattern).** A read-only muted
  `ElidedLabel` beneath the t_good row always shows the offset in time
  (`≈ 0.112 µs after t0`, from the preview run's bin width); in Manual mode
  it also shows the file's values (`File: offset 7 · last bin 2048`) so the
  user sees both at once. The existing t0 verdict's good-window checks
  (window at/before detected t0; pulsed window inside the pulse) already read
  the payload's `first_good_bin` and keep working unchanged.
- **D9 — out of scope, recorded as follow-ups:** a "Suggest" affordance that
  fills the offset from the measured pulse edge; making the flat Apply
  payload's absolute `t0_bin` per-run (today the preview run's absolute bin is
  applied to every follower in the same Apply and only the next resolve
  restores offset semantics — pre-existing, observed, not changed here).

## Code map (verified 2026-09-21)

| Concern | Where |
|---|---|
| Policy dataclasses, `to_dict`/`from_dict`, `GroupingProfile` fields | `core/project/profiles.py:150-530, 729-905` |
| `profile_from_payload` (lifts shareable keys, ignores per-run) | `profiles.py:992-1080` |
| Resolver order: copy facts → rebase → t0 policy → deadtime → background → alpha → beta | `profiles.py:1195-1295` |
| `_rebase_on_profile_groups` (file window base) | `profiles.py:1374-1422` |
| `_apply_t0_policy` (shifts `first_good_bin` by the t0 delta; not `last_good_bin`) | `profiles.py:1429-1520` |
| `good_window_for_groups` (single window rule) | `core/transform/t0.py:88-126` |
| Dialog t0 row construction, form rows, mode gating, t0 line | `gui/windows/grouping/dialog.py:700-770, 1005-1025, 2520-2600, 2818-2916` |
| Dialog spins + payload + draft sync | `dialog.py:715-727, 1705-1716, 2917-2924, 3003-3012, 5415-5470` |
| Dialog seed source (resolver) and override drafts | `dialog.py:1600-1665` |
| MainWindow apply per run | `gui/mainwindow.py:5230-5760` (window at `:5289-5306`) |
| MainWindow project fragment / extract | `mainwindow.py:4805-4900` |
| Loaders' window derivation | `io/nexus.py:788-802`, `io/psi.py:1153-1170`, `io/root.py:658-670` |
| Consumers | `core/transform/reduce.py:456-472` (first/last), `deadtime.py:64-95` (offset), `plot_panel.py:8268-8335` (display) |
| Docs | `docs/reference/detector_grouping.rst:441-500`, `docs/reference/project_files.rst`, `docs/reference/data_reduction/t0_search.rst` |
| Screenshot scenario pattern | `docs/screenshots/scenarios/grouping_window_t0_row.py`, registered in `docs/screenshots/capture.py:329` |
| Existing tests to extend | `tests/core/test_grouping_profiles.py` (t0 policy block 560-1040), `tests/gui/test_grouping_dialog.py:1711, 2922-3040`, `tests/gui/test_mainwindow_additional.py:3896, 4458` |

## Core API contract (Phase 1 delivers, Phase 2 consumes)

```python
# core/project/profiles.py
GOOD_WINDOW_POLICY_MODES = ("from_file", "manual")

@dataclass
class GoodWindowPolicy:
    mode: str = "from_file"
    first_offset_bins: int | None = None   # manual only; bins from effective t0
    last_offset_bins: int | None = None    # manual only; bins from effective t0
    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: Any) -> GoodWindowPolicy: ...   # lenient, default from_file

GroupingProfile.good_window_policy: GoodWindowPolicy   # emitted only when mode != from_file

def resolve_good_window(
    policy: GoodWindowPolicy, *, t0_bin: int, file_first_good: int, file_last_good: int, n_bins: int
) -> tuple[int, int]:
    """(first_good, last_good) — from_file returns the file window clamped; manual = t0 + offsets, clamped (D7)."""

def run_file_good_window(run: Run, grouping: dict[str, Any], *, common_t0_bin: int) -> tuple[int, int]:
    """The run's file-derived window for the payload's analysis groups: the
    per-detector tables through good_window_for_groups when the payload carries
    them, else the grouping's own first/last_good_bin."""
```

Resolver: `_apply_good_window_policy(grouping, profile.good_window_policy, run, n_hist)`
runs **immediately after `_apply_t0_policy`** and writes `first_good_bin`,
`last_good_bin`, `t_good_offset` (= first − t0_bin). In `from_file` it writes
nothing.

## Phases

Lead runs `python tools/harness.py validate` once at the end; agents run only
focused files and `--tier fast`. Agents leave the tree uncommitted; the lead
reviews the diff, reruns the focused tests, and commits.

- **P1 — core (Opus).** Policy dataclass + serialization, `GroupingProfile`
  field and dict round trip, `profile_from_payload` explicit read (D5),
  `resolve_good_window` / `run_file_good_window`, `_apply_good_window_policy`
  in the resolver, exports in `core/project/__init__.py`. Tests in
  `tests/core/test_grouping_profiles.py`: dict round trip each mode; default
  omitted from profile dict; from_file bit-identical to today on a tables run
  and on a no-tables run (pins the loader-dependence found above); manual
  offsets resolve onto a staggered run in every t0 mode and ride a manual t0
  shift; clamping; `profile_from_payload` never infers.
- **P2 — GUI (Opus).** Dialog: mode combo on the t_good row gating both spins
  (D3), seeding from the file window on switch, absolute↔offset conversion
  for Last Good Bin via the preview run's effective t0 (D2), preview-run
  switch re-seeds under the policy, provenance line (D8), payload carries
  `good_window_policy`, draft sync sets `self._draft.good_window_policy`
  directly (never inferred). MainWindow: per-run derivation in
  `_apply_grouping_settings_to_dataset` when the payload carries the policy
  (D6). Tests: dialog defaults/gating/payload/preview-switch; MainWindow
  applies different absolute windows to two runs with different t0 under one
  manual policy; **edit the offset on a profile-following run, save the
  project, reopen, and the offset is retained and displayed** (the bug this
  PR exists for).
- **P3 — docs (Sonnet).** `detector_grouping.rst`: replace the "per-run
  facts" paragraph with a "Good window" subsection quoting the UI strings
  verbatim; `project_files.rst` profile field; `t0_search.rst` cross-link;
  `CHANGELOG.md` `[Unreleased]`; a `grouping_window_good_window_row`
  screenshot scenario (crop of the t_good row, its line, and the Last Good
  Bin row) registered in `capture.py` and referenced from
  `detector_grouping.rst`; `docs/PLANS.md` entry pointing here.

## Acceptance

1. A project saved with a Manual good window reopens with the same offsets on
   every follower, and the grouping window shows them.
2. A project saved before this change opens byte-identically resolved.
3. `python tools/harness.py validate`, `gui-smoke` and `docs` are green.
