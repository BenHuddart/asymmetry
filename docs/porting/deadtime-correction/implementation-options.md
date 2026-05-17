# Deadtime Correction Implementation Options

## Decision Framing

The study shows that Asymmetry already has the shared file-based correction
formula. The implementation decision is therefore about scope, not about the
mathematics.

User decision recorded after the study pass:

- WiMDA is the chosen reference program for the implementation pass.
- The deadtime controls should be adapted into Asymmetry's grouping window.
- `estimate` should follow the same semantics as alpha estimate in Asymmetry:
  compute from the selected reference dataset, apply to all selected datasets,
  and let future loaded datasets inherit the resulting grouping payload.

## Option 1: Keep current Asymmetry behavior and only document parity

Description:

- Treat the existing file-based deadtime correction as the completed port.
- Do not add WiMDA manual or estimated modes.
- Do not add Mantid-style external deadtime table selection.

Pros:

- No behavior change.
- Matches the current Asymmetry docs and tests.
- Lowest risk.

Cons:

- Does not satisfy broader parity with WiMDA or Mantid workflows.
- Leaves deadtime provenance limited to file metadata.
- Does not help users who need imported or calibrated deadtime values.

When to choose it:

- If the target is only file-based ISIS/NeXus parity.

## Option 2: Extend Asymmetry to a normalized source model

Description:

- Keep the current core formula implementation.
- Introduce a normalized deadtime source contract at the loader/grouping
  boundary.
- Support multiple sources through adapters while reusing the existing core
  transform:
  - file
  - imported calibration file
  - external table
  - estimated, if later approved

Pros:

- Preserves the working core math.
- Aligns with Mantid's separation between source selection and correction.
- Makes WiMDA-style imported/manual modes possible without rewriting the
  transform.
- Gives project persistence and GUI code a stable provenance model.

Cons:

- Requires schema and GUI changes, not just transform changes.
- Needs clear product decisions around which source modes are user-facing.

When to choose it:

- If the goal is practical parity with the reference programs while keeping the
  current Asymmetry architecture.

## Option 3: Adapt WiMDA-style correction modes into Asymmetry's grouping workflow

Description:

- Model the grouping UI after WiMDA's `Deadtime Correction` group.
- Add `Off`, `Man`, `Auto Load`, and `Auto Estimate` controls directly to the
  grouping dialog.
- Store the selected mode and resolved per-detector payload in grouping state,
  then reuse the existing core correction transform.
- Make `estimate` compute from the reference run only, then apply the resolved
  payload to all selected targets and future loaded datasets through the same
  inheritance rule used by grouping/alpha settings today.

Pros:

- Most familiar to WiMDA users.
- Matches the user's chosen reference behavior.
- Makes manual, auto-load, and estimated workflows explicit in the UI.
- Reuses Asymmetry's existing selected-target apply path and future-load
  grouping inheritance path.

Cons:

- Needs careful separation so the UI chooses the source mode but the core still
  owns the correction math.
- Requires new grouping/project provenance fields.
- WiMDA's more specialized model-panel variants may not map cleanly into the
  initial Asymmetry implementation.

When to choose it:

- This is the chosen direction.

## Recommended Default

Recommended for the implementation pass: Option 3, implemented with Option 2's
core boundary discipline.

Reasoning:

- The user has selected WiMDA as the authoritative workflow.
- The existing Asymmetry core transform should still remain the owner of the
  correction formula.
- The best fit is therefore a WiMDA-style grouping UI that resolves into a
  normalized per-detector payload and provenance fields.
- This preserves scriptability and persistence while matching the chosen UX.

## Proposed First Implementation Slice

For the chosen WiMDA-first direction, the smallest implementation slice is:

1. Add a deadtime mode field to grouping payloads and project persistence.
2. Extend the grouping dialog with WiMDA-style mode controls:
  - `Off`
  - `Man`
  - `Auto Load`
  - `Auto Estimate`
3. Normalize each chosen mode into a resolved per-detector payload plus
  provenance fields.
4. Make `Auto Estimate` use the selected reference dataset only.
5. Apply that resolved payload to all selected datasets through the existing
  shared-grouping apply path.
6. Preserve the same future-load inheritance behavior that existing grouping
  settings already use.

Recommended first added source beyond current file metadata:

- estimated deadtime from the reference run

Reasoning:

- This is the user-selected gap that most directly changes behavior.
- The inheritance rule is already established elsewhere in Asymmetry.
- It forces the provenance contract to be designed correctly before optional
  manual/load polish expands further.

## Explicitly Deferred Unless Requested

- Mantid-style live ADS workspace selection
- WiMDA higher-order model-panel variants beyond the initial manual/load/
  estimate workflow
- full UI parity with every WiMDA instrument-specific branch