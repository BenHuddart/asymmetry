# Documentation style guide

This guide captures the writing voice for Asymmetry's user documentation. It is
derived from the maintainer's own writing — the MuFinder software paper
(*Comput. Phys. Commun.* **280**, 108488 (2022)), the PhD thesis (Durham, 2020),
and recent muon-spin spectroscopy papers — so that the docs read in a single,
consistent voice. When in doubt, match the patterns below.

A note on a false signal: the *Physical Review B* papers read American
(*behavior*, *modeling*, *polarized*) because that is the journal's house style.
The thesis and the MuFinder paper — written without that constraint — are
British, and so are these docs.

## Voice and person

- **Lead with motivation, then mechanics.** Say *why* a step or feature matters
  before explaining *how* it works. A feature is introduced by the problem it
  solves, not by its controls.
- **Reference and explanatory prose** is formal but accessible. The program or
  the physics is usually the subject: "Asymmetry estimates α from a
  transverse-field run by balancing the forward and backward groups." Use "we"
  sparingly, for recommendations ("we recommend fixing Δ when the relaxation is
  fast"). Avoid the first-person singular.
- **Tutorials and how-to pages** address the reader directly as "you" and use
  imperatives for actions: "Open the **Grouping** dialog and click
  **Estimate α**." This is the one deliberate departure from the papers'
  third-person register — step-by-step instructions read better as commands.
  Reserve "the user" for describing behaviour in reference text, not for
  instructions.

## Tense

- Present tense for how the program behaves and for established physics: "the
  muon precesses about the local field"; "the panel shows the residuals".
- Past tense only for genuinely completed or historical statements.

## Mechanical conventions

- **British spelling throughout**: analyse, behaviour, centre, colour, organise,
  optimise, normalise, minimise, characterise, recognise, modelling, labelled,
  signalling, polarisation, depolarisation, apodisation, spin-polarised.
- **Oxford comma**, always: "grouping, asymmetry, and rebinning".
- **Em-dash** for a parenthetical aside — like this — used sparingly; prefer
  commas or parentheses for shorter asides.
- **Hyphenate compound modifiers**: zero-field, transverse-field,
  longitudinal-field, time-domain, frequency-domain, count-domain, long-range,
  low-temperature, muon-spin. Do not hyphenate adverb + adjective ("externally
  applied field", "strongly damped relaxation").
- **Units**: a space between the number and the unit — 2.2 μs, 135.5 MHz, 39 K,
  500 mT, 16 ns. Use the μ glyph for the micro prefix.
- **Uncertainties** in parentheses on the trailing digit(s): T_N = 0.23(1) K.
- **Numbers**: spell out below ten in running prose ("two detector groups"); use
  numerals for ten and above, and wherever a quantity carries a unit ("8 bins",
  "16 ns").
- **Headings**: sentence case — "Detector grouping and α calibration", not Title
  Case.

## Mathematics in prose

- Name a symbol verbally as (or before) you give the equation, then interpret it
  immediately afterwards: concept → equation → meaning.
- Italicise variables (the `:math:` role handles this). Use inline `:math:` for
  simple symbols and `.. math::` for displayed or important relations.
- Keep notation consistent with the fit-function reference pages
  (γ_μ, A(t), σ, Δ, λ, ν, B_L, …).
- Only `:math:` / `.. math::` render — never write bare `$…$` (MathJax is not
  configured for dollar delimiters).

## Pedagogy and page structure

- Assume undergraduate physics, but do **not** assume the reader knows μSR
  specifics. Define terms on first use, and where it helps, anchor a concept by
  comparison to a sibling technique (NMR, ESR, Mössbauer).
- Ground an explanation in a concrete worked example with real input and output,
  and compare against an expected or known result where possible.
- Acknowledge limitations candidly: "in most cases…", "provided that…", "note
  that…".
- A feature or reference page generally follows: motivation → what it does → how
  to use it (with a worked example) → options and parameters → limitations and
  cross-references.

## Citations (APS style)

- Journal articles use the compact APS (Physical Review) form: spaced author
  initials, a **roman** journal abbreviation (not italic), a **bold volume**,
  the page, and the year — `A. B. Author, Phys. Rev. B **20**, 850 (1979).`
  Article titles are **omitted** (the house convention; do not add them). Use an
  Oxford comma and "and" before the final author; `*et al.*` (italic) is fine
  where the source uses it.
- Books use an italic title: `A. B. Author, *Title* (Publisher, City, Year).`
- Never fabricate a missing datum (year, page, volume, city, author). If a
  reference is incomplete, leave it and flag it rather than inventing a value.
- Use full formal entries in reference lists; brief in-text pointers and figure
  captions may use short forms (e.g. "see Blundell *et al.* Ch 6").

## Tone

- Authoritative but welcoming; emphasise accessibility ("…so that switching
  representation is a single click"). Plain where possible, technical where
  necessary, never sensational, and hedged where the evidence is partial.

## Quick reference

- **Reference voice:** "Asymmetry estimates α from a transverse-field run by
  balancing the forward and backward groups; the value is then applied to every
  dataset in the representation."
- **Tutorial voice:** "Open the **Grouping** dialog, choose the **Spin-rotated**
  preset, and click **Estimate α**. The fitted value appears in the status bar."
- **Primer voice:** "In a muon-spin relaxation experiment we follow the
  depolarisation of an ensemble of spin-polarised muons implanted in the sample,
  which reveals the distribution of local magnetic fields at the muon site."
