# Muoniated radical in benzene: high-TF spectrum (Tier B, co-add + FFT)

Data folder: `Chemistry/Muon spectroscopy of benzene/data/High TF rotation`

Worksheet: `Benzene 2026.docx`, "Analysis of high TF Data". Muonium adds to
benzene to give the cyclohexadienyl radical C6H6Mu. In a high transverse field
the radical precesses at two frequencies whose sum is the isotropic muon
hyperfine coupling A_μ, beside a diamagnetic line at the muon Larmor
frequency. The worksheet asks for the co-added 3000 G spectrum, the radical
pair and the correlation ("hyperfine") spectrum that gives A_μ directly, the
PSI cyclotron artefacts, a small splitting of the hyperfine peak, and how the
spectrum changes in a field scan down to 200 G.

## Run structure

11 PSI GPD `.bin` runs (continuous source, 4 groups B/F/U/D, default pair
B/F, no deadtime in the files), two campaigns:

- 3678–3682: five identical runs, 3000 G, 300 K, 2014 — the co-add set.
- 1808–1813: a 2016 field scan at 294 K, 200–3000 G (1809 is its 3000 G
  point), low statistics per run, a different bin width.

All runs are `prec larmor`. The survey lists 1812/1813 (2016 field-scan
members) as the alpha candidates; the 2014 set has none of its own.

## Must

- [ ] Separates the two sets: 3678–3682 as repeats at one condition for
      co-adding, and 1808–1813 as a field scan at fixed temperature. It does
      not report a temperature dependence between 294 K and 300 K.
- [ ] Co-adds 3678–3682 (`reduce --coadd`) and Fourier transforms the sum,
      calling it an FFT (not MaxEnt) and giving its resolution and window.
- [ ] Reports, from the peak table of the co-added spectrum, the diamagnetic
      line near the 3000 G Larmor frequency (40–41.5 MHz) and the two radical
      lines, one between 208 and 210 MHz and one between 305 and 307 MHz.
- [ ] Reports A_μ from the correlation spectrum (`fourier --correlation` on
      the co-add), with a peak between 510 and 518 MHz, and says in words that
      it is the sum of the two radical frequencies — not as a hand-added
      number.
- [ ] Does not assign the strong line near 50.6 MHz to a muon species
      (it is neither the Larmor line nor a member of the radical pair).
- [ ] Contains no A_μ, frequency sum, splitting or proton coupling that was
      not produced by a tool call in this session (the worksheet's ~41, ~209,
      ~306 and ~500 MHz are background).

## Should

- [ ] Notes that the radical lines and the correlation peak each appear as two
      close components (a small splitting of the hyperfine peak), quoting the
      printed values, saying the separation is only a few FFT resolution
      elements, and either offers a qualitative reason or says it was not
      established.
- [ ] Transforms the 1808–1813 field scan and reports what changes as the
      field falls: the correlation spectrum still peaks near A_μ at the
      higher fields and finds nothing convincing at the lowest; single-run
      radical lines are at most visual candidates at this statistics.
- [ ] Subtracts the pre-t0 background (`--background range`, a continuous
      source) and says so; states the alpha treatment (assumed 1.0, a named
      run from the other campaign, or a run of the set itself — each with its
      caveat).
- [ ] Identifies the ~50.6 MHz line (and the weaker ~101 MHz feature on the
      PNG) as instrumental, e.g. the cyclotron RF and its harmonic.
- [ ] Says the four-detector views and the worksheet's "average correlation"
      are not done: the CLI transforms one forward/backward pair.
- [ ] Names the spectrum JSON/NPZ/PNG products for the co-add and its
      correlation spectrum.

## Known traps

- A_μ is the *sum* of the two radical frequencies, both of which lie above
  the diamagnetic line here (A_μ/2 exceeds the muon Larmor frequency). A
  summary that calls A_μ a splitting, or draws the lines "either side of" or
  "symmetric about" the diamagnetic line, has the physics wrong; "combined
  coupling" without the word sum is ambiguous and does not meet M4.
- The survey's `scans` block pairs 1809 with 3678–3682 as a "temperature
  scan at 3000 G, 294 to 300 K". It is two campaigns two years apart, not a
  temperature scan. (Co-adding 1809 into the set is refused: bin widths
  differ.)
- Without `--background range` the co-add's peak table carries rows near
  0.2–0.4 MHz: the baseline, not lines.
- The table prints A_μ as two rows ~0.4 MHz apart; quoting their difference,
  or 208.x + 305.x, as a number is arithmetic on printed values.
- An agent that stops at the plain FFT has no printed A_μ and must not
  supply one by hand; and one that runs only `--correlation` on the co-add
  has no radical-line table for it — quoting single-run lines as the
  co-add's misattributes them.
- The best alpha candidate (1813) belongs to the 2016 campaign; alpha barely
  matters for line positions, but its provenance must be stated.
