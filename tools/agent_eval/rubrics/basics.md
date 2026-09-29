# Basics: instrument calibration exercises (Tier B, multi-purpose folder)

Data folder: `Basics/data`

Worksheet: `Basics 2026.docx`. The school's introductory sheet, not one
experiment: t0 and t_good, deadtime correction (a high-statistics ZF Ag run),
grouping and the alpha estimate (a MuSR Ag run), then four instrument
calibrations: a steering curve (initial asymmetry of a Ag mask over Fe2O3,
which depolarises, against the steering-magnet current), a range curve (100 G
diamagnetic asymmetry of quartz behind a growing stack of Ti foils; it
saturates once every muon stops in the foils), t0 from the field dependence
of the TF phase in Ag (a correct t0 gives a field-independent phase), and the
EMU frequency response from muonium precession in quartz in a few gauss TF
(the pulse width limits the highest resolvable frequency). A passing summary
recognises these as separate exercises and reports each on its own terms.

## Run structure

68 files, three instruments; EMU and MUSR share run numbers 44989–44997.

- EMU 18850, 18852–18864 (2010, no deadtime in file): Ag TF field scan,
  5–100 G. 18851 is absent. 18864 (45 G) was taken the next morning and sits
  in its own alpha block.
- EMU 18888–18899: quartz at 100 G TF, the range curve. The foil count is in
  the run notes only.
- EMU 19625–19643: quartz, 0.5–9 G TF, the muonium frequency-response scan
  (19625 is an extra 2 G run).
- EMU 34998 (ZF Ag, ~1.3×10⁹ events) and 34999 (100 G TF Ag): deadtime demo
  and reference.
- EMU 44989–44997 (2013): "Haematite on Ag", 100 G TF, steering current
  −1.0 to +1.0 A in the run notes. EMU 44998/44999: Ag TF/ZF "In variox".
- MUSR 44989–44997: "Ag on candlestick", header field 0 G and stamped ZF, but
  each precesses near 0.29 MHz: an unrecorded weak TF, the grouping/alpha demo.
- HIFI 62798: Ag, 20 G TF, "steered properly".

## Must

- [ ] Treats the folder as several separate calibration exercises. It names
      at least the Ag TF field scan, the quartz range curve, the quartz
      low-field muonium scan and the steering scan, and analyses them
      separately. It does not present the survey's 24-run 100 G "temperature
      scan" or its 2-run ZF one as a temperature dependence.
- [ ] Keeps the EMU and MUSR runs 44989–44997 apart and says which set is
      which (EMU: haematite on Ag at 100 G, the steering scan; MUSR: Ag
      calibration). It does not read the MUSR runs' ~0.29 MHz line as
      spontaneous order in Ag: the header says 0 G, but the line comes from
      an unrecorded weak transverse field.
- [ ] Quotes no logged 943 K, ~640 K or ~550 K reading as a sample
      temperature. It says these logged values are faulty or irrelevant to
      these room-temperature calibrations, and claims no temperature trend.
- [ ] Steering: fits the Ag-mask asymmetry against the current taken from
      the run notes, and reports it smallest near zero current (between −0.25
      and +0.25 A), rising on both sides.
- [ ] Range curve: reports the 100 G diamagnetic amplitude as small and
      roughly flat for thin stacks, then rising steeply and saturating for the
      thickest. It names, from its own fits, the foil count (as the run notes
      give it) where saturation begins — about 12–14 foils (a range such as
      "13–15" counts).
- [ ] Quartz scan: identifies the few-gauss lines as muonium (a line far
      faster per gauss than the bare muon's — the ratio itself need not be
      quoted). It reports the muonium amplitude
      falling as the frequency rises, and the line lost at the top fields,
      and presents the amplitude fall as the instrument's frequency-response
      limit (attributing the line's loss at the top fields to it too is a
      Should).
- [ ] Ag field scan: reports the fitted frequency tracking the applied field.
      It also reports the fitted phase against field. On the file's t0 the
      phase drifts steadily with field (a t0 offset), so calling it
      field-independent without a t0 test is a fail.
- [ ] States which run supplied alpha for each exercise it fits. It does not
      carry one alpha across the survey's ALPHA STEPs.
- [ ] Contains no t0 correction, stopping density (mg/cm²), steering centre,
      upper frequency, field offset or alpha that was not produced by a tool
      call in this session. The worksheet's Ti density and foil thickness are
      background.

## Should

- [ ] Reads each exercise's pattern the right way round: the steering
      scan's silver signal is smallest where the beam is centred on the
      (depolarising) sample; the range curve's step is where the muons stop
      passing through the degraders into the sample. An inverted reading is
      a gap even when the Must's pattern is reported.
- [ ] Does not dismiss a relative amplitude trend (the muonium frequency
      response) on the ground that alpha is uncertain: one alpha scales every
      amplitude alike.
- [ ] Deadtime: reduces ZF Ag 34998 with deadtime off and from the file, and
      describes the early-time sag in asymmetry that the correction removes,
      with both fits' χ²ᵣ. Says the 2010 files carry no deadtime values.
- [ ] Measures MUSR alpha on a MUSR Ag run and says why that is legitimate,
      since the survey lists no MUSR candidate because the header records
      0 G. Alternatively, it says plainly that it assumed 1.0.
- [ ] t0: tests `--t0-offset` (or fits `Linear` to phase against field),
      reports the shift in bins that flattens the phase, and checks the
      fitted frequency against the nominal field.
- [ ] Treats 18864 as its own alpha block, or excludes it and says why.
- [ ] Says that turning the saturating foil count into mg/cm² needs the foil
      thickness and Ti density, which the files lack, and does not present
      that conversion as Asymmetry output.

## Known traps

- Converting a fitted frequency to a field by hand ("about 21 G" from
  0.28 MHz) is arithmetic no command printed and fails the number rule;
  say the line is at the frequency printed, well away from zero field.
- The worksheet's run logs do not match the disk. 18851 is absent, and 18864
  and 19625 are unlisted. The worksheet's foil counts (0, 0, 1…10) disagree
  with the run notes (0, 4, 6, 8, 9, 9, 10…15), which are all the agent sees.
  Quoting the worksheet's counts breaks the number rule. The worksheet says
  the steering current is not logged, but the notes carry it.
- The survey's `scans` block merges the range curve, the steering scan,
  34999, 44998 and the 100 G Ag run into one "temperature scan". Neither the
  steering nor the range variable appears as a scan: both need `--order
  <name> --x …` from the notes. The `trend` "Next" hint (Linear on Lambda
  against current or foils) points at the wrong quantity. The physics is in
  the amplitude.
- The wizard on MUSR 44989 reports μ–F and Kubo–Toyabe pattern matches.
  Silver has neither, so reporting them is a fail.
- Logged temperatures: 943.48 K (railed) on 18890–18898, ~644 K on EMU
  44989–44998, and 550 K on 44999. MUSR logs 142 → 99 K against a stale
  290 K setpoint; that reading is plausible, and irrelevant for Ag.
- The top quartz fits (8.5, 9 G) fail, and nearly every run in that series
  is flagged. The loss of the line is the finding, not a fit to rescue.
