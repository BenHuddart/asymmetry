# t0 determination — comparison across WiMDA, Mantid, musrfit and Asymmetry

Line numbers refer to the local checkouts on 2026-09-17 (`$WIMDA_SRC`,
`$MANTID_SRC`, `$MUSRFIT_SRC`). Quotes are kept short; the cited files hold
the full context.

## 0. What t0 means — the facility and textbook definitions

| Source | Definition | Practical determination |
|---|---|---|
| ISIS, Cottrell *et al.* 2002, *The application of the NeXus data format to ISIS muon data* (NOBUGS 2002, [arXiv:cond-mat/0210439](https://arxiv.org/abs/cond-mat/0210439)), §4 | "An exact value for time zero (the time from the start of data acquisition to the centre of the muon pulse) is read and included in the file." §2: the file stores "histogram resolution, time zero and the first and last good histogram data bins". | Written by the data-acquisition system (CONVERT_NEXUS then, ICP/IBEX now); the paper does not say how the DAQ measures it. |
| WiMDA manual (F. Pratt, ISIS), §5.5–5.6, §10 ([WiMDA Manual.doc](http://shadow.nd.rl.ac.uk/wimda/WiMDA%20Manual.doc)) | §5.5: the interval from the timer starting to count until the middle of the muon pulse reaches the sample; ISIS examples ≈ 0.645 µs (channel 40) on MuSR, ≈ 0.278 µs (channel 18) on EMU. §5.6: good data begin only once the *entire* pulse has arrived (tgood); tgood − t0 is the "tgood offset", usually 7 bins at ISIS, and may be smaller for F/B than for TF analysis. | §10 (PSI data): run *Search for T0* and set the tgood offset to about 3. |
| PSI, MusrRoot definition (`$MUSRFIT_SRC/src/external/MusrRoot/doc/MusrRootDefinition.tex:553-563`, rendered `doc/html/musr-root.html:724-734`) | `DetectorInfo/Detector%03d/Time Zero Bin` is a `Double_t` "since for the high-field spectrometer an Int_t representation would be not good enough"; `First Good Bin`, `Last Good Bin` are `Int_t`. Example header: Time Zero Bin 3419.0, First Good Bin 3419. | Per detector; the definition does not say whether the DAQ or an offline step fills it. |
| PSI, RRF memo (`$MUSRFIT_SRC/doc/memos/rrf/rrf-notes.tex:104-105, 195-207`) | t0 is "the time of the muon implantation"; HAL-9500 prompt-peak figure caption: "It is *not* straight forward how to define t_0." One-channel t0 error at 9 T ≡ 1.7° phase. | — |
| Textbook: Blundell, De Renzi, Lancaster & Pratt, *Muon Spectroscopy: An Introduction* (OUP 2022), §15.3 pp. 223–224, §14.2 Fig. 14.4 | Continuous sources: t0 is the sharp **prompt peak** (beam positrons triggering both counters), good to a few tenths of ns. Pulsed sources: t0 is the **centre of the muon pulse**, "in practice found from the midpoint of the rising edge"; first good data after the pulse has fully arrived. | "never rely on information stored in the data file, if you have not recorded it yourself!" |

Two physical conventions therefore coexist and the file header encodes
whichever the facility chose: **prompt peak** (continuous) and **pulse
centre** (pulsed). Asymmetry's `find_t0` already selects the estimator by
source type; none of the three reference programs does.

## 1. Reading t0 from files

### WiMDA (`$WIMDA_SRC/src/muondata.pas`, `nexusunit.pas`)

Header values are per histogram in `mrun.info.t0bin[i]`, `tgood_beg[i]`,
`tgood_end[i]` (`muondata.pas:82`), all raw bin numbers. Readers:
ISIS NeXus v1 attribute `t0_bin` broadcast to every histogram
(`nexusunit.pas:1115-1128`); NeXus v2 `time_zero` µs → `round(t·1e3/tbinres)`
then overridden by `time_zero_bin` / attr `t0_bin` (`:1902-2042`; for
`pulsedTD` files all per-histogram arrays are then overwritten by element 1,
`:2036-2042`); PSI `.bin` `nt0[i]/ntini[i]/ntfin[i]` (`muondata.pas:1744-1748`);
`.mdu` `t0b div ibin` (`:1645-1649`); TRIUMF `hishead[8..10]` (`:2192-2195`);
**no MusrRoot reader exists**. On load (`muondata.pas:2493-2506`):

```pascal
if FileVals or not runloaded then begin
  for i := 1 to 32 do htzero[i] := mrun.info.t0bin[i];
  cgrp.tzero := htzero;
  cgrp.tgoodbeg := mrun.info.tgood_beg[1];
  cgrp.tgoodend := mrun.info.tgood_end[1];
  if cgrp.tgoodend = 0 then cgrp.tgoodend := mrun.info.lenhis;
  cgrp.toff := cgrp.tgoodbeg - cgrp.tzero[1];
```

Only histogram 1's first/last good bins are used. Header bins index the
1-based Pascal arrays verbatim — no ±1 anywhere.

### Mantid (`$MANTID_SRC/Framework/...`)

All loaders expose `TimeZero` (µs, common), `FirstGoodData`, `LastGoodData`,
`TimeZeroList`, optional `TimeZeroTable` (`Muon/src/LoadMuonNexus.cpp:83-96`,
`DataHandling/src/LoadMuonNexusV2.cpp:110-122`, `LoadPSIMuonBin.cpp:95-116`).

- NeXus v1: `run/histogram_data_1/time_zero` (µs); first/last good =
  `first_good_bin·resolution` with no 0/1-based adjustment
  (`Muon/src/LoadMuonNexus1.cpp:93-123`); axis from `corrected_time` treated
  as **bin centres** and converted to edges
  (`Muon/src/MuonNexusReader.cpp:211-222`).
- NeXus v2: `raw_data_1/detector_1/time_zero` may be an **array** (per
  spectrum) — `"Time zero list size does not match number of spectra"` throws
  (`DataHandling/src/LoadMuonNexusV2NexusHelper.cpp:219-233`); the loader
  subtracts the *common* value from `raw_time` edges
  (`LoadMuonNexusV2.cpp:150-153`).
- PSI bin: `realT0[i]` (float µs) preferred, else `integerT0[i]`; **common t0
  = max over histograms** (`LoadPSIMuonBin.cpp:186-195`); bin-vs-µs decided
  by magnitude (`:197-209`); first good = max over 16 slots, last good = slot
  0 only (`:183-184, 220-222`).

### musrfit (`$MUSRFIT_SRC/src/classes/PRunDataHandler.cpp`)

`PRawRunDataSet` holds `Double_t fTimeZeroBin`, `fTimeZeroBinEstimated`,
`Int_t fFirstGoodBin/fLastGoodBin` (`src/include/PMusr.h:657-711`), all
0-based bins. MusrRoot `DetectorInfo/.../Time Zero Bin` per detector
(`:1911-1931`); PSI-BIN `GetT0Vector()` — the **integer** t0 (the float
"real t0" at byte 792 is ignored), error if empty (`:2587-2610`; reader
clamps `t0 > length → 0`, `MuSR_td_PSI_bin.cpp:2001-2008`); MDU
`(t0b+1)·f − 1` (`:1553-1555`); MUD `t0_bin` (`:2895-2934`); NeXus IDF1/2
attribute `t0_bin` **one value for every histogram**, default −1
(`src/include/PRunDataHandler.h:651-690, 868-901`). Every non-NeXus reader
also stores an argmax **estimate** at read time (`:2620-2628`):

```cpp
    // estimate T0 from maximum of the data
    for (UInt_t j=0; j<histoData.size(); j++)
      if (histoData[j] > maxVal) { maxVal = histoData[j]; maxBin = j; }
```

(ROOT readers store `histo->GetMaximumBin()`, which is 1-based — a likely
off-by-one in the *estimate* only.)

### Asymmetry (`src/asymmetry/core/io/`)

`Histogram.t0_bin` is an integer; `time_axis = (arange − t0_bin)·w`
(`core/data/dataset.py:38-52`). PSI `.bin` header `i16` at 458/490/522 + 2i
(`psi.py:234-236`), MDU `(t0b+1)·f − 1` (`:341-343`), padded/clamped by
`_resize_ints` so a missing value is bin 0 (`:1515-1526`). MusrRoot
`DetectorInfo` "Time Zero Bin" **rounded** to int, missing → 0
(`root.py:578-583, 1008-1012`). NeXus: `counts.attrs['t0_bin']` else
`time_zero` (bin index in v1, µs in v2), nothing → 0 (`nexus.py:895-941`);
a 1-based heuristic decrements when the first ≤ 8 detectors' axes vote for
it (`:1027-1068`). PSI/ROOT write `t0_bin = common_t0_for_groups(F, B)`,
`detector_t0_bins`, per-detector good bins (`psi.py:1146-1155`,
`root.py:647-656`); **NeXus writes `t0_bin = histograms[0].t0_bin` and no
detector table** (`nexus.py:816-828`).

## 2. Automatic search

| | Statistic | Range / ties | Per what | Pulsed handling | tgood |
|---|---|---|---|---|---|
| WiMDA `SearchT0` (`$WIMDA_SRC/src/Group.pas:2225-2256`) | argmax of **group-summed raw** counts (a max-derivative variant is commented out) | bins `np`→2 descending, strict `>` ⇒ highest tie bin; `np` = last Regroup's *output* point count (`:1592`), can undercut the peak | per group | identical scan, but result copied into `htzero` only for continuous data; button disabled for pulsed (`SameT0` forced, `muondata.pas:2620`) and while FileValues is ticked | untouched |
| Mantid | **none** (swept `Framework/Muon`, `DataHandling`, Python algorithms, Muon GUI) | — | — | — | — |
| musrfit `musrt0 -g` (`$MUSRFIT_SRC/src/musrt0.cpp:145-160, 349-770`) | argmax of the raw histogram | whole histogram, strict `>` ⇒ lowest tie bin | per histogram (each F/B member and ADDRUN) | none; manual: "tries to estimate t0 from the prompt peak (maximum entry)" (`doc/html/user-manual.html:400`); musrFT warns "(assuming a prompt peak!!)" (`:355`) | `fgb := t0 + offset` if given, `lgb := length` |
| Asymmetry `find_t0` (`src/asymmetry/core/transform/t0.py:75-113`) | continuous: argmax (lowest tie); pulsed: half-maximum crossing of the leading edge walked back from the peak, linearly interpolated, rounded | whole histogram | per histogram; consensus = rounded **median**, `spread_bins` = range (`:116-158`) | strategy from facility/instrument tokens; unknown ⇒ pulsed (`:59-72`) | untouched (policy shifts `first_good_bin` with t0) |

musrfit's interactive `musrt0` (`src/classes/PMusrT0.cpp`) draws the file t0
as a reference line on key `s` (`:841-849`), jumps to the estimate on `T`
(`:930-960`), and zooms to t0 ± 75 bins (`:1192-1215`) — the closest any
reference program comes to a file-vs-detected comparison.

## 3. Policy: which value wins, and what the user override means

| | Default | Override semantics | Where the choice lives | Divergence check |
|---|---|---|---|---|
| WiMDA | `FileValues` ticked (`globals.pas:75`, `Group.pas:864`): header t0 per histogram | Unticked: `NewTz` **replaces every detector's t0** with the typed bin (`Group.pas:710-725`), or per group with `SameT0` off (`:2850-2862`); persists across later run loads (`muondata.pas:2493` guard) | binary `.mgp` record incl. `FileValues`, `SameT0`, `tzero[]`, `toff`, `tgoodend` (`Group.pas:2168-2180`) | none |
| Mantid Muon Analysis | "μs (From data file)" ticked, edit **disabled** (`home_instrument_widget/view.py:169-220`); user box initialised to the file value (`model.py:34-41`) | unticked: common `TimeOffset = t0_file − t0_user` ⇒ **replaces** the common t0 (`calculate_pair_and_group.py:65-71`); per-detector `TimeZeroTable` exists in `MuonPreProcess` (`Muon/src/MuonPreProcess.cpp:225-240`) but the GUI never passes it; first-good box holds an absolute corrected time that does **not** track t0 edits (`load_utils.py:238,251`) | GUI context | none; doc: "By default the time zero is taken from the file but it can be specified here" (`docs/source/interfaces/muon/muon_home_tab.rst:15-22`) |
| musrfit | RUN `t0` → GLOBAL `t0` → file (only if `> 0.0`) → argmax estimate with `**WARRNING** NO t0's found … NO WARRANTY THAT THIS OK!!` (`PRunSingleHisto.cpp:1795-1848`); the chosen value is written back to the `.mlog` (`PMsrHandler.cpp:1020-1083`, **not** for ISIS runs) | msr `t0` is an **absolute per-histogram bin** (`Double_t`); `addt0` per ADDRUN | msr file; manual order at `user-manual.html:1236-1239` | bounds `ERROR` if t0 ∉ [0, N] (`:1842-1847`); no file-vs-msr comparison; file fgb/lgb never consulted by the fit classes (`:1960-2031`, fallback `fgb = t0 + 10 ns`) |
| Asymmetry | `T0Policy.mode = "from_file"` (`core/project/profiles.py:387-418`) — but a fresh draft is **inferred** from the payload and mislabelled Manual when an out-of-group detector has a larger header t0 (`:895-928`) | Manual stores an **absolute** bin; applied per run as `delta = value − file_common_t0(run)` added to every detector's file t0, published as `effective_detector_t0_bins`, `first_good_bin += delta`; histograms never mutated (`:1288-1330`) | profile (`t0_policy` dict) or per-run override payload | none; `docs/reference/data_reduction/t0_search.rst:15-17` claims the found value "sits beside the file value" — it does not |

## 4. Aligning detectors with different t0

- **WiMDA**: each histogram is read at `histos[hn, i + htzero[hn]]` while
  summing into its group (`Group.pas:1496-1502`); one common `toff` is
  applied relative to each histogram's own t0; last good = `tgoodend −
  max(tzero)` bins after t0 (`:1309-1313`), i.e. truncation to the shortest.
- **Mantid**: loaders shift the whole X axis by the *common* t0 (PSI: the
  max); per-spectrum differences are ignored in the GUI.
- **musrfit**: group members and ADDRUNs are shifted by **integer** bins
  onto the first member's t0 (`PRunSingleHisto.cpp:1119-1125`); forward and
  backward are *not* shifted onto each other — instead `fgb − t0` is forced
  equal for both with a WARNING (`PRunAsymmetry.cpp:2110-2128`) and the
  asymmetry is formed bin-by-bin from each side's own good window.
- **Asymmetry**: `apply_grouping_aligned` zero-pads each detector at the
  front by `common − t0_i` and truncates to the shortest padded array
  (`core/transform/grouping.py:101-160`); `common_t0_for_groups` = max over
  the present F/B detectors (`:163-180`).

## 5. Time-axis convention (where t = 0 sits in the t0 bin)

| | Stamp of raw bin *i* | t = 0 |
|---|---|---|
| WiMDA (`Group.pas:1405-1422`): `t1 := (i−0.5)·tres; timed := (t1+t2)/2` | `(i − t0)·tres` = **centre** of bin | centre of bin t0 |
| musrfit (`PRunSingleHisto.cpp:1245-1275`): `DataTimeStart = dt·((fgb−0.5) + p/2 − t0)`; `musrt0` axis `−0.5 … N−0.5` (`PMusrT0.cpp:317-319`) | `(i − t0)·dt` = **centre**; packed point at block centre | centre of bin t0 (single-histo truncates a fractional t0 to int; asymmetry keeps the double, `PRunAsymmetry.cpp:1143`) |
| Mantid: `x[0] = −t0` after `timeAxis − timeZero` on **edges** (`LoadMuonNexusV2Test.h:60-75`; PSI `absTimeZero = x[floor(t0)]`, `LoadPSIMuonBin.cpp:197-209`) | edges shifted | **left edge** of bin t0 — half a bin later than WiMDA/musrfit |
| Asymmetry: `time = (k − t0)·w`, merged bins `t_start + (e0+e1−1)/2·w` (`core/transform/rebin.py:223, 265`) | `(k − t0)·w`; the merged formula equals musrfit's block centre | centre of bin t0 **numerically** — but the `rebin.py:196-201` docstring calls the stamps "left-edge", and the NeXus loader's own dataset axis is the file `corrected_time` (`nexus.py:740-744`), which Mantid treats as bin centres |

Asymmetry, WiMDA and musrfit agree; Mantid is offset by half a bin (8 ns at
16 ns binning). The Asymmetry docstring is the only inconsistency, plus the
loader-time NeXus axis not being rebuilt from `t0_bin`.

## 6. Downstream consumers (Asymmetry only — where the policy is and is not honoured)

| Consumer | Alignment source | Honours `effective_detector_t0_bins`? |
|---|---|---|
| `core/transform/reduce.py:245-296`, `grouping.py::group_forward_backward` | override → file | **yes** |
| `core/fourier/grouped.py:308-320`, `gui/mainwindow.py:8302-8304` | `common_t0_for_groups` on file t0 | **no** (`fourier/spectrum.py:449` puts the override in the cache digest only) |
| `core/maxent/engine.py:583-590` | axis from `grouping["t0_bin"]` (shifted), counts from grouped Fourier (file) | **inconsistent** |
| `core/fitting/grouped_time_domain.py:292, 346-349` (count-domain fits) | file t0 | **no** |
| `core/fitting/count_domain.py:740-744, 956-960` | free µs nuisance `t0`, `t_eval = time + t0` | n/a |
| `core/transform/promote.py:77-124` | `t0_bin += round(t0_us / w)` | sign inverted (see README §3); does not shift `first_good_bin` or the override |
| `core/io/nexus_writer.py:129-133, 188-196` | `time_zero` = file per-detector bins; `corrected_time`/`first_good_bin` from the (shifted) grouping | **mixed** |
| `core/transform/deadtime.py:98` (calibration window) | `histogram.t0_bin + offset` | no (raw-bin window; arguably correct) |
| `gui/panels/plot_panel.py:6268-6272` (good-window mask) | `histograms[0].time_axis` | no — detector 0, not the common t0 |
| `core/data/combine.py:382-419` | per-detector max(t0) across runs | file (consistent with loaders) |

## 7. Divergence table

| # | Topic | WiMDA | Mantid | musrfit | Asymmetry today | Proposed |
|---|---|---|---|---|---|---|
| T1 | Default source | file | file (edit disabled) | msr → file → estimate | file, but mislabelled Manual by inference | file; policy stored explicitly, never inferred |
| T2 | Automatic search | argmax, group sums, continuous only | none | argmax per histogram | argmax / pulse-edge midpoint, median consensus | keep; run always for display |
| T3 | File-vs-detected display | none | none | reference line in musrt0 | none (docs claim otherwise) | read-only detected value + Δ + spread in every mode |
| T4 | Divergence warning | none | none | bounds error only | none | tolerance per source; per-detector outlier check; missing/zero/out-of-range t0 flagged |
| T5 | Manual semantics | replace all detectors | replace common | absolute per histogram | absolute bin applied as per-run offset | store the offset; display resolved absolute |
| T6 | Per-detector alignment | own t0 per histogram | ignored (common max) | integer shift onto first member | shift + truncate to common max | keep; single resolver used by all consumers |
| T7 | Missing t0 in file | used verbatim | used verbatim | ≤ 0 ⇒ estimate + warning | silently 0 | flag "unknown", auto-detect, warn |
| T8 | t = 0 position | bin centre | left edge | bin centre | bin centre (docstring says edge) | keep centre; fix docstring; rebuild NeXus loader axis from `t0_bin` |
| T9 | tgood | file, hist 1 only; offset 7 (ISIS) / 3 (PSI) | file, absolute, does not track t0 | msr `data`; fallback t0 + 10 ns | file, shifted with t0 under policy | keep; warn if `first_good < t0` or, pulsed, before the pulse has ended |
| T10 | Fit-time t0 | free "t0 offset (ns)" parameter (`Analyse.pas:1358`) | none | none (phase absorbs it; RRF memo) | free µs nuisance + promotion | keep; fix promotion sign and shift the good window with it |
| T11 | NeXus per-detector t0 | arrays read then overwritten by element 1 | v2 array read, common applied | one value for all | attrs read per detector, but payload carries detector 0 only | write `detector_t0_bins` and group-max `t0_bin` like PSI/ROOT |
| T12 | 0/1-based header bins (ISIS) | verbatim into 1-based arrays | index × width, no adjustment | 0-based verbatim | axis-vote heuristic, ≤ 8 detectors, needs explicit `t0_bin` attr | keep heuristic; add a corpus test on the pulse position |

## 8. Uncertainties carried forward

- Whether ISIS writes `t0_bin`/`first_good_bin` 0- or 1-based is undocumented
  in all three references (WiMDA indexes 1-based arrays verbatim; Mantid and
  musrfit treat them as 0-based). Asymmetry's heuristic resolves it per file;
  verification-plan.md adds a corpus check of the pulse position.
- No PSI document in the musrfit tree states who fills `Time Zero Bin`
  (DAQ vs offline). The LEM reader sets `fgb := t0`, and example MusrRoot
  headers carry `First Good Bin == Time Zero Bin`.
- The WiMDA manual figures (5.4, 10.1) were not inspected; captions only.
- The half-bin question for NeXus v1 `corrected_time` rests on Mantid's
  reader comment ("Assume that values are bin centre times") and its tests.
