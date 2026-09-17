# ISIS NeXus header bins are 1-based — evidence

**Question.** Are the `t0_bin`, `first_good_bin` and `last_good_bin`
attributes on the ISIS `counts` dataset 0-based or 1-based? None of WiMDA,
Mantid or musrfit documents it (comparison.md § 8).

**Answer (2026-09-17).** They are **1-based, inclusive**. `t0_bin` is the
1-based index of the bin that *contains* the file's own `time_zero`
(microseconds); equivalently the 0-based bin containing t0 is `t0_bin − 1`.
Three independent lines of evidence, from 1,245 ISIS files on this machine
(EMU, MuSR, HiFi, ARGUS; HDF4 v1 from 2003–2014 and HDF5 v2 from 2010s–2024;
16 ns and 8 ns binning), all agree.

## Method

The DAQ writes the same quantity twice: `time_zero` in µs and `t0_bin` as an
index, plus `resolution`, `raw_time` (v1: one value per bin; v2: `n+1`
edges) and `corrected_time`. Dividing `time_zero` by `resolution` gives the
position `q` of t0 in bins; the stored integer then reveals the rule. The
survey script is `tests/porting/t0-determination/isis_t0_header_survey.py`
(run on a list of file paths; nothing from the private corpora is recorded
here beyond instrument, era and bin values).

## Evidence

### E1. `last_good_bin` equals the histogram length in every file

| Instrument | Files | `last_good_bin == n_bins` |
|---|---|---|
| EMU | 528 | 528 |
| HiFi | 478 | 478 |
| MuSR | 200 | 200 |
| ARGUS (2024, IBEX) | 38 | 38 |
| MuSR 2003 (`32482.NXS`) | 1 | 1 (1500 of 1500) |

A 0-based index can never equal the length. The good-bin attributes are
therefore 1-based and inclusive.

### E2. `t0_bin = floor(time_zero / resolution) + 1` in every unambiguous file

On files whose `time_zero / resolution` has a clearly fractional part
(0.05 < frac < 0.95), so that floor, round and ceiling give different
answers:

| Instrument | Files | `floor(q) + 1` | `floor(q)` | `round(q)` | other |
|---|---|---|---|---|---|
| EMU | 466 | 466 | 0 | 0 | 0 |
| HiFi | 478 | 478 | 0 | 0 | 0 |
| MuSR | 200 | 200 | 0 | 0 | 0 |

Observed `(q → t0_bin)` pairs: 10.437 → 11, 9.437 → 10, 8.813 → 9,
14.937 → 15, 34.375 → 35, 29.625 → 30, 13.75 → 14, 27.5 → 28, 11.25 → 12,
1.5 → 2, 1.75 → 2, 1.937 → 2, 2.75 → 3, 4.75 → 5, 29.875 → 30 (8 ns). The
two integer cases (EMU v2 0.160 µs → 9.99999 → 10; ARGUS 0.400 µs →
25.00001 → 26) follow the same rule once float32 noise is accounted for.
`round` is ruled out by every `.437`/`.375`/`.25` case.

Reading: 1-based index of the bin containing t0 — the ICP/VMS Fortran
heritage, and exactly how WiMDA (Pascal `histos[h, 1..lenhis]`) indexes
these values verbatim.

### E3. The file's own `corrected_time` agrees

`corrected_time[k]` is the bin-*centre* time minus `time_zero` (v1
`raw_time` holds centres; v2 `raw_time` holds `n+1` edges). In every file
the sign change of `corrected_time` straddles the 0-based bin `t0_bin − 1`,
e.g. EMU 0.167 µs: `ct[10] = +0.001`, `ct[11] = +0.017` (t0 inside 0-based
bin 10, `t0_bin = 11`); MuSR 2003: `ct[39] = 0.000`, `t0_bin = 40`.

### E4. The pulse itself

Half-maximum rising-edge midpoint of the summed counts, in edge units,
minus `time_zero / resolution` (bins):

| Instrument / era | n | mean | sd |
|---|---|---|---|
| HiFi v1 16 ns | 442 | −0.34 | 0.19 |
| MuSR v1 16 ns | 198 | −0.25 | 0.22 |
| EMU v2 16 ns | 105 | −1.17 | 0.52 |
| EMU v1 16 ns | 404 | −1.82 | 1.50 |
| EMU v1 8 ns | 19 | −6.14 | 0.11 |
| ARGUS 2024 | 38 | −15.34 | 0.02 |

On HiFi and MuSR the detected rise midpoint sits a third of a bin *before*
the header t0, consistent with `time_zero` marking the pulse centre (the
rise midpoint leads the centre of an asymmetric pulse). EMU headers run
1–2 bins late against the observed pulse, the 8 ns EMU runs ~50 ns late,
and the 2024 ARGUS calibration runs carry a `time_zero` of 0.400 µs while
the pulse rises at ~0.15 µs — a mis-set header. These are precisely the
cases the divergence warning is for, and they set the tolerance: **± 1 bin
is normal on HiFi/MuSR; EMU needs ≥ 2 bins; anything beyond ~3 bins is a
header problem worth flagging.**

### E5. A stale `time_zero` with a correct `t0_bin` (2003)

`32482.NXS` (MuSR, musrfit example data) stores `time_zero = 0.278 µs`
(EMU's value) but `t0_bin = 40` and `corrected_time` built from bin 40 —
the pulse rises at 40.1. The two fields can disagree in old files, so the
loader must cross-check them rather than trust either alone.

## What Asymmetry does today with these files

`_infer_v2_bin_index_offset` (`src/asymmetry/core/io/nexus.py:1027-1068`)
votes from `corrected_time` and decrements when the axis is closer to zero
at `t0_bin − 1`. On the survey files it returned `bin_index_base = 1` and
`t0_bin − 1` everywhere **except** the exact-edge EMU v2 case (0.160 µs,
`ct[9] = −0.008`, `ct[10] = +0.008`: a tie, no vote, `t0_bin` kept at 10 with
`bin_index_base = 0`). It also relied on `corrected_time` being present and
on the first eight detectors.

## Consequences for the design (feeds implementation-options.md R6)

1. **Decode deterministically**: for ISIS NeXus, 0-based `t0_bin` =
   attr − 1, `first_good_bin` = attr − 1, `last_good_bin` = attr − 1
   (inclusive), always. Keep the axis vote only as a *cross-check* that
   raises a loader warning when it disagrees (E5-type files).
2. **Carry the exact t0.** `time_zero` places t0 at an arbitrary sub-bin
   position (fractions .437, .375, .75, .5, .937 and .0 all occur). An
   integer `t0_bin` therefore mis-stamps the time axis by up to half a bin,
   8 ns at 16 ns binning: a phase error of 2π·f·8 ns, i.e. 8° at 20 mT
   and 39° at 0.1 T in a TF fit. Mantid subtracts the µs value exactly;
   musrfit keeps a `Double_t` t0 on its asymmetry path. Asymmetry should
   store `t0_time_us` (ISIS: `time_zero`; PSI MusrRoot: the `Double_t`
   `Time Zero Bin × width`) and stamp the time axis from it, keeping integer
   bins for detector alignment and the good window. This is a decision for
   the user (see implementation-options.md § B R10).
3. **Cross-check `time_zero` against `t0_bin`** at load: `floor(q) + 1 ≠
   t0_bin` ⇒ warn and prefer `t0_bin` (E5), recorded as `t0_source`.
