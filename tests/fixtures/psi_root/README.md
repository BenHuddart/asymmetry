# Synthetic PSI calibration fixtures

These files contain our own two tagged count arrays: `100+k` and `300+3*k`,
for zero-based raw bins `k=0..127`. The independently authored PSI BIN header
has run 4321, labels Forw/Back, t0 4/7, good bounds 16/127, and a float32
0.01 µs resolution. All text is synthetic.

The ROOT files were written by musrfit any2many 1.12.1, revision 708f777b,
with ROOT 6.40.04, from this BIN using public conversion commands:

```sh
any2many -f synthetic.bin -c PSI-BIN ROOT -o legacy-lem.root
any2many -f synthetic.bin -c PSI-BIN MusrRoot -o musrroot-folder.root
any2many -f synthetic.bin -c PSI-BIN MusrRootDir -o musrroot-directory.root
```

The legacy RunInfo TFolder contains a genuine serialized TLemRunHeader v5.
Its resolution is 10 ns, t0 is a 32-element array with active values 4/7,
and histograms are numbered 00/01. Modern forms use 001/002 and retain
Forw/Back labels. Each ROOT histogram has 129 bins: all 128 original counts
plus a terminal zero, retained by the loader. Legacy files supply no good-bin
bounds; the tests set an explicit common window rather than inventing them.
No reference class or implementation source is included.

`reference.json` records input SHA256 hashes and fresh musrfit ASCII exports
of time, observed value, and error; the theory column is excluded. Settings:
packing 1, t0 4/7, no deadtime, zero fixed background, alpha=beta=1.
Type 0 groups forward histograms 1/2 and uses `data 20 60`.
Type 2 uses forward 1, backward 2 and `data 20 60 23 63`.
Both select forward raw 20..59 and backward raw 23..62. The equivalent
Asymmetry window is inclusive 23..62 before its common-support crop of 3.
Counts are exact integers; numerical exports allow 5e-6 relative for their
six-significant-digit precision. Time allows 3e-8 relative for the BIN float32
width versus legacy decimal ns; cross-format F/B values/errors agree to
float64 precision. Asymmetry reports percent; musrfit reports fractions.

musrfit reads BIN, legacy ROOT and modern TFolder with matching selected
observations. Its generated TDirectory uses `decayAnaModule`, whereas its
reader requests `DecayAnaModule`; that file is an Asymmetry compatibility
fixture, not a musrfit numerical oracle. Encoded directory headers,
fractional t0, offsets, beam channels and split detectors have separate loader
regressions. These synthetic files establish no real-facility ROOT coverage.
