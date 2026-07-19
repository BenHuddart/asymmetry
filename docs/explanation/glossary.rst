Glossary
========

A short reference for the μSR terms and the Asymmetry-specific vocabulary used
throughout these pages. The definitions are deliberately brief; where a concept
has its own reference page, the entry points there for the full treatment.
For a narrative introduction rather than isolated definitions, read the
:doc:`musr_primer`.

.. glossary::
   :sorted:

   asymmetry
      The normalised difference between two opposed detector groups,
      :math:`A(t) = (F - \alpha B)/(F + \alpha B)`, which tracks the projection
      of the muon-spin polarisation onto the detector axis. It is the primary
      observable in most μSR analyses and the quantity Asymmetry fits by
      default.

   alpha
   α
      The calibration constant that balances the forward and backward detector
      groups, correcting for their unequal efficiency and solid angle. It is
      usually estimated from a transverse-field run in which the true asymmetry
      oscillates symmetrically about zero. See
      :doc:`/reference/data_reduction/alpha_calibration`.

   beta
   β
      The intrinsic-asymmetry balance :math:`\beta = A_{0,b}/A_{0,f}` from
      musrfit's asymmetry fit (fit type 2): it corrects for the two detector
      groups observing different asymmetry *amplitudes* (solid-angle and
      absorption effects), entering the corrected asymmetry as
      :math:`A = (F - \alpha B)/(\beta F + \alpha B)`. Unlike :math:`\alpha`
      it is invisible to count totals, so it cannot be estimated from count
      ratios; it is entered as a fixed value in the Grouping window's
      **β (asymmetry balance)** card and defaults to 1.

   ALC
   avoided level crossing
      A resonance seen in a longitudinal-field scan when a muon and a
      neighbouring nuclear (or electronic) spin level are brought close in
      energy, allowing polarisation to transfer and producing a dip or step in
      the time-integrated asymmetry. Asymmetry builds ALC scans in the
      :doc:`Integral scan mode </reference/alc_mode>`.

   Knight shift
      The small fractional shift of the muon precession frequency, relative to
      the frequency in a reference material, caused by the local field from
      polarised conduction electrons or a paramagnetic host. Its temperature or
      angle dependence probes the local susceptibility and the muon site; see
      :ref:`knight-shift`.

   Kubo–Toyabe
   KT
      The depolarisation function for muons in a random, quasi-static
      distribution of local fields — the canonical zero-field lineshape, with a
      characteristic dip and recovery to a one-third tail. Static and dynamic
      variants are documented at :ref:`fit-kubo-toyabe`.

   muonium
   Mu
      The bound state of a positive muon and an electron, :math:`\mathrm{Mu} =
      \mu^{+}e^{-}` — a light isotope of hydrogen whose spin dynamics are
      governed by the muon–electron hyperfine coupling. See :ref:`fit-muonium`.

   F–μ–F
      A muon bound between two fluorine nuclei, whose entangled dipolar
      three-spin state produces a distinctive, coherent zero-field oscillation
      that fingerprints the muon stopping site in a fluoride. See
      :ref:`fit-nuclear-dipolar`.

   RRF
   rotating reference frame
      A display transform that demodulates a fast transverse-field precession
      down to a slow beat about a chosen reference frequency, making the
      relaxation envelope legible without changing the fitted physics. See
      :doc:`/reference/rotating_frame`.

   TF
   transverse field
      A geometry in which the applied field is perpendicular to the initial
      muon-spin direction, so the spins precess and the asymmetry oscillates;
      the standard configuration for measuring precession frequencies and α.

   ZF
   zero field
      A geometry with no applied field, so the muon senses only the internal
      fields of the sample — the configuration for detecting spontaneous
      magnetic order and nuclear-dipolar structure.

   LF
   longitudinal field
      A geometry in which the applied field is parallel to the initial
      muon-spin direction, used to decouple the spin from static internal
      fields and thereby separate static from dynamic behaviour.

   t0
   time-zero
      The bin at which the muons arrive and the polarisation is maximal, from
      which the analysis clock is measured. An error in t0 biases early-time
      amplitudes and phases, so Asymmetry locates it explicitly; see
      :doc:`/reference/data_reduction/t0_search`.

   dead time
      The short interval after a detected event during which a detector cannot
      register a second one, which suppresses counts at high rate. Asymmetry
      corrects for dead time during reduction, using the values recorded
      alongside the detector grouping; see :doc:`/reference/detector_grouping`.

   good-bin range
      The span of time bins retained for analysis — after t0 and before the
      counts fall into noise — excluding the pre-pulse baseline and the
      exhausted tail.

   detector group
      A set of physical detectors summed together into one forward or backward
      histogram. The grouping defines which detectors oppose which and is where
      α and dead time are recorded; see :doc:`/reference/detector_grouping`.

   apodisation
      Multiplying the time signal by a smooth window before the Fourier
      transform, which suppresses the ringing (spectral leakage) that an abrupt
      cut-off would introduce, at the cost of some broadening. See
      :doc:`/reference/fourier_analysis`.

   MaxEnt
   maximum entropy
      A spectral estimator that reconstructs the frequency spectrum as the least
      structured distribution consistent with the data, giving sharper lines
      than a plain Fourier transform on short or noisy records. See
      :doc:`/reference/fourier_analysis`.

   penetration depth
      The length :math:`\lambda` over which a magnetic field is screened at the
      surface of a superconductor. Its temperature dependence, extracted from
      the transverse-field lineshape in the vortex state, reports on the
      superconducting gap structure; see :doc:`/reference/sc_penetration_depth`.

   superfluid density
      The density of paired charge carriers in a superconductor, proportional to
      :math:`1/\lambda^{2}`. Its temperature dependence distinguishes gap
      symmetries and is a standard target of vortex-state μSR.

   pulsed source
      A muon source that delivers muons in intense, widely spaced pulses (as at
      ISIS). The pulse width sets the highest resolvable precession frequency,
      but the low background between pulses favours slow relaxation and
      longer-lived signals. Contrast with a continuous source.

   continuous source
      A muon source that delivers muons one at a time (as at PSI). A single
      muon is timed individually, giving fine time resolution at the cost of a
      rate-limited data-collection window. Contrast with a pulsed source.

   decimation
      Thinning the points drawn on screen when a trace contains more samples
      than the plot can usefully show, so that panning and zooming stay
      responsive. Decimation affects only the display — fits always use the
      full data — and the plot shows a chip whenever it is active.

   representation
      In Asymmetry, the way the loaded data is currently expressed for viewing
      and fitting — F-B asymmetry, individual detector groups, the integral
      scan, or a frequency spectrum. Switching representation is a single click
      and does not alter the underlying reduction.

   integral scan
      An Asymmetry representation that reduces each run in a series to a single
      time-integrated asymmetry value and plots it against the swept variable —
      the basis of ALC and repolarisation scans; see
      :doc:`/reference/alc_mode`.

   project
   .asymp
      The Asymmetry project file, which persists a whole analysis session — the
      loaded datasets, reduction choices, fit setups, and trends — so that
      reopening it reproduces the state exactly. Raw count arrays are not
      embedded; see :doc:`/reference/project_files`.

   trend fit
      Fitting a physical model to a fitted parameter as a function of a scan
      variable — for example an order parameter against temperature, or the
      penetration depth against a gap model — rather than to a time spectrum.
      See :doc:`/reference/parameter_trending`.
