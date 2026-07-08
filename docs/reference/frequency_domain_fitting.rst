Frequency-domain fitting
========================

Frequency-domain fitting extends the same single-fit, global-fit, and
parameter-trending workflow used in the time domain to displayed Fourier
spectra.  The V1 workflow fits the real-valued spectrum currently shown in the
Frequency view.  It does not fit the complex FFT directly.

The spectrum you fit can come from either quantitative estimator — the FFT or
maximum entropy; pick one with :ref:`choosing-spectral-estimator`. (The Burg
*Resolution* view is a line-count diagnostic, not a fit target.) Computing and
conditioning that spectrum in the first place is covered by
:doc:`fourier_analysis`.

Workflow
--------

1. Compute a Fourier spectrum from the **Fourier** panel.
2. Switch to the **Frequency** workspace.
3. Open the **Fit** dock.
4. Fit the displayed spectrum with a Gaussian or Lorentzian peak plus a
   constant or linear background.
5. For a run series, select multiple runs with cached spectra and use the
   **Global** tab.
6. Inspect ``nu0`` and ``fwhm`` in the **Parameters** dock, alongside derived
   ``B0`` and ``Bwid`` field equivalents.

The fitting x axis is stored internally as absolute frequency in MHz.  Plotting
controls may show field in gauss or a reference-relative frequency axis, but fit
parameters remain canonical: ``nu0`` and ``fwhm`` are MHz quantities.

Fit range and seeding
----------------------

The fit range (``≤ ν ≤``, in MHz) restricts the fit to a band of the spectrum.
As in the time domain, the band is drawn on the plot as a shaded span with
dashed edges, and either edge can be dragged directly on the spectrum or typed
into the range fields.  When the frequency axis is displayed in gauss or
relative to a reference field the span follows the displayed units, but the
range is always entered and stored as absolute MHz.

Peak parameters are seeded automatically from the displayed spectrum: the
centre ``nu0``, height, and width are read from the dominant peak of the run
being fitted, so **Preview** draws the peak in place before you fit.  Because
the peak position tracks each run's applied field, these seeds are re-derived
for every run rather than carried across a run series.

When the model carries more than one peak (add a second ``GaussianPeak`` or
``LorentzianPeak`` from the fit-function builder), each peak component is
seeded from a distinct line in the spectrum — the strongest detected peak
seeds the first component, the next strongest the second, and so on.  Adding a
peak component is read as *"a line exists here"*, so a weak-but-real shoulder
is seeded rather than gated out.  If the spectrum shows fewer lines than the
model declares, the surplus components are spread across the fit window so they
stay visible in the preview instead of collapsing to an off-screen default.

Available components
--------------------

The fit-function builder is filtered by analysis domain: when fitting a
spectrum it offers only the frequency-domain components below (as a flat
list), and these components do not appear when fitting in the time domain.
Typing a component name from the other domain gives an explanatory error.

``GaussianPeak``
    Peak height, centre ``nu0``, and full width at half maximum ``fwhm``.

``LorentzianPeak``
    Peak height, centre ``nu0``, and full width at half maximum ``fwhm``.

``ConstantBackground``
    Flat spectral background ``bg``.

``LinearBackground``
    Background ``bg + slope * nu``.

Global fits and trends
----------------------

Global frequency-domain fitting uses the same parameter-role table as
time-domain global fitting.  Mark peak centre or width as ``Local`` to trend
them across a field or temperature series, or mark background terms as
``Global`` when they should be shared.

Successful global frequency fits are sent to the **Parameters** dock under the
``Frequency Domain`` group.  The parameter-trending tools can then fit
``nu0(T)``, ``fwhm(B)``, ``B0(T)``, or ``Bwid(B)`` using the usual trend-model
workflow.

Project files
-------------

Project files store frequency-fit state separately from time-domain fit state.
This lets a project reopen with both a time-domain model and a spectral peak
model intact.  Cached Fourier spectra are still stored in the Fourier spectrum
state; raw detector arrays remain referenced by source-file path rather than
embedded in the project.

.. note::

   The Gaussian and Lorentzian peak forms fitted here are the ordinary
   line shapes; the minimiser and its statistics are the shared engine
   documented in :doc:`fitting`. Because the fit target is the *displayed*
   real spectrum, an apodised or baseline-subtracted spectrum carries those
   conditioning choices into the fitted width and amplitude — see the
   apodisation caveat in :doc:`fourier_analysis` and the conditioning steps in
   :doc:`frequency_finishers`. This page therefore adds no new physics of its
   own; the underlying references are those of :doc:`fitting` and
   :doc:`fourier_analysis`.
