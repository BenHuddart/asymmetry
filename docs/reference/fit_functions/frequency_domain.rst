.. _fit-frequency-domain:

Frequency domain
================

These components fit the real-valued Fourier spectrum displayed in the
Frequency workspace, as a function of frequency :math:`\nu` in MHz (see
:doc:`../frequency_domain_fitting` for the workflow). They are only offered
when fitting in the frequency domain, where the picker presents them as a
flat list.

The physical motivation mirrors the time domain through the Fourier-transform
pairs of the standard relaxation envelopes: an exponentially relaxing
(dynamically broadened) signal :math:`e^{-\lambda t}` gives a Lorentzian line
of width :math:`\mathrm{FWHM} = \lambda/\pi`, while a Gaussian envelope
:math:`e^{-(\sigma t)^2}` (static, dense field distribution) gives a Gaussian
line of width :math:`\mathrm{FWHM} = 2\sigma\sqrt{\ln 2}/\pi`. Both peaks are
parameterised **directly by their full width at half maximum** — the
``fwhm`` parameter is the literal width of the line on the plot, which is why
the factors :math:`4\ln 2` and :math:`4` appear in the expressions below. A
fitted width therefore converts back to a time-domain rate via
:math:`\lambda = \pi\,\mathrm{FWHM}` or
:math:`\sigma = \pi\,\mathrm{FWHM}/(2\sqrt{\ln 2})`, and a peak centre to a
local field via :math:`B_0 = \nu_0/(\gamma_\mu/2\pi)`. Peak heights are in
the arbitrary units of the displayed spectrum (they depend on apodisation and
normalisation), so physical conclusions should rest on positions and widths
rather than absolute heights.

.. _fit-gaussian-peak:

GaussianPeak
------------

.. math::

   S(\nu) = h\,\exp\!\left[-4\ln 2\,
   \frac{(\nu-\nu_0)^2}{w^2}\right],
   \qquad w \equiv \mathrm{FWHM}

A Gaussian spectral line — appropriate when the underlying time-domain
envelope is Gaussian (static, dense field distribution): the spectral-domain
counterpart of :ref:`fit-gaussian`. The width :math:`w` is the literal full
width at half maximum, :math:`S(\nu_0 \pm w/2) = h/2` exactly.

==========  ==============  =====  ============================================
Name        Symbol          Unit   Description
==========  ==============  =====  ============================================
``height``  :math:`h`       a.u.   Peak height.
``nu0``     :math:`\nu_0`   MHz    Peak centre.
``fwhm``    :math:`w`       MHz    Full width at half maximum.
==========  ==============  =====  ============================================

.. _fit-lorentzian-peak:

LorentzianPeak
--------------

.. math::

   S(\nu) = \frac{h}{1 + 4\,(\nu-\nu_0)^2/w^2},
   \qquad w \equiv \mathrm{FWHM}

A Lorentzian spectral line — appropriate when the time-domain envelope is
exponential (dynamic broadening, or a dilute static field distribution): the
spectral-domain counterpart of :ref:`fit-exponential`. Parameters as for
``GaussianPeak``, with the same half-height convention. Note the heavy
Lorentzian tails: fit windows should extend several FWHM beyond the peak or
the width will be underestimated; a line that is neither Gaussian nor
Lorentzian usually signals overlapping sites, which are better fitted as two
peaks.

.. _fit-constant-background:

ConstantBackground
------------------

.. math::

   S(\nu) = b_g

A flat spectral baseline, absorbing white noise and the flat part of any
apodisation pedestal. Parameter: ``bg`` (a.u.).

.. _fit-linear-background:

LinearBackground
----------------

.. math::

   S(\nu) = b_g + m\,\nu

A linearly sloped baseline for spectra with a gentle trend across the fit
window, e.g. the shoulder of a distant intense line or low-frequency
1/f-like leakage. Parameters: ``bg`` (a.u.), ``slope`` (a.u./MHz). Prefer
``ConstantBackground`` unless the slope is clearly visible — the slope is
strongly correlated with the peak parameters in narrow windows.
