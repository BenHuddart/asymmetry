.. _fit-background:

Background
==========

.. _fit-constant:

Constant
--------

.. math::

   A(t) = A_{\mathrm{bg}}

A time-independent background asymmetry, used additively in essentially
every composite model. Physically it accounts for muons that stop outside
the sample — silver sample holders, cryostat walls and tails — and for any
residual detector imbalance that survives the α calibration. Because all
relaxation functions satisfy :math:`G(0) = 1`, the asymmetry amplitudes and
the background obey the sum rule :math:`\sum_i A_i + A_{\mathrm{bg}} = A_0`
with :math:`A_0` the calibrated total asymmetry; a fitted background that
drifts across a run series usually signals a changing beam spot rather than
sample physics, and is worth fixing from a calibration measurement.

==========  =======================  =====  ====================================
Name        Symbol                   Unit   Description
==========  =======================  =====  ====================================
``A_bg``    :math:`A_{\mathrm{bg}}`  %      Time-independent background level.
==========  =======================  =====  ====================================

Note the distinction from a *relaxing* background: muons stopping in silver
relax so slowly (:math:`\lambda \lesssim 0.01\;\mu s^{-1}`) that a constant
is almost always adequate, but a holder containing nuclear moments may need
its own slow ``Gaussian`` or ``StaticGKT_ZF`` term instead.
