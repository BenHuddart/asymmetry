Example data
============

Download the raw histograms for each analysis from the Rigi school archive or
the ISIS DataGateway. The workflow pages give the run selections, calibrations,
and fit settings used for the figures.

.. _rigi-school-data:

Rigi school data
----------------

On the `WiMDA homepage <https://shadow.nd.rl.ac.uk/wimda/>`__, download the
**2026 ISIS/PSI Muon School at Rigi Kaltbad** archive (ZIP, 2.2 GB).
Francis L. Pratt prepared the Rigi School training data.

Extract the archive and open the ``Wimda Workshop`` folder. The paths below
are relative to that folder. Use the instrument prefix as well as the run
number when locating ISIS files. For example, the Basics folder has both a
MuSR run 44989 for detector grouping and an EMU run 44989 for the
steering-current trend. Load the MuSR file in one tab, then use **File →
New Project** to open the EMU files in a separate tab.

EMU runs 18854, 19489, 51315, and 20721, and HiFi run 9023 open with
the instrument's **Longitudinal** preset. Their legacy files declare all-zero
grouping tables. For EMU, the desktop **LOG** reports this warning.

   The file’s detector-grouping table contains only zeros. Applied the standard
   EMU forward/backward grouping.

The HiFi warning names HiFi. To change the grouping, open **Grouping**, choose a
different **Preset** or edit the detectors in **Detector Layout…**, then click
**Apply** and save the project. See the :ref:`loading reference
<legacy-v1-zero-grouping>` for the conditions and recorded provenance.

.. list-table::
   :header-rows: 1
   :widths: 33 37 30

   * - Example
     - Folder
     - Raw runs
   * - :doc:`EMU calibration and grouping <calibration_grouping_emu>`
     - ``Basics/data/``
     - MuSR 44989. EMU 18850, 18854, 34998, and 44989–44997.
   * - :doc:`Copper diffusion and QLCR <dynamic_kt_copper>`
     - ``Nuclear magnetism and ionic motion/Muon diffusion and QLCR in copper/Data/``
     - EMU 20882–20917 and ARGUS 76924–76961.
   * - :doc:`Photo-μSR in silicon <photomusr_silicon_periods>`
     - ``Semiconductors/Photo-muSR in silicon/Data/``
     - HiFi 103277–103298 for the two-period analysis. HiFi 103299 for
       transverse-field calibration.
   * - :doc:`TCNQ resonance <alc_scan_tcnq>`
     - ``Chemistry/ALC resonance in TCNQ/Data/``
     - EMU 19485–19612, including the 350 K scan at 19489–19519.
   * - :doc:`Corannulene resonance <alc_scan_tcnq>`
     - ``Chemistry/Molecular dynamics of corannulene/data/``
     - HiFi 118133–118515, including the wide 40 K scan at 118259–118416.
   * - :doc:`EuO magnetic ordering <temperature_scan_magnetism>`
     - ``Magnetism/Magnetic ordering in EuO/data/``
     - PSI GPS ``.bin`` files 2923–2960 for the zero-field series.
   * - :doc:`Ferromagnetic nickel <temperature_scan_magnetism>`
     - ``Magnetism/Ferromagnetic nickel/Data/``
     - EMU 124218–124278, including the order-parameter series at
       124227–124236.
   * - :doc:`Garnet electrolyte (Al-LLZ) <global_fit_ionic_motion>`
     - ``Nuclear magnetism and ionic motion/Ionic motion in a solid electrolyte/Data/``
     - EMU 51315 for calibration and 51341–51379 for the temperature
       triplets.
   * - :doc:`PTFE <fmuf_entangled_states>`
     - ``Nuclear magnetism and ionic motion/The FmuF state in PTFE/Data/``
     - MuSR 17293 for calibration and 17294–17322 for the zero-field scan.
   * - :ref:`BiSCCO linewidths <biscco-linewidth>`
     - ``Superconductivity/A high-Tc cuprate/Data/``
     - MuSR 1276–1289 at 400 G and 1290, 1292–1294, and 1296–1301 at 200 G.
       Runs 1276 (125 K) and 1290 (120 K) are the respective normal-state references.
   * - :doc:`LiFeAs penetration depth <superconductor_penetration_depth>`
     - ``Superconductivity/LiFeAs/data/``
     - PSI GPS ``.bin`` files 3366–3373 for the 40 mT Sample 1 temperature scan.
   * - :doc:`Ca₃Co₂O₆ magnetic plateau <lf_decoupling_dynamics>`
     - ``Magnetism/Dynamics in a magnetic plateau system/Data/``
     - HiFi 9023–9051, including the 15 K field scan at 9031–9051.
   * - :ref:`CdS shallow donor <cds-shallow-donor>`
     - ``Semiconductors/Shallow donor state in cadmium sulphide/Data/``
     - EMU 20721 (5.175 K) and 20729 (50 K) for the CdS FFT and MaxEnt comparison.
   * - :ref:`High-field κ-Cl <kappa-cl-high-field>`
     - ``Magnetism/AFM transition in high TF/data/``
     - PSI HAL9500 ``.mdu`` files 686 and 693 for the 6 T FFT overlay.
       MaxEnt uses run 686.

ISIS DataGateway
----------------

- :ref:`Molecular antiferromagnet <molecular-antiferromagnet>` uses MuSR
  runs 17094–17104. Open **Download the ISIS runs** in that section for
  the individual files and the calibration run.
- The `Re₆Zr dataset <https://doi.org/10.5286/ISIS.E.24088625>`__ contains
  MuSR runs 38176–38275. See the
  :ref:`Re₆Zr temperature scans <re6zr-penetration-depth>` for the linewidth
  trends and the :ref:`longitudinal-field example <re6zr-lf>` for the
  zero-field and 10 mT spectra.
