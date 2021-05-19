Overview
========


SAR Calibration Tool (SCT) is the tool for the radiometric and geometric calibration of SAR data.

SCT can be used to analyse SAR products acquired over calibration sites, i.e. locations where calibration targets, like transponders or corner reflectors, are installed, to derive a set of useful information for each target, i.e.:

- Range and azimuth resolutions
- Range and azimuth Absolute Localization Errors (ALE)
- Radar Cross Section (RCS) and Signal to Clutter Ratio (SCR)
- Incidence, look and squint angles of observation

and more.

These information are collected in a database and can be used to support accurate radiometric and geometric calibration of a given SAR mission data.

In particular, ALE measurements are performed applying both instrument-related corrections (bistatic delay, Doppler shift, ...) and the most relevant geophysical corrections (plate tectonics and Solid Earth Tides (SET) displacements, ionospheric and tropospheric path delays).

Note: The current version of the tool is able to manage only Sentinel-1 data and related orbit files (Earth Explorer format), but it has been designed to be easily extended to other SAR mission data in the future.


Table of contents
-----------------

.. toctree::
   :maxdepth: 2

   install
   notebooks/index
   api/index
   changelog


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Citing
------

If you use SAR Calibration Tool, please add a citation:

    *SCT: SAR Calibration Tool, https://github.com/aresys-srl/sar-calibration-tool*

