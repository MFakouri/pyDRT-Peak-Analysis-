# pyDRT-Peak-Analysis

A modified and extended version of
[pyDRTtools](https://github.com/ciuccislab/pyDRTtools)
for Distribution of Relaxation Times (DRT) analysis and peak
deconvolution of Electrochemical Impedance Spectroscopy (EIS) data.

The original pyDRTtools was developed by the Ciucci Lab.
This repository retains the original MIT license and attribution.

## Main Extensions

- Direct import of EIS data from multiple instruments and file formats
- Active-area correction during data import
- Improved frequency and relaxation-time representation
- Automatic and custom regularization-parameter selection
- DRT peak deconvolution
- Automatic extraction of resistance, relaxation time, capacitance,
  peak frequency, and FWHM
- Peak-parameter table directly in the GUI
- Automatic CSV export of DRT, EIS regression, and peak parameters
- Percentage-normalized residual visualization
- Updated GUI and compatibility fixes for newer Python versions

## Installation

Clone or download this repository, then run:

```bash
pip install -r requirements.txt
python launch.py