# pyDRT Peak Analysis

An integrated Python GUI for EIS data import, Distribution of Relaxation Times (DRT) analysis, peak deconvolution, and automated extraction of peak parameters.
This repository provides an extended and modified version of pyDRTtools in which data conversion, DRT analysis, peak analysis, and parameter export are integrated into a single GUI.
<p align="center">
  <img src="tutorial/Nyquist.png" width="430">
</p>
<p align="center">
  Nyquist plot.
</p>

<p align="center">
  <img src="tutorial/DRT-Peaks.png" width="600">
</p>
<p align="center">
  DRT peaks.
</p>

## Main Features

- Direct import of EIS data exported from different instruments, including BioLogic (.mpr), Zahner (.ism), Gamry (.dta), Scribner/ZView/ZPlot (.z), MATLAB (.mat), and generic text/data files (.txt, .csv, .dat).
- Automatic conversion and standardization of imported EIS files for use in pyDRTtools.
- Optional active-area correction for impedance data reported in Ω.
- Direct use of data already normalized in Ω·cm² without additional area correction.
- DRT calculation and visualization within the same GUI.
- Peak deconvolution directly from the DRT curve.
- Automated extraction of the following parameters for each DRT peak:
  - Polarization resistance (R)
  - Capacitance
  - Relaxation time (τ)
  - Peak frequency
  - FWHM (Full Width at Half Maximum)
- Automatic export of the peak parameters to a CSV file.
- Automatic saving of the resulting figure.
- Standalone Windows executable available; no Python or MATLAB installation or additional configuration required.


## Installation

### Windows 10/11 (64-bit) - easiest source-code setup

1. Download the repository ZIP and extract it to a normal folder.
2. Double-click `run_windows.bat`.
3. On the first run, the setup script automatically:
   - downloads the official **CPython 3.13.15 64-bit** installer from `python.org` if Python 3.13 is not already available;
   - verifies the Python installer SHA-256 checksum before running it;
   - installs Python for the current Windows user (no system-wide Python installation is required);
   - creates an isolated `.venv` environment inside the project folder;
   - installs every package listed in `requirements.txt`; and
   - runs dependency and numerical smoke tests.
4. After setup, use `run_windows.bat` whenever you want to launch the GUI.

An internet connection is required for the first source-code installation. MATLAB is not required. The automatic installer is intended for Windows 10/11 x64.

### Manual Python installation

For manual installation, use **Python 3.11-3.13** (Python 3.13 is recommended):

```bash
python -m venv .venv
```

Activate the environment and install the dependencies:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Then launch the GUI:

```bash
python launch.py
```

The same dependencies are also declared in `setup.py`, so `pip install .` installs the complete runtime dependency set, including BioLogic `.mpr` (`galvani`) and Zahner `.ism` (`zahner-analysis`) support.

## Original pyDRTtools

This software is based on and extends the original **pyDRTtools** Python toolbox developed by the Ciucci Lab.

Original pyDRTtools repository:
https://github.com/ciuccislab/pyDRTtools

pyDRTtools is distributed under the MIT License.

Users are encouraged to cite the primary DRTtools/pyDRTtools reference:

T. H. Wan, M. Saccoccio, C. Chen, and F. Ciucci,
“Influence of the discretization methods on the distribution of relaxation times deconvolution: implementing radial basis functions with DRTtools,”
*Electrochimica Acta*, vol. 184, pp. 483–499, 2015.

https://doi.org/10.1016/j.electacta.2015.09.097

## Contact and Citation

Developed and modified by:

**Masood Fakouri Hasanabadi**  
Email: [fakourih@ualberta.ca](mailto:fakourih@ualberta.ca)

If you use this software in your research, please cite this GitHub repository using the **“Cite this repository”** option available on GitHub:

https://github.com/MFakouri/pyDRT-Peak-Analysis-

Please cite the software in any publications or presentations in which it is used.
Users should also cite the original pyDRTtools/DRTtools publication listed above when appropriate.
