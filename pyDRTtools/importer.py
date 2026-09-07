"""EIS data import and standardization utilities.

The public function ``read_eis_file`` returns an ``(N, 3)`` float array with
columns: frequency [Hz], Z' and Z''.
"""
from __future__ import annotations

from pathlib import Path
import csv
import io
import re
from typing import Iterable

import numpy as np
from scipy.io import loadmat

_FREQ_ALIASES = {"freq", "frequency"}
_REAL_ALIASES = {
    "zreal", "zre", "zprime", "impedanceprime", "realz", "rez",
    "zprimetotal", "re/ohm", "impedancer/ohm",
}
_IMAG_ALIASES = {
    "zimag", "zimaginary", "zim", "zprimeprime", "impedanceprimeprime",
    "imagz", "imz", "zprimeprimetotal", "im/ohm", "impedancei/ohm",
}
_MAG_ALIASES = {
    "z", "|z|", "abs", "mod", "mag", "magnitude", "modulus",
    "ztotal", "zmod", "zmag", "zabs", "absz", "modz",
    "zmagnitude", "zmodulus", "absolutez",
    "impedance", "totalimpedance", "impedancetotal",
    "impedancemagnitude", "impedancemodulus", "impedanceabs",
    "absoluteimpedance",
    "z/ohm", "|z|/ohm", "ztotal/ohm", "zmod/ohm", "zmag/ohm",
    "zabs/ohm", "absz/ohm", "zmagnitude/ohm", "zmodulus/ohm",
    "impedance/ohm", "totalimpedance/ohm", "impedancetotal/ohm",
    "impedancemagnitude/ohm", "impedancemodulus/ohm",
    "absoluteimpedance/ohm",
}
_PHASE_ALIASES = {
    "phase", "phaseangle", "phaseofz", "zphase", "zangle", "angle",
    "phi", "phiz", "theta",
    "phase/deg", "phase/degree", "phase/degrees",
    "phaseangle/deg", "zphase/deg", "zangle/deg", "angle/deg",
    "phi/deg", "phiz/deg", "theta/deg",
    "phase/rad", "phase/radian", "phase/radians",
    "phaseangle/rad", "zphase/rad", "zangle/rad", "angle/rad",
    "phi/rad", "phiz/rad", "theta/rad",
}


def _normalize_header(token: str) -> tuple[str, bool]:
    token = token.strip().replace("\ufeff", "").lower()
    token = token.replace("Ω", "ohm").replace("ω", "ohm")
    token = (token.replace("″", "primeprime")
                  .replace("′", "prime")
                  .replace("’", "prime")
                  .replace("'", "prime")
                  .replace('"', "primeprime"))

    # Match the MATLAB code: remove parenthesized/bracketed instrument units/suffixes.
    token = re.sub(r"\([^)]*\)", "", token)
    token = re.sub(r"\[[^\]]*\]", "", token).strip()

    negative = False
    if token.startswith("-"):
        negative = True
        token = token[1:]
    elif token.startswith("+"):
        token = token[1:]

    token = re.sub(r"[\s_\-]", "", token)
    return token, negative


def _classify_header(token: str) -> tuple[str | None, bool]:
    norm, negative = _normalize_header(token)
    if norm in _FREQ_ALIASES or norm.startswith("freq"):
        return "freq", False
    if norm in _REAL_ALIASES:
        return "real", False
    if norm in _IMAG_ALIASES:
        return "imag", negative
    if norm in _MAG_ALIASES:
        return "mag", False
    if norm in _PHASE_ALIASES or norm.startswith("phase") or norm.startswith("zphase"):
        return "phase", False
    return None, False


def _phase_is_radians(token: str) -> bool:
    return "rad" in token.lower()


def _split_header(line: str) -> tuple[list[str], str]:
    if "\t" in line:
        return line.split("\t"), "tab"
    if line.count(",") >= 2:
        return line.split(","), "comma"
    if line.count(";") >= 2:
        return line.split(";"), "semicolon"
    return re.split(r"\s{2,}", line.strip()), "multispace"


def _split_data(line: str, delimiter: str) -> list[str]:
    if delimiter == "tab":
        return line.split("\t")
    if delimiter == "comma":
        return line.split(",")
    if delimiter == "semicolon":
        return line.split(";")
    return re.split(r"\s+", line.strip())


def _parse_number(token: str) -> float:
    token = token.strip().replace('"', "").replace("\xa0", "")
    try:
        return float(token)
    except ValueError:
        return np.nan


def _read_intelligent_text(path: Path) -> np.ndarray:
    # utf-8-sig consumes BOM; errors='replace' makes vendor files more tolerant.
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    lines = re.split(r"\r\n|\n|\r", text)

    header_line = None
    freq_idx = real_idx = imag_idx = mag_idx = phase_idx = None
    negative_imag = False
    phase_radians = False
    delimiter = None

    for i, line in enumerate(lines):
        tokens, this_delimiter = _split_header(line)
        if len(tokens) < 3:
            continue

        tf = tr = ti = tm = tp = None
        tneg = False
        trad = False

        for j, token in enumerate(tokens):
            kind, neg = _classify_header(token)
            if kind == "freq" and tf is None:
                tf = j
            elif kind == "real" and tr is None:
                tr = j
            elif kind == "imag" and ti is None:
                ti = j
                tneg = neg
            elif kind == "mag" and tm is None:
                tm = j
            elif kind == "phase" and tp is None:
                tp = j
                trad = _phase_is_radians(token)

        has_cartesian = tf is not None and tr is not None and ti is not None
        has_polar = tf is not None and tm is not None and tp is not None

        if has_cartesian or has_polar:
            header_line = i
            freq_idx = tf
            delimiter = this_delimiter

            if has_cartesian:
                real_idx = tr
                imag_idx = ti
                negative_imag = tneg
            else:
                mag_idx = tm
                phase_idx = tp
                phase_radians = trad
            break

    if header_line is None:
        raise ValueError(
            "Could not identify valid EIS data.\n\n"
            "Supported file formats: .mat, .txt, .csv, .z, .dat, .dta, .mpr, .ism, .zmx, .pssession, .tdms\n\n"
            "Text-based files must contain either Frequency + Zreal + Zimag "
            "or Frequency + |Z|/Magnitude + Phase columns."
        )

    used_indices = [freq_idx]
    if real_idx is not None:
        used_indices.extend([real_idx, imag_idx])
    else:
        used_indices.extend([mag_idx, phase_idx])
    max_idx = max(used_indices)

    rows: list[list[float]] = []
    for line in lines[header_line + 1:]:
        if not line.strip():
            continue

        tokens = _split_data(line, delimiter)
        if len(tokens) <= max_idx:
            continue

        f = _parse_number(tokens[freq_idx])

        if real_idx is not None:
            zr = _parse_number(tokens[real_idx])
            zi = _parse_number(tokens[imag_idx])
            if negative_imag:
                zi = -zi
        else:
            magnitude = _parse_number(tokens[mag_idx])
            phase = _parse_number(tokens[phase_idx])

            if phase_radians:
                phase_rad = phase
            else:
                phase_rad = np.deg2rad(phase)

            zr = magnitude * np.cos(phase_rad)
            zi = magnitude * np.sin(phase_rad)

        if np.isfinite(f) and np.isfinite(zr) and np.isfinite(zi):
            rows.append([f, zr, zi])

    if not rows:
        raise ValueError("The required columns were found, but no numeric EIS rows could be read.")
    return np.asarray(rows, dtype=float)


def _read_plain_numeric(path: Path) -> np.ndarray:
    """Backward-compatible loader for classic 3-column DRTtools text/CSV files."""
    ext = path.suffix.lower()
    if ext == ".csv":
        arr = np.genfromtxt(path, delimiter=",")
    else:
        try:
            arr = np.loadtxt(path)
        except Exception:
            # Handle decimal formatting in legacy input files.
            text = path.read_text(encoding="utf-8-sig", errors="replace").replace(", ", ".")
            arr = np.loadtxt(io.StringIO(text))
    arr = np.atleast_2d(np.asarray(arr, dtype=float))
    if arr.shape[1] < 3:
        raise ValueError("The imported data must contain Frequency, Zreal, and Zimag.")
    return arr[:, :3]



def _eis_from_dataframe(df) -> np.ndarray:
    """Extract [frequency, Zreal, Zimag] from a native-reader DataFrame.

    This helper is used only by the additional native readers below.  Existing
    text, MAT, BioLogic MPR, and Zahner ISM import behavior is unchanged.
    """
    if df is None or getattr(df, "empty", True):
        raise ValueError("Empty measurement table.")

    freq_col = real_col = imag_col = mag_col = phase_col = complex_col = None
    negative_imag = False
    phase_radians = False

    for col in df.columns:
        name = str(col)
        kind, neg = _classify_header(name)
        norm, _ = _normalize_header(name)

        # Native libraries occasionally expose slightly different column labels
        # than text exports.  Keep these additions local to native-reader tables.
        if kind is None:
            if norm.startswith(("frequency", "freq")):
                kind = "freq"
            elif norm.startswith(("zreal", "zre", "realz", "rez")):
                kind = "real"
            elif norm.startswith(("zimag", "zim", "imagz", "imz")):
                kind = "imag"

        if kind == "freq" and freq_col is None:
            freq_col = col
        elif kind == "real" and real_col is None:
            real_col = col
        elif kind == "imag" and imag_col is None:
            imag_col = col
            negative_imag = neg
        elif kind == "mag" and mag_col is None:
            mag_col = col
        elif kind == "phase" and phase_col is None:
            phase_col = col
            phase_radians = _phase_is_radians(name)

        # Some native APIs expose impedance as one complex-valued column.
        if complex_col is None:
            try:
                values = np.asarray(df[col])
                if np.iscomplexobj(values) and (
                    norm == "z" or "impedance" in norm or norm.startswith("z")
                ):
                    complex_col = col
            except Exception:
                pass

    if freq_col is None:
        raise ValueError("No frequency column was found in the measurement table.")

    freq = np.asarray(df[freq_col], dtype=float).reshape(-1)

    if real_col is not None and imag_col is not None:
        zr = np.asarray(df[real_col], dtype=float).reshape(-1)
        zi = np.asarray(df[imag_col], dtype=float).reshape(-1)
        if negative_imag:
            zi = -zi
    elif complex_col is not None:
        impedance = np.asarray(df[complex_col], dtype=complex).reshape(-1)
        zr = impedance.real
        zi = impedance.imag
    elif mag_col is not None and phase_col is not None:
        magnitude = np.asarray(df[mag_col], dtype=float).reshape(-1)
        phase = np.asarray(df[phase_col], dtype=float).reshape(-1)
        phase_rad = phase if phase_radians else np.deg2rad(phase)
        zr = magnitude * np.cos(phase_rad)
        zi = magnitude * np.sin(phase_rad)
    else:
        raise ValueError(
            "Could not identify Zreal/Zimag, complex impedance, or Magnitude/Phase columns."
        )

    if not (freq.size == zr.size == zi.size):
        raise ValueError("Native-reader EIS columns do not have equal lengths.")

    return np.column_stack([freq, zr, zi])


def _read_gamry_dta(path: Path) -> np.ndarray:
    """Read Gamry Framework .dta EIS data, with the existing ASCII parser as fallback."""
    try:
        import gamry_parser as gp_module

        gp = gp_module.GamryParser()
        gp.load(filename=str(path))
        datasets = []
        count = int(getattr(gp, "curve_count", 0) or 0)

        for i in range(1, count + 1):
            try:
                data = _eis_from_dataframe(gp.curve(i))
                if data.size:
                    datasets.append(data)
            except Exception:
                continue

        if datasets:
            return np.vstack(datasets)
    except Exception:
        pass

    # DTA is ASCII, so preserve the importer behavior that already worked before
    # gamry-parser support was added.
    return _read_intelligent_text(path)


def _read_zahner_zmx(path: Path) -> np.ndarray:
    """Read Zahner .zmx EIS data as [frequency, Zreal, Zimag]."""
    try:
        import zahner_link as zl
    except ImportError as exc:
        raise ImportError(
            "Zahner .zmx support requires the 'zahner-link' package. "
            "Install it with: pip install zahner-link"
        ) from exc

    importer = zl.xml.ZXmlImporter()
    measurement = importer.import_from_file_as_measurement(str(path))
    datasets = []

    for dataset in measurement.get_datasets():
        try:
            freq = np.asarray(dataset.get_frequencies(), dtype=float).reshape(-1)
            impedance = np.asarray(
                dataset.get_impedance_data().get_calculated_complex_impedance_track(),
                dtype=complex,
            ).reshape(-1)
        except Exception:
            continue

        if freq.size and freq.size == impedance.size:
            datasets.append(np.column_stack([freq, impedance.real, impedance.imag]))

    if not datasets:
        raise ValueError("No readable EIS dataset was found in the Zahner ZMX file.")

    return np.vstack(datasets)


def _read_palmsens_session(path: Path) -> np.ndarray:
    """Read PalmSens PSTrace .pssession EIS data as [frequency, Zreal, Zimag]."""
    try:
        import pypalmsens as ps
    except ImportError as exc:
        raise ImportError(
            "PalmSens .pssession support requires the 'pypalmsens' package. "
            "Install it with: pip install pypalmsens"
        ) from exc

    measurements = ps.load_session_file(path)
    datasets = []

    for measurement in measurements:
        # Prefer the explicit EIS representation when the native API provides it.
        try:
            eis_data = getattr(measurement, "eis_data", None)
            if eis_data is not None and hasattr(eis_data, "to_dataframe"):
                data = _eis_from_dataframe(eis_data.to_dataframe())
                if data.size:
                    datasets.append(data)
                    continue
        except Exception:
            pass

        # Fall back to the raw measurement dataset and identify its EIS columns.
        try:
            dataset = getattr(measurement, "dataset", None)
            if dataset is not None and hasattr(dataset, "to_dataframe"):
                data = _eis_from_dataframe(dataset.to_dataframe())
                if data.size:
                    datasets.append(data)
        except Exception:
            continue

    if not datasets:
        raise ValueError("PalmSens session does not contain identifiable EIS data.")

    return np.vstack(datasets)


def _read_tdms_eis(path: Path) -> np.ndarray:
    """Read EIS channels from a TDMS file, including I.E.D./Nordic EC4 exports."""
    try:
        from nptdms import TdmsFile
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "TDMS support requires the 'nptdms' package. "
            "Install it with: pip install nptdms"
        ) from exc

    tdms = TdmsFile.read(str(path))
    datasets = []

    for group in tdms.groups():
        columns = {}
        for channel in group.channels():
            try:
                columns[str(channel.name)] = channel[:]
            except Exception:
                continue

        if not columns:
            continue

        try:
            data = _eis_from_dataframe(pd.DataFrame(columns))
            if data.size:
                datasets.append(data)
        except Exception:
            continue

    if not datasets:
        raise ValueError("TDMS file does not contain identifiable EIS channels.")

    return np.vstack(datasets)


def _read_biologic_mpr(path: Path) -> np.ndarray:
    """Read BioLogic EC-Lab .mpr EIS data as [frequency, Zreal, Zimag]."""
    try:
        from galvani import BioLogic
    except ImportError as exc:
        raise ImportError(
            "BioLogic .mpr support requires the 'galvani' package. "
            "Install it with: pip install galvani"
        ) from exc

    mpr = BioLogic.MPRfile(str(path))
    names = set(mpr.data.dtype.names or ())
    required = {"freq/Hz", "Re(Z)/Ohm", "-Im(Z)/Ohm"}
    missing = required - names
    if missing:
        raise ValueError(
            "BioLogic .mpr file does not contain the required EIS columns: "
            + ", ".join(sorted(missing))
        )

    freq = np.asarray(mpr.data["freq/Hz"], dtype=float).reshape(-1)
    zr = np.asarray(mpr.data["Re(Z)/Ohm"], dtype=float).reshape(-1)
    # BioLogic stores -Im(Z); the importer standard is Zimag.
    zi = -np.asarray(mpr.data["-Im(Z)/Ohm"], dtype=float).reshape(-1)
    return np.column_stack([freq, zr, zi])


def _read_zahner_ism(path: Path) -> np.ndarray:
    """Read Zahner .ism EIS data as [frequency, Zreal, Zimag]."""
    try:
        from zahner_analysis.file_import.ism_import import IsmImport
    except ImportError as exc:
        raise ImportError(
            "Zahner .ism support requires the 'zahner_analysis' package. "
            "Install it with: pip install zahner_analysis"
        ) from exc

    ism = IsmImport(str(path))
    freq = np.asarray(ism.getFrequencyArray(), dtype=float).reshape(-1)
    impedance = np.asarray(ism.getComplexImpedanceArray(), dtype=complex).reshape(-1)

    if freq.size != impedance.size:
        raise ValueError("Zahner .ism frequency and impedance arrays do not have equal lengths.")

    return np.column_stack([freq, impedance.real, impedance.imag])


def read_eis_file(filename: str | Path) -> np.ndarray:
    """Read and standardize an EIS file to [frequency, Zreal, Zimag]."""
    path = Path(filename)
    ext = path.suffix.lower()

    if ext == ".mpr":
        data = _read_biologic_mpr(path)

    elif ext == ".ism":
        data = _read_zahner_ism(path)

    elif ext == ".zmx":
        data = _read_zahner_zmx(path)

    elif ext == ".pssession":
        data = _read_palmsens_session(path)

    elif ext == ".tdms":
        data = _read_tdms_eis(path)

    elif ext == ".dta":
        data = _read_gamry_dta(path)

    elif ext == ".mat":
        mat = loadmat(path, squeeze_me=True)
        try:
            freq = np.asarray(mat["freq"], dtype=float).reshape(-1)
            zr = np.asarray(mat["Z_prime"], dtype=float).reshape(-1)
            zi = np.asarray(mat["Z_double_prime"], dtype=float).reshape(-1)
        except KeyError as exc:
            raise ValueError("MAT file must contain freq, Z_prime, and Z_double_prime.") from exc
        if not (freq.size == zr.size == zi.size):
            raise ValueError("MAT file EIS vectors do not have equal lengths.")
        data = np.column_stack([freq, zr, zi])

    elif ext in {".txt", ".csv", ".z", ".dat"}:
        try:
            data = _read_intelligent_text(path)
        except Exception as intelligent_error:
            if ext not in {".txt", ".csv"}:
                raise intelligent_error
            try:
                data = _read_plain_numeric(path)
            except Exception:
                raise intelligent_error
    else:
        raise ValueError(
            f"Unsupported file type: {ext}\n\n"
            "Supported file formats: .mat, .txt, .csv, .z, .dat, .dta, .mpr, .ism, .zmx, .pssession, .tdms"
        )

    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data[:, :3]).all(axis=1)]
    data = data[data[:, 0] != 0]
    if data.size == 0:
        raise ValueError("No valid EIS data rows were found.")

    # Match MATLAB DRTtools: descending frequency order.
    if data[0, 0] < data[-1, 0]:
        data = data[::-1].copy()
    return data[:, :3]


def apply_active_area(data: np.ndarray, area_cm2: float) -> np.ndarray:
    area = float(area_cm2)
    if not np.isfinite(area) or area <= 0:
        raise ValueError("Active area must be a positive finite number.")
    out = np.asarray(data, dtype=float).copy()
    if out.ndim != 2 or out.shape[1] < 3:
        raise ValueError("The imported data must contain Frequency, Zreal, and Zimag.")
    out[:, 1:3] *= area
    return out
