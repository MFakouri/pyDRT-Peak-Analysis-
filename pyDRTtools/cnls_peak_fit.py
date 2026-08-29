"""CNLS fitting driven by DRT peak-deconvolution parameters.

Current behavior:
- The number of RC branches comes from the requested number of peaks.
- Initial R, C and frequency for every RC branch come directly from the
  peak-deconvolution table, ordered high frequency -> low frequency.
- Peak frequency is fitted inside a frequency-dependent interval. For each
  table frequency f0:
      delta(f0) = 0.30000540010800214 - 5.400108002160044e-05*f0
      b = log10(f0)
      endpoint_low  = 10**(b - 2*delta(f0))
      endpoint_high = 10**(b + delta(f0))
  and the optimizer interval is the sorted pair of these two endpoints.
- R and C are allowed to change. R starts from the table resistance. Frequency
  starts from the table frequency and is fitted inside the interval above.
  C is determined at every optimizer evaluation from the physical RC relation
      C = 1/(2*pi*f*R)
  so R, C and f remain mutually consistent.
- Series inductance L and series resistance R are also fitted.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from .export_utils import compute_peak_summary


_DELTA_SLOPE = -5.400108002160044e-05
_DELTA_INTERCEPT = 0.30000540010800214
_LOWER_DELTA_MULTIPLIER = 2.0
_LOG_R_SCALE_LIMIT = 50.0


def frequency_bounds_from_table_frequency(frequency_hz: float) -> tuple[float, float]:
    """Return the requested frequency-dependent log-domain interval.

    The user's rule is applied directly to the table frequency f:
        delta(f) = 0.30000540010800214 - 5.400108002160044e-05*f
        b = log10(f)
        endpoint_1 = 10**(b - 2*delta(f))
        endpoint_2 = 10**(b + delta(f))

    The two endpoints are sorted before being returned so the optimizer always
    receives a valid lower/upper pair, including when delta becomes negative.
    """
    f0 = float(frequency_hz)
    if not np.isfinite(f0) or f0 <= 0.0:
        raise ValueError("Peak-table frequency must be finite and positive.")

    delta = _DELTA_INTERCEPT + _DELTA_SLOPE * f0
    b = np.log10(f0)
    endpoint_1 = 10.0 ** (b - _LOWER_DELTA_MULTIPLIER * delta)
    endpoint_2 = 10.0 ** (b + delta)
    lower = float(min(endpoint_1, endpoint_2))
    upper = float(max(endpoint_1, endpoint_2))
    return lower, upper


def _validate_rc_inputs(
    rc_resistances: np.ndarray,
    rc_capacitances: np.ndarray,
    rc_frequencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Rk = np.asarray(rc_resistances, dtype=float).reshape(-1)
    Ck = np.asarray(rc_capacitances, dtype=float).reshape(-1)
    fk = np.asarray(rc_frequencies, dtype=float).reshape(-1)
    if not (Rk.size == Ck.size == fk.size):
        raise ValueError("RC resistance, capacitance and frequency arrays must have the same length.")
    if Rk.size == 0:
        raise ValueError("At least one RC branch is required for CNLS fitting.")
    for name, values in (("R", Rk), ("C", Ck), ("frequency", fk)):
        if np.any(~np.isfinite(values)) or np.any(values <= 0):
            raise ValueError(f"All RC {name} values must be finite and positive.")
    return Rk, Ck, fk


def _rc_values_from_variables(
    rc_R0: np.ndarray,
    log_r_scales: np.ndarray,
    frequencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return fitted R and physically consistent C for fitted frequencies."""
    R0 = np.asarray(rc_R0, dtype=float).reshape(-1)
    sR = np.asarray(log_r_scales, dtype=float).reshape(-1)
    fk = np.asarray(frequencies, dtype=float).reshape(-1)
    if not (R0.size == sR.size == fk.size):
        raise ValueError("RC resistance, scale and frequency arrays must have the same length.")
    if np.any(fk <= 0.0) or np.any(~np.isfinite(fk)):
        raise ValueError("Fitted RC frequencies must be finite and positive.")

    Rk = R0 * np.exp(sR)
    Ck = 1.0 / (2.0 * np.pi * fk * Rk)
    return Rk, Ck


def impedance_model_variable_frequency(
    omega: np.ndarray,
    L: float,
    R_series: float,
    rc_resistances: np.ndarray,
    rc_capacitances: np.ndarray,
    rc_frequencies: np.ndarray,
) -> np.ndarray:
    """Return series L+R plus parallel RC branches with variable frequencies.

    R, C and f are required to obey f = 1/(2*pi*R*C). The model itself uses
    the standard parallel-RC expression R/(1+j*w*R*C).
    """
    omega = np.asarray(omega, dtype=float).reshape(-1)
    Rk, Ck, fk = _validate_rc_inputs(rc_resistances, rc_capacitances, rc_frequencies)

    f_from_rc = 1.0 / (2.0 * np.pi * Rk * Ck)
    if not np.allclose(f_from_rc, fk, rtol=5e-9, atol=0.0):
        raise ValueError("Fitted R, C and frequency are internally inconsistent.")

    Z = np.full(omega.shape, complex(float(R_series), 0.0), dtype=complex)
    Z += 1j * omega * float(L)
    for R_i, C_i in zip(Rk, Ck):
        Z += R_i / (1.0 + 1j * omega * R_i * C_i)
    return Z


def _decode_optimization_vector(
    x: np.ndarray,
    rc_R0: np.ndarray,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Decode [L, R_series, log_R_scales..., log_frequencies...]."""
    x = np.asarray(x, dtype=float).reshape(-1)
    R0 = np.asarray(rc_R0, dtype=float).reshape(-1)
    n_rc = R0.size
    if x.size != 2 + 2 * n_rc:
        raise ValueError("Optimization vector has the wrong size.")

    L = float(x[0])
    R_series = float(x[1])
    log_r_scales = x[2:2+n_rc]
    frequencies = np.exp(x[2+n_rc:2+2*n_rc])
    Rk, Ck = _rc_values_from_variables(R0, log_r_scales, frequencies)
    return L, R_series, Rk, Ck, frequencies


def normalized_complex_residual(
    x: np.ndarray,
    omega: np.ndarray,
    Z_exp: np.ndarray,
    rc_R0: np.ndarray,
) -> np.ndarray:
    """Real/imag residual vector normalized by |Z_exp|, as in the MATLAB code."""
    L, R_series, Rk, Ck, fk = _decode_optimization_vector(x, rc_R0)
    Z_model = impedance_model_variable_frequency(
        omega, L, R_series, Rk, Ck, fk
    )

    denom = np.abs(Z_exp)
    denom = np.where(denom == 0.0, np.finfo(float).eps, denom)
    delta = Z_model - Z_exp
    return np.concatenate((np.real(delta) / denom, np.imag(delta) / denom))


def rc_parameters_from_peak_deconvolution(entry, expected_n_peaks: int | None = None):
    """Read initial R/C/f from the peak table and build frequency bounds.

    Rows are returned in high-frequency -> low-frequency order. Initial values
    are the values displayed by Peak Analysis; no independent peak extraction
    is performed here.
    """
    df = compute_peak_summary(entry)
    if df.empty:
        raise ValueError("Peak deconvolution did not produce RC parameters.")

    peaks = df[df["Series_Name"] != "Total Sum"].copy()
    peaks = peaks.sort_values(
        "Frequency_Hz", ascending=False, na_position="last"
    ).reset_index(drop=True)

    if expected_n_peaks is not None and len(peaks) != int(expected_n_peaks):
        raise ValueError(
            f"Peak deconvolution returned {len(peaks)} RC branches, but "
            f"{int(expected_n_peaks)} were requested."
        )

    required = ["R_Ohm_cm2", "Capacitance_F_per_cm2", "Frequency_Hz"]
    for col in required:
        vals = np.asarray(peaks[col], dtype=float)
        if vals.size == 0 or not np.all(np.isfinite(vals)) or np.any(vals <= 0):
            raise ValueError(f"All peak-table {col} values must be finite and positive.")

    R0 = np.asarray(peaks["R_Ohm_cm2"], dtype=float)
    C0 = np.asarray(peaks["Capacitance_F_per_cm2"], dtype=float)
    f0 = np.asarray(peaks["Frequency_Hz"], dtype=float)

    # At the initial point, R/C/f should represent the same RC time constant.
    f_from_table_rc = 1.0 / (2.0 * np.pi * R0 * C0)
    if not np.allclose(f_from_table_rc, f0, rtol=5e-9, atol=0.0):
        raise ValueError(
            "Peak-table R and C are inconsistent with the peak-table frequency. "
            "The CNLS initial point requires f = 1/(2*pi*R*C)."
        )

    rows = []
    for i in range(len(peaks)):
        f_lower, f_upper = frequency_bounds_from_table_frequency(f0[i])
        rows.append({
            "component": f"RC{i+1}",
            "R_initial": float(R0[i]),
            "C_initial": float(C0[i]),
            "frequency_initial": float(f0[i]),
            "frequency_lower": float(f_lower),
            "frequency_upper": float(f_upper),
            "R": float(R0[i]),
            "C": float(C0[i]),
            "frequency": float(f0[i]),
            "tau": float(1.0 / (2.0 * np.pi * f0[i])),
        })
    return rows


def fit_with_frequency_bounds(entry, expected_n_peaks: int | None = None):
    """Fit L, series R, and all RC branches with bounded peak frequencies.

    Initial R, C and f come from the Peak Analysis table.  During fitting:
    - each branch R is free and positive (log-scaled around table R),
    - each branch f is free only inside the requested frequency-dependent bounds,
    - each branch C changes consistently according to C=1/(2*pi*f*R),
    - L and series R are free and nonnegative.
    """
    rc_rows = rc_parameters_from_peak_deconvolution(entry, expected_n_peaks)
    R0 = np.array([row["R_initial"] for row in rc_rows], dtype=float)
    C0 = np.array([row["C_initial"] for row in rc_rows], dtype=float)
    f0 = np.array([row["frequency_initial"] for row in rc_rows], dtype=float)
    f_lower = np.array([row["frequency_lower"] for row in rc_rows], dtype=float)
    f_upper = np.array([row["frequency_upper"] for row in rc_rows], dtype=float)

    freq = np.asarray(entry.freq, dtype=float).reshape(-1)
    Z_exp = np.asarray(entry.Z_exp, dtype=complex).reshape(-1)
    if freq.size != Z_exp.size or freq.size == 0:
        raise ValueError("EIS frequency and impedance arrays are unavailable or inconsistent.")
    if np.any(~np.isfinite(freq)) or np.any(freq <= 0):
        raise ValueError("All EIS frequencies must be finite and positive.")
    omega = 2.0 * np.pi * freq

    L0 = float(getattr(entry, "L", 0.0) or 0.0)
    R_series0 = float(getattr(entry, "R", 0.0) or 0.0)
    if not np.isfinite(L0) or L0 < 0:
        L0 = 0.0
    if not np.isfinite(R_series0) or R_series0 < 0:
        R_series0 = 0.0

    # Exact table values are the initial point:
    # log_R_scale=0 -> R=R_table; log(f)=log(f_table).
    log_f0 = np.log(f0)
    log_f_lower = np.log(f_lower)
    log_f_upper = np.log(f_upper)

    x0 = np.concatenate((
        [L0, R_series0],
        np.zeros(len(rc_rows), dtype=float),
        log_f0,
    ))
    lower = np.concatenate((
        [0.0, 0.0],
        np.full(len(rc_rows), -_LOG_R_SCALE_LIMIT),
        log_f_lower,
    ))
    upper = np.concatenate((
        [np.inf, np.inf],
        np.full(len(rc_rows), _LOG_R_SCALE_LIMIT),
        log_f_upper,
    ))

    # If delta(f0)=0, both requested endpoints equal f0. scipy's
    # least_squares requires strict lb < ub, so such a frequency is kept
    # effectively fixed using the smallest practical floating interval.
    equal_f_bounds = ~(log_f_upper > log_f_lower)
    if np.any(equal_f_bounds):
        eps = np.finfo(float).eps * 16.0
        idx0 = 2 + len(rc_rows)
        for i in np.where(equal_f_bounds)[0]:
            lower[idx0+i] = log_f0[i] - eps
            upper[idx0+i] = log_f0[i] + eps

    result = least_squares(
        normalized_complex_residual,
        x0,
        bounds=(lower, upper),
        args=(omega, Z_exp, R0),
        method="trf",
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
        max_nfev=100000,
    )

    L_fit, R_series_fit, R_fit, C_fit, f_fit = _decode_optimization_vector(result.x, R0)

    # Clamp only microscopic roundoff at a bound; optimizer is already bounded.
    f_fit = np.minimum(np.maximum(f_fit, f_lower), f_upper)
    C_fit = 1.0 / (2.0 * np.pi * f_fit * R_fit)

    for i, row in enumerate(rc_rows):
        row["R"] = float(R_fit[i])
        row["C"] = float(C_fit[i])
        row["frequency"] = float(f_fit[i])
        row["tau"] = float(1.0 / (2.0 * np.pi * f_fit[i]))

    Z_fit = impedance_model_variable_frequency(
        omega, L_fit, R_series_fit, R_fit, C_fit, f_fit
    )
    denom = np.abs(Z_exp)
    denom = np.where(denom == 0.0, np.finfo(float).eps, denom)
    residual_real_pct = np.abs(np.real(Z_fit - Z_exp) / denom) * 100.0
    residual_imag_pct = np.abs(np.imag(Z_fit - Z_exp) / denom) * 100.0

    fit = {
        "success": bool(result.success),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "cost": float(result.cost),
        "L": float(L_fit),
        "R": float(R_series_fit),
        "rc": rc_rows,
        "freq": freq.copy(),
        "Z_exp": Z_exp.copy(),
        "Z_fit": Z_fit.copy(),
        "residual_real_pct": residual_real_pct,
        "residual_imag_pct": residual_imag_pct,
    }
    entry.cnls_fit = fit
    return fit


# Compatibility aliases for older integrated GUI code / external imports.
def fit_with_fixed_frequency(entry, expected_n_peaks: int | None = None):
    return fit_with_frequency_bounds(entry, expected_n_peaks)


def fit_lr_with_fixed_rc(entry, expected_n_peaks: int | None = None):
    return fit_with_frequency_bounds(entry, expected_n_peaks)
