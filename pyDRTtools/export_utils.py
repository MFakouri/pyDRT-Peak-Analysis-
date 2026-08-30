"""Utilities for exporting DRT, EIS, and peak-analysis results."""
from __future__ import annotations

from pathlib import Path
import csv
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid


def _series_from_entry(entry):
    """Return Gamma-vs-tau series for peak-parameter export.

    The first series is the total DRT, followed by the individual peak curves.
    """
    tau = np.asarray(entry.out_tau_vec, dtype=float).reshape(-1)
    gamma = np.asarray(entry.gamma, dtype=float).reshape(-1)
    series = [("DRT", tau, gamma)]
    if getattr(entry, 'method', '') == 'peak' and hasattr(entry, 'gamma_gauss_mat'):
        mat = np.asarray(entry.gamma_gauss_mat, dtype=float)
        for i in range(mat.shape[1]):
            series.append((f"Peak_{i+1}", tau, mat[:, i]))
    return series


def _integrated_area(x, y):
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x, y = x[mask], y[mask]
    if x.size < 2:
        return np.nan
    idx = np.argsort(x)
    xs, ys = x[idx], y[idx]
    return float(trapezoid(ys, np.log(xs)))


def compute_peak_summary(entry):
    """Port of computePeakSummary(..., ignoreFirst=true) from MATLAB."""
    series = _series_from_entry(entry)
    if len(series) <= 1:
        return pd.DataFrame(columns=[
            'ID','Series_Name','Area_Ohm_cm2','R_Ohm_cm2','Tau_s',
            'Capacitance_F_per_cm2','Frequency_Hz','FWHM_lnTau'
        ])

    all_R = np.array([_integrated_area(x, y) for _, x, y in series], dtype=float)
    R_total = float(np.nansum(all_R[1:]))
    R_cell_1 = float(all_R[0]) if all_R.size else np.nan
    R_corr_ratio = 1.0
    if R_total != 0 and np.isfinite(R_cell_1) and np.isfinite(R_total):
        R_corr_ratio = R_cell_1/R_total

    rows = []
    for i, (name, x, y) in enumerate(series[1:], start=1):
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
        x, y = x[mask], y[mask]
        area = all_R[i]
        tau_pk = cap = freq_hz = fwhm = np.nan
        if x.size >= 2:
            idx = np.argsort(x)
            xs, ys = x[idx], y[idx]
            imax = int(np.argmax(ys))
            tau_pk = float(xs[imax])
            peak_val = float(ys[imax])
            if np.isfinite(peak_val) and peak_val > 0:
                half = peak_val/2.0
                ln_xs = np.log(xs)
                left_candidates = np.where(ys[:imax+1] <= half)[0]
                right_rel = np.where(ys[imax:] <= half)[0]
                if left_candidates.size and right_rel.size:
                    ileft = int(left_candidates[-1])
                    iright = int(imax + right_rel[0])
                    if ileft < imax and iright > imax:
                        y1, y2 = ys[ileft], ys[ileft+1]
                        if y2 != y1:
                            ln_left = ln_xs[ileft] + (half-y1)*(ln_xs[ileft+1]-ln_xs[ileft])/(y2-y1)
                        else:
                            ln_left = ln_xs[ileft]
                        y1, y2 = ys[iright-1], ys[iright]
                        if y2 != y1:
                            ln_right = ln_xs[iright-1] + (half-y1)*(ln_xs[iright]-ln_xs[iright-1])/(y2-y1)
                        else:
                            ln_right = ln_xs[iright]
                        fwhm = float(ln_right-ln_left)
            if np.isfinite(tau_pk) and tau_pk > 0:
                freq_hz = float(1/(2*np.pi*tau_pk))
            if np.isfinite(area) and area != 0:
                R_corr = area*R_corr_ratio
                if R_corr != 0:
                    cap = float(tau_pk/R_corr)
        rows.append({
            'Series_Name': name,
            'Area_Ohm_cm2': area,
            'R_Ohm_cm2': area*R_corr_ratio if np.isfinite(area) else np.nan,
            'Tau_s': tau_pk,
            'Capacitance_F_per_cm2': cap,
            'Frequency_Hz': freq_hz,
            'FWHM_lnTau': fwhm,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values('Frequency_Hz', ascending=False, na_position='last').reset_index(drop=True)
        df.insert(0, 'ID', np.arange(1, len(df)+1))
        total = pd.DataFrame([{
            'ID': len(df)+1, 'Series_Name': 'Total Sum',
            'Area_Ohm_cm2': df['Area_Ohm_cm2'].sum(skipna=True),
            'R_Ohm_cm2': df['R_Ohm_cm2'].sum(skipna=True),
            'Tau_s': np.nan, 'Capacitance_F_per_cm2': np.nan,
            'Frequency_Hz': np.nan, 'FWHM_lnTau': np.nan,
        }])
        df = pd.concat([df, total], ignore_index=True)
    return df


def _finite_or_blank(value):
    """Return a plain Python float for CSV, or an empty string for non-finite values."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ''
    return value if np.isfinite(value) else ''


def export_parameters_csv(entry, output_folder, original_name):
    """Export DRT and bounded-CNLS parameters to one CSV file.

    The CSV layout follows the requested combined parameter table: DRT values
    first, then the fitted RC values. Peak/RC numbering is always assigned
    from the highest characteristic frequency to the lowest.
    """
    if getattr(entry, 'method', '') != 'peak':
        return None

    drt = compute_peak_summary(entry)
    if drt.empty:
        return None

    peaks = drt[drt['Series_Name'] != 'Total Sum'].copy()
    peaks = peaks.sort_values(
        'Frequency_Hz', ascending=False, na_position='last'
    ).reset_index(drop=True)

    rows = []

    # DRT section. Column 1 is intentionally left blank under the section title,
    # matching the requested row/column arrangement.
    rows.append(['DRT parameters', '', '', '', '', '', '', ''])
    rows.append([
        '', 'Component', 'Value', 'Tau (s)', 'Frequency (Hz)',
        'R (Ohm cm2)', 'Capacitance (F/cm2)', 'FWHM (LnTau)'
    ])
    rows.append(['', 'L (H cm2)', _finite_or_blank(getattr(entry, 'L', np.nan)), '', '', '', '', ''])
    rows.append(['', 'R_ohmic (Ohm cm2)', _finite_or_blank(getattr(entry, 'R', np.nan)), '', '', '', '', ''])

    for i, row in peaks.iterrows():
        rows.append([
            '', f'Peak_{i+1}', '',
            _finite_or_blank(row['Tau_s']),
            _finite_or_blank(row['Frequency_Hz']),
            _finite_or_blank(row['R_Ohm_cm2']),
            _finite_or_blank(row['Capacitance_F_per_cm2']),
            _finite_or_blank(row['FWHM_lnTau']),
        ])

    # Three blank rows between DRT and fitting sections.
    rows.extend([[''] * 8 for _ in range(3)])

    rows.append(['Fitting parameters', '', '', '', '', '', '', ''])
    rows.append([
        '', 'Component', 'Value', 'Tau (s)', 'Frequency (Hz)',
        'R (Ohm cm2)', 'Capacitance (F/cm2)', ''
    ])

    fit = getattr(entry, 'cnls_fit', None)
    if fit:
        rows.append(['', 'L (H cm2)', _finite_or_blank(fit.get('L')), '', '', '', '', ''])
        rows.append(['', 'R_ohmic (Ohm cm2)', _finite_or_blank(fit.get('R')), '', '', '', '', ''])

        rc_rows = sorted(
            fit.get('rc', []),
            key=lambda rc: float(rc.get('frequency', -np.inf)),
            reverse=True,
        )
        for i, rc in enumerate(rc_rows):
            rows.append([
                '', f'RC{i+1}', '',
                _finite_or_blank(rc.get('tau')),
                _finite_or_blank(rc.get('frequency')),
                _finite_or_blank(rc.get('R')),
                _finite_or_blank(rc.get('C')),
                '',
            ])

    path = Path(output_folder) / f"{original_name} parameters.csv"
    with path.open('w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerows(rows)
    return path


def export_drt_csv(entry, path, drt_type='Gamma vs Tau'):
    path = Path(path)
    tau = np.asarray(entry.out_tau_vec, dtype=float).reshape(-1)
    omega_like = 1.0/tau
    freq_hz = omega_like/(2*np.pi)
    gamma = np.asarray(entry.gamma, dtype=float).reshape(-1) if hasattr(entry, 'gamma') else None

    if drt_type == 'Gamma vs Tau':
        x, xname, mult = tau, 'tau', 1.0
        yname = 'gamma(tau)'
    elif drt_type == 'Gamma vs Frequency':
        x, xname, mult = freq_hz, 'freq', 1.0
        yname = 'gamma(freq)'
    elif drt_type == 'g vs Tau':
        x, xname, mult = tau, 'tau', omega_like
        yname = 'g(tau)'
    else:
        x, xname, mult = freq_hz, 'freq', omega_like
        yname = 'g(freq)'

    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if entry.method == 'simple':
            w.writerow([xname, yname])
            for vals in zip(x, gamma*mult): w.writerow(vals)
        elif entry.method == 'credit':
            labels = [xname, f'MAP {yname.split("(")[0].strip()}', f'Mean {yname.split("(")[0].strip()}',
                      f'Upperbound {yname.split("(")[0].strip()}', f'Lowerbound {yname.split("(")[0].strip()}']
            w.writerow(labels)
            for vals in zip(x, gamma*mult, np.asarray(entry.mean)*mult,
                            np.asarray(entry.upper_bound)*mult, np.asarray(entry.lower_bound)*mult): w.writerow(vals)
        elif entry.method == 'BHT':
            base = 'gamma' if 'Gamma' in drt_type else 'g'
            w.writerow([xname, f'{base}_Re', f'{base}_Im'])
            for vals in zip(x, np.asarray(entry.mu_gamma_fine_re)*mult, np.asarray(entry.mu_gamma_fine_im)*mult): w.writerow(vals)
        elif entry.method == 'peak':
            prefix = 'gamma_gauss' if 'Gamma' in drt_type else 'g_gauss'
            headers = [xname, yname] + [f'{prefix}_{i+1}' for i in range(entry.N_peaks)]
            w.writerow(headers)
            peaks = np.asarray(entry.gamma_gauss_mat)*np.asarray(mult).reshape(-1,1) if np.ndim(mult) else np.asarray(entry.gamma_gauss_mat)*mult
            for i in range(x.size):
                w.writerow([x[i], (gamma*mult)[i], *peaks[i,:].tolist()])


def export_eis_csv(entry, path):
    """Export EIS regression data using the requested Input/DRT/Fitting layout.

    For the DRT section, the existing regression values (mu_Z_*) are kept.
    DRT and bounded-CNLS residuals are exported as percentages using the same
    normalization as the GUI residual plots: residual / |Z_exp| * 100.
    """
    path = Path(path)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)

        # Keep the existing BHT-specific export untouched.
        if entry.method == 'BHT':
            for key in ['s_res_re','s_res_im']:
                w.writerow([key, *np.asarray(entry.out_scores[key]).reshape(-1).tolist()])
            for key in ['s_mu_re','s_mu_im','s_HD_re','s_HD_im','s_JSD_re','s_JSD_im']:
                w.writerow([key, entry.out_scores[key]])
            w.writerow(['freq','mu_Z_re','mu_Z_im','Z_H_re','Z_H_im','Z_H_re_band','Z_H_im_band','Z_H_re_res','Z_H_im_res'])
            for vals in zip(entry.freq, entry.mu_Z_re, entry.mu_Z_im, entry.mu_Z_H_re_agm,
                            entry.mu_Z_H_im_agm, entry.band_re_agm, entry.band_im_agm,
                            entry.res_H_re, entry.res_H_im):
                w.writerow(vals)
            return path

        if entry.method == 'none':
            return None

        freq = np.asarray(entry.freq, dtype=float).reshape(-1)
        Z_input = np.asarray(entry.Z_exp, dtype=complex).reshape(-1)
        drt_re = np.asarray(entry.mu_Z_re, dtype=float).reshape(-1)
        drt_im = np.asarray(entry.mu_Z_im, dtype=float).reshape(-1)
        drt_res_re = np.asarray(entry.res_re, dtype=float).reshape(-1)
        drt_res_im = np.asarray(entry.res_im, dtype=float).reshape(-1)

        # Match the Re/Im residual plots in the GUI: residual / |Z_exp| * 100.
        mod = np.abs(Z_input)
        drt_res_re = np.divide(
            drt_res_re, mod,
            out=np.zeros_like(drt_res_re, dtype=float),
            where=mod != 0
        ) * 100
        drt_res_im = np.divide(
            drt_res_im, mod,
            out=np.zeros_like(drt_res_im, dtype=float),
            where=mod != 0
        ) * 100

        n = freq.size
        arrays = [Z_input, drt_re, drt_im, drt_res_re, drt_res_im]
        if any(np.asarray(a).size != n for a in arrays):
            raise ValueError("EIS regression arrays have inconsistent lengths.")

        # Requested two-line grouped header. Residual columns are percentages
        # normalized by |Z_exp|, matching the GUI Re/Im residual plots.
        w.writerow(['', 'Input', '', 'DRT', '', '', '', 'Fitting', '', '', ''])
        w.writerow([
            'freq',
            'Z_re (Ohm cm2)', 'Z_im (Ohm cm2)',
            'Z_re (Ohm cm2)', 'Z_im (Ohm cm2)', 'Z_re_res (%)', 'Z_im_res (%)',
            'Z_re (Ohm cm2)', 'Z_im (Ohm cm2)', 'Z_re_res (%)', 'Z_im_res (%)',
        ])

        # Fitting columns are populated only when RC/CNLS fitting has already
        # been completed.  Peak analysis creates entry.cnls_fit automatically.
        fit = getattr(entry, 'cnls_fit', None)
        fit_re = fit_im = fit_res_re = fit_res_im = None
        if fit:
            fit_freq = np.asarray(fit.get('freq', []), dtype=float).reshape(-1)
            Z_fit = np.asarray(fit.get('Z_fit', []), dtype=complex).reshape(-1)
            Z_fit_input = np.asarray(fit.get('Z_exp', []), dtype=complex).reshape(-1)

            if fit_freq.size != n or Z_fit.size != n:
                raise ValueError("RC fitting and DRT regression frequency arrays have inconsistent lengths.")
            if not np.allclose(fit_freq, freq, rtol=1e-10, atol=1e-12):
                raise ValueError("RC fitting frequencies do not match the EIS regression frequencies.")
            if Z_fit_input.size == n and not np.allclose(Z_fit_input, Z_input, rtol=1e-10, atol=1e-12):
                raise ValueError("RC fitting input impedance does not match the EIS input impedance.")

            fit_re = np.real(Z_fit)
            fit_im = np.imag(Z_fit)
            fit_res_re_raw = fit_re - np.real(Z_input)
            fit_res_im_raw = fit_im - np.imag(Z_input)
            fit_res_re = np.divide(
                fit_res_re_raw, mod,
                out=np.zeros_like(fit_res_re_raw, dtype=float),
                where=mod != 0
            ) * 100
            fit_res_im = np.divide(
                fit_res_im_raw, mod,
                out=np.zeros_like(fit_res_im_raw, dtype=float),
                where=mod != 0
            ) * 100

        for i in range(n):
            row = [
                freq[i],
                np.real(Z_input[i]), np.imag(Z_input[i]),
                drt_re[i], drt_im[i], drt_res_re[i], drt_res_im[i],
            ]
            if fit_re is None:
                row.extend(['', '', '', ''])
            else:
                row.extend([fit_re[i], fit_im[i], fit_res_re[i], fit_res_im[i]])
            w.writerow(row)

    return path

