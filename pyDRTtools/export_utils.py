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


def export_peak_parameters(entry, output_folder, original_name):
    if getattr(entry, 'method', '') != 'peak':
        return None
    df = compute_peak_summary(entry)
    if df.empty:
        return None
    path = Path(output_folder)/f"{original_name} DRT parameters.csv"
    export_df = df.rename(columns={'ID': 'Peak_ID'})
    export_df.to_csv(path, index=False)
    return path


def export_fitting_parameters(entry, output_folder, original_name):
    fit = getattr(entry, 'cnls_fit', None)
    if not fit:
        return None

    rows = [
        {
            'Component': 'L',
            'Value': fit['L'],
            'R': np.nan,
            'C': np.nan,
            'Frequency (Hz)': np.nan,
            'Tau (s)': np.nan,
        },
        {
            'Component': 'R',
            'Value': fit['R'],
            'R': np.nan,
            'C': np.nan,
            'Frequency (Hz)': np.nan,
            'Tau (s)': np.nan,
        },
    ]
    for rc in fit['rc']:
        rows.append({
            'Component': rc['component'],
            'Value': np.nan,
            'R': rc['R'],
            'C': rc['C'],
            'Frequency (Hz)': rc['frequency'],
            'Tau (s)': rc['tau'],
        })

    path = Path(output_folder)/f"{original_name} fitting parameters.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
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
        if entry.method in {'simple','credit','peak'}:
            w.writerow(['L', getattr(entry,'L',0)])
            w.writerow(['R', getattr(entry,'R',0)])
        elif entry.method == 'BHT':
            w.writerow(['L', entry.mu_L_0])
            w.writerow(['R', entry.mu_R_inf])

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
    path = Path(path)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
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
        elif entry.method != 'none':
            w.writerow(['freq','mu_Z_re','mu_Z_im','Z_re_res','Z_im_res'])
            for vals in zip(entry.freq, entry.mu_Z_re, entry.mu_Z_im, entry.res_re, entry.res_im):
                w.writerow(vals)
