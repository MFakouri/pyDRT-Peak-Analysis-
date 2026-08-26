"""Headless numerical smoke test; does not require PyQt5."""
import numpy as np
from pyDRTtools import EIS_object, simple_run, peak_analysis

f = np.logspace(4, -1, 18)
w = 2*np.pi*f
R0, Rp, tau_true = 0.1, 0.6, 0.02
Z = R0 + Rp/(1 + 1j*w*tau_true)
entry = EIS_object(f, Z.real, Z.imag)
entry = simple_run(entry, rbf_type='Gaussian', data_used='Combined Re-Im Data',
                   induct_used=1, der_used='2nd order', cv_type='custom',
                   reg_param=1e-3, shape_control='FWHM Coefficient', coeff=0.5)
entry = peak_analysis(entry, rbf_type='Gaussian', data_used='Combined Re-Im Data',
                      induct_used=1, der_used='2nd order', cv_type='custom',
                      reg_param=1e-3, shape_control='FWHM Coefficient', coeff=0.5,
                      N_peaks=1)
f_peak = 1/(2*np.pi*entry.out_tau_vec[np.argmax(entry.gamma)])
assert np.isfinite(f_peak)
assert entry.method == 'peak'
print('PASS')
print('Recovered peak frequency [Hz]:', f_peak)
print('Expected RC peak frequency [Hz]:', 1/(2*np.pi*tau_true))
