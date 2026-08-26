"""Headless test for automatic lambda selection and boundary diagnostics."""
import contextlib
import io
from pathlib import Path
import numpy as np
import pandas as pd
from pyDRTtools import EIS_object, simple_run
from pyDRTtools import basics

root = Path(__file__).resolve().parent
sample = pd.read_csv(root / 'tutorial' / 'data' / '1ZARC.csv')
f = sample['Freq'].to_numpy(float)
zre = sample['Real'].to_numpy(float)
zim = sample['Imag'].to_numpy(float)

values = []
for entered_lambda in (1e-3, 1e-1):
    entry = EIS_object(f.copy(), zre.copy(), zim.copy())
    with contextlib.redirect_stdout(io.StringIO()):
        entry = simple_run(
            entry,
            rbf_type='Gaussian',
            data_used='Combined Re-Im Data',
            induct_used=1,
            der_used='2nd order',
            cv_type='GCV',
            reg_param=entered_lambda,
            shape_control='FWHM Coefficient',
            coeff=0.5,
        )
    values.append(float(np.asarray(entry.lambda_value).reshape(-1)[0]))

assert np.isfinite(values).all()
assert np.isclose(values[0], values[1], rtol=1e-5, atol=1e-12), values
assert not basics.optimal_lambda.last_diagnostics['boundary_hit']
print('PASS - automatic lambda is independent of the custom lambda entry')
print('PASS - sample GCV optimum is an interior solution')
print('GCV lambda:', values[0])
