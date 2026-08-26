"""Small non-negative QP wrapper.

Uses CVXOPT when available (matching pyDRTtools), with a SciPy L-BFGS-B
fallback so the toolbox remains usable on Python installations where a
CVXOPT wheel is unavailable.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

try:
    from cvxopt import matrix, solvers  # type: ignore
    HAVE_CVXOPT = True
    solvers.options['show_progress'] = False
except Exception:
    matrix = solvers = None
    HAVE_CVXOPT = False


def solve_nonnegative_qp(H, c, x0=None):
    """Minimize 0.5*x.T@H@x + c.T@x subject to x >= 0."""
    H = np.asarray(H, dtype=float)
    c = np.asarray(c, dtype=float).reshape(-1)
    H = 0.5 * (H + H.T)
    n = c.size

    if HAVE_CVXOPT:
        G = matrix(-np.eye(n))
        h = matrix(np.zeros(n))
        sol = solvers.qp(matrix(H), matrix(c), G, h)
        status = str(sol.get('status', ''))
        if status not in {'optimal', 'optimal_inaccurate'}:
            raise RuntimeError(f"CVXOPT QP failed: {status}")
        return np.asarray(sol['x'], dtype=float).reshape(-1)

    if x0 is None:
        x0 = np.ones(n, dtype=float)
    else:
        x0 = np.maximum(np.asarray(x0, dtype=float).reshape(-1), 0.0)

    def fun(x):
        return 0.5 * float(x @ H @ x) + float(c @ x)

    def jac(x):
        return H @ x + c

    res = minimize(
        fun,
        x0,
        jac=jac,
        method='L-BFGS-B',
        bounds=[(0.0, None)] * n,
        options={'ftol': 1e-14, 'gtol': 1e-10, 'maxiter': 100000, 'maxls': 50},
    )
    if not res.success:
        # SLSQP is slower but can be more forgiving for ill-conditioned cases.
        res2 = minimize(
            fun,
            np.maximum(res.x, 0),
            jac=jac,
            method='SLSQP',
            bounds=[(0.0, None)] * n,
            options={'ftol': 1e-12, 'maxiter': 100000, 'disp': False},
        )
        if res2.success or res2.fun <= res.fun:
            res = res2
    if not np.all(np.isfinite(res.x)):
        raise RuntimeError(f"Non-negative QP failed: {res.message}")
    return np.maximum(np.asarray(res.x, dtype=float), 0.0)
