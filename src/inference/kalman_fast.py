from __future__ import annotations

try:
    from numba import njit

    HAS_NUMBA = True
except Exception:  # pragma: no cover - depends on optional dependency
    njit = None
    HAS_NUMBA = False


if HAS_NUMBA:  # pragma: no cover - exercised only when numba is installed
    import numpy as np

    @njit(cache=True)
    def _kalman_loglike_numba(y, Theta1, C, Theta0, Q, H, Psi0, Psi2, s_bar, P_bar):
        T = y.shape[0]
        n_y = y.shape[1]
        n_s = Theta1.shape[0]
        const = n_y * np.log(2.0 * np.pi)
        ll_sum = 0.0
        eye = np.eye(n_s)

        for t in range(T):
            s_hat = C + Theta1 @ s_bar
            P_hat = Theta1 @ P_bar @ Theta1.T + Theta0 @ Q @ Theta0.T
            P_hat = 0.5 * (P_hat + P_hat.T)

            e_t = y[t, :] - Psi0 - Psi2 @ s_hat
            S_t = Psi2 @ P_hat @ Psi2.T + H

            try:
                L = np.linalg.cholesky(S_t)
            except Exception:
                return -np.inf

            logdet = 0.0
            for j in range(n_y):
                logdet += 2.0 * np.log(L[j, j])

            z = np.linalg.solve(L, e_t)
            Sinv_e = np.linalg.solve(L.T, z)

            ZP = Psi2 @ P_hat.T
            tmp = np.linalg.solve(L, ZP)
            K = np.linalg.solve(L.T, tmp).T

            s_bar = s_hat + K @ e_t
            I_KZ = eye - K @ Psi2
            P_bar = I_KZ @ P_hat @ I_KZ.T + K @ H @ K.T
            P_bar = 0.5 * (P_bar + P_bar.T)

            ll_sum += -0.5 * (const + logdet + e_t @ Sinv_e)

        return ll_sum


def kalman_loglike_numba(y, Theta1, C, Theta0, Q, H, Psi0, Psi2, s_bar, P_bar):
    if not HAS_NUMBA:
        return None
    return float(_kalman_loglike_numba(y, Theta1, C, Theta0, Q, H, Psi0, Psi2, s_bar, P_bar))
