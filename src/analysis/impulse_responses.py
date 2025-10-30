import numpy as np 
import matplotlib.pyplot as plt
from src.inference.likelihoods import *

def compute_irfs(
    draws_work,
    registry,
    equations,
    y_t,
    y_tp1,
    eps_t,
    *,
    y_tm1=None,
    eta_t=None,
    measurement=None,
    horizon=20,
    burn_in=0,
    quantiles=(0.16, 0.5, 0.84),
    shock_indices=None,
    observable_names=None,
    shock_names=None,
    div=0.0,):

    draws = np.asarray(draws_work, dtype=float)
    draws = draws[burn_in:] if burn_in > 0 else draws

    n_shocks = len(eps_t)
    shock_indices = shock_indices or range(n_shocks)

    if shock_names is None:
        shock_names = [str(sym) for sym in eps_t]
    else:
        shock_names = list(shock_names)

    Psi2_example = None
    valid_draws = []
    for theta in draws:
        Theta1, Theta0, eu, Psi0, Psi2 = st_sp(theta,
                                    equations,y_t,
                                    y_tp1,
                                    eps_t,
                                    registry,
                                    y_tm1=y_tm1,
                                    eta_t=eta_t,
                                    measurement=measurement,
                                    div=div)
        
        if eu[0] < 1 or eu[1] < 1:
            continue
        try:
            Q = registry.build_Q(theta, n_shocks)
            L = np.linalg.cholesky(Q)
        except np.linalg.LinAlgError:
            continue
        valid_draws.append((theta, Theta1, Theta0, Psi0, Psi2, L))
        if Psi2_example is None:
            Psi2_example = Psi2

    if not valid_draws:
        raise RuntimeError("Ningún draw produce una solución BK válida y SPD en Q.")

    n_obs = Psi2_example.shape[0]
    if observable_names is None:
        observable_names = [str(sym) for sym in y_t[:n_obs]]
    else:
        observable_names = list(observable_names)

    irf_store = {shock_names[j]: [] for j in shock_indices}

    for theta, Theta1, Theta0, Psi0, Psi2, L in valid_draws:
        for j in shock_indices:
            eps_path = np.zeros((n_shocks, horizon))
            # choque unitario (1 std) en t=1
            eps_path[:, 1] = L[:, j]

            state = np.zeros((Theta1.shape[0], horizon))
            obs = np.zeros((n_obs, horizon))
            obs[:, 0] = Psi0  # baseline

            for h in range(1, horizon):
                state[:, h] = Theta1 @ state[:, h - 1] + Theta0 @ eps_path[:, h]
                obs[:, h] = Psi2 @ state[:, h]

            irf_store[shock_names[j]].append(obs)

    results = {}
    quant_array = np.array(quantiles, dtype=float)
    for shock_name, paths in irf_store.items():

        arr = np.stack(paths, axis=0) 
        q_values = np.quantile(arr, quant_array, axis=0)  
        results[shock_name] = {
            "observables": observable_names,
            "horizon": horizon,
            "quantiles": quant_array,
            "summary": q_values,
            "raw": arr,}

    return results


def plot_irf_bands(irf_dict, shocks=None, start=1, figsize=(7, 4), colors=None):

    shocks = shocks or list(irf_dict.keys())
    colors = colors or {
        "median": "#222222",
        "band": "#B0BEC5",}

    for shock in shocks:
        info = irf_dict[shock]
        qvals = info["quantiles"]
        data = info["summary"] 
        obs_names = info["observables"]
        horizon = info["horizon"]
        h_grid = np.arange(horizon)

        low_idx = np.argmin(np.abs(qvals - qvals[0]))
        med_idx = np.argmin(np.abs(qvals - qvals[len(qvals)//2]))
        high_idx = np.argmin(np.abs(qvals - qvals[-1]))

        fig, axes = plt.subplots(len(obs_names), 1, sharex=True, figsize=figsize)
        if len(obs_names) == 1:
            axes = [axes]

        fig.suptitle(f"IRFs to {shock}", fontweight="bold")

        for ax, obs_name, low, med, high in zip(axes,obs_names,
            data[low_idx],
            data[med_idx],
            data[high_idx],):

            ax.plot(h_grid[start:], med[start:], color=colors["median"], linewidth=1.8)
            ax.fill_between(
                h_grid[start:], low[start:], high[start:],
                color=colors["band"], alpha=0.4)
            
            ax.axhline(0.0, color="#888888", linewidth=0.8, linestyle="--")
            ax.set_ylabel(obs_name)

        axes[-1].set_xlabel("Horizonte")
        plt.tight_layout()
        plt.show()