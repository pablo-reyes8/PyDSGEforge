import matplotlib.pyplot as plt
import numpy as np 
from src.inference.likelihoods import *


def plot_mcmc_diagnostics(
    draws_work: np.ndarray,
    registry: ParamRegistry,
    subset: Union[int, Sequence[str]],
    *,
    bins: int = 20,
    burn_in: int = 0,
    figsize=(10, 4)):
    
    """
    Diagnósticos MCMC por parámetro: histograma + traza.

    Parámetros
    ----------
    draws_work : np.ndarray
        Matriz (R × k) con los draws en el espacio de trabajo.
    registry : ParamRegistry
        Registro que permite convertir valores al espacio económico.
    subset : int o list[str]
        - Si es un entero N, se grafican los primeros N parámetros según el orden del registro.
        - Si es una lista de nombres, se usan esas claves explícitamente.
    bins : int, opcional
        Número de bins del histograma.
    burn_in : int, opcional
        Número de draws iniciales a descartar.
    figsize : tuple, opcional
        Tamaño del gráfico por parámetro.
    """
    draws = np.asarray(draws_work, dtype=float)
    if burn_in > 0:
        if burn_in >= draws.shape[0]:
            raise ValueError("burn_in no puede ser mayor o igual al número total de draws.")
        draws = draws[burn_in:]

    if isinstance(subset, int):
        names = registry.names[:subset]
    else:
        names = list(subset)

    for name in names:
        try:
            econ_vals = registry.to_econ_dict_subset(draws, names=[name])
        except KeyError:
            print(f"Parámetro '{name}' no encontrado en el registro; se omite.")
            continue

        series = econ_vals[name].to_numpy().reshape(-1)
        fig, axes = plt.subplots(ncols=2, figsize=figsize)
        fig.suptitle(name, fontweight="bold")

        axes[0].hist(series, bins=bins, color="#4c78a8", alpha=0.8, edgecolor="white")
        axes[0].set_title("Posterior")
        axes[0].set_xlabel(name)
        axes[0].set_ylabel("Frecuencia")

        axes[1].plot(series, color="#f58518", linewidth=0.8)
        axes[1].set_title("Trace")
        axes[1].set_xlabel("Iteración")
        axes[1].set_ylabel(name)

        plt.tight_layout()
        plt.show()