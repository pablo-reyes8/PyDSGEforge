import numpy as np
import sympy as sp 

from src.specification.param_registry_class import * 
from src.specification.param_specifications import *
from src.transformations.param_transformations import *

def _as_residuals(equations):
    """
    Convierte ecuaciones simbólicas en residuales (lhs - rhs).

    - Si `eq` es una igualdad SymPy (sp.Equality), devuelve lhs - rhs.
    - Si `eq` ya es una expresión residual (Expr), la deja igual.

    Retorna
    -------
    list[sp.Expr]
        Lista de expresiones residuales.
    """

    res = []
    for eq in equations:
        if isinstance(eq, sp.Equality):
            res.append(eq.lhs - eq.rhs)
        else:
            res.append(eq)
    return res

def _subs_numeric(expr, subs):
    """
    Sustituye símbolos por valores numéricos en una matriz SymPy.

    Parámetros
    ----------
    expr : sp.Matrix
        Matriz simbólica.
    subs : dict
        Diccionario de sustitución {símbolo: valor}.

    Retorna
    -------
    sp.Matrix
        Matriz con los valores sustituidos (sigue siendo SymPy).
    """
    return expr.subs(subs) if subs else expr


def _to_float_array(M):
    """
    Convierte una matriz SymPy en un array NumPy float64.

    Maneja casos vacíos y vectores columna automáticamente.

    Retorna
    -------
    np.ndarray
        Matriz NumPy equivalente.
    """

    if M.shape == (0, 0):
        return np.zeros((0, 0), dtype=np.float64)
    A = np.array(M.tolist(), dtype=np.float64)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    return A


def build_matrices(
    equations,
    y_t,
    y_tp1=None,
    eps_t=None,
    *,
    y_tm1=None,
    eta_t=None,
    param_values=None,
    steady_values=None,
    measurement=None):
    
    """
    Construye las matrices lineales del modelo DSGE:
        Γ0 y_t = Γ1 y_{t-1} + Ψ eps_t + Π eta_t + c

    Soporta variables adelantadas (t+1), rezagadas (t-1)
    y shocks de expectativas (eta_t). Cuando hay leads,
    arma un estado extendido z_t = [y_t; y_{t+1}] y devuelve
    Γ0_aug, Γ1_aug, etc.
    """

    f = sp.Matrix(_as_residuals(equations))

    if isinstance(param_values, ParamRegistry):
        theta0 = np.zeros(len(param_values.params))
        param_values = param_values.to_sympy_subs(theta0)

    y_curr = sp.Matrix(list(y_t))
    y_lead = sp.Matrix(list(y_tp1)) if y_tp1 else sp.Matrix([])
    y_lag  = sp.Matrix(list(y_tm1)) if y_tm1 else sp.Matrix([])
    eps    = sp.Matrix(list(eps_t)) if eps_t else sp.Matrix([])
    eta    = sp.Matrix(list(eta_t)) if eta_t else sp.Matrix([])

    G0_sym   = f.jacobian(y_curr)
    Glead_sym = f.jacobian(y_lead) if y_lead.shape != (0, 0) else sp.zeros(f.rows, 0)
    Glag_sym  = f.jacobian(y_lag)  if y_lag.shape  != (0, 0) else sp.zeros(f.rows, 0)
    Psi_sym  = f.jacobian(eps)     if eps.shape    != (0, 0) else sp.zeros(f.rows, 0)
    Pi_sym   = f.jacobian(eta)     if eta.shape    != (0, 0) else sp.zeros(f.rows, 0)


    subs = dict(param_values or {})
    if steady_values:
        subs.update(steady_values)
    else:
        subs.update({sym: 0.0 for sym in y_curr})
        subs.update({sym: 0.0 for sym in y_lead})
        subs.update({sym: 0.0 for sym in y_lag})
        subs.update({sym: 0.0 for sym in eps})
        subs.update({sym: 0.0 for sym in eta})

    Gamma0  = _to_float_array(_subs_numeric(G0_sym, subs))
    Gamma_lead = _to_float_array(_subs_numeric(Glead_sym, subs))
    Gamma_lag  = _to_float_array(_subs_numeric(Glag_sym, subs))
    Psi  = _to_float_array(_subs_numeric(Psi_sym, subs))
    Pi  = _to_float_array(_subs_numeric(Pi_sym, subs))
    c   = _to_float_array((-f).subs(subs)).reshape(-1)

    meas = measurement or MeasurementSpec()
    Psi0, Psi2 = meas.build(n_state=len(y_t))

    n_eq = Gamma0.shape[0]
    n    = len(y_t)
    m    = len(y_tp1) if y_tp1 else 0

    if m > 0:
        y_curr_names = [str(sym) for sym in y_t]
        y_lead_names = [str(sym) for sym in y_tp1]

        lead_indices = []
        name_to_idx = {nm: i for i, nm in enumerate(y_curr_names)}
        for nm in y_lead_names:
            if nm in name_to_idx:
                lead_indices.append(name_to_idx[nm])
            else:
                base = nm.replace("tp1", "t").replace("_t1", "_t")
                if base in name_to_idx:
                    lead_indices.append(name_to_idx[base])
                else:
                    raise ValueError(f"No puedo mapear {nm} a una variable actual.")

        # Γ0 extendido: 
        G0_top = np.hstack([Gamma0, -Gamma_lead])
        G0_bot = np.zeros((m, n + m), dtype=np.float64)
        for j, idx_curr in enumerate(lead_indices):
            G0_bot[j, idx_curr] = 1.0
            G0_bot[j, n + j]    = -1.0
        Gamma0_aug = np.vstack([G0_top, G0_bot])

        Gamma1_top = np.zeros((n_eq, n + m), dtype=np.float64)
        if y_tm1:
            name_to_idx = {str(sym): i for i, sym in enumerate(y_t)}
            for col, lag_sym in enumerate(y_tm1):
                base = str(lag_sym).replace("tm1", "t").replace("_t1", "_t")
                if base not in name_to_idx:
                    raise ValueError(f"No puedo mapear la variable rezagada {lag_sym}.")
                Gamma1_top[:, name_to_idx[base]] = Gamma_lag[:, col]
        Gamma1_bot = np.zeros((m, n + m), dtype=np.float64)
        Gamma1_aug = np.vstack([Gamma1_top, Gamma1_bot])

        Psi_aug = np.vstack([Psi, np.zeros((m, Psi.shape[1]))]) if Psi.size else np.zeros((n_eq + m, 0))
        Pi_aug  = np.vstack([Pi, np.zeros((m, Pi.shape[1]))])   if Pi.size else np.zeros((n_eq + m, 0))
        c_aug   = np.concatenate([c, np.zeros(m, dtype=np.float64)])

        Psi2_aug = np.hstack([Psi2, np.zeros((Psi2.shape[0], m), dtype=np.float64)])

        return Gamma0_aug, Gamma1_aug, Psi_aug, Pi_aug, Psi0, Psi2_aug, c_aug

    else:
        Gamma1 = np.zeros_like(Gamma0)
        if y_tm1:
            name_to_idx = {str(sym): i for i, sym in enumerate(y_t)}
            for col, lag_sym in enumerate(y_tm1):
                base = str(lag_sym).replace("tm1", "t").replace("_t1", "_t")
                if base not in name_to_idx:
                    raise ValueError(f"No puedo mapear la variable rezagada {lag_sym}.")
                Gamma1[:, name_to_idx[base]] = Gamma_lag[:, col]
        return Gamma0, Gamma1, Psi, Pi, Psi0, Psi2, c


def dataframe_to_numpy(df, columns):
    """
    Convierte un DataFrame en una matriz NumPy (float64) seleccionando
    columnas específicas en el orden indicado.

    Útil para construir la matriz de observaciones y_t (T × n_y)
    antes de usar un modelo en forma estado-espacio.

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame con las series observadas.

    columns : list[str]
        Lista ordenada con los nombres de las columnas que se desean
        extraer. El orden determina el orden de las columnas en la matriz
        resultante.

    Retorna
    -------
    np.ndarray
        Matriz NumPy de tamaño (T, len(columns)) y tipo float64.

    Errores
    -------
    TypeError
        Si df no es un pandas.DataFrame.

    Ejemplo
    --------
    >>> df = pd.DataFrame({"y": [1, 2], "pi": [3, 4]})
    >>> dataframe_to_numpy(df, ["y", "pi"])
    array([[1., 3.],
           [2., 4.]])
    """

    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df debe ser un pandas.DataFrame")
    y = df.loc[:, columns].to_numpy(dtype=np.float64)
    return y

def measurement_from_registry(
    registry: ParamRegistry,
    theta_work: np.ndarray,
    state_dim: int,) -> MeasurementSpec:
    """
    Construye (Psi0, Psi2) a partir de los valores económicos actuales.
    Si el registro define un measurement_builder, se usa; si no, se devuelve
    Psi2 = I y Psi0 = 0 (observamos directamente los estados).
    """
    econ = registry.to_econ_dict(theta_work)

    if registry.measurement_builder is not None:
        psi0, psi2 = registry.measurement_builder(econ)
        psi0 = np.asarray(psi0, dtype=float).reshape(-1)
        psi2 = np.asarray(psi2, dtype=float)
    else:
        psi0 = np.zeros(state_dim, dtype=float)
        psi2 = np.eye(state_dim, dtype=float)

    return MeasurementSpec(Psi0=psi0, Psi2=psi2)
