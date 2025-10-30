from __future__ import annotations
import numpy as np
import sympy as sp
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union

Sym = sp.Symbol
Expr = sp.Expr

def _collect_symbols(seq: Optional[Iterable[Sym]]) -> List[Sym]:
    return list(seq) if seq else []


def _name_to_curr(sym: Sym) -> str:
    """Normaliza nombres de y_{t+1}, y_{t-1} -> y_t por convención de nombres."""
    s = str(sym)
    return s.replace("tp1", "t").replace("_t1", "_t").replace("tm1", "t")


def _map_to_current(targets: List[Sym], y_curr: List[Sym]) -> Dict[Sym, Sym]:
    """Crea un mapeo de {lead/lag_sym -> current_sym} por nombre."""
    name_to_curr = {str(s): s for s in y_curr}
    subs = {}
    for z in targets:
        base = _name_to_curr(z)
        if base in name_to_curr:
            subs[z] = name_to_curr[base]
        else:
            raise ValueError(f"[steady] No puedo mapear {z} a una variable actual {base}.")
    return subs

def _as_residuals(equations: Iterable[Union[Expr, sp.Eq]]) -> List[Expr]:
    res = []
    for eq in equations:
        if isinstance(eq, sp.Equality):
            res.append(eq.lhs - eq.rhs)
        else:
            res.append(eq)
    return res


@dataclass
class SteadyConfig:
    tol_f: float = 1e-12      # tolerancia en residuales (norma infinito)
    tol_x: float = 1e-12      # tolerancia en paso
    max_iter: int = 200
    use_nsolve: bool = True   # intenta nsolve primero si hay Jacobiano bien condicionado
    damping: bool = True      # amortiguación en newton manual
    verbose: bool = False

def build_static_system(
    equations: Iterable[Union[Expr, sp.Eq]],
    y_t: Iterable[Sym],
    y_tp1: Optional[Iterable[Sym]] = None,
    y_tm1: Optional[Iterable[Sym]] = None,
    eps_t: Optional[Iterable[Sym]] = None,
    eta_t: Optional[Iterable[Sym]] = None,
    param_values: Optional[Dict[Sym, float]] = None,) -> Tuple[sp.Matrix, sp.Matrix, List[Sym]]:
    """
    Devuelve (F(ȳ), J(ȳ), ȳ_symbols) del sistema estático:
        - y(+1) -> y, y(-1) -> y
        - shocks -> 0
        - sustituye parámetros
    """
    y_cur = _collect_symbols(y_t)
    y_lead = _collect_symbols(y_tp1)
    y_lag  = _collect_symbols(y_tm1)
    eps    = _collect_symbols(eps_t)
    eta    = _collect_symbols(eta_t)

    F = sp.Matrix(_as_residuals(equations))

    # mapear leads/lags a current
    if y_lead:
        F = F.subs(_map_to_current(y_lead, y_cur))
    if y_lag:
        F = F.subs(_map_to_current(y_lag,  y_cur))

    # shocks a cero
    if eps:
        F = F.subs({e: 0 for e in eps})
    if eta:
        F = F.subs({h: 0 for h in eta})

    # parámetros (acepta dict {sym->float} o tu ParamRegistry ya evaluado)
    if param_values:
        F = F.subs(param_values)

    #  Jacobiano respecto a ȳ
    ybar = sp.Matrix(y_cur)
    J = F.jacobian(ybar)

    return F, J, list(ybar)




######## Solvers ###########

def _norm_inf(vec: np.ndarray) -> float:
    return float(np.max(np.abs(vec))) if vec.size else 0.0

def _newton_damped(F_fun, J_fun, y0: np.ndarray, cfg: SteadyConfig) -> Tuple[np.ndarray, Dict]:
    y = y0.copy().astype(float)
    report = {"converged": False, "it": 0, "f_norm": None}
    for it in range(1, cfg.max_iter + 1):
        Fv = np.atleast_1d(np.array(F_fun(y), dtype=float).squeeze())
        Jm = np.array(J_fun(y), dtype=float)
        fn = _norm_inf(Fv)
        if cfg.verbose:
            print(f"[steady-newton] it={it} ||F||_inf={fn:.3e}")
        if fn < cfg.tol_f:
            report.update({"converged": True, "it": it, "f_norm": fn})
            return y, report
        try:
            step = np.linalg.solve(Jm, Fv)
        except np.linalg.LinAlgError:
            # pseudo-inversa como fallback
            step = np.linalg.pinv(Jm) @ Fv
        # damping line search simple
        alpha = 1.0
        for _ in range(20 if cfg.damping else 1):
            y_try = y - alpha * step
            F_try = np.atleast_1d(np.array(F_fun(y_try), dtype=float).squeeze())
            if _norm_inf(F_try) < fn:
                y = y_try
                break
            alpha *= 0.5
        if np.linalg.norm(alpha * step, ord=np.inf) < cfg.tol_x:
            Fv = np.atleast_1d(np.array(F_fun(y), dtype=float).squeeze())
            report.update({"converged": True, "it": it, "f_norm": _norm_inf(Fv)})
            return y, report
    report.update({"converged": False, "it": cfg.max_iter, "f_norm": _norm_inf(np.atleast_1d(np.array(F_fun(y), dtype=float).squeeze()))})
    return y, report

def solve_steady(
    equations: Iterable[Union[Expr, sp.Eq]],
    y_t: Iterable[Sym],
    y_tp1: Optional[Iterable[Sym]] = None,
    y_tm1: Optional[Iterable[Sym]] = None,
    eps_t: Optional[Iterable[Sym]] = None,
    eta_t: Optional[Iterable[Sym]] = None,
    param_values: Optional[Dict[Sym, float]] = None,
    init_guess: Optional[Dict[Sym, float]] = None,
    cfg: Optional[SteadyConfig] = None,) -> Tuple[Dict[Sym, float], Dict]:
    """
    Resuelve F(ȳ)=0. Intenta:
      (i) solución cerrada (sp.solve),
      (ii) nsolve (si J está disponible),
      (iii) newton amortiguado.
    Devuelve (steady_values: dict {sym->float}, report).
    """
    cfg = cfg or SteadyConfig()
    F, J, ybar_syms = build_static_system(equations, y_t, y_tp1, y_tm1, eps_t, eta_t, param_values)

    try:
        sol = sp.solve(F, ybar_syms, dict=True)
        if sol:
            sv = {sym: float(sol[0][sym]) for sym in ybar_syms}
            return sv, {"method": "solve", "converged": True, "it": 1, "f_norm": 0.0}
    except Exception:
        pass
    y_vec = sp.Matrix(ybar_syms)
    F_fun = sp.lambdify([y_vec], F, "numpy")
    J_fun = sp.lambdify([y_vec], J, "numpy")

    if init_guess is None:
        # Heurística: cero (ideal para NK en desviaciones)
        y0 = np.zeros(len(ybar_syms), dtype=float)
    else:
        y0 = np.array([init_guess.get(sym, 0.0) for sym in ybar_syms], dtype=float)

    if cfg.use_nsolve:
        try:
            sol_vec = sp.nsolve(F, y_vec, y0, tol=cfg.tol_f, maxsteps=cfg.max_iter, prec=50)
            sv = {sym: float(sol_vec[i]) for i, sym in enumerate(ybar_syms)}
            fn = _norm_inf(np.array(F_fun(np.array(list(sv.values()))), dtype=float))
            return sv, {"method": "nsolve", "converged": True, "it": None, "f_norm": fn}
        except Exception as e:
            if cfg.verbose:
                print(f"[steady] nsolve falló: {e}")

    # Newton amortiguado
    y_star, rep = _newton_damped(F_fun, J_fun, y0, cfg)
    sv = {sym: float(y_star[i]) for i, sym in enumerate(ybar_syms)}
    return sv, {"method": "newton", **rep}
