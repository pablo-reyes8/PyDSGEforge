from __future__ import annotations
import sympy as sp
from typing import Dict, Iterable, Optional, List

Sym = sp.Symbol

# ------------------ Helpers para detección ------------------

def _names(syms: Iterable[Sym]) -> List[str]:
    return [str(s) for s in syms]

def _has_any(names: List[str], *candidates: str) -> bool:
    return any(c in names for c in candidates)

def _get_param(theta: Dict[str, float], keys: Iterable[str], default: float) -> float:
    """Devuelve el primer parámetro disponible en 'keys' o 'default'."""
    for k in keys:
        if k in theta:
            return float(theta[k])
    return float(default)

def _same_econ_name(sym: Sym, base: str) -> bool:
    """¿El símbolo es la versión 't' de un base (e.g., 'x_t' vs base='x_t')?"""
    return str(sym) == base


def _guess_nk_deviations(y_t: Iterable[Sym]) -> Dict[Sym, float]:
    """NK en desviaciones: SS canónico = 0 para todas las variables de estado."""
    return {s: 0.0 for s in y_t}

def _guess_nk_levels(y_t: Iterable[Sym], theta: Dict[str, float]) -> Dict[Sym, float]:
    """
    NK en niveles: usa pi_target (si existe) y r* ~ 1/beta - 1.
    Intenta mapear por nombres: 'pi_t', 'i_t', 'x_t'.
    """
    names = _names(y_t)
    beta  = _get_param(theta, ["beta"], 0.99)
    rstar = 1.0 / beta - 1.0
    # inflación objetivo (si está):
    pi_star = _get_param(theta, ["pi_star", "pi_target", "pi_bar"], 0.0)  # 0 si no se define
    guess = {s: 0.0 for s in y_t}
    for s in y_t:
        sname = str(s)
        if sname in ("x_t", "gap_t", "output_gap_t"):
            guess[s] = 0.0
        elif sname in ("pi_t", "inflation_t"):
            guess[s] = pi_star
        elif sname in ("i_t", "rate_t", "policy_t"):
            guess[s] = rstar + pi_star
    return guess

def _guess_rbc_levels(y_t: Iterable[Sym], theta: Dict[str, float]) -> Dict[Sym, float]:
    """
    RBC sin crecimiento ni trabajo (heurística mínima):
      y = A * k^alpha
      i = delta * k
      c = y - i
    Ecuación de Euler en SS: 1 = beta * (alpha*A*k^{alpha-1} + 1 - delta)
      => k = ((1/beta - 1 + delta) / (alpha*A))^(1/(alpha-1))
    """
    names = _names(y_t)
    alpha = _get_param(theta, ["alpha"], 0.33)
    delta = _get_param(theta, ["delta"], 0.025)
    A     = _get_param(theta, ["A", "TFP", "zbar"], 1.0)
    beta  = _get_param(theta, ["beta"], 0.99)

    # k_bar
    num = 1.0 / beta - 1.0 + delta
    den = alpha * A
    if den <= 0 or alpha == 1.0:
        k_bar = 10.0
    else:
        k_bar = (num / den) ** (1.0 / (alpha - 1.0))  # (alpha-1) < 0, está bien

    y_bar = A * (k_bar ** alpha)
    i_bar = delta * k_bar
    c_bar = max(y_bar - i_bar, 1e-8)  # positividad

    guess = {}
    for s in y_t:
        sname = str(s)
        if sname in ("k_t", "kap_t", "capital_t"):
            guess[s] = k_bar
        elif sname in ("y_t", "output_t"):
            guess[s] = y_bar
        elif sname in ("i_t", "invest_t"):
            guess[s] = i_bar
        elif sname in ("c_t", "cons_t"):
            guess[s] = c_bar
        else:
            guess[s] = 0.0
    return guess

#  Detección de tipo de modelo 

def _detect_model_family(
    equations: Iterable[sp.Expr | sp.Eq],
    y_t: Iterable[Sym],
    y_tp1: Optional[Iterable[Sym]] = None,):
    """
    Devuelve 'nk_deviations', 'nk_levels', 'rbc_levels' o 'generic'.
    Heurísticas simples por nombres presentes y leads.
    """
    y_names = set(_names(y_t))
    lead_names = set(_names(y_tp1 or []))

    nk_core = {"x_t", "pi_t", "i_t"}
    nk_leads = {"x_tp1", "pi_tp1"}
    if nk_core.issubset(y_names) and len(nk_leads & lead_names) > 0:
        # Si el usuario define pi_target -> niveles; si no -> desviaciones
        return "nk_levels" if any(k in ("pi_star", "pi_target", "pi_bar") for k in sp.symbols.__dict__) else "nk_deviations"

    rbc_state_names = {"k_t", "c_t", "y_t", "i_t"}
    if len(rbc_state_names & y_names) >= 2:
        return "rbc_levels"

    return "generic"

#  API principal 

def guess_factory(
    equations: Iterable[sp.Expr | sp.Eq],
    y_t: Iterable[Sym],
    *,
    y_tp1: Optional[Iterable[Sym]] = None,
    theta_econ: Optional[Dict[str, float]] = None,
    hint: Optional[str] = None) -> Dict[Sym, float]:
    """
    Construye un init_guess para el estado estacionario.
    - Si 'hint' se proporciona: usa esa familia ('nk_deviations'|'nk_levels'|'rbc_levels').
    - Si no: intenta detectar tipo de modelo por nombres/leads.
    Devuelve dict {y_t_symbol -> guess_value}.
    """

    theta = theta_econ or {}
    family = hint or _detect_model_family(equations, y_t, y_tp1)

    if family == "nk_deviations":
        return _guess_nk_deviations(y_t)
    if family == "nk_levels":
        return _guess_nk_levels(y_t, theta)
    if family == "rbc_levels":
        return _guess_rbc_levels(y_t, theta)
    # fallback genérico: ceros
    return {s: 0.0 for s in y_t}
