from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple, Union, Any, Iterator
from numpy.random import Generator


import numpy as np
import sympy as sp
import re

from src.analysis.impulse_responses import compute_irfs, plot_irf_bands
from src.inference.likelihoods import log_like, st_sp
from src.inference.map import run_map
from src.inference.mcmc import run_metropolis

from src.model_builders.linear_system import (
    build_matrices,
    measurement_from_registry,
    MeasurementSpec,)

from src.model_builders.steady import (
    SteadyConfig,
    complete_steady_values,
    solve_steady)

from src.solvers.gensys import gensys
from src.specification.param_registry_class import ParamRegistry


@dataclass(frozen=True)
class ModelSignature:
    """
    Fotografía compacta del bloque estructural: dimensiones y nombres clave.
    Pensado para mostrarse en `__repr__` sin listar ecuaciones completas.
    """
    n_equations: int
    n_states: int
    n_leads: int
    n_lags: int
    n_shocks: int

    def __str__(self) -> str:
        return (
            f"{self.n_equations} eqs | "
            f"{self.n_states} states "
            f"(+{self.n_leads} leads, {self.n_lags} lags) | "
            f"{self.n_shocks} shocks")



def _escape_latex(text: str) -> str:
    replacements = {
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "$": r"\$",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)



class DSGE:
    """
    Cascarón inicial de la fachada `DSGE`. Por ahora sólo gestiona estructura
    y métodos mágicos “quality of life” que no tenemos en Dynare.
    """

    def __init__(
        self,
        equations,
        y_t: Sequence[sp.Symbol],
        *,
        y_tp1: Optional[Sequence[sp.Symbol]] = None,
        y_tm1: Optional[Sequence[sp.Symbol]] = None,
        eps_t: Optional[Sequence[sp.Symbol]] = None,
        eta_t: Optional[Sequence[sp.Symbol]] = None,
        metadata: Optional[dict] = None,):
        
        self._equations = tuple(equations)
        self._y_t = tuple(y_t)
        self._y_tp1 = tuple(y_tp1 or ())
        self._y_tm1 = tuple(y_tm1 or ())
        self._eps_t = tuple(eps_t or ())
        self._eta_t = tuple(eta_t or ())
        self._metadata = dict(metadata or {})
        self._measurement = None

        self.signature = ModelSignature(
            n_equations=len(self._equations),
            n_states=len(self._y_t),
            n_leads=len(self._y_tp1),
            n_lags=len(self._y_tm1),
            n_shocks=len(self._eps_t),)


    def __len__(self) -> int:
        """Número de ecuaciones del modelo."""
        return len(self._equations)

    def __iter__(self):
        """Permite iterar directamente sobre las ecuaciones."""
        return iter(self._equations)

    def __getitem__(self, idx: Union[int, str]):
        """
        Acceso indexado:
        - `model[i]` devuelve la i-ésima ecuación.
        - `model['pi_t']` busca una ecuación cuyo lhs contenga ese símbolo.
        """
        if isinstance(idx, int):
            return self._equations[idx]
        if isinstance(idx, str):
            target = sp.Symbol(idx)
            for eq in self._equations:
                if isinstance(eq, sp.Equality):
                    lhs = eq.lhs
                else:
                    lhs = eq
                if target in lhs.free_symbols:
                    return eq
            raise KeyError(f"No equation found involving symbol '{idx}'")
        raise TypeError("DSGE indices must be ints or str names")

    def __contains__(self, item) -> bool:
        """
        - SymPy equation/residual: busca coincidencia exacta.
        - Symbol/str: verifica si aparece en alguna ecuación.
        """
        if isinstance(item, (sp.Expr, sp.Equality)):
            return item in self._equations
        if isinstance(item, sp.Symbol):
            return any(item in eq.free_symbols for eq in self._equations)
        if isinstance(item, str):
            sym = sp.Symbol(item)
            return sym in self
        return False

    def __repr__(self) -> str:
        """Resumen compacto, útil para logs y debugging."""
        meta = ""
        if self._metadata:
            meta = " | " + ", ".join(f"{k}={v}" for k, v in self._metadata.items())
        return f"<DSGE {self.signature}{meta}>"

    def __str__(self) -> str:
        """Salida legible en texto plano (buena para consola/logs)."""
        header = repr(self)
        lines = []
        for idx, eq in enumerate(self._equations, start=1):
            if isinstance(eq, sp.Equality):
                lhs = sp.sstr(eq.lhs)
                rhs = sp.sstr(eq.rhs)
                lines.append(f"[{idx}] {lhs} = {rhs}")
            else:
                residual = sp.sstr(eq)
                lines.append(f"[{idx}] {residual} = 0")
        return header + "\n" + "\n".join(lines)
    
    def _repr_latex_(self) -> str:

        def _esc(x):
            return _escape_latex(str(x))

        sig = _esc(getattr(self, "signature", ""))

        meta = ""
        if getattr(self, "_metadata", None):
            items = [f"{_esc(k)}={_esc(v)}" for k, v in self._metadata.items()]
            meta = r"\quad\text{\small(" + ", ".join(items) + ")}"

        header = (
            r"{\normalsize \mathbf{DSGE\ Model}}"
            r"\\[2pt]"
            r"{\scriptsize " + sig + r"}"
            + meta
            + r"\\[4pt]")

        rows = []
        for idx, eq in enumerate(self._equations, start=1):
            if isinstance(eq, sp.Equality):
                lhs, rhs = sp.latex(eq.lhs), sp.latex(eq.rhs)
            else:
                lhs, rhs = sp.latex(eq), "0"
            rows.append(
                rf"{lhs} & = & {rhs} & \quad [\,{idx}\,] \\[-2pt]")

        body = "\n".join(rows)

        latex = (
            r"\begin{aligned}"
            + header
            + r"\left\{\,\begin{array}{rcll}"
            + body
            + r"\end{array}\right."
            + r"\end{aligned}")
        return latex

    def __hash__(self) -> int:
        """
        Nos permite usar el modelo como clave de diccionarios (cache) siempre
        que esté congelado simbólicamente.
        """
        return hash((self._equations, self._y_t, self._y_tp1, self._y_tm1, self._eps_t))


    def summary(self):
        """Versión textual más corta que __str__ (sin ecuaciones)."""
        meta = ""
        if self._metadata:
            meta = " | " + ", ".join(f"{k}={v}" for k, v in self._metadata.items())
        return f"DSGE({self.signature}){meta}"

    def symbols(self):
        """Devuelve el conjunto de símbolos de estado (orden canónico)."""
        return self._y_t
    

    ######## DGSE INFERENCE ############

    def compute(
        self,
        registry: ParamRegistry,
        theta_struct: Union[Sequence[float], Dict[str, float]],
        *,
        data: np.ndarray,
        compute_steady: bool = True,
        steady_cfg: Optional[SteadyConfig] = None,
        steady_guess: Optional[Dict[sp.Symbol, float]] = None,
        measurement: Optional[
            Union[MeasurementSpec, Callable[[np.ndarray], MeasurementSpec]]] = None,
            
        div: float = 1.0 + 1e-6,
        map: bool = True,
        map_bounds: Optional[Sequence[Tuple[float, float]]] = None,
        map_start: Optional[Sequence[float]] = None,
        map_kwargs: Optional[Dict[str, Any]] = None,
        run_mcmc: bool = False,
        mcmc_draws: int = 4000,
        mcmc_cov: Optional[np.ndarray] = None,
        mcmc_start: Optional[Sequence[float]] = None,
        mcmc_rng: Optional[np.random.Generator] = None,
        mcmc_kwargs: Optional[Dict[str, Any]] = None,
        log_summary: bool = True) -> Dict[str, Any]:
        
        if isinstance(theta_struct, dict):
            theta_econ = dict(theta_struct)
            theta_work = registry.from_econ_dict(theta_econ)
        else:
            theta_work = np.asarray(theta_struct, dtype=float).reshape(-1)
            if theta_work.size != len(registry.params):
                raise ValueError(
                    f"theta tiene longitud {theta_work.size}; "
                    f"se esperaban {len(registry.params)}.")
            theta_econ = registry.to_econ_dict(theta_work)


        self.registry = registry
        self.theta_work = theta_work
        self.theta_econ = theta_econ

        if measurement is not None:
            self._measurement = measurement

        # Steady opcional 
        steady_full = None

        if compute_steady:
            steady_core, report = solve_steady(
                equations=self._equations,
                y_t=self._y_t,
                y_tp1=self._y_tp1,
                y_tm1=self._y_tm1,
                eps_t=self._eps_t,
                eta_t=self._eta_t,
                param_values=registry.to_sympy_subs(theta_work),
                init_guess=steady_guess,
                cfg=steady_cfg)

            self._steady_values = {"values": steady_core, "report": report}

            steady_full = complete_steady_values(
                steady_core,
                self._y_t,
                y_tp1=self._y_tp1,
                y_tm1=self._y_tm1,
                eps_t=self._eps_t,
                eta_t=self._eta_t,)
        else:
            self._steady_values = None

        #  MAP opcional 
        map_result = None
        if map:
            theta0 = (
                np.asarray(map_start, dtype=float).reshape(-1)
                if map_start is not None
                else theta_work)
            
            kwargs_map = dict(map_kwargs or {})
            map_result = run_map(
                theta0=theta0,
                y=data,
                equations=self._equations,
                y_t=self._y_t,
                y_tp1=self._y_tp1,
                eps_t=self._eps_t,
                registry=registry,
                y_tm1=self._y_tm1,
                eta_t=self._eta_t,
                measurement=self._measurement,
                steady=steady_full,
                bounds=map_bounds,
                div=div,
                **{k: v for k, v in kwargs_map.items()
                if k in {"method", "hess_step", "tau_scale", "include_jacobian_prior"}})
            
        self.map_result = map_result

        #  Metropolis
        mcmc_result = None
        if run_mcmc:
            if map_result is not None:
                theta_start = map_result["theta_map"]
                cov_prop = map_result.get("cov_proposal")
            else:
                theta_start = (
                    np.asarray(mcmc_start, dtype=float).reshape(-1)
                    if mcmc_start is not None
                    else theta_work)
                
                cov_prop = None

            if mcmc_cov is not None:
                cov_prop = mcmc_cov
            if cov_prop is None:
                cov_prop = np.eye(theta_work.size) * 1e-3

            rng = mcmc_rng or np.random.default_rng()
            kwargs_mcmc = dict(mcmc_kwargs or {})

            mcmc_result = run_metropolis(
                theta_start=theta_start,
                y=data,
                equations=self._equations,
                y_t=self._y_t,
                y_tp1=self._y_tp1,
                eps_t=self._eps_t,
                registry=registry,
                y_tm1=self._y_tm1,
                eta_t=self._eta_t,
                measurement=self._measurement,
                steady=steady_full,
                cov_proposal=cov_prop,
                div=div,
                R=mcmc_draws,
                rng=rng,
                **{k: v for k, v in kwargs_mcmc.items()
                if k in {"adapt", "warmup", "adapt_block", "low", "high",
                            "shrink", "expand", "stuck_shrink", "min_scale",
                            "max_scale", "logs"}},)
            
        self.mcmc_result = mcmc_result

        if log_summary:
            print("\n[DSGE.compute] summary")

            if self._steady_values is not None:
                steady_vals = self._steady_values.get("values", {})
                if steady_vals:
                    print("  Steady state values:")
                    for sym, val in steady_vals.items():
                        print(f"    - {sym}: {val:.6g}")
                report = self._steady_values.get("report", {})
                if report:
                    parts = []
                    if report.get("method"):
                        parts.append(f"method={report['method']}")
                    if report.get("it") is not None:
                        parts.append(f"it={report['it']}")
                    if report.get("f_norm") is not None:
                        parts.append(f"||F||_inf={report['f_norm']:.3e}")
                    if parts:
                        print(f"  Steady solver info: {', '.join(parts)}")

            if map_result is not None:
                theta_map = np.asarray(map_result.get("theta_map", ()), dtype=float)
                print("  MAP theta (work):", np.array2string(theta_map, precision=6, suppress_small=True))
                if map_result.get("neglogpost_map") is not None:
                    print(f"  MAP neg-log posterior: {map_result['neglogpost_map']:.6f}")
                if theta_map.size:
                    theta_map_econ = registry.to_econ_dict(theta_map)
                    if theta_map_econ:
                        print("  MAP theta (econ):")
                        for name, val in theta_map_econ.items():
                            print(f"    - {name}: {val:.6g}")
            else:
                print("  MAP stage: not executed.")

            if mcmc_result is not None:
                acc = mcmc_result.get("acceptance_rate")
                draws = mcmc_result.get("draws")
                total_draws = draws.shape[0] if isinstance(draws, np.ndarray) else mcmc_draws
                if acc is not None:
                    print(f"  MCMC acceptance rate: {acc:.3f} ({int(round(acc * total_draws))}/{total_draws})")
            elif run_mcmc:
                print("  MCMC stage requested but no result returned.")

        return {
            "steady": self._steady_values,
            "map": self.map_result,
            "mcmc": self.mcmc_result}
