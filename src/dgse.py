from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, Optional, Sequence, Tuple, Union

import numpy as np
import sympy as sp

from src.analysis.mcmc_diagnostics import plot_mcmc_diagnostics
from src.inference.map import run_map
from src.inference.mcmc import run_metropolis, unpack_draws
from src.model_builders.linear_system import MeasurementSpec
from src.model_builders.steady import (
    SteadyConfig,
    complete_steady_values,
    solve_steady,
)
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
    def __init__(
        self,
        equations: Iterable[Union[sp.Equality, sp.Expr]],
        y_t: Sequence[sp.Symbol],
        *,
        y_tp1: Optional[Sequence[sp.Symbol]] = None,
        y_tm1: Optional[Sequence[sp.Symbol]] = None,
        eps_t: Optional[Sequence[sp.Symbol]] = None,
        eta_t: Optional[Sequence[sp.Symbol]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self._equations = tuple(equations)
        self._y_t = tuple(y_t)
        self._y_tp1 = tuple(y_tp1 or ())
        self._y_tm1 = tuple(y_tm1 or ())
        self._eps_t = tuple(eps_t or ())
        self._eta_t = tuple(eta_t or ())
        self._metadata = dict(metadata or {})
        self._measurement: Optional[
            Union[
                MeasurementSpec,
                Callable[[np.ndarray], MeasurementSpec],
            ]
        ] = None

        self.signature = ModelSignature(
            n_equations=len(self._equations),
            n_states=len(self._y_t),
            n_leads=len(self._y_tp1),
            n_lags=len(self._y_tm1),
            n_shocks=len(self._eps_t),
        )

        self.registry: Optional[ParamRegistry] = None
        self.theta_work: Optional[np.ndarray] = None
        self.theta_econ: Optional[Dict[str, float]] = None
        self._steady_values: Optional[Dict[str, Any]] = None
        self.map_result: Optional[Dict[str, Any]] = None
        self.mcmc_result: Optional[Dict[str, Any]] = None
        self._mcmc_draws_full: Optional[np.ndarray] = None
        self._mcmc_draws_after_burn: Optional[np.ndarray] = None
        self._mcmc_default_burn: int = 0

    # ------------------------------------------------------------------
    # Python niceties
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._equations)

    def __iter__(self) -> Iterator[Union[sp.Equality, sp.Expr]]:
        return iter(self._equations)

    def __getitem__(self, idx: Union[int, str]) -> Union[sp.Equality, sp.Expr]:
        if isinstance(idx, int):
            return self._equations[idx]
        target = sp.Symbol(idx)
        for eq in self._equations:
            lhs = eq.lhs if isinstance(eq, sp.Equality) else eq
            if target in lhs.free_symbols:
                return eq
        raise KeyError(f"No equation found involving symbol '{idx}'")

    def __contains__(self, item: Union[sp.Symbol, str, sp.Expr, sp.Equality]) -> bool:
        if isinstance(item, (sp.Expr, sp.Equality)):
            return item in self._equations
        sym = sp.Symbol(item) if isinstance(item, str) else item
        return any(sym in (eq.lhs if isinstance(eq, sp.Equality) else eq).free_symbols for eq in self._equations)

    def __repr__(self) -> str:
        meta = ""
        if self._metadata:
            meta = " | " + ", ".join(f"{k}={v}" for k, v in self._metadata.items())
        return f"<DSGE {self.signature}{meta}>"

    def __str__(self) -> str:
        header = repr(self)
        lines = []
        for idx, eq in enumerate(self._equations, start=1):
            if isinstance(eq, sp.Equality):
                lhs, rhs = sp.sstr(eq.lhs), sp.sstr(eq.rhs)
                lines.append(f"[{idx}] {lhs} = {rhs}")
            else:
                lines.append(f"[{idx}] {sp.sstr(eq)} = 0")
        return header + "\n" + "\n".join(lines)

    def summary(self) -> str:
        parts = [
            f"equations={len(self._equations)}",
            f"states={len(self._y_t)}",
            f"leads={len(self._y_tp1)}",
            f"lags={len(self._y_tm1)}",
            f"shocks={len(self._eps_t)}",
        ]
        if self.registry is not None:
            parts.append(f"params={len(self.registry.params)}")
        if self._metadata:
            parts.append("meta=" + ", ".join(f"{k}={v}" for k, v in self._metadata.items()))
        return "DSGE(" + "; ".join(parts) + ")"

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


    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

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
            Union[MeasurementSpec, Callable[[np.ndarray], MeasurementSpec]]
        ] = None,
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
        log_summary: bool = False,
    ) -> Dict[str, Any]:
        if isinstance(theta_struct, dict):
            theta_econ = dict(theta_struct)
            theta_work = registry.from_econ_dict(theta_econ)
        else:
            theta_work = np.asarray(theta_struct, dtype=float).reshape(-1)
            if theta_work.size != len(registry.params):
                raise ValueError(
                    f"theta tiene longitud {theta_work.size}; se esperaban {len(registry.params)}."
                )
            theta_econ = registry.to_econ_dict(theta_work)

        self.registry = registry
        self.theta_work = theta_work
        self.theta_econ = theta_econ

        if measurement is not None:
            self._measurement = measurement

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
                cfg=steady_cfg,
            )
            self._steady_values = {"values": steady_core, "report": report}
            steady_full = complete_steady_values(
                steady_core,
                self._y_t,
                y_tp1=self._y_tp1,
                y_tm1=self._y_tm1,
                eps_t=self._eps_t,
                eta_t=self._eta_t,
            )
        else:
            self._steady_values = None

        map_info = None
        if map:
            theta0 = (
                np.asarray(map_start, dtype=float).reshape(-1)
                if map_start is not None
                else theta_work
            )
            map_kwargs = dict(map_kwargs or {})
            allowed_map_keys = {
                "method",
                "hess_step",
                "tau_scale",
                "include_jacobian_prior",
            }
            filtered_map_kwargs = {k: v for k, v in map_kwargs.items() if k in allowed_map_keys}

            map_info = run_map(
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
                **filtered_map_kwargs)
            
        self.map_result = map_info

        if map_info != None: 
            print("MAP success:", self.map_result["success"], self.map_result["message"])
            print("MAP optimization Complete: Starting Metropolis Hastings")
            print()
        else:
            pass

        self._mcmc_draws_full = None
        self._mcmc_draws_after_burn = None
        self._mcmc_default_burn = 0

        
        mcmc_info = None
        if run_mcmc:
            if map_info is not None:
                theta_start = map_info["theta_map"]
                cov_prop = map_info.get("cov_proposal")
            else:
                theta_start = (
                    np.asarray(mcmc_start, dtype=float).reshape(-1)
                    if mcmc_start is not None
                    else theta_work
                )
                cov_prop = None

            if mcmc_cov is not None:
                cov_prop = mcmc_cov
            if cov_prop is None:
                cov_prop = np.eye(theta_work.size) * 1e-3

            rng = mcmc_rng or np.random.default_rng()
            mcmc_kwargs = dict(mcmc_kwargs or {})
            allowed_mcmc = {
                "adapt",
                "warmup",
                "adapt_block",
                "low",
                "high",
                "shrink",
                "expand",
                "stuck_shrink",
                "min_scale",
                "max_scale",
                "logs",
                "log_every",
            }
            filtered_mcmc = {k: v for k, v in mcmc_kwargs.items() if k in allowed_mcmc}

            mcmc_info = run_metropolis(
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
                **filtered_mcmc,
            )

            draws_full = np.asarray(mcmc_info.get("draws", np.empty((0, 0))), dtype=float)
            default_burn = int(mcmc_kwargs.get("warmup", 0) or 0)
            if draws_full.size:
                default_burn = max(0, min(default_burn, draws_full.shape[0]))
                draws_after = draws_full[default_burn:] if default_burn else draws_full
            else:
                default_burn = 0
                draws_after = draws_full

            mcmc_info["draws_full"] = draws_full
            mcmc_info["draws_after_burn"] = draws_after
            mcmc_info["burn_in"] = default_burn

            self._mcmc_draws_full = draws_full
            self._mcmc_draws_after_burn = draws_after
            self._mcmc_default_burn = default_burn

        self.mcmc_result = mcmc_info

        if log_summary:
            print("\n[DSGE.compute] summary")
            if self._steady_values is not None:
                values = self._steady_values.get("values", {})
                if values:
                    print("  Steady state values:")
                    for sym, val in values.items():
                        print(f"    - {sym}: {val:.6g}")
            if map_info is not None:
                print("  MAP success:", map_info.get("success"), map_info.get("message"))
            if mcmc_info is not None:
                acc = mcmc_info.get("acceptance_rate")
                if acc is not None:
                    total = mcmc_info.get("draws", np.empty((0, 0))).shape[0]
                    print(f"  MCMC acceptance rate: {acc:.3f} ({int(round(acc * total))}/{total})")

        return {
            "steady": self._steady_values,
            "map": self.map_result,
            "mcmc": self.mcmc_result,
        }

    # ------------------------------------------------------------------
    # Posterior analysis
    # ------------------------------------------------------------------

    def analyze_posteriors(
        self,
        subset: Optional[Union[int, Sequence[str]]] = None,
        *,
        burn_in: Optional[int] = None,
        use_full_chain: bool = False,
        bins: int = 20,
        figsize: Tuple[float, float] = (10, 4),
        plot: bool = True,
        describe: bool = True,
        return_data: bool = False,
    ):
        registry = getattr(self, "registry", None)
        if registry is None:
            raise RuntimeError("No hay registro asociado; ejecuta `compute` primero.")

        if self.mcmc_result is None or self._mcmc_draws_full is None:
            raise RuntimeError("No se encontró resultado MCMC; ejecuta `compute(..., run_mcmc=True)`.")

        draws_full = self._mcmc_draws_full
        default_burn = self._mcmc_default_burn if not use_full_chain else 0
        effective_burn = default_burn if burn_in is None else int(burn_in)
        if effective_burn < 0 or effective_burn >= draws_full.shape[0]:
            raise ValueError("burn_in debe ser un entero entre 0 y R-1.")

        draws_work = draws_full[effective_burn:] if effective_burn else draws_full
        kept = draws_work.shape[0]
        total = draws_full.shape[0]

        draws_econ = unpack_draws(draws_work, registry)

        if plot:
            subset_for_plot: Union[int, Sequence[str]]
            subset_for_plot = min(4, draws_work.shape[1]) if subset is None else subset
            plot_mcmc_diagnostics(
                draws_work,
                registry,
                subset_for_plot,
                bins=bins,
                burn_in=0,
                figsize=figsize,
            )

        summary = None
        if describe:
            summary = draws_econ.describe().T
            cols = [c for c in ["mean", "std", "25%", "50%", "75%"] if c in summary.columns]
            summary_to_print = summary[cols] if cols else summary

            acc = self.mcmc_result.get("acceptance_rate")
            acc_msg = f"{acc:.3f}" if acc is not None else "n/d"
            print(
                f"\nPosterior (espacio económico)"
                f" | draws usados: {kept}/{total} (burn-in={effective_burn})"
                f" | tasa de aceptación: {acc_msg}"
            )
            print(summary_to_print.to_string(float_format=lambda x: f"{x:0.4g}"))

        if return_data:
            return {
                "draws_econ": draws_econ,
                "summary": summary,
                "burn_in": effective_burn,
                "total_draws": total,
            }
        return None
