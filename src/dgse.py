from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, Optional, Sequence, Tuple, Union

import numpy as np
import sympy as sp

from src.analysis.mcmc_diagnostics import plot_mcmc_diagnostics
from src.analysis.impulse_responses import compute_irfs, plot_irf_bands
from src.inference.map import run_map
from src.inference.mcmc import run_metropolis, unpack_draws
from src.model_builders.linear_system import MeasurementSpec
from src.model_builders.steady import (
    SteadyConfig,
    complete_steady_values,
    solve_steady)

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
        "^": r"\textasciicircum{}"}
    
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
        metadata: Optional[Dict[str, Any]] = None):

        self._equations = tuple(equations)
        self._y_t = tuple(y_t)
        self._y_tp1 = tuple(y_tp1 or ())
        self._y_tm1 = tuple(y_tm1 or ())
        self._eps_t = tuple(eps_t or ())
        self._eta_t = tuple(eta_t or ())
        self._metadata = dict(metadata or {})

        self._measurement: Optional[Union[
                MeasurementSpec, Callable[[np.ndarray], MeasurementSpec],]] = None

        self.signature = ModelSignature(
            n_equations=len(self._equations),
            n_states=len(self._y_t),
            n_leads=len(self._y_tp1),
            n_lags=len(self._y_tm1),
            n_shocks=len(self._eps_t))
        

        self.registry: Optional[ParamRegistry] = None
        self.theta_work: Optional[np.ndarray] = None
        self.theta_econ: Optional[Dict[str, float]] = None
        self._steady_values: Optional[Dict[str, Any]] = None
        self._steady_full: Optional[Dict[sp.Symbol, float]] = None
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
            f"shocks={len(self._eps_t)}",]
        
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
        include_jacobian_prior: bool = False,
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
        log_summary: bool = False,) -> Dict[str, Any]:

        if isinstance(theta_struct, dict):
            theta_econ = dict(theta_struct)
            theta_work = registry.from_econ_dict(theta_econ)
        else:
            theta_work = np.asarray(theta_struct, dtype=float).reshape(-1)
            if theta_work.size != len(registry.params):
                raise ValueError(
                    f"theta tiene longitud {theta_work.size}; se esperaban {len(registry.params)}.")
            
            theta_econ = registry.to_econ_dict(theta_work)

        self.registry = registry
        self.theta_work = theta_work
        self.theta_econ = theta_econ
        self._include_jacobian_prior = bool(include_jacobian_prior)

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
                cfg=steady_cfg,)
            
            self._steady_values = {"values": steady_core, "report": report}
            steady_full = complete_steady_values(
                steady_core,
                self._y_t,
                y_tp1=self._y_tp1,
                y_tm1=self._y_tm1,
                eps_t=self._eps_t,
                eta_t=self._eta_t,)
            
            self._steady_full = steady_full
        else:
            self._steady_values = None
            self._steady_full = None
            steady_full = None

        map_info = None
        map_kwargs = dict(map_kwargs or {})
        map_include_jac = bool(map_kwargs.pop("include_jacobian_prior", include_jacobian_prior))
        if map_bounds == "auto":
            map_bounds = registry.suggest_work_bounds()

        if map:
            theta0 = (
                np.asarray(map_start, dtype=float).reshape(-1)
                if map_start is not None else theta_work)

            allowed_map_keys = {
                "method",
                "hess_step",
                "tau_scale",}
            
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
                include_jacobian_prior=map_include_jac,
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

        
        mcmc_kwargs = dict(mcmc_kwargs or {})
        mcmc_include_jac = bool(mcmc_kwargs.pop("include_jacobian_prior", include_jacobian_prior))

        mcmc_info = None
        if run_mcmc:
            if map_info is not None:
                theta_start = map_info["theta_map"]
                cov_prop = map_info.get("cov_proposal")
            else:
                theta_start = (
                    np.asarray(mcmc_start, dtype=float).reshape(-1)
                    if mcmc_start is not None else theta_work)
                
                cov_prop = None

            if mcmc_cov is not None:
                cov_prop = mcmc_cov
            if cov_prop is None:
                cov_prop = np.eye(theta_work.size) * 1e-3

            rng = mcmc_rng or np.random.default_rng()
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
                "log_every"}
            
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
                include_jacobian=mcmc_include_jac,
                **filtered_mcmc)

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
            if mcmc_info is not None:
                acc = mcmc_info.get("acceptance_rate")
                if acc is not None:
                    total = mcmc_info.get("draws", np.empty((0, 0))).shape[0]
                    print(f"  MCMC acceptance rate: {acc:.3f} ({int(round(acc * total))}/{total})")

        # Auto-plot posterior vs prior at the end of compute, if possible
        if (
            self._mcmc_draws_full is not None
            and self._mcmc_draws_full.size
            and self.registry is not None):
            try:
                self.posterior(title="Posterior vs Prior (auto)")
            except Exception as exc:
                print("[DSGE.compute] No se pudo graficar posterior automáticamente:", exc)

        return {
            "steady": self._steady_values,
            "map": self.map_result,
            "mcmc": self.mcmc_result}

    

    @staticmethod
    def _build_prior_distribution(prior_spec):
        try:
            from scipy.stats import (
                gamma as stats_gamma,
                invgamma as stats_invgamma,
                norm as stats_norm,
                uniform as stats_uniform)
        except ImportError as exc:
            raise ImportError("Se requiere SciPy para evaluar priors.") from exc

        fam = prior_spec.family.lower()
        params = prior_spec.params

        if fam == "gamma":
            return stats_gamma(a=params["a"], scale=params["scale"]), (0.0, np.inf)
        if fam == "invgamma":
            return stats_invgamma(a=params["a"], scale=params["scale"]), (0.0, np.inf)
        if fam == "normal":
            return stats_norm(loc=params["loc"], scale=params["scale"]), (-np.inf, np.inf)
        if fam == "uniform":
            loc = params.get("loc", 0.0)
            scale = params.get("scale", 1.0)
            return stats_uniform(loc=loc, scale=scale), (loc, loc + scale)
        raise ValueError(f"Familia de prior no soportada: {prior_spec.family}")

    

    # ------------------------------------------------------------------
    # Prior visualization
    # ------------------------------------------------------------------

    def prior(
        self,
        registry: Optional[ParamRegistry] = None,
        *,
        include: Optional[Sequence[str]] = None,
        exclude: Optional[Sequence[str]] = None,
        quantile_bounds: Tuple[float, float] = (0.01, 0.99),
        grid_points: int = 400,
        figsize: Tuple[float, float] = (10, 6),
        sharey: bool = False,
        title: Optional[str] = None):
        """
        Grafica las densidades de las priors en el espacio económico.

        Este método puede llamarse antes de ejecutar `compute`, siempre
        que se provea explícitamente un `ParamRegistry`.
        """

        reg = registry or self.registry
        if reg is None:
            raise RuntimeError(
                "No hay ParamRegistry asociado. Pasa uno mediante `registry=` "
                "o ejecuta primero `compute(...)`.")

        try:
            import matplotlib.pyplot as plt  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Se requiere matplotlib para graficar priors."
            ) from exc

        lo_q, hi_q = quantile_bounds
        if not (0.0 < lo_q < hi_q < 1.0):
            raise ValueError("quantile_bounds debe cumplir 0 < lo < hi < 1.")

        include_set = set(include or [])
        exclude_set = set(exclude or [])

        specs = [
            p for p in reg.params
            if p.prior is not None
            and (not include_set or p.name in include_set)
            and (p.name not in exclude_set)]

        if not specs:
            raise ValueError(
                "No hay parámetros con prior para graficar "
                "(revisa include/exclude).")

        n_plots = len(specs)
        n_cols = min(3, n_plots)
        n_rows = int(np.ceil(n_plots / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=sharey)
        axes = np.atleast_1d(axes).ravel()

        for ax, spec in zip(axes, specs):
            prior_spec = spec.prior
            dist, support = self._build_prior_distribution(prior_spec)
            q_low, q_high = dist.ppf([lo_q, hi_q])

            if (not np.isfinite(q_low) or not np.isfinite(q_high)
                    or q_low == q_high):
                mean = dist.mean()
                std = dist.std()
                if np.isfinite(mean) and np.isfinite(std) and std > 0:
                    q_low = mean - 4 * std
                    q_high = mean + 4 * std
                else:
                    q_low, q_high = support

            lower_support, upper_support = support
            if np.isfinite(lower_support):
                q_low = max(q_low, lower_support + 1e-8)
            if np.isfinite(upper_support):
                q_high = min(q_high, upper_support - 1e-8)

            if not np.isfinite(q_low):
                q_low = lower_support if np.isfinite(lower_support) else -10.0
            if not np.isfinite(q_high):
                q_high = upper_support if np.isfinite(upper_support) else 10.0

            if q_high <= q_low:
                q_high = q_low + 1.0

            grid = np.linspace(q_low, q_high, grid_points)
            logpdf_vals = np.array([prior_spec.logpdf(val) for val in grid])
            pdf_vals = np.exp(logpdf_vals)
            pdf_vals[~np.isfinite(pdf_vals)] = np.nan

            ax.plot(grid, pdf_vals, color="#1565C0", lw=1.8)
            ax.set_title(f"{spec.name} ({prior_spec.family})")
            ax.set_xlabel("Valor económico")
            ax.set_ylabel("densidad")
            ax.grid(alpha=0.15)

        for ax in axes[n_plots:]:
            ax.axis("off")

        if title:
            fig.suptitle(title)

        fig.tight_layout()
        return fig, axes.reshape(n_rows, n_cols)

    # ------------------------------------------------------------------
    # Posterior visualization
    # ------------------------------------------------------------------

    def posterior(
    self,
    *,
    registry: Optional[ParamRegistry] = None,
    draws_work: Optional[np.ndarray] = None,
    include: Optional[Sequence[str]] = None,
    exclude: Optional[Sequence[str]] = None,
    burn_in: Optional[int] = None,
    use_full_chain: bool = False,
    quantile_bounds: Tuple[float, float] = (0.01, 0.99),
    grid_points: int = 400,
    figsize: Tuple[float, float] = (10, 6),
    sharey: bool = False,
    kde_bw: Optional[Union[str, float]] = None,
    title: Optional[str] = None):
        """
        Grafica prior vs posterior utilizando los draws MCMC.

        Comportamiento por defecto:
        - Si no pasas `draws_work`, usa SIEMPRE los draws *sin* burn-in que
            se guardaron en `compute` (self._mcmc_draws_after_burn).
        - Si no existe `self._mcmc_draws_after_burn`, aplica el burn-in
            por defecto `self._mcmc_default_burn` a `self._mcmc_draws_full`.

        Opciones:
        - `use_full_chain=True` fuerza a usar la cadena completa (`self._mcmc_draws_full`);
            si además pasas `burn_in`, se aplica sobre esa cadena completa.
        - Si `use_full_chain=False` (por defecto), puedes pasar un `burn_in` adicional
            (entero >= 0) que se aplica sobre la cadena ya sin burn-in.

        Parámetros:
        - `include`/`exclude`: filtran parámetros por nombre económico.
        - `quantile_bounds`: recorta el soporte de las curvas para evitar colas extremas.
        - `kde_bw`: bandwidth para gaussian_kde (ej. 'scott', 'silverman' o float).
        """

        reg = registry or self.registry
        if reg is None:
            raise RuntimeError(
                "No hay ParamRegistry asociado. Ejecuta `compute(...)` "
                "o pasa `registry=` explícitamente."
            )

        try:
            import matplotlib.pyplot as plt  # type: ignore
        except ImportError as exc:
            raise ImportError("Se requiere matplotlib para graficar posterior.") from exc

        try:
            from scipy.stats import gaussian_kde, norm as stats_norm  # type: ignore
        except ImportError as exc:
            raise ImportError("Se requiere SciPy para graficar posterior.") from exc

        def _to_scalar(x) -> float:
            """Convierte cualquier entrada (esc/array) a float escalar para usar en if/comparaciones."""
            arr = np.asarray(x)
            return float(arr.reshape(-1)[0])

        if draws_work is None:

            if (self._mcmc_draws_full is None or self._mcmc_draws_full.size == 0) and \
            (self._mcmc_draws_after_burn is None or getattr(self._mcmc_draws_after_burn, "size", 0) == 0):
                raise RuntimeError("No hay draws de MCMC almacenados; ejecuta `compute(..., run_mcmc=True)`.")

            if use_full_chain:
                base = np.asarray(self._mcmc_draws_full, dtype=float)
                if base.ndim != 2 or base.shape[0] == 0:
                    raise RuntimeError("La cadena completa MCMC no está disponible o está vacía.")
                if burn_in is not None:
                    if burn_in < 0 or burn_in >= base.shape[0]:
                        raise ValueError("burn_in debe estar entre 0 y R-1.")
                    draws = base[burn_in:]
                else:
                    draws = base

            else:
                if hasattr(self, "_mcmc_draws_after_burn") and self._mcmc_draws_after_burn is not None \
                and self._mcmc_draws_after_burn.size > 0:
                    base = np.asarray(self._mcmc_draws_after_burn, dtype=float)
                else:
                    base_full = np.asarray(self._mcmc_draws_full, dtype=float)
                    default_burn = int(getattr(self, "_mcmc_default_burn", 0) or 0)
                    if default_burn < 0 or default_burn >= base_full.shape[0]:
                        default_burn = 0
                    base = base_full[default_burn:]

                if base.ndim != 2 or base.shape[0] == 0:
                    raise RuntimeError("No hay draws disponibles tras aplicar el burn-in por defecto.")

                if burn_in is not None:
                    if burn_in < 0 or burn_in >= base.shape[0]:
                        raise ValueError("burn_in debe estar entre 0 y R-1.")
                    draws = base[burn_in:]
                else:
                    draws = base
        else:
            draws = np.asarray(draws_work, dtype=float)
            if draws.ndim != 2 or draws.size == 0:
                raise ValueError("draws_work debe ser una matriz no vacía (R x k).")
            if burn_in is not None:
                if burn_in < 0 or burn_in >= draws.shape[0]:
                    raise ValueError("burn_in debe estar entre 0 y R-1.")
                draws = draws[burn_in:]

        if draws.ndim != 2 or draws.shape[0] == 0:
            raise ValueError("No quedan draws para graficar después del burn-in indicado.")

        draws_df = reg.to_econ_dict_subset(draws, names=reg.names)

        include_names = list(reg.names) if include is None else list(include)
        exclude_set = set(exclude) if exclude is not None else set()
        final_names = [nm for nm in include_names if nm in draws_df.columns and nm not in exclude_set]

        if not final_names:
            raise ValueError("No hay parámetros seleccionados para graficar posterior.")

        lo_q, hi_q = quantile_bounds
        if not (0.0 < lo_q < hi_q < 1.0):
            raise ValueError("quantile_bounds debe cumplir 0 < lo < hi < 1.")

        specs = {p.name: p for p in reg.params}
        n_plots = len(final_names)
        n_cols = min(3, n_plots)
        n_rows = int(np.ceil(n_plots / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=sharey)
        axes = np.atleast_1d(axes).ravel()

        for ax, name in zip(axes, final_names):
            values = draws_df[name].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                ax.text(0.5, 0.5, "Sin draws válidos", ha="center", va="center")
                ax.set_title(name)
                ax.axis("off")
                continue

            prior_spec = specs[name].prior
            prior_dist = support = None

            if prior_spec is not None:
                prior_dist, support = self._build_prior_distribution(prior_spec)
                q_low = _to_scalar(prior_dist.ppf(lo_q))
                q_high = _to_scalar(prior_dist.ppf(hi_q))
            else:
                q_low = float(np.nanmin(values))
                q_high = float(np.nanmax(values))

            vmin = float(np.nanmin(values))
            vmax = float(np.nanmax(values))
            span = max(1e-6, vmax - vmin)
            margin = 0.10 * span

            lo = (vmin - margin) if not np.isfinite(q_low) else float(np.nanmin([q_low, vmin - margin]))
            hi = (vmax + margin) if not np.isfinite(q_high) else float(np.nanmax([q_high, vmax + margin]))

            if support is not None:
                lower_support, upper_support = support
                lower_support = _to_scalar(lower_support) if lower_support is not None else -np.inf
                upper_support = _to_scalar(upper_support) if upper_support is not None else  np.inf
                if np.isfinite(lower_support):
                    lo = max(lo, lower_support + 1e-8)
                if np.isfinite(upper_support):
                    hi = min(hi, upper_support - 1e-8)

            if not np.isfinite(lo):
                lo = vmin - margin
            if not np.isfinite(hi):
                hi = vmax + margin
            if hi <= lo:
                hi = lo + 1e-3

            grid = np.linspace(lo, hi, grid_points)

            post_pdf = None
            if values.size > 1:
                try:
                    kde = gaussian_kde(values, bw_method=kde_bw)
                    post_pdf = kde(grid)
                except np.linalg.LinAlgError:
                    post_pdf = None

            if post_pdf is None:
                mean_val = float(np.nanmean(values))
                std_val = float(np.nanstd(values))
                if not np.isfinite(std_val) or std_val == 0.0:
                    std_val = max(1e-3, 0.05 * abs(mean_val) if np.isfinite(mean_val) else 1e-3)
                post_pdf = stats_norm(loc=mean_val, scale=std_val).pdf(grid)

            if prior_spec is not None:
                prior_logpdf = np.array([prior_spec.logpdf(val) for val in grid], dtype=float)
                prior_pdf = np.exp(prior_logpdf)
                prior_pdf[~np.isfinite(prior_pdf)] = np.nan
                ax.plot(grid, prior_pdf, label="Prior", color="#546E7A", linestyle="--", linewidth=1.4)

            ax.plot(grid, post_pdf, label="Posterior", color="#C62828", linewidth=1.8)
            ax.fill_between(grid, 0, post_pdf, color="#FFCDD2", alpha=0.35)

            post_mean = float(np.nanmean(values))
            if np.isfinite(post_mean):
                ax.axvline(post_mean, color="#B71C1C", linestyle=":", linewidth=1.5)

            ax.set_title(name)
            ax.set_xlabel("Valor económico")
            ax.set_ylabel("densidad")
            ax.grid(alpha=0.15)
            ax.legend(loc="upper right", frameon=False)

        for ax in axes[n_plots:]:
            ax.axis("off")

        if title:
            fig.suptitle(title)

        fig.tight_layout()
        return fig, axes.reshape(n_rows, n_cols)

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
        return_data: bool = False,):

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
                figsize=figsize)

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



    def impulse_responses(
        self,
        draws_work: Optional[np.ndarray] = None,
        *,
        burn_in: Optional[int] = None,
        horizon: int = 20,
        quantiles: Sequence[float] = (0.16, 0.5, 0.84),
        shock_indices: Optional[Sequence[int]] = None,
        observable_names: Optional[Sequence[str]] = None,
        shock_names: Optional[Sequence[str]] = None,
        div: float = 0.0,
        plot: bool = True,
        plot_kwargs: Optional[Dict[str, Any]] = None):

        if self.registry is None:
            raise RuntimeError("No hay registro asociado; ejecuta `compute` primero.")

        if draws_work is None:
            if self._mcmc_draws_full is None:
                raise RuntimeError("No draws disponibles; provee `draws_work` o ejecuta MCMC.")
            if burn_in is None:
                draws = (
                    self._mcmc_draws_after_burn
                    if self._mcmc_draws_after_burn is not None
                    else self._mcmc_draws_full)
                
            else:
                if burn_in < 0 or burn_in >= self._mcmc_draws_full.shape[0]:
                    raise ValueError("burn_in debe estar entre 0 y el número de draws almacenados.")
                draws = self._mcmc_draws_full[burn_in:]
        else:
            draws = np.asarray(draws_work, dtype=float)
            if draws.ndim != 2:
                raise ValueError("`draws_work` debe ser una matriz (R x k).")
            if burn_in is not None:
                if burn_in < 0 or burn_in >= draws.shape[0]:
                    raise ValueError("burn_in debe estar entre 0 y R-1.")
                draws = draws[burn_in:]

        if draws.size == 0:
            raise ValueError("No hay draws disponibles después de aplicar el burn-in.")

        irf_result = compute_irfs(
            draws,
            self.registry,
            self._equations,
            self._y_t,
            self._y_tp1,
            self._eps_t,
            steady=self._steady_full,
            y_tm1=self._y_tm1,
            eta_t=self._eta_t,
            measurement=self._measurement,
            horizon=horizon,
            burn_in=0,
            quantiles=quantiles,
            shock_indices=shock_indices,
            observable_names=observable_names,
            shock_names=shock_names,
            div=div)

        if plot:
            plot_kwargs = plot_kwargs or {}
            plot_irf_bands(irf_result, **plot_kwargs)

        return irf_result
