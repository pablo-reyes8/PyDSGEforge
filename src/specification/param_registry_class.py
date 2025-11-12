import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
from src.specification.param_specifications import *

@dataclass
class ParamRegistry:
    """
    Registro global de parámetros del modelo.

    Esta clase:
    - Mantiene el orden canónico de los parámetros.
    - Mapea entre el espacio de trabajo ("theta_work", sin restricciones)
      y el espacio económico interpretado (restricciones cumplidas).
    - Evalúa la log-prior conjunta del vector de parámetros.
    - Construye las matrices de covarianza de shocks (Q) y de error de
      medición (H) usando los valores económicos actuales.

    Es el objeto que debería vivir junto al solver / likelihood, porque
    garantiza consistencia entre:
      * el optimizador / sampler (trabaja en R^k),
      * el modelo económico (requiere valores con significado económico),
      * la especificación estocástica (Q, H),
      * y la prior bayesiana.

    Atributos
    ---------
    params : List[ParamSpec]
        Lista ordenada de especificaciones de parámetros individuales.
        Cada ParamSpec sabe:
        - su nombre,
        - su símbolo de SymPy,
        - su transformación (id, exp, logistic, tanh01),
        - su prior (opcional),
        - y metadatos (rol, bounds sugeridos, etc.).

        El orden en esta lista define el orden esperado en theta_work.

    qspec : Optional[QSpec]
        Especificación para construir la matriz Q (covarianza de shocks
        estructurales). Si es None, se usa identidad por defecto.

    hspec : Optional[HSpec]
        Especificación para construir la matriz H (covarianza de error
        de medición). Si es None, se asume H = 0.

    Propiedades
    -----------
    names : List[str]
        Nombres (claves) de todos los parámetros en el orden interno.

    symbols : List[sympy.Symbol]
        Símbolos de SymPy correspondientes, en el mismo orden, útiles
        para hacer sustituciones simbólicas en matrices Γ0, Γ1, Ψ, Π, etc.

    Métodos principales
    -------------------
    to_econ_dict(theta_work)
        Convierte un vector en el espacio de trabajo → diccionario
        {nombre_param: valor_económico} aplicando las transformaciones
        directas definidas en cada ParamSpec.

    to_sympy_subs(theta_work)
        Igual que arriba, pero devuelve {símbolo_SymPy: valor_económico}
        listo para hacer `.subs()` en expresiones/matrices simbólicas.

    log_prior(theta_work, include_jacobian=False)
        Evalúa la suma de log-priors individuales en los valores
        económicos actuales. Opcionalmente puede incluir el término
        de cambio de variable (Jacobiano) si quieres densidad en el
        espacio de trabajo en lugar del espacio económico.

    build_Q(theta_work, n_eps)
        Construye la matriz Q de covarianza de shocks estructurales,
        usando qspec y los valores económicos actuales.

    build_H(theta_work, n_y)
        Construye la matriz H de error de medición,
        usando hspec y los valores económicos actuales.

    from_econ_dict(econ)
        Operación inversa: toma un diccionario con valores económicos
        propuestos y devuelve el vector en el espacio de trabajo
        (aplicando la inversa de cada transformación).
    """

    params: List[ParamSpec]
    qspec: Optional[QSpec] = None
    hspec: Optional[HSpec] = None
    measurement_builder: Optional[
        Callable[[Dict[str, float]], Tuple[np.ndarray, np.ndarray]]] = None


    @property
    def names(self):
        """
        Lista de nombres de parámetros en el mismo orden interno
        en que se espera theta_work.
        """

        return [p.name for p in self.params]

    @property
    def symbols(self):
        """
        Lista de símbolos SymPy asociados a cada parámetro,
        en el mismo orden interno.
        """

        return [p.symbol for p in self.params]

    def to_econ_dict(self, theta_work: Sequence[float]):
        """
        Convierte un vector θ_work (valores libres en R^k) a un diccionario
        de valores económicos restringidos.

        Parámetros
        ----------
        theta_work : Sequence[float]
            Vector de parámetros en el espacio de trabajo. Debe tener
            la misma longitud que self.params.

        Retorna
        -------
        Dict[str, float]
            Diccionario {nombre_param: valor_económico_transformado}.

        Errores
        -------
        ValueError
            Si la dimensión de theta_work no coincide con el número
            de parámetros registrados.

        Notas
        -----
        Para cada parámetro p:
            econ[p.name] = p.fwd(theta_work[i])
        donde p.fwd aplica la transformación elegida ('exp', 'logistic', etc.).
        """

        theta_work = np.asarray(theta_work, dtype=float).reshape(-1)
        if len(theta_work) != len(self.params):
            raise ValueError(f"theta tiene {len(theta_work)} elementos, se esperaban {len(self.params)}")
        econ = {p.name: p.fwd(theta_work[i]) for i, p in enumerate(self.params)}
        return econ


    def to_econ_dict_subset(
        self,
        theta_work: Sequence[float],
        names: Optional[Sequence[str]] = None):
        """
        Convierte θ_work al espacio económico. Acepta:
        - vector 1D → dict {param: valor}
        - matriz 2D → DataFrame con columnas 'names'
        """
        theta = np.asarray(theta_work, dtype=float)

        if theta.ndim == 1:
            if names is None:
                if len(theta) > len(self.params):
                    raise ValueError("theta tiene más elementos de los registrados")
                names = self.names[:len(theta)]
            else:
                if len(names) != len(theta):
                    raise ValueError(f"names tiene {len(names)} elementos, theta {len(theta)}")

            name_to_spec = {p.name: p for p in self.params}
            return {
                nm: name_to_spec[nm].fwd(val)
                for nm, val in zip(names, theta)
            }

        elif theta.ndim == 2:
            n_draws, k = theta.shape
            if names is None:
                if k > len(self.params):
                    raise ValueError("theta tiene más columnas de las registradas")
                names = self.names[:k]
            else:
                if len(names) != k:
                    raise ValueError(f"names tiene {len(names)} elementos, pero theta tiene {k}")

            name_to_spec = {p.name: p for p in self.params}
            data = {
                nm: name_to_spec[nm].fwd(theta[:, j])
                for j, nm in enumerate(names)}
            
            import pandas as pd
            return pd.DataFrame(data)

        else:
            raise ValueError("theta_work debe ser vector o matriz 2D")


    def to_sympy_subs(self, theta_work: Sequence[float]):
        """
        Devuelve un diccionario {símbolo_sympy: valor_económico} listo
        para hacer sustituciones en ecuaciones/matrices simbólicas.

        Esto es crítico para:
        - construir Γ0, Γ1, Ψ, Π numéricas a partir de su forma simbólica,
        - evaluar el modelo linealizado con un set de parámetros concreto.

        Parámetros
        ----------
        theta_work : Sequence[float]
            Vector en el espacio de trabajo.

        Retorna
        -------
        Dict[sympy.Symbol, float]
            Mapeo símbolo -> valor económico numérico.
        """

        econ = self.to_econ_dict(theta_work)
        return {p.symbol: econ[p.name] for p in self.params if p.symbol is not None}

    def log_prior(self, theta_work: Sequence[float], include_jacobian: bool = False):
        """
        Calcula la suma de log-densidades a priori para todos los parámetros.

        Parámetros
        ----------
        theta_work : Sequence[float]
            Vector de parámetros en el espacio de trabajo.

        include_jacobian : bool, opcional
            Si False (default), se asume que cada prior está definida
            sobre el ESPACIO ECONÓMICO directamente, y simplemente se usa:
                log p(θ_econ)
            para cada parámetro.

            Si True, se añade el término de cambio de variable para obtener
            la densidad inducida en el espacio de trabajo:
                log p(θ_econ) + log|dθ_econ/dθ_work|
            parámetro a parámetro.

            Esto es útil si quieres que tu sampler opere en θ_work pero
            que la prior esté conceptualmente definida en θ_econ; es decir,
            haces la corrección jacobiana típicamente necesaria en MCMC
            cuando se muestrea en un espacio transformado.

        Retorna
        -------
        float
            Log-prior conjunta (suma sobre parámetros con prior definida).
        """

        econ = self.to_econ_dict(theta_work)
        lp = 0.0
        for p in self.params:
            if p.prior is not None:
                lp += p.prior.logpdf(econ[p.name])
                if include_jacobian:
                    # jacobiano opcional (desactivado por default para mimetizar tu .jl)
                    if p.transform == "exp":
                        lp += float(np.log(econ[p.name]))  # d/dx exp(x) = exp(x)
                    elif p.transform == "logistic":
                        # y = e^x/(1+e^x) => dy/dx = y(1-y)
                        y = econ[p.name]
                        lp += float(np.log(y*(1.0 - y)))
                    elif p.transform == "tanh01":
                        # y = (tanh x + 1)/2 => dy/dx = (1 - tanh^2 x)/2 = (1 - (2y-1)^2)/2
                        y = econ[p.name]
                        lp += float(np.log( (1.0 - (2.0*y - 1.0)**2) / 2.0 + 1e-300 ))
        return float(lp)


    def build_Q(self, theta_work: Sequence[float], n_eps: int):
        """
        Construye la matriz Q (covarianza de shocks estructurales).

        Parámetros
        ----------
        theta_work : Sequence[float]
            Vector de parámetros en el espacio de trabajo.

        n_eps : int
            Número de shocks estructurales (dimensión de ε_t).

        Retorna
        -------
        np.ndarray
            Matriz Q (n_eps x n_eps).

        Notas
        -----
        - Si no hay qspec, usa identidad como valor por defecto
          (equivalente a shocks con varianza 1 e independientes).
        - Si hay qspec, se arma usando los valores económicos actuales,
          y se valida la dimensión.
        """

        if self.qspec is None:
            return np.eye(n_eps)  
        econ = self.to_econ_dict(theta_work)
        Q = self.qspec.build(econ)
        if Q.shape != (n_eps, n_eps):
            raise ValueError(f"Q debe ser {n_eps}x{n_eps}, recibido {Q.shape}")
        return Q

    def build_H(self, theta_work: Sequence[float], n_y: int):
        """
        Construye la matriz H (covarianza de errores de medición).

        Parámetros
        ----------
        theta_work : Sequence[float]
            Vector de parámetros en el espacio de trabajo.

        n_y : int
            Dimensión del vector observado y_t.

        Retorna
        -------
        np.ndarray
            Matriz H (n_y x n_y).

        Notas
        -----
        - Si no hay hspec, se asume sin error de medición (H = 0).
        - Si hay hspec, se arma con los valores económicos actuales y
          se valida la dimensión.
        """

        if self.hspec is None:
            return np.zeros((n_y, n_y))
        econ = self.to_econ_dict(theta_work)
        H = self.hspec.build(econ, n_y)
        if H.shape != (n_y, n_y):
            raise ValueError(f"H debe ser {n_y}x{n_y}, recibido {H.shape}")
        return H

    def from_econ_dict(self, econ: Dict[str, float]):
        """
        Operación inversa de `to_econ_dict`.

        Toma un diccionario con valores económicos (restringidos,
        interpretables) y devuelve el vector θ_work que produciría
        exactamente esos valores al aplicar las transformaciones forward.

        Parámetros
        ----------
        econ : Dict[str, float]
            Diccionario {nombre_param: valor_económico}.

        Retorna
        -------
        np.ndarray
            Vector θ_work en R^k, listo para usarse como inicialización
            de un optimizador o una cadena MCMC.

        Errores
        -------
        KeyError
            Si falta algún parámetro requerido.

        Uso típico
        ----------
        - Para setear "starting values" en el sampler a partir de una
          calibración económica.
        - Para hacer warm start desde una solución calibrada/estimada
          previamente.
        """

        theta = []
        for p in self.params:
            if p.name not in econ:
                raise KeyError(f"Falta valor económico para {p.name}")
            theta.append(p.inv(econ[p.name]))
        return np.asarray(theta, dtype=float)

    def suggest_work_bounds(
        self,
        *,
        logistic_tol: float = 1e-6,
        exp_min: float = 1e-6,
        exp_max: float = 100.0) -> List[Tuple[float, float]]:
        """
        Genera bounds heurísticos en el espacio de trabajo.

        - Para transformaciones logísticas/tanh01 → evita acercarse a 0/1.
        - Para transformaciones exponenciales → asegura positividad
          con un mínimo configurable y opcionalmente un máximo.
        - Para transformaciones identidad → usa límites de `meta` si existen.
        """

        bounds: List[Tuple[float, float]] = []
        for spec in self.params:
            meta = spec.meta or {}
            transform = spec.transform.lower()

            if transform in {"logistic", "tanh01"}:
                lower_econ = meta.get("lower", logistic_tol)
                upper_econ = meta.get("upper", 1.0 - logistic_tol)
                lower = spec.inv(lower_econ)
                upper = spec.inv(upper_econ)

            elif transform == "exp":
                lower_econ = max(exp_min, meta.get("lower", exp_min))
                upper_econ = meta.get("upper")
                lower = spec.inv(lower_econ)
                if upper_econ is None:
                    upper = np.inf
                else:
                    upper = spec.inv(max(upper_econ, lower_econ * (1.0 + 1e-6)))

            else:  # identidad u otras sin restricciones explícitas
                lower = meta.get("lower", -np.inf)
                upper = meta.get("upper", np.inf)

            bounds.append((float(lower), float(upper)))

        return bounds
