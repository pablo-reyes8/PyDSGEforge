import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
try:
    from scipy.stats import gamma as _Gamma
    from scipy.stats import uniform as _Uniform
    from scipy.stats import norm as _Normal
    from scipy.stats import invgamma as _InvGamma
except Exception:
    _Gamma = _Uniform = _Normal = _InvGamma = None 
import sympy as sp

from src.transformations.param_transformations import *

_TRANSFORMS = {
    "id": (tf_id, inv_id),
    "exp": (tf_exp, inv_exp),
    "logistic": (tf_logistic, inv_logistic),
    "tanh01": (tf_tanh01, inv_tanh01)}


@dataclass
class PriorSpec:
    """
    Representa la especificación de una distribución *a priori* univariada
    definida en el ESPACIO ECONÓMICO del parámetro.

    Esta clase se usa para encapsular tanto el tipo de distribución (familia)
    como los parámetros numéricos que la definen, de forma que sea posible 
    evaluar su log-verosimilitud (`logpdf`) durante procesos de estimación 
    o muestreo bayesiano.

    Atributos
    ----------
    family : str
        Nombre de la familia de la prior. Admite:
        - 'gamma'
        - 'uniform'
        - 'normal'
        - 'invgamma'

    params : Dict[str, float]
        Diccionario con los parámetros específicos de la distribución,
        de acuerdo con las convenciones de `scipy.stats`:
        - Gamma:      {'a': shape, 'scale': θ}
        - Uniform:    {'loc': a, 'scale': b-a}
        - Normal:     {'loc': μ, 'scale': σ}
        - InvGamma:   {'a': shape, 'scale': θ}

    Métodos
    --------
    logpdf(x: float) -> float
        Evalúa el logaritmo de la densidad de probabilidad en el valor `x`.

    Ejemplo
    --------
    >>> prior_sigma = PriorSpec(family="invgamma", params={"a": 2.0, "scale": 0.5})
    >>> prior_sigma.logpdf(0.7)
    -0.5148  # log densidad de la prior evaluada en x = 0.7

    Notas
    -----
    - Las priors se definen en el espacio económico, no en el transformado.
      Por ejemplo, si un parámetro tiene restricción de positividad, 
      su prior gamma se aplica sobre el valor positivo resultante de la transformación exp(x).

    - La función requiere SciPy (`scipy.stats`), y utiliza las siguientes abreviaturas internas:
        _Gamma     → scipy.stats.gamma
        _Uniform   → scipy.stats.uniform
        _Normal    → scipy.stats.norm
        _InvGamma  → scipy.stats.invgamma
    """
    family: str                      # 'gamma', 'uniform', 'normal', 'invgamma'
    params: Dict[str, float]        

    def logpdf(self, x):
        """
        Evalúa el logaritmo de la densidad de probabilidad (log-pdf) de la prior.

        Parámetros
        ----------
        x : float
            Valor del parámetro (en el espacio económico) en el que se 
            evaluará la función de densidad.

        Retorna
        -------
        float
            Valor del logaritmo de la densidad en `x`.

        Errores
        -------
        ImportError
            Si SciPy no está disponible en el entorno.

        ValueError
            Si la familia especificada no está soportada.
        """

        if _Gamma is None:
            raise ImportError("Se requiere SciPy para evaluar priors (scipy.stats).")
        fam = self.family.lower()

        if fam == "gamma":
            # SciPy gamma: shape=a, scale=θ
            return float(_Gamma.logpdf(x, a=self.params["a"], scale=self.params["scale"]))
        elif fam == "uniform":
            return float(_Uniform.logpdf(x, loc=self.params.get("loc", 0.0),scale=self.params.get("scale", 1.0)))
        
        elif fam == "normal":
            return float(_Normal.logpdf(x, loc=self.params["loc"], scale=self.params["scale"]))
        
        elif fam == "invgamma":
            return float(_InvGamma.logpdf(x, a=self.params["a"], scale=self.params["scale"]))
        
        else:
            raise ValueError(f"Prior desconocida: {self.family}")
        
    
    
@dataclass
class ParamSpec:
    """
    Representa la especificación completa de un parámetro económico 
    dentro de un modelo estructural o bayesiano.

    Este objeto conecta:
    - su representación simbólica (para las ecuaciones del modelo),
    - la transformación entre el espacio de trabajo y el espacio económico,
    - la prior bayesiana (opcional),
    - y metadatos útiles para calibración o estimación.

    Atributos
    ----------
    name : str
        Nombre del parámetro. Usualmente la clave interna 
        (ejemplo: 'phi', 'sigma_eps', 'beta').

    symbol : sp.Symbol
        Símbolo simbólico de SymPy asociado al parámetro, 
        útil para generar matrices estructurales (Γ₀, Γ₁, Ψ, Π, etc.)
        de forma automática a partir de ecuaciones.

    transform : str, opcional
        Nombre de la transformación que mapea el parámetro desde el
        espacio de trabajo (sin restricciones) al espacio económico.
        Opciones admitidas:
            - 'id'       → sin restricción (ℝ → ℝ)
            - 'exp'      → positivo (ℝ → ℝ⁺)
            - 'logistic' → intervalo (0,1)
            - 'tanh01'   → intervalo (0,1) con tanh suave
        Default: 'id'.

    prior : Optional[PriorSpec], opcional
        Distribución *a priori* sobre el parámetro, 
        definida en el espacio económico. Puede ser `None`
        si el parámetro se fija o se estima sin prior explícita.

    role : str, opcional
        Rol del parámetro dentro del modelo, útil para clasificación:
            - 'struct'     → parámetro estructural (en ecuaciones)
            - 'shock_std'  → desviación estándar de shocks
            - 'meas'       → parámetro de medición
            - 'steady'     → valor de estado estacionario
        Default: 'struct'.

    meta : Dict[str, float], opcional
        Diccionario libre para almacenar metadatos auxiliares
        (por ejemplo, límites sugeridos, etiquetas, comentarios, etc.).

    Métodos
    --------
    fwd(x_work: float) -> float
        Aplica la transformación directa: pasa del espacio de trabajo 
        (sin restricciones) al espacio económico (restringido).

    inv(x_econ: float) -> float
        Aplica la transformación inversa: pasa del espacio económico 
        al espacio de trabajo.

    Ejemplo
    --------
    >>> phi_spec = ParamSpec(
    ...     name="phi",
    ...     symbol=sp.Symbol("phi"),
    ...     transform="logistic",
    ...     prior=PriorSpec("beta", {"a": 2.0, "b": 5.0}),
    ...     role="struct"
    ... )
    >>> phi_spec.fwd(0.3)
    0.5744
    >>> phi_spec.inv(0.5744)
    0.3
    >>> phi_spec.prior.logpdf(phi_spec.fwd(0.3))
    -0.89

    Notas
    -----
    - `_TRANSFORMS` es un diccionario global que mapea nombres de 
      transformaciones ('id', 'exp', 'logistic', 'tanh01') a pares de funciones:
          `_TRANSFORMS[name] = (fwd_fn, inv_fn)`
      donde `fwd_fn` y `inv_fn` son las funciones definidas en tu módulo de 
      transformaciones (`tf_*` e `inv_*`).
    
    - Esta estructura facilita integrar los parámetros dentro de un pipeline
      bayesiano o un solver DSGE, garantizando consistencia entre espacios
      y evitando errores de escala o dominio.

    - En el contexto de un modelo DSGE/BVAR:
        * `x_work` es lo que manipula el optimizador o sampler MCMC.
        * `x_econ` es el valor económico interpretado (positivo, acotado, etc.).
    """

    name: str                  
    symbol: sp.Symbol           
    transform: str = "id"       
    prior: Optional[PriorSpec] = None
    role: str = "struct"       
    meta: Dict[str, float] = field(default_factory=dict)  

    def fwd(self, x_work: float):
        """
        Transforma un valor desde el espacio de trabajo al espacio económico.

        Parámetros
        ----------
        x_work : float
            Valor en el espacio sin restricciones.

        Retorna
        -------
        float
            Valor transformado al espacio económico (restringido).
        """
        return _TRANSFORMS[self.transform][0](x_work)

    def inv(self, x_econ: float):
        """
        Transforma un valor desde el espacio económico al espacio de trabajo.

        Parámetros
        ----------
        x_econ : float
            Valor en el espacio económico (restringido).

        Retorna
        -------
        float
            Valor transformado al espacio de trabajo (sin restricciones).
        """
        return _TRANSFORMS[self.transform][1](x_econ)
    



@dataclass
class QSpec:
    """
    Especifica cómo construir la matriz Q, correspondiente a la 
    covarianza de los shocks estructurales del modelo.

    Por defecto, Q se asume diagonal, donde cada elemento de la diagonal 
    corresponde a la varianza de un shock estructural (σᵢ²).

    Atributos
    ----------
    diag_params : List[str]
        Lista con los nombres de los parámetros asociados a las 
        desviaciones estándar de los shocks, en el mismo orden 
        en que aparecen en el vector de innovaciones εₜ.

    Métodos
    --------
    build(pvals_econ: Dict[str, float]) -> np.ndarray
        Construye y devuelve la matriz Q evaluada con los valores 
        económicos actuales de los parámetros.

    Ejemplo
    --------
    >>> qspec = QSpec(diag_params=["sigma_a", "sigma_b"])
    >>> pvals = {"sigma_a": 0.2, "sigma_b": 0.5}
    >>> qspec.build(pvals)
    array([[0.04, 0.  ],
           [0.  , 0.25]])

    Notas
    -----
    - Cada σᵢ se interpreta como la desviación estándar del i-ésimo shock.
      Por tanto, la matriz Q resultante es diag(σ₁², σ₂², ..., σₙ²).

    - En modelos más complejos, esta clase puede extenderse para permitir
      correlaciones cruzadas o parametrizaciones más ricas de Q.
    """

    diag_params: List[str]  # nombres de parámetros (std de shocks) en el orden de eps_t

    def build(self, pvals_econ):
        """
        Construye la matriz Q a partir de los valores económicos actuales 
        de los parámetros (σᵢ).

        Parámetros
        ----------
        pvals_econ : Dict[str, float]
            Diccionario que asocia nombres de parámetros con sus valores
            numéricos actuales en el espacio económico.

        Retorna
        -------
        np.ndarray
            Matriz Q (diagonal), donde Q[i,i] = σᵢ².

        Errores
        -------
        KeyError:
            Si algún parámetro especificado en `diag_params` no está presente
            en `pvals_econ`.
        """
        sigmas = []
        for nm in self.diag_params:
            if nm not in pvals_econ:
                raise KeyError(f"QSpec: falta parámetro '{nm}' en pvals.")
            sigmas.append(pvals_econ[nm])
        sigmas = np.asarray(sigmas, dtype=float)
        return np.diag(sigmas**2)
    

@dataclass
class HSpec:
    """
    Especifica la matriz H de errores de medición del modelo.

    Permite tres configuraciones:
      1. **Fija** (`fixed` ≠ None): se usa una matriz H dada explícitamente.
      2. **Diagonal parametrizada** (`diag_params` ≠ None): 
         cada error de medición tiene una desviación estándar libre (σᵢ).
      3. **Sin error de medición** (por defecto): H = 0.

    Atributos
    ----------
    fixed : Optional[np.ndarray]
        Matriz fija H (ny × ny). Si se proporciona, se utiliza directamente.

    diag_params : Optional[List[str]]
        Lista con los nombres de los parámetros asociados a las desviaciones
        estándar de los errores de medición. Si se proporciona, se construye
        una matriz diagonal H = diag(σ₁², ..., σₙ²).

    Métodos
    --------
    build(pvals_econ: Dict[str, float], ny: int) -> np.ndarray
        Construye la matriz H correspondiente a la configuración elegida.

    Ejemplo
    --------
    >>> hspec = HSpec(diag_params=["meas_y", "meas_pi"])
    >>> pvals = {"meas_y": 0.1, "meas_pi": 0.05}
    >>> hspec.build(pvals, ny=2)
    array([[0.01  , 0.    ],
           [0.    , 0.0025]])

    Notas
    -----
    - Si `fixed` se usa, su dimensión debe coincidir con (ny × ny).
    - Si no se define `fixed` ni `diag_params`, H se asume nula.
    - Esta matriz entra típicamente en la ecuación de medición del 
      modelo de estado:
      
      y_t = Z_t alpha_t + eta_t,  eta_t  mathcal{N}(0, H)
    """

    fixed: Optional[np.ndarray] = None
    diag_params: Optional[List[str]] = None  # si no es None, usa var = param^2

    def build(self, pvals_econ: Dict[str, float], ny: int):
        if self.fixed is not None:
            H = np.asarray(self.fixed, dtype=float)
            if H.shape != (ny, ny):
                raise ValueError(f"H fijo debe ser {ny}x{ny}, recibido {H.shape}")
            return H
        
        if self.diag_params is not None:
            if len(self.diag_params) != ny:
                raise ValueError("diag_params de H debe tener longitud n_y.")
            
            stds = [pvals_econ[nm] for nm in self.diag_params]
            stds = np.asarray(stds, dtype=float)
            return np.diag(stds**2)

        return np.zeros((ny, ny), dtype=float)
    

@dataclass
class MeasurementSpec:
    """
    Especifica la estructura del bloque de medición del modelo de estado:

        y_t = Psi0 + Psi2 * x_t + eta_t,   eta_t ~ N(0, H)

    donde:
    - y_t : vector de variables observadas (dimensión n_y)
    - x_t : vector de estados latentes (dimensión n_state)
    - Psi0 : vector de constantes o sesgos de medición
    - Psi2 : matriz de carga de los estados sobre los observables

    Comportamiento por defecto
    ---------------------------
    - Si no se proporciona nada:
        Psi2 = I_n_state   → se observan directamente los estados.
        Psi0 = 0           → sin intercepto en la ecuación de medición.

    Esto equivale a suponer que tus observables son idénticos a los estados
    del modelo (por ejemplo, en un VAR reducido o en un DSGE linealizado sin
    variables latentes adicionales).

    Atributos
    ----------
    Psi0 : Optional[np.ndarray], por defecto None
        Vector constante en la ecuación de medición. Si se omite,
        se asume un vector de ceros.

    Psi2 : Optional[np.ndarray], por defecto None
        Matriz de carga de los estados sobre los observables.
        Si se omite, se usa la identidad de tamaño n_state.

    Métodos
    --------
    build(n_state: int) -> Tuple[np.ndarray, np.ndarray]
        Devuelve las matrices (Psi0, Psi2) en forma numérica y con
        dimensiones validadas.

    Ejemplo
    --------
    >>> meas = MeasurementSpec(
    ...     Psi0=np.array([0.0, 0.0]),
    ...     Psi2=np.array([[1.0, 0.0], [0.0, 0.5]])
    ... )
    >>> Psi0, Psi2 = meas.build(n_state=2)
    >>> Psi0
    array([0., 0.])
    >>> Psi2
    array([[1. , 0. ],
           [0. , 0.5]])

    Notas
    -----
    - El vector Psi0 tiene longitud igual al número de filas de Psi2
      (una constante por observable).
    - Psi2 debe tener tantas columnas como estados (n_state).
    - Esta clase no incluye el ruido de medición H; eso se controla con HSpec.

    En el contexto de un modelo DSGE linealizado:
        y_t = Psi0 + Psi2 * x_t
    típicamente se usa para mapear el vector de estados modelados (por ejemplo,
    “output gap”, “inflation gap”, “policy rate deviation”) a las variables
    observadas reales (PIB, inflación observada, tasa de política, etc.).
    """
    Psi0: Optional[np.ndarray] = None
    Psi2: Optional[np.ndarray] = None

    def build(self, n_state: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.Psi2 is None:
            Psi2 = np.eye(n_state, dtype=float)
        else:
            Psi2 = np.asarray(self.Psi2, dtype=float)
            if Psi2.shape[1] != n_state:
                raise ValueError(f"Psi2 columnas = {Psi2.shape[1]} no coincide con n_state={n_state}")
        if self.Psi0 is None:
            Psi0 = np.zeros((Psi2.shape[0],), dtype=float)
        else:
            Psi0 = np.asarray(self.Psi0, dtype=float).reshape(Psi2.shape[0],)
        return Psi0, Psi2