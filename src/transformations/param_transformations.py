import numpy as np 

def _maybe_array(result):
    """
    Converts 0-d arrays back to Python floats while keeping higher
    dimensional arrays intact (useful for vectorized transforms).
    """
    if np.isscalar(result):
        return float(result)
    arr = np.asarray(result)
    return float(arr) if arr.ndim == 0 else arr


def tf_id(x: float):
    """
    Identidad: deja el parámetro sin transformar.

    Parámetros
    ----------
    x : float
        Valor del parámetro en el espacio de trabajo.

    Retorna
    -------
    float
        Mismo valor (sin transformación).

    Uso
    ----
    Esta transformación se usa para parámetros sin restricciones,
    como coeficientes que pueden tomar cualquier valor real.
    """
    return _maybe_array(np.asarray(x, dtype=float))

def tf_exp(x: float):
    """
    Exponencial: transforma un parámetro libre en un valor estrictamente positivo.

    Parámetros
    ----------
    x : float
        Valor del parámetro (puede ser cualquier número real).

    Retorna
    -------
    float
        Valor transformado en el espacio económico: exp(x) > 0.

    Uso
    ----
    Ideal para parámetros como desviaciones estándar, tasas o 
    coeficientes que deben ser estrictamente positivos.
    """

    return _maybe_array(np.exp(np.asarray(x, dtype=float)))

def tf_logistic(x: float):
    """
    Logística: mapea un número real al intervalo (0, 1).

    Parámetros
    ----------
    x : float
        Valor en el espacio de trabajo.

    Retorna
    -------
    float
        Valor transformado: exp(x)/(1+exp(x)) ∈ (0, 1).

    Uso
    ----
    Común para probabilidades o parámetros de tipo "peso" que 
    deben estar acotados entre 0 y 1.
    """

    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim == 0:
        x_val = float(x_arr)
        if x_val >= 0:
            z = np.exp(-x_val)
            return float(1.0 / (1.0 + z))
        z = np.exp(x_val)
        return float(z / (1.0 + z))

    result = np.empty_like(x_arr, dtype=float)
    mask = x_arr >= 0
    result[mask] = 1.0 / (1.0 + np.exp(-x_arr[mask]))
    exp_pos = np.exp(x_arr[~mask])
    result[~mask] = exp_pos / (1.0 + exp_pos)
    return result

def tf_tanh01(x: float): 
    """
    Transformación hiperbólica: mapea un número real al intervalo (0, 1)
    usando la función tangente hiperbólica.

    Parámetros
    ----------
    x : float
        Valor en el espacio de trabajo.

    Retorna
    -------
    float
        Valor transformado: 0.5 * (tanh(x) + 1.0) ∈ (0, 1).

    Uso
    ----
    Alternativa suave a la logística, útil cuando se desea 
    una transformación más simétrica o derivadas más estables.
    """

    return _maybe_array(0.5 * (np.tanh(np.asarray(x, dtype=float)) + 1.0))

def inv_id(y: float): 
    """
    Inversa de la identidad: deja el valor sin modificar.

    Parámetros
    ----------
    y : float
        Valor del parámetro en el espacio económico.

    Retorna
    -------
    float
        Mismo valor (sin transformación).
    """

    return float(y)

def inv_exp(y: float): 
    """
    Inversa de la exponencial: mapea valores positivos a reales.

    Parámetros
    ----------
    y : float
        Valor en el espacio económico (debe ser > 0).

    Retorna
    -------
    float
        log(y), correspondiente al espacio de trabajo.

    Errores
    -------
    ValueError:
        Si y ≤ 0.

    Uso
    ----
    Permite recuperar el valor en el espacio de trabajo de un
    parámetro originalmente positivo (como una varianza).
    """

    if y <= 0: 
        raise ValueError("inv_exp: y debe ser > 0")
    return float(np.log(y))

def inv_logistic(p: float):
    """
    Inversa de la función logística: mapea (0,1) → R.

    Parámetros
    ----------
    p : float
        Valor en el intervalo (0,1).

    Retorna
    -------
    float
        Valor transformado: log(p / (1 - p)).

    Errores
    -------
    ValueError:
        Si p no está en el intervalo (0,1).

    Uso
    ----
    Recupera el valor en el espacio no restringido de un parámetro
    modelado como probabilidad.
    """

    if not (0 < p < 1): 
        raise ValueError("inv_logistic: p debe estar en (0,1)")
    return float(np.log(p / (1.0 - p)))

def inv_tanh01(p: float):
    """
    Inversa de la transformación basada en tanh: mapea (0,1) → R.

    Parámetros
    ----------
    p : float
        Valor en el intervalo (0,1).

    Retorna
    -------
    float
        Valor transformado: atanh(2p - 1).

    Errores
    -------
    ValueError:
        Si p no está en el intervalo (0,1).

    Derivación
    -----------
    Dado p = 0.5 * (tanh(x) + 1),
    se tiene tanh(x) = 2p - 1, y por tanto x = atanh(2p - 1).

    Uso
    ----
    Permite volver al espacio libre de un parámetro acotado en (0,1)
    mediante la función tanh.
    """
    if not (0 < p < 1): 
        raise ValueError("inv_tanh01: p debe estar en (0,1)")

    return float(np.arctanh(2.0*p - 1.0))

_TRANSFORMS = {
    "id": (tf_id, inv_id),
    "exp": (tf_exp, inv_exp),
    "logistic": (tf_logistic, inv_logistic),
    "tanh01": (tf_tanh01, inv_tanh01)}
