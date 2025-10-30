import numpy as np 
import scipy as sc 
from scipy.linalg import qz
from scipy.linalg import ordqz, svd, solve
from numpy.typing import NDArray

def gensys_from_schur(
    S: NDArray[np.complex128],
    T: NDArray[np.complex128],
    Q: NDArray[np.complex128],
    Z: NDArray[np.complex128],
    c: NDArray[np.float64],
    Psi: NDArray[np.float64],
    Pi: NDArray[np.float64],
    div: float):
    """
    Compute the linear equilibrium law of motion of a linear rational expectations model
    using a (generalized) Schur / QZ decomposition, à la Sims (1991, 2001), with a
    Blanchard–Kahn style check for existence and uniqueness.

    This solver assumes you already ran a QZ decomposition of the generalized system
        A x_t = B x_{t-1} + c + Psi * eps_t + Pi * eta_t
    and you are passing in the QZ factors (S,T,Q,Z) such that:
        Q^H A Z = S
        Q^H B Z = T
    where `H` is conjugate transpose. The function reorders the QZ decomposition to
    separate stable and unstable eigenvalues, checks BK conditions, and constructs:

        x_t = G1 x_{t-1} + C + impact * eps_t

    Parameters
    ----------
    S : (n, n) complex ndarray
        Upper (quasi-)triangular matrix from the generalized Schur decomposition
        of A (i.e. Q^H A Z = S). Diagonal elements pair with T to give generalized
        eigenvalues alpha/beta.

    T : (n, n) complex ndarray
        Upper (quasi-)triangular matrix from the generalized Schur decomposition
        of B (i.e. Q^H B Z = T).

    Q : (n, n) complex ndarray
        Left unitary matrix from the QZ decomposition. Rows of Q' rotate original states.

    Z : (n, n) complex ndarray
        Right unitary matrix from the QZ decomposition. Columns of Z are the new state basis.

    c : (n,) float ndarray
        Constant term in the original model. Enters the affine shift C in the solved
        law of motion.

    Psi : (n, k) float ndarray
        Impact matrix of exogenous shocks eps_t. These are shocks that *can* hit the
        equilibrium law of motion contemporaneously (they show up in `impact`).

    Pi : (n, m) float ndarray
        Matrix multiplying "expectational errors" / forward-looking residuals eta_t.
        This matrix is crucial for determinacy: Pi tells us which equations contain
        non-predetermined (jump) components.

        Intuición: if an unstable root is not “killed” by a jump variable, BK fails.

    div : float
        Threshold that classifies eigenvalues as "stable" vs "unstable".
        We treat |beta_i| > div * |alpha_i| as unstable roots. Typical choice is
        something like div = 1 + 1e-6 (just above 1) so that |lambda| = |alpha/beta| > 1
        counts as explosive.

    Returns
    -------
    G1 : (n, n) float ndarray
        State transition matrix. Gives how x_t depends on x_{t-1}.

    C : (n,) float ndarray
        Deterministic constant / fixed point shift in the policy rule.

    impact : (n, k) float ndarray
        Impact matrix of structural shocks eps_t on x_t.

    eu : (2,) int ndarray
        Blanchard–Kahn existence/uniqueness flags:
        - eu[0] = 1 if a solution exists (enough jump variables to pin down
                  all unstable roots), 0 otherwise.
          (We set eu[0] = -2 if we detect a degenerate singular case early.)
        - eu[1] = 1 if the solution is unique (no extra free jump components),
                  0 if multiple solutions.

        So:
           eu = [1,1] → unique stable equilibrium
           eu = [1,0] → multiple equilibria
           eu = [0,0] → no stable equilibrium
           eu = [-2,-2] → degenerate singular case caught early

    Notes
    -----
    Outline of the algorithm:

    1. Rebuild A,B:
       We invert the effect of Q,Z to get A,B back in the original coordinate system:
           A = Q S Z^H
           B = Q T Z^H

    2. QZ reordering (ordqz):
       We reorder the generalized eigenvalues so that all "stable" eigenvalues
       (roughly |alpha/beta| < div^{-1}) come first. After reordering:
           A2, B2, Q2, Z2
       such that the first block corresponds to stable roots and the last block
       to unstable roots.

       Let:
           nunstab = number of unstable roots
           k       = n - nunstab  (size of stable block)

    3. BK existence:
       We look at the subspace associated with unstable eigenvalues. We ask:
       do we have enough forward-looking variables (columns of Pi) to kill those
       explosive directions? This is tested via an SVD on qt2' * Pi, where
       qt2 are the rows of Q2' aligned with the unstable block.

       Rank conditions here set eu[0].

    4. BK uniqueness:
       We also test whether there is "too much flexibility" in the stable block,
       which would imply multiple equilibria. Another SVD-based projection checks
       if forward-looking components spill into the stable space. This sets eu[1].

    5. Construct law of motion:
       We build a transformed system (G0, G1c) where we pin jump variables so that
       the unstable block is forced to satisfy the appropriate jump condition.
       Then we invert G0 to get the reduced form:

           x_t = G1 x_{t-1} + C + impact * eps_t

       Finally we rotate back with Z and take real parts, since the underlying
       model is real even if intermediate steps are complex.

    References
    ----------
    - Sims, Christopher A. (2001). "Solving Linear Rational Expectations Models."
    - Blanchard, Olivier J. and Charles M. Kahn (1980). "The Solution of Linear
      Difference Models under Rational Expectations."
    - Klein, Paul (2000). "Using the Generalized Schur Form to Solve a Multivariate
      Linear Rational Expectations Model."
    """

    # Normalización de tipos y validaciones básicas
    S = np.asarray(S, dtype=np.complex128)
    T = np.asarray(T, dtype=np.complex128)
    Q = np.asarray(Q, dtype=np.complex128)
    Z = np.asarray(Z, dtype=np.complex128)

    if S.ndim != 2 or T.ndim != 2:
        raise ValueError("S y T deben ser matrices 2D.")
    if S.shape != T.shape or S.shape[0] != S.shape[1]:
        raise ValueError("S y T deben ser cuadradas y del mismo tamaño.")

    n = S.shape[0]

    # Entradas reales
    c = np.asarray(c, dtype=np.float64).reshape(n,)
    Psi = np.asarray(Psi, dtype=np.float64)
    Pi = np.asarray(Pi, dtype=np.float64)

    if Psi.ndim != 2 or Pi.ndim != 2:
        raise ValueError("Psi y Pi deben ser matrices 2D.")
    if Psi.shape[0] != n or Pi.shape[0] != n:
        raise ValueError("Psi y Pi deben tener n filas, con n = S.shape[0].")
    if Z.ndim != 2 or Q.ndim != 2:
        raise ValueError("Q y Z deben ser matrices 2D de tamaño (n, n).")
    if Q.shape != (n, n) or Z.shape != (n, n):
        raise ValueError("Q y Z deben ser (n, n), con n = S.shape[0].")

    eps = 1e-6
    eu = np.array([0, 0], dtype=int)

    if np.any((np.abs(np.diag(S)) < eps) & (np.abs(np.diag(T)) < eps)):
        G1 = np.empty((0, 0), dtype=np.float64)
        C = np.empty((0,), dtype=np.float64)
        impact = np.empty((0, 0), dtype=np.float64)
        eu[:] = [-2, -2]
        return G1, C, impact, eu

    # Reconstrucción A, B y reordenamiento 
    A = Q @ (S @ Z.conj().T)
    B = Q @ (T @ Z.conj().T)

    def _select(alpha, beta):
        return ~(np.abs(beta) > div * np.abs(alpha))

    res = ordqz(A, B, sort=_select, output="complex")

    if len(res) == 6 and res[2].ndim == 2 and res[3].ndim == 2:
        A2, B2, Q2, Z2, alpha, beta = res
    # Caso 2 (SciPy devuelve AA, BB, alpha, beta, Q, Z)
    elif len(res) == 6 and res[2].ndim == 1 and res[3].ndim == 1 and res[4].ndim == 2 and res[5].ndim == 2:
        A2, B2, alpha, beta, Q2, Z2 = res
    else:
        raise RuntimeError(
            f"ordqz() devolvió una tupla inesperada: tipos {[x.ndim if hasattr(x,'ndim') else type(x) for x in res]}")

    # Asegurar formas 2D correctas para lo que sigue
    if Q2.ndim != 2 or Z2.ndim != 2:
        raise ValueError(f"ordqz devolvió Q2/Z2 no-2D: Q2.ndim={Q2.ndim}, Z2.ndim={Z2.ndim}")
    if Q2.shape != (n, n) or Z2.shape != (n, n):
        raise ValueError(f"ordqz devolvió Q2/Z2 con forma inesperada: Q2={Q2.shape}, Z2={Z2.shape}, n={n}")

    a, b = A2, B2
    qt, z = Q2, Z2  

    # Conteo de inestables y particionamiento
    sel = ~(np.abs(np.diag(b)) > div * np.abs(np.diag(a))) 
    nunstab = int(n - np.sum(sel))
    k = n - nunstab
    if k < 0 or k > n:
        raise ValueError(f"Bloque estable inválido: k={k}, n={n}, nunstab={nunstab}")

    if qt.ndim != 2:
        raise ValueError(f"Q/qt debe ser 2D; recibido qt.ndim={qt.ndim}, shape={qt.shape}")

    qt1 = qt[:, :k]
    qt2 = qt[:, k:]
    neta = Pi.shape[1]

    # Existencia (SVD sobre bloque inestable)
    if nunstab == 0:
        ueta = np.zeros((0, 0), dtype=np.complex128)
        deta = np.zeros((0, 0), dtype=np.complex128)
        veta = np.zeros((neta, 0), dtype=np.complex128)
        bigev_len = 0
    else:
        etawt = qt2.conj().T @ Pi                   
        U, s, Vh = svd(etawt, full_matrices=False)   
        keep = np.where(s > eps)[0]
        bigev_len = len(keep)
        ueta = U[:, keep]                             
        deta = np.diag(s[keep]) if bigev_len else np.zeros((0, 0), dtype=np.complex128)  
        veta = Vh.conj().T[:, keep]                  

    if bigev_len >= nunstab:
        eu[0] = 1  # existencia

    # Unicidad (SVD sobre bloque estable)
    if nunstab == n:
        ueta1 = np.zeros((0, 0), dtype=np.complex128)
        deta1 = np.zeros((0, 0), dtype=np.complex128)
        veta1 = np.zeros((neta, 0), dtype=np.complex128)
    else:
        etawt1 = qt1.conj().T @ Pi
        U1, s1, Vh1 = svd(etawt1, full_matrices=False)
        keep1 = np.where(s1 > eps)[0]
        ueta1 = U1[:, keep1]
        deta1 = np.diag(s1[keep1]) if keep1.size else np.zeros((0, 0), dtype=np.complex128)
        veta1 = Vh1.conj().T[:, keep1]

    if veta1.size == 0:
        unique = True
    else:
        if veta.size == 0:
            proj = np.zeros_like(veta1)
        else:
            proj = veta @ (veta.conj().T @ veta1)
        loose = veta1 - proj
        _, sL, _ = svd(loose, full_matrices=False)
        nloose = np.sum(np.abs(sL) > (eps * n))
        unique = (nloose == 0)

    if unique:
        eu[1] = 1

    # Construcción de tmat y de G0, G1 (complejos intermedios)
    if nunstab == 0:
        right_block_T = np.zeros((k, 0), dtype=np.complex128)
    else:
        if deta.size:
            X = solve(deta, veta.conj().T)
        else:
            X = np.zeros((0, veta.conj().T.shape[1]), dtype=np.complex128)
        M1 = ueta @ X                            
        M2 = M1 @ veta1                             
        M3 = deta1 @ (ueta1.conj().T) if deta1.size else np.zeros((0, k), dtype=np.complex128)  
        M = M2 @ M3                                
        right_block_T = -M.conj().T                

    tmat = np.hstack([np.eye(k, dtype=np.complex128), right_block_T])  

    upper = tmat @ a
    lower = np.hstack([np.zeros((nunstab, k), dtype=np.complex128),
                       np.eye(nunstab, dtype=np.complex128)])
    G0 = np.vstack([upper, lower])

    upper1 = tmat @ b
    lower1 = np.zeros((nunstab, n), dtype=np.complex128)
    G1c = np.vstack([upper1, lower1])

    G0I = np.linalg.inv(G0)
    G1c = G0I @ G1c

    # Constante 
    c_col = c.astype(np.complex128).reshape(n, 1)
    usix = np.arange(k, n)
    Busix = b[np.ix_(usix, usix)]
    Ausix = a[np.ix_(usix, usix)]

    part1 = tmat @ (qt.conj().T @ c_col)         
    rhs2 = qt[:, k:].conj().T @ c_col              

    if Ausix.size == 0:
        x2 = np.zeros((0, 1), dtype=np.complex128)
    else:
        x2 = solve(Ausix - Busix, rhs2)

    Cc = G0I @ np.vstack([part1, x2])          

    # Impacto
    top = tmat @ (qt.conj().T @ Psi.astype(np.complex128))    
    bot = np.zeros((nunstab, Psi.shape[1]), dtype=np.complex128)
    impactc = G0I @ np.vstack([top, bot])                      

    # Mapear con z y tomar la parte real
    G1 = np.real(z @ (G1c @ z.conj().T)).astype(np.float64)
    C = np.real(z @ Cc).reshape(-1).astype(np.float64)
    impact = np.real(z @ impactc).astype(np.float64)

    return G1, C, impact, eu



def gensys(
    Gamma0,
    Gamma1,
    c,
    Psi,
    Pi,
    div: float = 0.0):
    """
    Solve a linear rational expectations (LRE) model in canonical form via generalized Schur (QZ) decomposition.

    Canonical system:
        Gamma0 @ y_t = Gamma1 @ y_{t-1} + c + Psi @ eps_t + Pi @ eta_t

    Under standard assumptions (i.i.d. shocks eps_t and expectational errors eta_t),
    the solution—when it exists and is unique—has the form:
        y_t = G1 @ y_{t-1} + C + impact @ eps_t

    Parameters
    ----------
    Gamma0 : (n, n) ndarray
        Left-hand-side matrix (post-linearization).
    Gamma1 : (n, n) ndarray
        Right-hand-side lag matrix.
    c : (n,) ndarray
        Constant term.
    Psi : (n, k) ndarray
        Impact matrix for i.i.d. structural shocks eps_t.
    Pi : (n, m) ndarray
        Matrix multiplying one-step-ahead expectational errors eta_t.
    div : float, optional (default = 0.0)
        Threshold for classifying generalized eigenvalues as (un)stable.
        If 0.0, a data-driven threshold will be computed downstream (like Sims/Julia’s `new_div`).

    Returns
    -------
    G1 : (n, n) ndarray
        State transition matrix for y_t (without contemporaneous shocks).
    C : (n,) ndarray
        Constant vector in the solved law of motion.
    impact : (n, k) ndarray
        Contemporaneous impact of structural shocks eps_t on y_t.
    eu : (2,) ndarray of ints
        Existence/Uniqueness flags:
            eu[0] = 1 if a stable (bounded) solution exists, else 0 or negative codes
            eu[1] = 1 if the solution is unique, else 0

        In case of a low-level LAPACK failure during QZ, returns eu = [-3, -3]
        and empty arrays for G1, C, impact.

    Notes
    -----
    * This function performs light validation (square matrices, compatible dimensions).
    * It intentionally does NOT compute any default `div` itself.
    """

    #normalize shapes
    Gamma0 = np.asarray(Gamma0, dtype=np.float64)
    Gamma1 = np.asarray(Gamma1, dtype=np.float64)
    Psi = np.asarray(Psi, dtype=np.float64)
    Pi = np.asarray(Pi, dtype=np.float64)
    c = np.asarray(c,dtype=np.float64).reshape(-1)

    if Gamma0.ndim != 2 or Gamma1.ndim != 2:
        raise ValueError("Gamma0 and Gamma1 must be 2D arrays.")
    
    n0, m0 = Gamma0.shape
    n1, m1 = Gamma1.shape

    if n0 != m0 or n1 != m1 or n0 != n1:
        raise ValueError("Gamma0 and Gamma1 must be square and of the same size.")
    
    n = n0

    if Psi.ndim != 2 or Pi.ndim != 2:
        raise ValueError("Psi and Pi must be 2D arrays.")
    
    if Psi.shape[0] != n or Pi.shape[0] != n:
        raise ValueError("Psi and Pi must have the same number of rows as Gamma0.")

    if c.shape[0] != n:
        raise ValueError("c must be length-n, where n = Gamma0.shape[0].")

    #  QZ decomposition (complex, like Matlab)
    try:
        S, T, Q, Z = qz(
            Gamma0.astype(np.complex128, copy=False),
            Gamma1.astype(np.complex128, copy=False),
            output="complex")
        
    except Exception:
        G1 = np.empty((0, 0), dtype=np.float64)
        C = np.empty((0,),    dtype=np.float64)
        impact = np.empty((0, 0), dtype=np.float64)
        eu = np.array([-3, -3], dtype=int)
        return G1, C, impact, eu

    return gensys_from_schur(S, T, Q, Z, c, Psi, Pi, div)