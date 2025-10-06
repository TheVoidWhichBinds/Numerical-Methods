import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import wishart

#### seed matrix ####
n = 5000 #matrix size
A = wishart.rvs(df=n, scale=np.eye(n))



#### Cholesky decomposition function ####
def mychol(A):
    """
    Cholesky factorization: A = C^T C with C upper triangular.
    Requires A be symmetric positive definite.
    """
    A = np.array(A, dtype=float, copy=True)
    n, m = A.shape
    if n != m:
        raise ValueError("Matrix A must be square")
    if not np.allclose(A, A.T, atol=1e-12):
        raise ValueError("Matrix A must be symmetric")

    C = np.zeros_like(A)
    for i in range(n):
        #diagonal r_ii = sqrt(a_ii - sum_{m=0}^{i-1} r_{mi}^2)
        diag = A[i, i] - np.dot(C[:i, i], C[:i, i])
        if diag <= 0:
            raise ValueError("Matrix A is not positive definite")
        C[i, i] = np.sqrt(diag)

        #off-diagonals r_{ik} = (a_{ik} - sum_{m=0}^{i-1} r_{mi} r_{mk}) / r_{ii},  for k>i
        for k in range(i+1, n):
            off_diag = A[i, k] - np.dot(C[:i, i], C[:i, k])
            C[i, k] = off_diag / C[i, i]

    return C



start_my = time.perf_counter()
C_my = mychol(A)
end_my = time.perf_counter()
print(f"My Cholesky Time: {end_my - start_my:.6f} seconds")

start_py = time.perf_counter()
C_py = np.linalg.cholesky(A).T
end_py = time.perf_counter()
print(f"Numpy Cholesky Time: {end_py - start_py:.6f} seconds")



