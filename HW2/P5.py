import numpy as np, os, time, uuid
from scipy.linalg import solve_triangular

#### Generate L and b ####
seed = int.from_bytes(os.urandom(8), "little") ^ time.time_ns() ^ hash(uuid.uuid4())
rng = np.random.default_rng(seed)

n = 35000
L = np.tril(rng.random((n, n), dtype=float) * 1e-3)  # scaled down randoms
np.fill_diagonal(L, 1.0)  # set diagonal entries to 1
b = rng.random(n, dtype=float) * 1e-3

np.seterr(all="ignore")  # suppress harmless fp warnings


#### Forward substitution functions ####
def forward1(L, b):
    """
    Naive forward substitution (double loop).
    Solves Lx = b for x, where L is lower triangular.
    """
    n, m = L.shape
    if n != m:
        raise ValueError("Matrix L must be square")
    if len(b) != n:
        raise ValueError("Size of b must match L")

    x = np.zeros(n)
    for i in range(n): #loop over rows
        s = 0.0
        for j in range(i): #loop over columns 
            s += L[i, j] * x[j]
        x[i] = (b[i] - s) / L[i, i] #solve for x[i]
    return x


def forward2(L, b):
    """
    Vectorized forward substitution (single loop).
    Solves Lx = b for x, where L is lower triangular.
    """
    n, m = L.shape
    if n != m:
        raise ValueError("Matrix L must be square")
    if len(b) != n:
        raise ValueError("Size of b must match L")

    x = np.zeros(n)
    for i in range(n): #loop over rows
        x[i] = (b[i] - L[i, :i] @ x[:i]) / L[i, i] #solve for x[i] using vectorized dot product
    return x


def forward3(L, b):
    n, m = L.shape
    if n != m:
        raise ValueError("Matrix L must be square")
    if len(b) != n:
        raise ValueError("Size of b must match L")

    b = b.astype(float, copy=True)
    for k in range(n): #loop over columns
        b[k] = b[k] / L[k,k]              
        if k < n-1: #only update if rows remain IS THIS NECESSARY?
            b[k+1:n] -= b[k] * L[k+1:n, k]
    return b


#### Built-in function ####
x = solve_triangular(L, b, lower=True)


#### Timing function ####
def stopwatch():
    # --- Forward1 timing ---
    start1 = time.perf_counter()
    x1 = forward1(L.copy(), b.copy())
    end1 = time.perf_counter()
    print(f"Forward1 Time: {end1 - start1:.6f} seconds")

    # --- Forward2 timing ---
    start2 = time.perf_counter()
    x2 = forward2(L.copy(), b.copy())
    end2 = time.perf_counter()
    print(f"Forward2 Time: {end2 - start2:.6f} seconds")

    # --- Forward3 timing ---
    start3 = time.perf_counter()
    x3 = forward3(L.copy(), b.copy())
    end3 = time.perf_counter()
    print(f"Forward3 Time: {end3 - start3:.6f} seconds")

    # --- Solve_triangular timing ---
    start_builtin = time.perf_counter()
    x_builtin = solve_triangular(L, b, lower=True)
    end_builtin = time.perf_counter()
    print(f"Built-in Solver Time: {end_builtin - start_builtin:.6f} seconds")

stopwatch()
