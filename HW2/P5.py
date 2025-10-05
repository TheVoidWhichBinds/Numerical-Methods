import numpy as np, os, time, uuid


#### Generate L and b ####
seed = int.from_bytes(os.urandom(8), "little") ^ time.time_ns() ^ hash(uuid.uuid4())
rng = np.random.default_rng(seed)

n = 5
L = np.tril(rng.random((n, n))) + np.eye(n) # lower triangular with nonzero diagonal
b = rng.random(n)




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

    for k in range(n): #loop over columns
        b[k] = b[k] / L[k,k]              
        if j < n-1: #only update if rows remain IS THIS NECESSARY?
            b[k+1:n] -= b[k] * L[k+1:n, k]
    return b

    