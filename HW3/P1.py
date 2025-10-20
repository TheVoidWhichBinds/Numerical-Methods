import os
import numpy as np
import matplotlib.pyplot as plt
downloads_dir = os.path.expanduser('~/Downloads') 

# Ensure float64 everywhere
U, _ = np.linalg.qr(np.random.rand(30, 30).astype(np.float64))
V, _ = np.linalg.qr(np.random.rand(30, 30).astype(np.float64))
S = np.diag((2.0 ** np.arange(-1, -31, -1)).astype(np.float64))
A = (U @ S @ V).astype(np.float64)

#--------------- Functions ---------------# 
def QR_MGS(A):
    A = A.astype(np.float64, copy=True)
    m, n = A.shape
    Q = np.zeros((m, n), dtype=np.float64)
    R = np.zeros((n, n), dtype=np.float64)
    for k in range(n):
        v = A[:, k].copy()
        for j in range(k):
            R[j, k] = Q[:, j] @ v
            v -= R[j, k] * Q[:, j]
        R[k, k] = np.linalg.norm(v)
        if R[k, k] == 0:
            continue
        Q[:, k] = v / R[k, k]
    return Q, R

def QR_Householder(A):
    A = A.astype(np.float64, copy=True)
    m, n = A.shape
    Q = np.eye(m, dtype=np.float64)
    R = A.copy()
    for k in range(min(m, n)):
        x = R[k:, k]
        normx = np.linalg.norm(x)
        if normx == 0:
            continue
        sign = 1.0 if x[0] >= 0 else -1.0
        u1 = x[0] + sign * normx
        v = x.copy()
        v[0] = u1
        v /= np.linalg.norm(v)
        R[k:, k:] -= 2.0 * np.outer(v, v @ R[k:, k:])
        Q[:, k:] -= 2.0 * (Q[:, k:] @ v[:, None]) @ v[None, :]
    Q = Q[:, :n]
    R = R[:n, :]
    return Q, R

Q_MGS, R_MGS = QR_MGS(A)
Q_HH, R_HH = QR_Householder(A)
err_MGS = np.linalg.norm(A - Q_MGS @ R_MGS)
err_HH = np.linalg.norm(A - Q_HH @ R_HH)
err_I_MGS = np.linalg.norm(np.eye(30, dtype=np.float64) - Q_MGS.T @ Q_MGS)
err_I_HH = np.linalg.norm(np.eye(30, dtype=np.float64) - Q_HH.T @ Q_HH)
print(f'Reconstruction error (MGS): {err_MGS:.2e}')
print(f'Reconstruction error (Householder): {err_HH:.2e}')
print(f'Orthogonality error (MGS): {err_I_MGS:.2e}')
print(f'Orthogonality error (Householder): {err_I_HH:.2e}')
