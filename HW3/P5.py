import os 
import numpy as np
import matplotlib.pyplot as plt
downloads_dir = os.path.expanduser('~/Downloads') 

L = np.array(
    [[0, 0, 1, 1/2],
     [1/3, 0, 0, 0],
     [1/3, 1/2, 0, 1/2],
     [1/3, 1/2, 0, 0]], dtype=float)

def power_method(A, x_0):
    x_k = x_0/np.linalg.norm(x_0,2)
    lam_k = 0.9
    k = 0
    while (np.linalg.norm(A @ x_k - lam_k * x_k, 2) / np.linalg.norm(x_k, 2) > 1e-4):
        x_k = A @ x_k / np.linalg.norm(A @ x_k, 2)
        lam_k = x_k @ (A @ x_k)
        k = k + 1
        if k > 1e4:
            print('Too many iterations. Try different trial eigenvector')
            break
    return lam_k, x_k

lam_1, x_1 = power_method(L, np.array([1, 1, 1, 1]))

print('The dominant eigenvalue is', np.round(lam_1, 0), 'and the its eigenvector is', np.round(x_1, 2))