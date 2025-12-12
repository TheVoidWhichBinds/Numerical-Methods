import numpy as np



n = 3
A = np.random.rand(n,n)
b = np.random.rand(n).reshape(-1,1)
Aug = np.append(A,b,1)

def gaussian_elimination():
    A = Aug.copy()
    for r in range(0,n-1):
        for i in range(r+1,n):
            A[i,:] -= (A[i,r] / A[r,r]) * A[r,:]
            print(A)
            print()
    return 
        

gaussian_elimination()