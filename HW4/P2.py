import numpy as np

n = 10 #dimension of square matrix nxn
coeff = np.poly(np.arange(1,n+1)) #coefficients of Wilkinson's polynomial
A_p = np.diag(np.ones(n-1), k=-1) #off-diag ones
A_p[:,-1] = -coeff[1:][::-1] #negative coefficients of polynomial in last column


#--------- Iterative QR Decomposition Eigenvalue Solver ---------#
def QR_Eigensolver(A_0):      
    A_k = A_0
    for k in range(n**4):
        Q_k, R_k = np.linalg.qr(A_k) #QR decomposition
        A_k = R_k @ Q_k 
    return A_k #converged final iteration 


A_diag = np.diag(QR_Eigensolver(A_p))
eigenvalues = np.round(A_diag, 3) 
print("the eigenvalues are:", eigenvalues)

