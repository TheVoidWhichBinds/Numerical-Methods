import numpy as np

#-------- Iterative Methods --------#
def Jacobi(n):
    #constructing A:
    A = (np.diag(2.01*np.ones(n)) + 
    np.diag(-1*np.ones(n-1),1) + 
    np.diag(-1*np.ones(n-1),-1))
    b = np.ones((n,1)) #destination of sol map Ax^*
    x_0 = np.zeros((n,1)) #initial guess sol
    x_k = x_0 #initializing iteration
    r_k = 2*b #initializing residual 
    k = 0 #initializing iteration count
    max_iter = 10000 #max # of iter allowed

    #ratio of norms: kth residual over initial residual:
    while np.linalg.norm(r_k)/np.linalg.norm(b) > 10**(-8) and (k < max_iter): 
        r_k = b - A @ x_k #calculating residual
        k += 1
        for i in range(n):
            x_k[i] += (1/A[i,i])*r_k[i] #iterating sol component-wise
    return k


def Gauss_Seidel(n):
    #constructing A:
    A = (np.diag(2.01*np.ones(n)) + 
    np.diag(-1*np.ones(n-1),1) + 
    np.diag(-1*np.ones(n-1),-1))
    b = np.ones((n,1)) #destination of sol map Ax^*
    x_0 = np.zeros((n,1)) #initial guess sol
    x_k = x_0 #initializing guess sol iter
    r_k = 2*b #initializing residual 
    k = 0 #initializing iteration count
    max_iter = 10000 #max # of iter allowed
    
    #ratio of norms: kth residual over initial residual:
    while (np.linalg.norm(r_k)/np.linalg.norm(b) > 10**(-8)) and (k < max_iter): 
        x_old = x_k.copy() #holds x^k for this sweep
        for i in range(n):
            #lower part (uses updated entries)
            left = A[i,:i] @ x_k[:i,0]
            #upper part (uses old entries)
            right = A[i,i+1:] @ x_old[i+1:,0]
            r_i = b[i] - left - right - A[i,i]*x_k[i]
            x_k[i] = x_k[i] + (1/A[i,i]) * r_i
        r_k = b - A @ x_k
        k += 1
    return k


#-------- Iterative Cost to Converge --------#
def Iterator(sizes):
    for n in sizes:
        print(f'Jacobi      , dimension: {n}, iterations: {Jacobi(n)}')
        print(f'Gauss-Seidel, dimension: {n}, iterations: {Gauss_Seidel(n)}')
        print()
    return


#Calling the function to print out the # of iterations
#for a solution to be found before the specified tolerance
#for a family of matrices of different dimensions n:
Iterator([10**2, 10**3, 10**4]) #spd matrix sizes

        
        