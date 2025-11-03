import numpy as np




#------ Iterative Methods ------#
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

    #ratio of norms: kth residual over initial residual:
    while (np.linalg.norm(r_k)/np.linalg.norm(b) < 10**(-8)): 
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

    #ratio of norms: kth residual over initial residual:
    while (np.linalg.norm(r_k)/np.linalg.norm(b) < 10**(-8)): 
        k += 1

        for i in range(n):
            sum_k = np.sum((A[i,j]*x_k[j]))
            sum_kk = A[i,j]*x_kk[j]
            r_k[i] = b[i] - sum_kk - sum_k #residual component
            x_k[i] += (1/A[i,i])*r_k[i] #iterating sol component-wise
    
    return k



#------ Iterative Cost to Converge ------#
def Iterator(sizes):
    for n in sizes:
        print(f'Jacobi, dimension: {n}, iterations: {Jacobi(n)}')
        print(f'GS, dimension: {n}, iterations: {Gauss_Seidel(n)}')
    
    return


Iterator([10**2, 10**3, 10**4]) #spd matrix sizes

        
        