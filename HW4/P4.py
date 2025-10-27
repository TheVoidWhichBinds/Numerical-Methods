import numpy as np




#------ Iterative Methods -----#
def Jacobi(n, x_0):


    return



def Gauss_Seidel(n, x_0):


    return



#------ Iterative Cost to Converge ------#
def Iterator(sizes, x_0):
    """
    ------ Purpose ------------------------------------------------
    Finds the iterations required to converge below a final/initial
    residual threshold of 10^-8.

    ------ Parameters -----------------------------------------
    sizes = np.array([a,b,...]) where a,b,... are the different
        matrix dimensions you wish to call the function for.

    x_0 = initial estimate of the solution. Always equal to an
        nx1 array with all entries equal to zero

    """
    for n in sizes:
        x_0 = np.zeros(n,1)
        print(f'Jacobi iter for convergence: {Jacobi(n, x_0)}')
        print(f'GS iter for convergence: {Gauss_Seidel(n, x_0)}')
    return


        
        