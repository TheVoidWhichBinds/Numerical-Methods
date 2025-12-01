import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt


######################################### SUBSECTION A ###########################################
#----- Global Variables -----#
a = 0 # lower bound
b = 5 # upper bound


def data_generator(N): 
    return np.linspace(a, b, N) # x values at trapezoid boundaries


def sqrt_func(x):
    return np.sqrt(x)
    

def trapez(f, x):
    N = len(x)
    h = (b - a)/(N-1)
    return h*(0.5*f[0] + np.sum(f[1:N-1]) + 0.5*f[N-1])


def simpson(f, x):
    N = len(x)
    
    if (N - 1) % 2 != 0:
        raise ValueError("Simpson requires an even number of subintervals.")
    h = (b - a) / (N - 1)
    sum = f[0] + f[-1]
    sum += 4 * np.sum(f[1:N-1:2])
    sum += 2 * np.sum(f[2:N-2:2])
    return 1/3 * h * sum

def integrals(N):
    x = data_generator(N)
    f = sqrt_func(x)
    int_trap = trapez(f, x)
    int_simp = simpson(f,x)
    return int_trap, int_simp


def error_plotting(N_range):
    plt.figure()
    plt.title('Quadrature Integration Error')
    plt.xlabel('Resolution N')
    plt.ylabel('Error')
    #
    int_exact = quad(sqrt_func, a, b)[0]
    trap_N = []
    simp_N = []
    for N in N_range:
        int_trap, int_simp = integrals(N)
        trap_N.append(int_trap)
        simp_N.append(int_simp)
    #
    plt.loglog(N_range, np.abs(int_exact - np.array(trap_N)), c='blue', label='Trapezoidal Error')
    plt.loglog(N_range, np.abs(int_exact - np.array(simp_N)), c='purple', label='Simpson Error')
    plt.legend()
    plt.grid(True)
    plt.savefig('Integral_Errors.png')
    plt.show()
    #
    return()


# Calling functions:
error_plotting(np.arange(11,1001,10))





######################################### SUBSECTION B ###########################################
#----- Error Order Estimation -----#

def error_fit(N_range):
    #
    int_exact = quad(sqrt_func, a, b)[0]
    trap_err = []
    simp_err = []
    #
    for N in N_range:
        int_trap, int_simp = integrals(N)
        trap_err.append(np.abs(int_exact - int_trap))
        simp_err.append(np.abs(int_exact - int_simp))
    #
    trap_err = np.array(trap_err)
    simp_err = np.array(simp_err)
    #
    logN = np.log(N_range)
    #
    A = np.vstack([np.ones_like(logN), logN]).T
    #
    coeff_trap, _, _, _ = np.linalg.lstsq(A, np.log(trap_err), rcond=None)
    coeff_simp, _, _, _ = np.linalg.lstsq(A, np.log(simp_err), rcond=None)
    #
    d_trap, kappa_trap = coeff_trap
    d_simp, kappa_simp = coeff_simp
    c_trap = np.exp(d_trap)
    c_simp = np.exp(d_simp)
    #
    return c_trap, kappa_trap, c_simp, kappa_simp


# Calling functions:
N_vals = np.arange(11,1001,10)
c_trap, kappa_trap, c_simp, kappa_simp = np.round(error_fit(N_vals),2)

print(f'Trapezoidal fit:  c={c_trap},  kappa = {kappa_trap}')
print(f'Simpson fit: c= {c_simp}, kappa={kappa_simp}')
