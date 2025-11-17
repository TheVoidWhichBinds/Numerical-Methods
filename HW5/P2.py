import numpy as np
import matplotlib.pyplot as plt
import sympy as sp   # no longer strictly needed, but harmless
import math

#------ Global Variables ------#
N = 14  # polynomial degree for the node comparison plot
x_cont = np.linspace(-1, 1, 100)

#----- Exact Function -----#
def f(x):
    return 1 / (1 + 12*x**2)

f_exact = f(x_cont)

# Equidistant Nodes Polynomial (for part (g) picture)
def equidistant(N):
    x_equid = np.linspace(-1, 1, N+1)  # equidistant nodes
    c_equid = np.polyfit(x_equid, f(x_equid), deg=N)
    p_equid = np.polyval(c_equid, x_cont)
    return x_equid, p_equid

x_equid, p_equid = equidistant(N)

# Chebyshev Nodes Polynomial (for plotting on x_cont)
def chebyshev(N):
    k = np.arange(0, N+1)
    x_cheb = np.cos((2*k + 1)*np.pi / (2*(N+1)))
    c_cheb = np.polyfit(x_cheb, f(x_cheb), deg=N)
    p_cheb = np.polyval(c_cheb, x_cont)
    return x_cheb, p_cheb

x_cheb, p_cheb = chebyshev(N)

#--------------- Plotting for 2(g) ----------------#
plt.figure()
plt.title('Polynomial Fit Node Comparison')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.plot(x_cont, f_exact, color='blue', label='Exact Function')
plt.scatter(x_equid, f(x_equid), color='green')
plt.plot(x_cont, p_equid, color='green', label='Polynomial with Equidistant Nodes')
plt.scatter(x_cheb, f(x_cheb), color='red')
plt.plot(x_cont, p_cheb, color='red', label='Polynomial with Chebyshev Nodes')
plt.legend()
plt.savefig('Nodes.png')
# plt.show()

#------- Error, Error Estimate for 2(h) -------#
grid = 10000
s = np.linspace(-1, 1, grid)

# Max interpolation error on [-1,1] for Chebyshev nodes
def exact_error(N):
    k = np.arange(0, N+1)
    x_cheb = np.cos((2*k + 1)*np.pi / (2*(N+1)))
    c_cheb = np.polyfit(x_cheb, f(x_cheb), deg=N)
    p_cheb = np.polyval(c_cheb, s)
    E_f = np.abs(f(s) - p_cheb)
    return np.max(E_f)

# Choose a range of larger N values (at least two bigger than 14)
N_vals = np.array([14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
                   41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70])

def error_plots(N_vals):
    E_f = np.zeros(len(N_vals))
    for i, N_val in enumerate(N_vals):
        E_f[i] = exact_error(N_val)

    # Plotting max error vs N on a log scale (semilogy)
    plt.figure()
    plt.title('Maximal Chebyshev Interpolation Error vs N')
    plt.xlabel('N')
    plt.ylabel('Max Error (sup norm)')
    plt.semilogy(N_vals, E_f, 'o-', color='purple', label='Max error (Chebyshev)')
    plt.legend()
    plt.savefig('Poly_Error.png')
    plt.show()

error_plots(N_vals)
