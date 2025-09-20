import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

x_0 = np.ones(16)*np.pi/4 #the value to evaluate the function at
deriv_exact = 3.101766393836051 #exact derivative value of func at x_0
k = np.arange(1,17)
h = 1/(10**(k))

def func(x): #the function itself
    sol = np.exp(x)/((np.cos(x))**3 + (np.sin(x))**3)
    return sol

sol = func(x_0) #gives variable to the solution at the exact point so it doesn't have to be repeatedly called


forw_diff = (func(x_0 + h) - sol)/h
cent_diff = (func(x_0 + h) - func(x_0 - h))/(2*h)
complex_diff = func(x_0 + 1j*h).imag/h


plt.figure()
plt.title("Derivative Approximations")
plt.xlabel("Step Size")
plt.ylabel("Absolute Error")
plt.xscale("log")
plt.yscale("log")
plt.plot(h, np.abs(forw_diff - deriv_exact), label='forward difference', color='tab:blue',
         path_effects=[pe.Stroke(linewidth=5, foreground='tab:blue', alpha=0.3), pe.Normal()])
plt.plot(h, np.abs(cent_diff - deriv_exact), label='centered difference', color='tab:orange',
         path_effects=[pe.Stroke(linewidth=5, foreground='tab:orange', alpha=0.3), pe.Normal()])
plt.plot(h, np.abs(complex_diff - deriv_exact), label='complex difference', color='tab:green',
         path_effects=[pe.Stroke(linewidth=3, foreground='tab:green', alpha=0.3), pe.Normal()])
plt.legend()
plt.savefig('Derivative_Approximations')

