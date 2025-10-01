import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate



#### global variables ####
a = 0
b = 4
N_min = int(1e1)
N_max = int(1e4)



#### function definitions ####
def f(x): #function
    return np.exp(x)/(1 + 4*x**2)

def trap(h, N): #trapezoidal integral
    total = 0.0
    for i in range(1, N):
        total += f(a + i*h)
    return (h/2)*f(a) + h*total + (h/2)*f(b)

u_exact = integrate.quad(lambda x: f(x), a, b)[0] #exact value



#### choose N values (logarithmic spacing) ####
N_values = np.logspace(np.log10(N_min), np.log10(N_max), num=20, dtype=int)

u_h = np.empty(len(N_values)) #array of approximated solutions at diff step sizes
h_vals = (b - a)/N_values #array of step sizes

for i, N in enumerate(N_values):
    h = (b - a)/N #single step-size for each N_value
    u_h[i] = trap(h, N) #entry into approx integral array for each step-size and N_value



#### plotting ####
plt.figure()
plt.title("Trapezoidal Rule Convergence")
plt.xlabel("Step Size")
plt.ylabel(r"Approximation Error  $|u_h - u|$")
plt.loglog(h_vals, np.abs(u_h - u_exact), marker='o')
plt.grid(True, which="both", ls=":")
plt.savefig("Convergence_Order.png")

