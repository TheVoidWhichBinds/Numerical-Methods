import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

#### global variables ####
a = 0
b = 4
N_min = int(1e1)
N_max = int(1e4)


#### function definitions ####
def f(x):
    """Function to integrate."""
    return np.exp(x) / (1 + 4*x**2)


def trap(a, b, N):
    """Trapezoidal rule for N subintervals."""
    h = (b - a) / N
    total = 0.0
    for i in range(1, N):
        total += f(a + i*h)
    return (h/2)*f(a) + h*total + (h/2)*f(b)


#### exact integral value ####
u_exact = integrate.quad(lambda x: f(x), a, b)[0]


#### choose N values (logarithmic spacing) ####
N_values = np.logspace(np.log10(N_min), np.log10(N_max), num=20, dtype=int)
h_vals = (b - a)/N_values  # step sizes

#### compute approximate solutions ####
u_h = np.empty(len(N_values))
for i, N in enumerate(N_values):
    u_h[i] = trap(a, b, N)

#### compute errors ####
errors = np.abs(u_h - u_exact)

#### --- Method 1: ratio-based estimate of convergence order ---
p_ratio = np.log(errors[:-1] / errors[1:]) / np.log(h_vals[:-1] / h_vals[1:])
p_avg = np.mean(p_ratio[-5:])  # average of last few for asymptotic rate

#### --- Method 2: slope of log(error) vs. log(h) ---
coeffs = np.polyfit(np.log(h_vals), np.log(errors), 1)
p_slope = -coeffs[0]  # negative of slope

print(f"Estimated convergence order (ratio method): {p_avg:.4f}")
print(f"Estimated convergence order (slope method): {p_slope:.4f}")


#### --- Plotting convergence behavior ---
plt.figure(figsize=(6,4))
plt.loglog(h_vals, errors, 'o-', label='Error $|u_h - u|$')
plt.xlabel("Step size $h$")
plt.ylabel("Approximation Error")
plt.title("Trapezoidal Rule Convergence Order")
plt.grid(True, which="both", ls=":")
plt.legend()
plt.tight_layout()
plt.savefig("Convergence_Order.png", dpi=300)
plt.show()
