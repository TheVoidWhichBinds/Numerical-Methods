import numpy as np
import matplotlib.pyplot as plt

x = np.arange(1.920, 2.020, 0.01)
factor_rep = (x-2)**9
coeff_rep = x**9 - 18*x**8 + x**7 - x**6 + x**5 - x**4 + x**3 - x**2 + x - 512

plt.figure()
plt.title("Numerical Precision")
plt.xlabel("x")
plt.ylabel("p(x)")
plt.plot(x, factor_rep, label='factored representation')

