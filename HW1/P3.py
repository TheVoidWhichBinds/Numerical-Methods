import numpy as np
import matplotlib.pyplot as plt

x = np.arange(1.920, 2.080, 0.001)
factor_rep = (x-2)**9
coeff_rep = x**9 - 18*x**8 + 144*x**7 - 672*x**6 + 2016*x**5 - 4032*x**4 + 5376*x**3 - 4608*x**2 + 2304*x - 512

plt.figure()
plt.title("Numerical Precision")
plt.xlabel("x")
plt.ylabel("p(x)")
plt.plot(x, factor_rep, label='factored representation')
plt.plot(x, coeff_rep, label='coefficient representation')
plt.legend()
plt.savefig('Numerical_Precision')
