import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,9,1000)

#------- Laguerre Polynomial Recursion ------#
def my_l(k, x):
    """
    Arguments -------------------------------------------------
    k: scalar - index of term of recurrence
    x: 1D array - points to be evaluated by the polynomial l_k

    Returns ---------------------------------------------------
    l_k(x): 1D vector - polynomial evaluation at each input x_i
    """
    l_0 = np.ones_like(x)
    l_1 = 1 - x

    if k == 0:
        return l_0
    if k == 1:
        return l_1

    l_f = l_0
    l_s = l_1
    for i in range(1, k):
        l_t = ((2*i + 1 - x) * l_s - i * l_f) / (i + 1)
        l_f = l_s
        l_s = l_t

    return l_t

#---------------- Plotting --------------#
plt.figure()
plt.title('Laguerre Polynomials') 
plt.xlabel('x')
plt.ylabel('y')
for i in range(0, 5):
    plt.plot(x, my_l(i, x), label=f'k={i}')

plt.legend()
plt.savefig('Recursion.png')
plt.show()
