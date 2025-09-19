import numpy as np
import math
import matplotlib.pyplot as plt

n = 30
Taylor_machine = np.zeros(30)
import numpy as np

exact = np.ones(30) * math.exp(-5.5)


def exponential(x):
  sum = 0
  for i in range(n):
    sum = sum + x**i/math.factorial(i)
    Taylor_machine[i] = sum
  return Taylor_machine

plt.figure()
plt.title("Roundoff Error in the Exponential Taylor Series")
plt.xlabel("Number of Terms in Taylor Series")
plt.ylabel("Absolute Error")
plt.plot(np.arange(0,n), exponential(-5.5) - exact, label = 'exp(-5.5)')
plt.plot(np.arange(0,n), (1/exponential(5.5)) - exact, label = '1/exp(5.5)')
plt.plot(np.arange(0,n), (1/exponential(0.5))**11 - exact, label = '(1/exp(0.5))^11')
plt.legend()
plt.savefig('Rounding_Taylor')

plt.figure()
plt.title("Roundoff Error in the Exponential Taylor Series")
plt.xlabel("Number of Terms in Taylor Series")
plt.ylabel("Absolute Error")     
plt.ylim(-0.01, 1)
plt.plot(np.arange(0,n), exponential(-5.5) - exact, label = 'exp(-5.5)')
plt.plot(np.arange(0,n), (1/exponential(5.5)) - exact, label = '1/exp(5.5)')
plt.plot(np.arange(0,n), (1/exponential(0.5))**11 - exact, label = '(1/exp(0.5))^11')
plt.legend()
plt.savefig('Rounding_Taylor_Zoom')



