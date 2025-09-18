import numpy as np
import matplotlib.pyplot as plt

n = 30
Taylor = np.zeros(30,1)

def exponential(x)
  sum = 0
  for i in range n
    sum = sum + x**i/i!
    Taylor(i,1) = sum
return Taylor

print(exponential(-5.5))
print(1/exponential(5.5))
print((1/exponential(0.5))**11)



