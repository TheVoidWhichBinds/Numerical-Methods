import numpy as np
import matplotlib.pyplot as plt

#----- Generating random data -----#
t = 2 * np.random.rand(20, 1)
b = 1 + 0.8*t - 0.5*t**2 + 0.3*np.exp(t) + 0.2*np.random.randn(20, 1)









#----- Plotting the random data -----#
plt.plot(t, b, 'ro')
plt.xlabel('t')
plt.ylabel('b')
plt.savefig('Data.png')