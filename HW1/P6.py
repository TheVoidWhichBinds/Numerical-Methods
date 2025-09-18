import numpy as np
import matplotlib.pyplot as plt

n = 30
Taylor_machine = np.zeros(30)
import numpy as np

Taylor_exact = np.array([
    1.0,
    -4.5,
    10.625,
    -17.104166666666668,
    21.023437499999996,
    -20.916927083333338,
    17.52840711805555,
    -12.678641183035719,
    8.08870452396453,
    -4.602451185868954,
    2.3776844545394624,
    -1.112383365664746,
    0.48723105192884963,
    -0.18952889397613304,
    0.07634108477225304,
    -0.021144574102155186,
    0.012366121135922642,
    0.0015244256177209918,
    0.0048371659149492735,
    0.0038782147762779287,
    0.004141926339412548,
    0.004072859025258243,
    0.00409012585379682,
    0.004085996829581073,
    0.004086943064297181,
    0.004086734892659637,
    0.004086778928967579,
    0.004086769958608554,
    0.004086771720643363,
    0.004086771386464348
], dtype=float)



def exponential(x):
  sum = 0
  for i in range(n):
    sum = sum + x**i/np.math.factorial(i)
    Taylor_machine[i] = sum
  return Taylor_machine


plt.title("Roundoff Error in the Exponential Taylor Series")
plt.xlabel("number of terms in Taylor series expansion n")
plt.ylabel("Difference between exact expansion and computer expansion")
plt.plot(np.arange(0,n), Taylor_machine - Taylor_exact)
plt.savefig('Rounding_Taylor')





