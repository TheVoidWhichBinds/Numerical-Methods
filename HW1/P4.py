import numpy as np

print(np.spacing(1))
print(np.spacing(np.float32(1)))
print(np.spacing(2**40))
print(np.spacing(np.float32(2**40)))

print()

a = 0.5
b = 0.7 - 0.2
print(a == b)
