import numpy as np
import matplotlib.pyplot as plt


k_opt = -0.404036086 #weight parameter
k_range = np.arange(-0.7,0,0.02)

#------- Functions -------#
#Exact function:
def f(x):
    return x*np.exp(x) - 1

#Iterative solver:
def fixed_point(k):
    x_0 = 0.5 #initialization
    max_iter = 10000 #max iterations
    x = x_0 #starting value

    for i in range(max_iter):
        x_new = x + k*f(x)
        if np.abs(x_new - x) < 1E-8:
            break
        x = x_new
    
    return i


#--------- Plotting ---------#
plt.figure()
plt.title('Fixed Point Weight Choice')
plt.xlabel('Weight k Value')
plt.ylabel('Iterations to Threshold')
iters = []
for k in k_range:
    iters.append(fixed_point(k))
plt.scatter(k_range, iters, color='magenta')
plt.show()
plt.savefig('fixed_pt.png')

print(f'k_opt iterations: {fixed_point(k_opt)}')
    

