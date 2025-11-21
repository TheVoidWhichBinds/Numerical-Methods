import numpy as np
import matplotlib.pyplot as plt

########################################## SUBSECTION A ########################################## 

k_opt = -0.404036086 #weight parameter
k_range = np.arange(-0.7,0,0.02)

#------- Functions -------#
#Exact function:
def f1(x):
    return x*np.exp(x) - 1

#Iterative solver:
def fixed_point(k):
    x_0 = 0.5 #initialization
    max_iter = 10000 #max iterations
    x = x_0 #starting value

    for i in range(max_iter):
        x_new = x + k*f1(x)
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
#plt.show()
plt.savefig('fixed_pt.png')

print(f'k_opt iterations: {fixed_point(k_opt)}')
    





########################################## SUBSECTION C ########################################## 



def f2(x):
    return 1/x + 2*np.log(x) - 2

def f2_prime(x):
    return -1/x**2 + 2/x 

def newton_method(init_guess):
    x = init_guess #initial guess
    num_iter = 10
    for i in range(num_iter):
        x_new = x - f2(x)/f2_prime(x)
        x = x_new
        print(x_new)

    return x_new


x = np.linspace(0.1,3,100)


plt.figure()
plt.title('')
plt.plot(x, f2(x), label='f(x)')
plt.scatter([newton_method(0.1), newton_method(2)], [0,0], label='roots')
plt.grid(True)
#plt.show()