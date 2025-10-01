import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

#### global variables ####
N_max = 1000 #number of steps
a = 0
b = 4
h = (b-a)/N
########################


#### function definitions ####
def f(x): #function definition
  return np.exp(x)/(1 + 4*x**2)

def trap(): #trapezoidal integral 
  N 
  sum = 0
  for i in range(1,N): 
    sum += f(a + i*h)
  return (h/2)*f(a) + h*sum + (h/2)*f(b)

integ_exact = integrate.quad(lambda x: f(x), a, b) #limit of integral
#####################################################################


#### plotting arrays ####
u_h = np.empty()




plt.figure()
plt.title("Trapezoidal Rule Convergence")
plt.xlabel(" ")
plt.ylabel(" ")
plt.plot(
  
