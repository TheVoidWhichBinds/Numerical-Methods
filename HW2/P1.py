import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

#### global variables ####
a = 0
b = 4
N = np.linspace(1e-6, 1e-1, 100)
########################


#### function definitions ####
def step_size():

  return (b-a)/N



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

  
