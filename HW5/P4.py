import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange



#----------- Polynomial Bases Coefficient Sovlers ---------#
#Exact Function:
def exact(x):
    f= np.exp(3*x)
    return f

#Solves Matrix Equation for Monomial Basis:
def monomial_solver(x_0, x_1, x_2):
    A= np.array([[1,x_0,x_0**2],
                 [1,x_1,x_1**2],
                 [1,x_2,x_2**2]])
    
    b= np.array([exact(x_0),exact(x_1),exact(x_2)])
    coeff= np.linalg.solve(A,b)
    return coeff


#-------------- Plotting ---------------#
t= np.linspace(0,1.2,100)
#Theoretical fit:
f= exact(t)
#Monomial fit:
m_0, m_1, m_2 = monomial_solver(0, 0.5, 1)
P_mon= m_0 + m_1*t + m_2*t**2
#Lagrange fit:
lag_coeff= lagrange([0,0.5,1],[exact(0),exact(0.5),exact(1)])
P_lag= lag_coeff(t)
#Newton fit:
P_newt = (2 - 4*np.exp(1.5) + 2*np.exp(3))*t**2 \
    + (-3 + 4*np.exp(1.5) - np.exp(3))*t \
    + 1




plt.figure()
plt.title('Polynomial Fit of Various Bases')
plt.xlabel('t')
plt.ylabel('y')
plt.plot(t, f, label='Theoretical Fit exp(3t)')
plt.plot(t, P_mon, label='Monomial Fit')
plt.plot(t, P_lag, label='Lagrange Fit')
plt.plot(t, P_newt, label='Newton Fit')
plt.scatter(np.array([0,0.5,1]), np.array([exact(0),exact(0.5),exact(1)]))
#plt.plot(t, P_herm, label='Hermite Fit')
plt.legend()
plt.savefig('Poly_Fits.png')