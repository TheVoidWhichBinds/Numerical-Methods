import numpy as np
import matplotlib.pyplot as plt

T = np.array([
  -4.100000000000000,
  -3.370000000000000,
  -2.760000000000000,
  -1.590000000000000,
  -0.010000000000000,
   0.850000000000000,
   1.150000000000000,
   1.790000000000000,
   2.510000000000000,
   4.210000000000000])
F = np.array([
    0.044100210121794,
    0.156493511195020,
    0.299664809130155,
    0.367754561511594,
    0.094225747280148,
    0.293699892415956,
    0.289707388033342,
    0.091759658119148,
    0.065019274935861,
    0.005349325963815])
NUM_GAUSS = 2


def residual(x):
    phi = np.zeros((len(T))) #initializing Gaussian superposition
    for g in range(NUM_GAUSS):
       #x[3g]=omega, x[3g+1]=sigma, x[3g+2]=tau:
       phi +=  x[3*g] * np.exp(-1/(2 * x[3*g + 1]**2) * (T - x[3*g + 2])**2)
    
    res = phi - F #calculating the residual
    return res


def res_jacobian(x):
    J = np.empty((len(T),3*NUM_GAUSS))
    for i in range(len(T)): #rows = data points
        for g in range(0,NUM_GAUSS):  #columns = wrt 6 Gaussian parameters
            #x[3g]=omega, x[3g+1]=sigma, x[3g+2]=tau:
            J[i,3*g] = np.exp(-1/(2 * x[3*g + 1]**2) * (T[i] - x[3*g + 2])**2)
            J[i,3*g+1] = (x[3*g] * np.exp(-1/(2 * x[3*g + 1]**2) * (T[i] - x[3*g + 2])**2) 
                         * (T[i] - x[3*g + 2])**2 / x[3*g + 1]**3)
            J[i,3*g+2] = (x[3*g] * np.exp(-1/(2 * x[3*g + 1]**2) * (T[i] - x[3*g + 2])**2) 
                         * (T[i] - x[3*g + 2]) / x[3*g + 1]**2)
    return J
    

def gauss_newton(x_0):
    x = x_0
    x_k = [x.copy()]
    max_iter = 100
    Delta_x = 1
    for k in range(max_iter):
        Q,R = np.linalg.qr(res_jacobian(x))
        Delta_x = np.linalg.solve(R,-Q.T@residual(x))
        x += Delta_x
        x_k.append(x.copy())
        if np.linalg.norm(Delta_x) < 1E-16:
            break 
    return np.array(x_k)


def error_array():
    iterations = gauss_newton(np.array([1,1,-3,1,1,1.5])) #calling gauss-newton solver
    x_star = iterations[-1] #solution 
    errors = np.linalg.norm((iterations - x_star), axis=1) #error = 2-norm(iteration sol - final sol)
    return errors


def convergence_order():
    iterations = gauss_newton(np.array([1,1,-3,1,1,1.5])) #solution at each iteration
    x_star = iterations[-1] #iteratively solved solution
    errors = np.linalg.norm(iterations - x_star, axis=1) #errors at each iteration
    e = errors[errors > 1e-15] #removing ~ zero errors so no log(0) issue arises
    #Convergence order:
    p_vals = []
    for k in range(1, len(e)-1):
        p_vals.append(np.log(e[k+1]/e[k]) / np.log(e[k] / e[k-1]))
    p_vals = np.array(p_vals)
    return p_vals[-10:].mean() #average of last 3 estimates (G-N errors initially erratic)
    
print(np.round(convergence_order(),3))


