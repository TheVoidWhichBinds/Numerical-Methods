import numpy as np

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

def phi(num_gaussians,x):
    phi = 0
    for g in range(num_gaussians):
       #x[3g]=omega, x[3g+1]=sigma, x[3g+2]=tau
       phi +=  x[3*g] * np.exp(-1/(2 * x[3*g + 1]**2) * (T - x[3*g + 2])**2)

    


def residual(num_gaussians,x):
    return phi(num_gaussians,x) - F


def res_jacobian(num_gaussians,x):
    J = np.empty((len(T),3*num_gaussians))
    for i in range(len(T)):
        for g in range(0,num_gaussians):
            J[i,3*g] = np.exp(-1/(2 * x[3*g + 1]**2) * (T[i] - x[3*g + 2])**2)
            J[i,3*g+1] = 
            J[i,3*g+2] = 
    
    


def gauss_newton():
