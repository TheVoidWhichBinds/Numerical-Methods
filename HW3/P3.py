import numpy as np
import matplotlib.pyplot as plt

m = 20 #number of data points/rows
t = 2 * np.random.rand(m, 1) #generating t array
b = 1 + 0.8*t - 0.5*t**2 + 0.3*np.exp(t) + 0.2*np.random.randn(m, 1) #noisy data
t_val = t.ravel() #2D-->1D
b_val = b.ravel()

m_out = 21 #number of data points including outlier
t_out = np.append(t_val, [1]) #outlier data pt
b_out = np.append(b_val, [8])



#--------------- Functions ---------------#
# Matrix QR factorization via Householder:
def QR_Householder(A):
    A = A.astype(float).copy()
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy()
    for k in range(min(m, n)):
        x = R[k:, k]
        normx = np.linalg.norm(x)
        if normx == 0:
            continue
        sign = 1.0 if x[0] >= 0 else -1.0
        u1 = x[0] + sign * normx
        v = x.copy()
        v[0] = u1
        v /= np.linalg.norm(v)
        R[k:, k:] -= 2.0 * np.outer(v, v @ R[k:, k:])
        Q[:, k:] -= 2.0 * (Q[:, k:] @ v[:, None]) @ v[None, :]
    Q = Q[:, :n]
    R = R[:n, :]
    return Q, R

# Backwards substitution solver:
def back_sub(M, y):
    n = M.shape[0]
    x = np.empty(n, dtype=M.dtype)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - M[i, i+1:] @ x[i+1:]) / M[i, i]
    return x



#----------- Plotting -----------#
# Generating A matrix:
A = np.empty((len(t_val), 4))
A[:,0] = 1
A[:,1] = t_val
A[:,2] = t_val**2
A[:,3] = np.exp(t_val)
Q_hh, R_hh = QR_Householder(A) #QR Decomp
x_hh = back_sub(R_hh, Q_hh.T @ b_val) #solution
y_hh = (A @ x_hh) #approximate b

idx = np.argsort(t_val) #sorting randomized t_val
t_plot = t_val[idx] #ordering t
y_plot = y_hh[idx] #ordering approximate b using sol

plt.figure(figsize=(7, 5))
plt.scatter(t_val, b_val, s=18, label="Data") #noisy data
plt.plot(t_plot, y_plot, linewidth=2, linestyle="--", color='orange', label="Fit (Householder)") 

b_real = 1 + 0.8*t_plot - 0.5*t_plot**2 + 0.3*np.exp(t_plot) #noiseless data SUSPICIOUS!!!!!!!!
plt.plot(t_plot, b_real, linewidth=2, color='black', label='Fit (Noiseless)')
plt.ylabel("b")
plt.title("Linear Least Squares Fit using QR Decomposition")
plt.legend()
plt.tight_layout()
plt.savefig('LLS.png', dpi=300)



#------------ Plotting Outlier -------------#
A_out = np.empty((m_out, 4))
A_out[:,0] = 1
A_out[:,1] = t_out
A_out[:,2] = t_out**2
A_out[:,3] = np.exp(t_out)
Q_hh_out, R_hh_out = QR_Householder(A_out)
x_hh_out = back_sub(R_hh_out, Q_hh_out.T @ b_out)
y_hh_out = (A_out @ x_hh_out)

idxo = np.argsort(t_out)
t_plot_o = t_out[idxo]
y_plot_o = y_hh_out[idxo]

plt.figure(figsize=(7, 5))
plt.scatter(t_out, b_out, s=18, label="Data with Outlier")
plt.plot(t_plot_o, y_plot_o, linewidth=2, linestyle="--", color='orange', label="Fit (Householder with Outlier)")
plt.xlabel("t")
plt.ylabel("b")
plt.title("Linear Least Squares Fit with Outlier using QR Decomposition")
plt.legend()
plt.tight_layout()
plt.savefig('LLS_outlier.png', dpi=300)



#----------- 1-Norm Minimization ------------#
# Diagonal matrix:
def D(x, eps):
    
    r = np.ones(m_out) * (A_out @ x - b_out) #residue
    elements = np.empty(m_out)
    for i in range(m_out):
        elements[i] = 1/np.sqrt(np.dot(r[i],r[i]) + eps) #elements of D matrix

    D = np.diag(elements)
    return D


# Nonlinear ODE, set to zero, to be minimized:
def G_minimizer(x_0, eps):

    iter_max = 1000 #max allowed iterations
    x = np.empty((iter_max, 4)) #initializing x iteration array
    x[0,:] = x_0 #initial guess
    eps = eps

    for i in range(1,iter_max):
        Q,R = QR_Householder(A_out.T @ D(x[i-1], eps) @ A_out) #takes iteration equation, QR decomp
        x[i] = back_sub(R,  Q.T @ A_out.T @ D(x[i-1], eps) @ b_out) #solves for x_i
        G_norm = np.linalg.norm(A_out.T @ D(x[i], eps) @ (A_out @ x[i] - b_out), 2) #2 NORM??? 1 NORM???
        iter = i #documents iteration
        if G_norm < 1E-6: #threshold to stop loop
            break

    return x[i] #returns final x after iterations


x_L1 = G_minimizer(([0.5,0.5,0.5,0.5]), 0.05) #initial guess input for x parameters, eps
    


#----------- Plotting Outlier with L1-Norm Solution -----------#
plt.figure(figsize=(7, 5))
plt.title("Linear Least Squares Fit with Outlier using 1-Norm Minimization")
plt.scatter(t_out, b_out, s=18, label="Data with Outlier")
plt.plot(t_plot_o, y_plot_o, linewidth=2, linestyle="--", color='orange', label="Fit (Householder with Outlier)")
plt.plot(t_plot_o, A_out[idxo] @ x_L1, linewidth=2, color='green', label='Fit (1-Norm Minimization)')
plt.xlabel("t")
plt.ylabel("b")
plt.legend()
plt.tight_layout()
plt.savefig('LLS_L1.png', dpi=300)
