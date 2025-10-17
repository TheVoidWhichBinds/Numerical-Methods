import numpy as np
import matplotlib.pyplot as plt

#----- Generating random data -----#
m = 20
n = 4
t = 2 * np.random.rand(m, 1)
b = 1 + 0.8*t - 0.5*t**2 + 0.3*np.exp(t) + 0.2*np.random.randn(m, 1)
t_val = t.ravel()

A = np.empty((len(t), n))
A[:,0] = 1
A[:,1] = t_val
A[:,2] = t_val**2
A[:,3] = np.exp(t_val)



def QR_Mod_GS(A):
    Q = np.empty((m, n))
    R = np.zeros((n, n))

    for i in range(n):
        u = A[:, i].copy() #intermediate vector u of current A column
        # Orthogonality by subtracting other vectors:
        for s in range(i):
            R[s, i] = np.dot(Q[:, s], u)
            u -= R[s, i] * Q[:, s]
        # Normalize the orthogonalized vector:
        R[i, i] = np.linalg.norm(u)
        Q[:, i] = u / np.linalg.norm(u)

    return Q, R

Q_GS, R_GS = QR_Mod_GS




def QR_Household():
    Q = np.empty((), dtype = np.float32)
    R = np.empty((), dtype = np.float32)


    return R, Q.T
    
Q_HH, R_HH = QR_Household




#----- Plotting the random data -----#
x_GS = np.linalg.solve(R_GS, b)
x_HH = np.linalg.solve(R_HH, b)

plt.figure()
plt.scatter(t, b, 'ro')
plt.scatter(t, (A @ x_GS).ravel, 'go')
plt.scatter(t, (A @ x_HH).ravel, 'bo')
plt.xlabel('t')
plt.ylabel('b')
plt.savefig('Data.png')
