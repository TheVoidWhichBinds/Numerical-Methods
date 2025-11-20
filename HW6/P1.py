import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


#Exact Function:
def exact_func(t):
    f= np.cos(np.pi*t)*np.exp(t/2)
    return f

#Specific Data Points:
t_knots= np.array([0,1,2,3])
y_pts= exact_func(t_knots)



#-------- Custom Matrix of Cubic Spline Equations --------#
def matrix_splines():
    A = np.array([
        # Data Point Matching
        [1, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0],
        [1, 1, 1, 1,   0, 0, 0, 0,   0, 0, 0, 0],
        [0, 0, 0, 0,   1, 1, 1, 1,   0, 0, 0, 0],
        [0, 0, 0, 0,   1, 2, 4, 8,   0, 0, 0, 0],
        [0, 0, 0, 0,   0, 0, 0, 0,   1, 2, 4, 8],
        [0, 0, 0, 0,   0, 0, 0, 0,   1, 3, 9, 27],
        # First Derivative Continuity
        [0, 1,  2,  3,     0, -1, -2, -3,     0, 0, 0, 0],
        [0, 0,  0,  0,     0,  1,  4, 12,     0, -1, -4, -12],
        # Second Derivative Continuity
        [0, 0,  2,  6,     0,  0, -2, -6,     0, 0, 0, 0],
        [0, 0,  0,  0,     0,  0,  2, 12,     0, 0, -2, -12],
        # End Conditions (natural)
        [0, 0,  2,  0,     0, 0, 0, 0,        0, 0, 0, 0],
        [0, 0,  0,  0,     0, 0, 0, 0,        0, 0, 2, 18]
    ], dtype=float)

    b = np.array([
        1,
        -np.exp(0.5),
        -np.exp(0.5),
        np.e,
        np.e,
        -np.exp(1.5),
        0, 0,
        0, 0,
        0, 0
    ], dtype=float)

    coeff = np.linalg.solve(A, b)

    # Labels in correct spline order
    labels = [
        "a_0^0", "a_1^0", "a_2^0", "a_3^0",
        "a_0^1", "a_1^1", "a_2^1", "a_3^1",
        "a_0^2", "a_1^2", "a_2^2", "a_3^2"
    ]

    #Printing spline coefficients:
    print("Spline Coefficients:")
    print("--------------------")
    for lbl, val in zip(labels, coeff):
        print(f"{lbl} = {val:.10f}")

    return coeff


#Calling func and extracting coefficients:
coeff = matrix_splines() #length-12 vector
coeff_mat = coeff.reshape(3, 4) #rows: [a0^j, a1^j, a2^j, a3^j]



#------------------------ Plotting ---------------------------#
plt.figure()
plt.title('Cubic Splines Interpolation (Zero Conditions on 2nd Derivative)')
plt.xlabel('t')
plt.ylabel('y')
#Plotting each spline segment on its interval:
for j in range(3):
    t_start = t_knots[j]
    t_end   = t_knots[j+1]
    t_seg = np.linspace(t_start, t_end, 200)
    a0, a1, a2, a3 = coeff_mat[j]
    s_seg = a0 + a1*t_seg + a2*t_seg**2 + a3*t_seg**3

    plt.plot(t_seg, s_seg, lw=4, label=f"s_{j}(t) on [{t_start},{t_end}]")


#Plotting exact function:
t= np.linspace(0,3.1,100)
plt.plot(t, exact_func(t), color='black', label='Exact Function')
plt.scatter(t_knots, exact_func(t_knots), c='black')

#Plotting Python solver function:
spline_sol= CubicSpline(t_knots, y_pts)
plt.plot(t, spline_sol(t), color='red', label='SciPy Splines')

plt.legend(loc='lower left')
plt.show()



    
    