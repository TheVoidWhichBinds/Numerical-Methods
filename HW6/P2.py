import numpy as np
import matplotlib.pyplot as plt




########################################## SUBSECTION A ########################################## 

#------- Function -------#
# Exact function:
def f1(x):
    return x*np.exp(x) - 1


#------- Fixed-Point Method -------#
def fixed_point(k):
    x_0 = 0.5 # initialization
    max_iter = 10000 # max iterations
    num_iter = 0
    x = x_0 #starting value
    #
    while np.abs(f1(x)) > 1E-8 and num_iter < max_iter:
        x = x + k*f1(x)
        num_iter += 1
    #
    return num_iter


#--------------- Plotting ---------------#
def a_plotting():
    k_range = np.arange(-0.7,0,0.02)
    #
    plt.figure()
    plt.title('Fixed Point Weight Choice')
    plt.xlabel('Weight k Value')
    plt.ylabel('Iterations to Threshold')
    iters = []
    for k in k_range:
        iters.append(fixed_point(k))
    plt.scatter(k_range, iters, color='magenta')
    plt.savefig('fixed_pt.png')
    plt.show()
    print(f'k_opt iterations: {fixed_point(-0.404036086)}') #optimal k as analytically found
    return()
        



########################################## SUBSECTION C ########################################## 

#----- Function & Derivative -----#
# Function:
def f2(x):
    return 1/x + 2*np.log(x) - 2

# Function Derivative:
def f2_prime(x):
    return -1/x**2 + 2/x 


#------------ 1D Newton Method ------------#
def newton1D(init_guess, f, f_prime):
    x = np.asarray(init_guess, dtype=float)
    max_iter = 10000
    x_k = [x.copy()]
    Delta_x = 1
    num_iter = 0
    #
    while np.abs(f(x)) > 1E-8 and num_iter < max_iter:
        Delta_x = - f(x)/f_prime(x)
        x = x + Delta_x
        x_k.append(x.copy()) 
        num_iter += 1
    # 
    return x, np.array(x_k)


#--------- Plotting ---------#
def c_plotting():
    x = np.linspace(0.1,3,100)
    plt.figure()
    plt.title('1D Newton Method')
    plt.plot(x, f2(x), label='f(x)')
    root1 = newton1D(0.1, f2, f2_prime)[0]
    root2 = newton1D(2, f2, f2_prime)[0]
    plt.scatter([root1, root2], [0,0], label='roots')
    plt.grid(True)
    plt.savefig('1D_Newton.png')
    plt.show()




########################################## SUBSECTION D ########################################## 

#----- Functions and Jacobian -----#
# Parabola and ellipse:
def intersect(coord):
    x, y = coord
    F1 = x**2 / 16 + y**2 / 4 - 1
    F2 = y - (x**2 - 2) 
    return np.array([F1, F2])

# Jacobian:
def intersect_jac(coord):
    x, y = coord
    J = np.array([
        [x / 8.0, y / 2.0],
        [-2.0 * x, 1.0],
    ])
    return J


#------------------- 2D Newton Method --------------------#
def newton2D(init_guess, F, J):
    xy = np.array(init_guess, dtype=float)  # forces a copy
    xy_k = [xy.copy()]
    num_iter = 0
    max_iter = 10000
    Delta_xy = np.full_like(xy, np.inf)
    #
    while np.linalg.norm(F(xy),2) > 1E-8 and num_iter < max_iter:
        F_k = F(xy)
        J_k = J(xy)
        Delta_xy = -np.linalg.solve(J_k, F_k)
        xy += Delta_xy
        xy_k.append(xy.copy())
        num_iter += 1
    #
    return np.array(xy_k)


#----------- Plotting -----------#
def d_plotting(guess_coords):
    plt.figure()
    plt.title('2D Newton Method')
    plt.xlabel('x')
    plt.ylabel('y')
    #
    x = np.linspace(-4,4,10000)
    y_ellipse = np.sqrt(4 - x**2/4)
    y_parabola = x**2 - 2
    #
    plt.plot(x, y_ellipse, color='black',label='ellipse')
    plt.plot(x, -y_ellipse, color='black')
    plt.plot(x, y_parabola, color='grey', label='parabola')
    #
    for i, g in enumerate(guess_coords):
        iter_coords = newton2D(g, intersect, intersect_jac)
        cmap = plt.cm.tab10(i+1)
        alphas = np.linspace(0.3, 1.0, len(iter_coords))
        for k in range(len(iter_coords)):
            plt.scatter(iter_coords[k,0], iter_coords[k,1], color=cmap, alpha=alphas[k])
    #
    plt.legend()
    plt.grid(True)
    plt.savefig('2D_Newton.png')
    plt.show()
    return


#------------------ Error & Convergence -----------------#
# Prints convergence order for each iteration:
def convergence_order(guess_coords):
    for g in guess_coords:
        xy_iter = newton2D(g, intersect, intersect_jac)[1]
        xy_star = xy_iter[-1] # iteratively solved solution
        errors = np.linalg.norm(xy_iter - xy_star, axis=1) #errors at each iteration
        e = errors[errors > 1e-15] # removing ~ zero errors so no log(0) issue arises
        #
        p_vals = []
        for k in range(1, len(e)-1):
            p_vals.append(np.log(e[k+1]/e[k]) / np.log(e[k] / e[k-1]))
        p_vals = np.array(p_vals)
        #
        print(f'Convergence order for guess {g}:')
        for i, p in enumerate(p_vals):
            print(f'p[{i}] = {p}')
    #
    return




########################################## SUBSECTIONS A,C,D FUNCTION CALLS ########################################## 
#a_plotting()
#c_plotting()
d_plotting(np.array([[0.01, 3],[-0.01, 3],[2, -2]]))
#convergence_order(np.array([[0.01, 3],[0.01, -3],[2, -2]]))
