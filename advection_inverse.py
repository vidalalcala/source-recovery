# advection_inverse.py

from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from advection_solver import forward as forward
from advection_solver import forwardGradient as forwardGradient

def gradient(x, u_final_data, f, psi, T, R, M):
    """ Gradient of the objective function
    $$
    F = 0.5 * || u_final - u_final_data ||_2^2
    $$

    Args:
        x: physical space grid
        f: source in advection equation
        u_final_data: observation of the field at time $T$
        psi: advection coefficient
        T: final time
        M: Nb of time intervals
        R: Source support is contained in $[- R / 2, R / 2]$

    Returns:
        F_f: gradient with respect to the source term
        F_psi: derivative with respect to advection coefficient

    """
    # grid related definitions
    dt = T / M
    N = len(x)
    dx = np.zeros(N)
    dx[0] = x[0] - (- R / 2.0)
    dx[1 : N] = x[1 :  N] - x[0 : N - 1]

    # we will accumulate the gradient over time
    F_f = np.zeros(len(x))
    F_psi = 0.0

    u_final = forward(x, np.zeros(len(x)), psi, f, T, R)

    for i in range(1, M + 1):
        t = i * dt
        # solve adjoint problem in w
        w = forward(x, u_final_data - u_final, -psi, np.zeros(len(x)), T - t, R)
        # solve forward problem for u_x
        u_x = forwardGradient(x, np.zeros(len(x)), psi, f, t, R)
        # accumulate gradient
        F_f = F_f - dt * w
        F_psi = F_psi - np.dot(u_x * w, dx                                                                                                                                                                                                                                                                                                                                              ) * dt
    
    return F_f, F_psi

def gradientTest():
    # problem parameters
    T = 0.0001
    R = 1.0
    N = 100  # Nb of grid points in physical space
    M = 10  # Nb of grid points in time space

    # synthetic data
    x = np.linspace(- R / 2.0 + R / N, R / 2.0, N)
    t_f = T
    f_optimal = (
        (1.0 / t_f) * np.exp( - x * x / (4.0 * t_f)) /
        np.sqrt( 4.0 * np.pi * t_f))
    psi_optimal = 2000
    u_final_data = forward(x, np.zeros(len(x)), psi_optimal, f_optimal, T, R)

    # initial coefficients
    f = np.zeros(len(x))
    psi = 0.0

    F_f, F_psi = gradient(x, u_final_data, f, psi, T, R, M)

    return F_f, F_psi

def recoverDemo():
    # problem parameters
    T = 0.0001
    R = 1.0
    N = 1000  # Nb of grid points in physical space
    M = 10  # Nb of grid points in time space
    nb_grad_steps = 1000  # Nb of updates with the gradient
    alpha_f = 100000000.0  # gradient update size
    alpha_psi = 1000.0
    
    # plots
    pp = PdfPages('./images/recover_demo.pdf')

    # synthetic data
    x = np.linspace(- R / 2.0 + R / N, R / 2.0, N)
    t_f = T
    f_optimal = (
        (1.0 / t_f) * np.exp( - x * x / (4.0 * t_f)) /
        np.sqrt( 4.0 * np.pi * t_f))
    psi_optimal = 1000
    u_final_data = forward(x, np.zeros(len(x)), psi_optimal, f_optimal, T, R)

    # initial coefficients   
    #f = np.zeros(len(x))
    f = (
        (1.0 / t_f) * np.exp( - (x - 0.1) * (x - 0.1) / (4.0 * t_f)) /
       np.sqrt( 4.0 * np.pi * t_f))
    psi = 1000.0
    
    # plot f
    plt.figure()    
    plt.plot(x, f, 'r', label="initial source")
    plt.plot(x, f_optimal, 'b', label="true source")
    plt.legend(
        bbox_to_anchor=(0., 1.02, 1., .102), loc=3, 
        ncol=2, mode="expand", borderaxespad=0.)
    plt.xlabel('$x$')
    plt.ylabel('$f$')
    pp.savefig()
    
    for i in range(0, nb_grad_steps):
        F_f, F_psi = gradient(x, u_final_data, f, psi, T, R, M)
        f = f - alpha_f * F_f
        psi = psi - alpha_psi * F_psi
    
    print("psi_recovered : ", psi)
    print("psi_optimal : ", psi_optimal)
    print("F_psi : ", F_psi)
    
    u_final_recovered = forward(x, np.zeros(len(x)), psi, f, T, R)    
    
    # plot u_final_data
    plt.figure()
    plt.plot(x, u_final_data, 'b', label="data")
    plt.plot(x, u_final_recovered, 'r--', label="recovered")
    plt.legend(
        bbox_to_anchor=(0., 1.02, 1., .102), loc=3, 
        ncol=2, mode="expand", borderaxespad=0.)
    plt.xlabel('$x$')
    plt.ylabel('$u_{T}$')
    pp.savefig()    
    
    # plot f
    plt.figure()    
    plt.plot(x, f, 'r', label="source recovered")
    plt.plot(x, f_optimal, 'b', label="true source")
    plt.legend(
        bbox_to_anchor=(0., 1.02, 1., .102), loc=3, 
        ncol=2, mode="expand", borderaxespad=0.)
    plt.xlabel('$x$')
    plt.ylabel('$f$')
    pp.savefig()

    #plot F_f
    plt.figure()
    plt.plot(x, F_f, label="source sensitivity")
    plt.xlabel('$x$')
    plt.ylabel('$F_f$')
    pp.savefig()
    
    # show and close pdf file
    pp.close()    
    plt.show()
    


# if module runs as a script, run Test
if __name__ == "__main__":
    recoverDemo()
