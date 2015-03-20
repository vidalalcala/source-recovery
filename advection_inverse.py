# advection_inverse.py

from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from random import uniform as uniform

from advection_solver import forward as forward
from advection_solver import forwardGradient as forwardGradient

def gradient(x, u_final_data, f, psi, T, R, M=100):
    """ Gradient of the objective function
    $$
    F = 0.5 * || u_final - u_final_data ||_2^2 + 
        0.5 * gamma2_source * || f ||_2^2 +
        0.5 * gamma2_advection * \psi^2
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
        w = forward(x, u_final - u_final_data, -psi, np.zeros(len(x)), T - t, R)
        # solve forward problem for u_x
        u_x = forwardGradient(x, np.zeros(len(x)), psi, f, t, R)
        # accumulate gradient
        F_f = F_f + dt * w
        F_psi = F_psi - np.dot(w, u_x * dx) * dt
    return F_f, F_psi

def gradientTest():
    # problem parameters
    T = 0.0001 
    R = 1.0
    N = 100  # Nb of grid points in physical space
    M = 1  # Nb of grid points in time space 
    gamma2_source = 0.0 
    gamma2_advection = 0.0

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
    
    print(" F_psi : ", F_psi)
    print(" F_f : ", F_f)
    
    
    return F_f, F_psi

# if module runs as a script, run Test
if __name__ == "__main__":
    gradientTest()

