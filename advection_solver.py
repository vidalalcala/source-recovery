# advection-solver.py

from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from nufft import nufft1 as nufft_fortran
from nufft import nufft3 as nufft3
from random import uniform as uniform

def nufftfreqs(M, df=1):
    """Compute the frequency range used in nufft for M frequency bins"""
    return df * np.arange(-(M // 2), M - (M // 2))

def nudft(x, y, xi, iflag=1):
    """Non-Uniform Direct Fourier Transform"""
    sign = -1 if iflag < 0 else 1
    return (1.0 / len(x)) *np.dot(y, np.exp(sign * 1j * xi * x[:, np.newaxis]))
    
def forward_dft(x, u_initial, psi, f, T, R):
    """Solves advection-diffusion equation forward in time.

    The equation is
    $$
    u_t - u_{xx} - \psi u_x - f(x) = 0 \:,
    $$
    with initial condition $u(x,0) = u_initial$. We assume that 
    the support of f(x) is a subset of [-R/2, R/2].

    Args:
        x: uniform grid in physical space, dx = $R / N$
        u_initial: initial field value
        psi: advection coefficient
        f: time independent source
        T: final time
        R: The support of the source is a subset of $[-R,R]$
    
    Returns:
        u_final: final field value

    """
    N = len(x)
    # nufft frecuencies
    xi = np.arange(-(N // 2), N - (N // 2))
    zero_freq_index = N // 2
    
    # Fourier transform
    u_hat_initial = R * nudft(x, u_initial, xi, 1)
    f_hat = R * nudft(x, f, xi, 1)
    
    # Solve ODE's analitically in Fourier space
    a = (1j * psi - xi) * xi
    u_hat_final = np.zeros(N)
    u_hat_final = np.array(u_hat_final, dtype=complex)
    
    for n in range (0, zero_freq_index):
        u_hat_final[n] = (
            u_hat_initial[n] * np.exp(a[n] * T) - 
            (f_hat[n] / a[n]) * (1.0 - np.exp(a[n] * T)))
    for n in range (zero_freq_index + 1, N):
        u_hat_final[n] = (
            u_hat_initial[n] * np.exp(a[n] * T) - 
            (f_hat[n] / a[n]) * (1.0 - np.exp(a[n] * T)))    
    u_hat_final[zero_freq_index] = (
        u_hat_initial[zero_freq_index] + 
        (T + 0j) * f_hat[zero_freq_index])
    
    # inverse Fourier transform
    u_final = (1.0 / (2.0 * np.pi)) * N * nudft(xi, u_hat_final, x, -1)
    return np.real(u_final)

def forward(x, u_initial, psi, f, T, R):
    """Solves advection-diffusion equation forward in time using the FFT.

    The equation is
    $$
    u_t - u_{xx} - \psi u_x - f(x) = 0 \:,
    $$
    with initial condition $u(x,0) = u_initial$. We assume that 
    the support of f(x) is a subset of [-R/2, R/2].

    Args:
        x: grid in physical space
        u_initial: initial field value
        psi: advection coefficient
        f: time independent source
        T: final time
        R: The support of the source is a subset of $[-R / 2, R / 2]$
    
    Returns:
        u_final: final field value

    """
    # add $R / 2$ to the grid, and evaluate Riemann sums on the right endpoint 
    x = np.append(x, R / 2.0)
    u_initial = np.append(u_initial, 0.0)
    f = np.append(f, 0.0)
    N = len(x)
    dx = np.zeros(N)
    dx[0] = x[0] - (- R / 2.0)
    dx[1 : N] = x[1 :  N] - x[0 : N - 1]
    
    # nufft frecuencies
    xi = (np.arange(-N // 2, N // 2) + N % 2)
    zero_freq_index = N // 2
    
    # Fourier transform, NOTICE that nufft3 calculates the normalized
    # sum: line 71 in https://github.com/dfm/python-nufft/blob/master/nufft/nufft.py
    u_hat_initial = N * nufft3(x, u_initial * dx, xi)
    f_hat = N * nufft3(x, f * dx, xi)
    
    # solve ODE's analitically in Fourier space
    a = (1j * psi - xi) * xi
    u_hat_final = np.zeros(N)
    u_hat_final = np.array(u_hat_final, dtype=complex)
    
    for n in range (0, zero_freq_index):
        u_hat_final[n] = (
            u_hat_initial[n] * np.exp(a[n] * T) - 
            (f_hat[n] / a[n]) * (1.0 - np.exp(a[n] * T)))
    for n in range (zero_freq_index + 1, N):
        u_hat_final[n] = (
            u_hat_initial[n] * np.exp(a[n] * T) - 
            (f_hat[n] / a[n]) * (1.0 - np.exp(a[n] * T)))    
    u_hat_final[zero_freq_index] = (
        u_hat_initial[zero_freq_index] + 
        (T + 0j) * f_hat[zero_freq_index])
    
    # inverse Fourier transform,  NOTICE that nufft3 calculates the normalized
    # sum: line 71 in https://github.com/dfm/python-nufft/blob/master/nufft/nufft.py
    u_final = (
        (1.0 / (2 * np.pi)) * N * 
        nufft3(xi, u_hat_final, x, iflag=-1))
    return np.real(u_final[0 : N - 1])

def forward_fft(x, u_initial, psi, f, T, R):
    """Solves advection-diffusion equation forward in time using the FFT.

    The equation is
    $$
    u_t - u_{xx} - \psi u_x - f(x) = 0 \:,
    $$
    with initial condition $u(x,0) = u_initial$. We assume that 
    the support of f(x) is a subset of [-R/2, R/2].

    Args:
        x: uniform grid in physical space, $dx = R / N$
        u_initial: initial field value
        psi: advection coefficient
        f: time independent source
        T: final time
        R: The support of the source is a subset of $[-R,R]$
    
    Returns:
        u_final: final field value

    """
    N = len(x)
    # nufft frecuencies
    xi = (np.arange(-N // 2, N // 2) + N % 2)
    zero_freq_index = N // 2
    
    # Fourier transform
    u_hat_initial = R * nufft_fortran(x, u_initial, N)
    f_hat = R * nufft_fortran(x, f, N)
    
    # Solve ODE's analitically in Fourier space
    a = (1j * psi - xi) * xi
    u_hat_final = np.zeros(N)
    u_hat_final = np.array(u_hat_final, dtype=complex)
    
    for n in range (0, zero_freq_index):
        u_hat_final[n] = (
            u_hat_initial[n] * np.exp(a[n] * T) - 
            (f_hat[n] / a[n]) * (1.0 - np.exp(a[n] * T)))
    for n in range (zero_freq_index + 1, N):
        u_hat_final[n] = (
            u_hat_initial[n] * np.exp(a[n] * T) - 
            (f_hat[n] / a[n]) * (1.0 - np.exp(a[n] * T)))    
    u_hat_final[zero_freq_index] = (
        u_hat_initial[zero_freq_index] + 
        (T + 0j) * f_hat[zero_freq_index])
    
    # inverse Fourier transform
    u_final = (
        (1.0 / (2 * np.pi)) * N * 
        nufft_fortran(xi * (R / N), u_hat_final, N, iflag=-1))
    return np.real(u_final)

def forwardGradient(x, u_initial, psi, f, T, R):
    """Solves for u_{x} in the advection-diffusion equation,
    forward in time using the FFT.

    The equation is
    $$
    u_t - u_{xx} - \psi u_x - f(x) = 0 \:,
    $$
    with initial condition $u(x,0) = u_initial$. We assume that 
    the support of f(x) is a subset of [-R/2, R/2].

    Args:
        x: grid in physical space. 
        u_initial: initial field value
        psi: advection coefficient
        f: time independent source
        T: final time
        R: The support of the source is a subset of $[-R,R]$
    
    Returns:
        u_final: final field value

    """
    # add $R / 2$ to the grid, and evaluate Riemann sums on the right endpoint 
    x = np.append(x, R / 2.0)
    u_initial = np.append(u_initial, 0.0)
    f = np.append(f, 0.0)
    N = len(x)
    dx = np.zeros(N)
    dx[0] = x[0] - (- R / 2.0)
    dx[1 : N] = x[1 :  N] - x[0 : N - 1]
    
    # nufft frecuencies
    xi = (np.arange(-N // 2, N // 2) + N % 2)
    zero_freq_index = N // 2
    
    # Fourier transform
    u_hat_initial = N * nufft3(x, u_initial * dx, xi)
    f_hat = N * nufft3(x, f * dx, xi)
    
    # Solve ODE's analitically in Fourier space
    a = (1j * psi - xi) * xi
    u_hat_final_x = np.zeros(N)
    u_hat_final_x = np.array(u_hat_final_x, dtype=complex)
    
    # $u_x$ in Fourier space is equivalent to multiplication by $-i * \xi$
    u_hat_initial_x = u_hat_initial * (-1j * xi)
    f_hat_x = f_hat * (-1j * xi)
    
    for n in range (0, zero_freq_index):
        u_hat_final_x[n] = (
            u_hat_initial_x[n] * np.exp(a[n] * T) - 
            (f_hat_x[n] / a[n]) * (1.0 - np.exp(a[n] * T)))
    for n in range (zero_freq_index + 1, N):
        u_hat_final_x[n] = (
            u_hat_initial_x[n] * np.exp(a[n] * T) - 
            (f_hat_x[n]/a[n]) * (1.0 - np.exp(a[n] * T)))    
    u_hat_final_x[zero_freq_index] = (
        u_hat_initial_x[zero_freq_index] + 
        (T + 0j) * f_hat_x[zero_freq_index])
    
    # inverse Fourier transform
    u_final_x = (
        (1.0 / (2.0 * np.pi)) * N * 
        nufft3(xi, u_hat_final_x, x, iflag=-1))
    return np.real(u_final_x[0 : N - 1])

def forwardDemo():
    """ Quick demo of the forward functions """
    N = 1000  # number of observations
    T = 0.0001
    t = 0.0001
    psi = 2000
    R = 1
    dx = R / N
    x = np.linspace(- R / 2.0 + dx, R / 2.0, N )  # position along the rod
    u_initial = np.exp(- x * x / (4.0 * t)) / np.sqrt(4.0 * np.pi * t)
    f = 1000 * np.exp( - x * x / (4.0 * t)) / np.sqrt( 4.0 * np.pi * t)
    u_final = forward(x, u_initial, psi, f, T, R)
    u_final_x = forwardGradient(x, u_initial, psi, f, T, R)
    
    # plots
    pp = PdfPages('./images/advection_solver_demo.pdf')
    
    # plot u
    plt.plot(x, u_initial, 'r', label="$t=0$")
    plt.plot(x, u_final, 'b', label="$t=T$")
    plt.legend(
        bbox_to_anchor=(0., 1.02, 1., .102), loc=3, 
        ncol=2, mode="expand", borderaxespad=0.)
    plt.xlabel('$x$')
    plt.ylabel('$u$')
    pp.savefig()
    
    # plot u_x
    plt.figure()
    plt.plot(x, u_final_x, 'b', label="$t=T$")
    plt.legend(
        bbox_to_anchor=(0., 1.02, 1., .102), loc=3, 
        ncol=2, mode="expand", borderaxespad=0.)
    plt.xlabel('$x$')
    plt.ylabel('$u_x$')
    pp.savefig()
    
    # show plots and close pdf
    plt.show()
    pp.close()
    
    
 
def u_advection(x, t, psi):
    """Analytical solution of advection-diffusion equation forward in time.

    The equation is
    $$
    u_t - u_{xx} - \psi u_x = 0 \:,
    $$
    with initial condition $u(x,0) = \delta_{x=0}$.

    Args:
        x: grid in physical space
        t: time
        psi: advection coefficient, units [x]/[t]
    
    Returns:
        u: field value at time t evaluated over the vector x

    """	
    u = (
        np.exp(- (x - psi * t) * (x - psi * t) / (4.0 * t)) / 
        np.sqrt(4.0 * np.pi * t))
    return u
 
def forwardTest():
    """ Quick test of the forward function, with ZERO SOURCE"""
    N = 10000 # number of observations
    T = 0.0001
    t = 0.0001
    psi = 1000.0
    R = 2.0
    
    # position along the rod
    x = np.zeros(N)
    for i in range(0, N):
        x[i] = uniform(-R / 2, R / 2)
    x = np.sort(x)

    u_initial = u_advection(x, t, psi)
    f = np.zeros(N)
    u_final = forward(x, u_initial, psi, f, T, R)
    u_final_analytical = u_advection(x, t + T, psi)
    plt.plot(x, u_initial, 'g', label="$t = 0$")
    print("length x : ", len(x))
    print("length u_final : ", len(u_final))
    plt.plot(x, u_final, 'b', label="numerical soln, $t = T$")
    plt.plot(x, u_final_analytical, 'r--', label="analytical soln, $t = T$")
    plt.legend(
        bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", 
        borderaxespad=0.)
    plt.xlabel('$x$')
    plt.ylabel('$u$')
    pp = PdfPages('./images/advection_solver_test.pdf')
    pp.savefig()
    plt.show()
    pp.close()


# if module runs as a script run demo
if __name__ == "__main__":
    forwardDemo()
