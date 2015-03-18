# advection-solver.py

from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from nufft import nufft1 as nufft_fortran

def nufftfreqs(M, df=1):
    """Compute the frequency range used in nufft for M frequency bins"""
    return df * np.arange(-(M // 2), M - (M // 2))

def nudft(x, y, xi, iflag=1):
    """Non-Uniform Direct Fourier Transform"""
    sign = -1 if iflag < 0 else 1
    return (1.0/len(x)) *np.dot(y, np.exp(sign * 1j * xi * x[:, np.newaxis]))
    
def forward(x, u_initial, psi, f, T, R):
    """Solves advection-diffusion equation forward in time.

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
        R: The support of the source is a subset of $[-R,R]$
    
    Returns:
        u_final: final field value

    """
    N = len(x)
    # nufft frecuencies
    xi = np.arange(-(N // 2), N - (N // 2))
    zero_freq_index = N // 2
    
    # Fourier transform
    u_hat_initial = R*nudft(x, u_initial, xi, 1)
    f_hat = R*nudft(x, f, xi, 1)
    
    # Solve ODE's analitically in Fourier space
    a = (1j*psi - xi) * xi
    u_hat_final = np.zeros(N)
    u_hat_final = np.array(u_hat_final, dtype=complex)
    
    for n in range (0, zero_freq_index):
        u_hat_final[n] = (u_hat_initial[n] * np.exp(a[n] * T) - (f_hat[n]/a[n]) * (1.0 - np.exp(a[n] * T)))
    for n in range (zero_freq_index + 1, N):
        u_hat_final[n] = (u_hat_initial[n] * np.exp(a[n] * T) - (f_hat[n]/a[n]) * (1.0 - np.exp(a[n] * T)))    
    u_hat_final[zero_freq_index] = u_hat_initial[zero_freq_index] + (T + 0j) * f_hat[zero_freq_index]
    
    # inverse Fourier transform
    u_final = (1.0/(2*np.pi))*N*nudft(xi, u_hat_final, x, -1)
    return u_final

def forward_nufft(x, u_initial, psi, f, T, R):
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
        R: The support of the source is a subset of $[-R,R]$
    
    Returns:
        u_final: final field value

    """
    N = len(x)
    # nufft frecuencies
    xi = (np.arange(-N // 2, N // 2) + N % 2)
    zero_freq_index = N // 2
    
    # Fourier transform
    u_hat_initial = R*nufft_fortran(x, u_initial, N)
    f_hat = R*nufft_fortran(x, f, N)
    
    # Solve ODE's analitically in Fourier space
    a = (1j*psi - xi) * xi
    u_hat_final = np.zeros(N)
    u_hat_final = np.array(u_hat_final, dtype=complex)
    
    for n in range (0, zero_freq_index):
        u_hat_final[n] = (u_hat_initial[n] * np.exp(a[n] * T) - (f_hat[n]/a[n]) * (1.0 - np.exp(a[n] * T)))
    for n in range (zero_freq_index + 1, N):
        u_hat_final[n] = (u_hat_initial[n] * np.exp(a[n] * T) - (f_hat[n]/a[n]) * (1.0 - np.exp(a[n] * T)))    
    u_hat_final[zero_freq_index] = u_hat_initial[zero_freq_index] + (T + 0j) * f_hat[zero_freq_index]
    
    # inverse Fourier transform
    u_final = (1.0/(2*np.pi))*N*nufft_fortran(xi*(R/N), u_hat_final, N, iflag=-1)
    return u_final

def forwardDemo():
    """ Quick demo of the forward function """
    N = 1000  # number of observations
    T = 0.0001
    t = 0.0001
    psi = 2000
    R = 1
    dx = R/N
    x = np.linspace(- R/2.0 + dx, R/2.0, N ) # position along the rod
    u0 = np.exp(-(x)*(x)/(4.0*t))/np.sqrt(4.0*np.pi*t)
    f = 1000*np.exp(-(x)*(x)/(4.0*t))/np.sqrt(4.0*np.pi*t)
    uT = forward(x, u0, psi, f, T, R)
    plt.plot(x, u0, 'r', label="$t=0$")
    plt.plot(x, uT, 'b', label="$t=T$")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, 
    	    ncol=2, mode="expand", borderaxespad=0.)
    plt.xlabel('$x$')
    plt.ylabel('$u$')
    pp = PdfPages('./images/advection-solver-demo.pdf')
    pp.savefig()
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
    u = np.exp(-(x - psi*t)*(x - psi*t)/(4.0*t))/np.sqrt(4.0*np.pi*t)
    return u
 
def forwardTest():
    """ Quick test of the forward function, with ZERO SOURCE"""
    N = 1000 # number of observations
    T = 0.0001
    t = 0.0001
    psi = 1000.0
    R = 2.0
    dx = R/N
    x = np.linspace(- R/2.0 , R/2.0 - dx , N ) # position along the rod
    u0 = u_advection(x, t, psi)
    f = np.zeros(N)
    uT = forward(x, u0, psi, f, T, R)
    uT_analytical = u_advection(x, t + T, psi)
    plt.plot(x, u0, 'g', label="$t=0$")
    plt.plot(x, uT, 'b', label="numerical soln, $t=T$")
    plt.plot(x, uT_analytical, 'r--', label="analytical soln, $t=T$")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, 
    	    ncol=2, mode="expand", borderaxespad=0.)
    plt.xlabel('$x$')
    plt.ylabel('$u$')
    pp = PdfPages('./images/advection-solver-test.pdf')
    pp.savefig()
    plt.show()
    pp.close()

