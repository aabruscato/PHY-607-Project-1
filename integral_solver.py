import numpy as np
from scipy.integrate import trapz, simps, simpson

def calc_energy(y, m1, m2, k1, k2, k3):

    x1, x2, v1, v2 = y

    KE = 0.5 * m1 * v1**2 + 0.5 * m2 * v2**2

    # Potential energy from springs
    PE1 = 0.5 * k1 * x1**2
    PE2 = 0.5 * k2 * (x2 - x1)**2
    PE3 = 0.5 * k3 * x2**2

    PE = PE1 + PE2 + PE3

    E = T + U
    return E

def riemann_sum(t, f):

    dt = np.diff(t)
    
    return np.sum(f[:-1] * dt)

def trapezoidal_rule(t, f):
    
    return np.trapz(f, t)

from scipy.integrate import simpson

def simpsons_rule(t, f):

    return simpson(f, t)

from scipy.integrate import trapz, simps

def compare_to_scipy(f_vals, t):

    return {
        "trapz": trapz(f_vals, t),
        "simps": simps(f_vals, t)
    }

