import numpy as np
from scipy.linalg import eigh
from scipy.integrate import solve_ivp
import pandas as pd

'''
This module solves the ODE for the undamped/ideal coupled oscillator system of
two masses m1 and m2 and three spring contstants k1, k2, k3. Set up appears as
|   k1    m1   k2   m2   k3   |
|-=-=-=-=-O-=-=-=-=-O-=-=-=-=-|
|                             |
'''

def get_w1w2(m1, m2, k1, k2, k3):
    
    """
    Given the matrix equation: [M][xdd] + [K][x] = [0]
    We solve for the determinant = 0 which gives the equation below,
    (w^2*m1 - k1 - k2)*(w^2*m2 - k3 - k2) - k2^2 = 0
    Which simplifies to the quadratic equation of w^2 below,
    a*w^4 - bw^2 + c = 0 where a = m1*m2
                               b = m1(k2 + k3) + m2(k1 + k2)
                               c = (k1 + k2)*(k2 + k3) - k2^2
    
    We can then solve for w1 and w2 using the quadratic formula in w^2.
    """
    
    # w^2 quadratic coeficients
    a = m1 * m2
    b = m2 * (k1 + k2) + m1 * (k2 + k3)
    c = (k1 + k2) * (k2 + k3) - k2**2
    
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        raise ValueError("ERROR: Can't solve for w1 & w2 ---> check parameters.")

    w1_sqrd = (b - np.sqrt(discriminant)) / (2 * a)
    w2_sqrd = (b + np.sqrt(discriminant)) / (2 * a)

    w1 = np.sqrt(w1_sqrd)
    w2 = np.sqrt(w2_sqrd)

    w1, w2 = sorted([w1, w2])
    
    return w1, w2

def calc_v1v2(m1, m2, k1, k2, k3):
    
    """
    [K][v] = w^2[M][v]
    eigenvalues: w^2
    eignenvectors: v
    """
    
    M = np.array([[m1, 0],
                  [0, m2]])
    
    K = np.array([[k1 + k2, -k2],
                  [-k2, k2 + k3]])

    eigvals, eigvecs = eigh(K, M)

    # Sort eigenvalues and corresponding eigenvectors
    i = eigvals.argsort()
    eigvecs = eigvecs[:, i]

    # Normalize eigenvectors
    v1 = eigvecs[:, 0] / eigvecs[0, 0]
    v2 = eigvecs[:, 1] / eigvecs[0, 1]

    return v1, v2

def analytic_solution(t, m1, m2, k1, k2, k3, x0, v0):

    w1, w2 = get_w1w2(m1, m2, k1, k2, k3)
    v1, v2 = calc_v1v2(m1, m2, k1, k2, k3)

    V = np.column_stack((v1, v2))

    # sine and cosine coefficeints
    C = np.linalg.solve(V, x0)
    D = np.linalg.solve(V, v0 / np.array([w1, w2]))

    x = np.zeros((2, len(t))) # positions of m1 & m2 @ t
    
    for i, ti in enumerate(t):
    
        cos_terms = C * np.cos(np.array([w1, w2]) * ti)
        sin_terms = D * np.sin(np.array([w1, w2]) * ti)

        A1 = cos_terms[0] + sin_terms[0]
        A2 = cos_terms[1] + sin_terms[1]

        x1 = v1[0] * A1 + v2[0] * A2
        x2 = v1[1] * A1 + v2[1] * A2

        x[0, i] = x1
        x[1, i] = x2

    return x

def ode_rhs(t, y, m1, m2, k1, k2, k3):

    """
    y = [x1, x2, v1, v2]
    """
    
    x1, x2, v1, v2 = y

    dx1dt = v1
    dx2dt = v2

    F1 = -k1 * x1 + k2 * (x2 - x1)
    F2 = -k3 * x2 - k2 * (x2 - x1)

    dv1dt = F1 / m1
    dv2dt = F2 / m2

    return np.array([dx1dt, dx2dt, dv1dt, dv2dt])

def euler_method(f, y0, t, m1, m2, k1, k2, k3):

    ys = np.zeros((len(t), len(y0)))
    ys[0] = y0

    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        ys[i] = ys[i-1] + dt * f(t[i-1], ys[i-1], m1, m2, k1, k2, k3)

    return ys.T

def rk4_method(f, y0, t, m1, m2, k1, k2, k3):

    ys = np.zeros((len(t), len(y0)))
    ys[0] = y0

    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        ti = t[i-1]
        yi = ys[i-1]

        k1_ = f(ti, yi, m1, m2, k1, k2, k3)
        k2_ = f(ti + dt/2, yi + dt*k1_/2, m1, m2, k1, k2, k3)
        k3_ = f(ti + dt/2, yi + dt*k2_/2, m1, m2, k1, k2, k3)
        k4_ = f(ti + dt, yi + dt*k3_, m1, m2, k1, k2, k3)

        ys[i] = yi + (dt/6) * (k1_ + 2*k2_ + 2*k3_ + k4_)


    return ys.T

def using_scipy(y0, t, m1, m2, k1, k2, k3):

    sol = solve_ivp(
        fun=lambda t, y: ode_rhs(t, y, m1, m2, k1, k2, k3),
        t_span=(t[0], t[-1]),
        y0=y0,
        t_eval=t,
        method='RK45'
    )
    return sol.y

def save_results(t, analytic_x, euler_x, rk4_x, scipy_x, filename='results.xlsx'):

    data = {
        'time': t,
        'analytic_x1': analytic_x[0],
        'analytic_x2': analytic_x[1],
        'euler_x1': euler_x[0],
        'euler_x2': euler_x[1],
        'rk4_x1': rk4_x[0],
        'rk4_x2': rk4_x[1],
        'scipy_x1': scipy_x[0],
        'scipy_x2': scipy_x[1]
    }
    df = pd.DataFrame(data)
    df.to_excel(filename, index=False)
    
    print(f"Results saved to {filename}")


    
