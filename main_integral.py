import numpy as np
import matplotlib.pyplot as plt
from ode_solver import analytic_solution, euler_method, rk4_method, using_scipy, ode_rhs
from integral_solver import calc_energy, riemann_sum, trapezoidal_rule, simpsons_rule, compare_to_scipy

# parameters & initial conditions
m1 = m2 = 1.0
k1 = k2 = k3 = 1.0

x1_0, x2_0 = 1.0, 0.0
v1_0, v2_0 = 0.0, 0.0
y0 = np.array([x1_0, x2_0, v1_0, v2_0])

t0, tf, dt = 0.0, 20.0, 0.01
t = np.arange(t0, tf + dt, dt)

# solving ODEs
analytic_x = analytic_solution(t, m1, m2, k1, k2, k3, [x1_0, x2_0], [v1_0, v2_0])
analytic_v = np.gradient(analytic_x, t, axis=1)
analytic_states = np.vstack([analytic_x, analytic_v])

euler_res = euler_method(ode_rhs, y0, t, m1, m2, k1, k2, k3)
rk4_res   = rk4_method(ode_rhs, y0, t, m1, m2, k1, k2, k3)
scipy_res = using_scipy(y0, t, m1, m2, k1, k2, k3)

# energy calculation function
def decomp_energy(states, m1=m1, m2=m2, k1=k1, k2=k2, k3=k3):
    x1, x2, v1, v2 = states
    KE = 0.5*m1*v1**2 + 0.5*m2*v2**2
    PE = 0.5*k1*x1**2 + 0.5*k2*(x2 - x1)**2 + 0.5*k3*x2**2
    return KE, PE, KE + PE

# compute energies for each method
KE_analytic, PE_analytic, E_analytic = decomp_energy(analytic_states)
KE_euler,    PE_euler,    E_euler    = decomp_energy(euler_res)
KE_rk4,      PE_rk4,      E_rk4      = decomp_energy(rk4_res)
KE_scipy,    PE_scipy,    E_scipy    = decomp_energy(scipy_res)

# plot energy drift
plt.figure(figsize=(8,5))
plt.plot(t, E_euler - E_euler[0], label="Euler")
plt.plot(t, E_rk4 - E_rk4[0], label="RK4")
plt.plot(t, E_scipy - E_scipy[0], label="SciPy RK45")
plt.xlabel("Time [s]")
plt.ylabel("Energy Drift")
plt.title("Energy Drift for Different Solvers")
plt.legend()
plt.grid(True)
plt.show()

# plot KE and PE for RK4
plt.figure(figsize=(8,5))
plt.plot(t, KE_rk4, label="Kinetic Energy")
plt.plot(t, PE_rk4, label="Potential Energy")
plt.plot(t, E_rk4, label="Total Energy", linestyle="--", color="green")
plt.xlabel("Time [s]")
plt.ylabel("Energy [J]")
plt.title("KE + PE Fluctuation using RK4")
plt.legend()
plt.grid(True)
plt.show()

# energy drift summary and Simpson integration
for label, E in [("Analytic", E_analytic),
                 ("Euler", E_euler),
                 ("RK4", E_rk4),
                 ("SciPy", E_scipy)]:
    drift = E[-1] - E[0]
    integral = simpsons_rule(t, E)
    print(f"{label:8s} | Drift = {drift:+.2e}, Simpson integral = {integral:.6f}")

# plot total energies
plt.figure(figsize=(8,5))
plt.plot(t, E_analytic, label="Analytic")
plt.plot(t, E_euler, label="Euler")
plt.plot(t, E_rk4, label="RK4")
plt.plot(t, E_scipy, label="SciPy RK45")
plt.xlabel("Time [s]")
plt.ylabel("Total Energy [J]")
plt.title("Energy vs Time")
plt.legend()
plt.grid(True)
plt.show()
