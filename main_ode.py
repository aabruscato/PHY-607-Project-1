import numpy as np
import matplotlib.pyplot as plt
from ode_solver import (
    analytic_solution,
    ode_rhs,
    euler_method,
    rk4_method,
    using_scipy,
    save_results
)

# constants
m1 = 1.0
m2 = 1.0
k1 = 1.0
k2 = 1.0
k3 = 1.0

# initial conditions
x1_0 = 1.0
x2_0 = 0.0
v1_0 = 0.0
v2_0 = 0.0

y0 = np.array([x1_0, x2_0, v1_0, v2_0])  # state vector

t0 = 0.0
tf = 20.0
dt = 0.01
t = np.arange(t0, tf + dt, dt)

# solving the ODE
analytic_x = analytic_solution(t, m1, m2, k1, k2, k3, [x1_0, x2_0], [v1_0, v2_0])
euler_x = euler_method(ode_rhs, y0, t, m1, m2, k1, k2, k3)
rk4_x = rk4_method(ode_rhs, y0, t, m1, m2, k1, k2, k3)
scipy_x = using_scipy(y0, t, m1, m2, k1, k2, k3)

# results
save_results(t, analytic_x, euler_x, rk4_x, scipy_x, filename="ode_results.xlsx")

plt.figure(figsize=(10, 6))

plt.plot(t, analytic_x[0], label="Analytic x1", color="black", linestyle="--")
plt.plot(t, analytic_x[1], label="Analytic x2", color="gray", linestyle="--")

plt.plot(t, euler_x[0], label="Euler x1", alpha=0.7)
plt.plot(t, euler_x[1], label="Euler x2", alpha=0.7)

plt.plot(t, rk4_x[0], label="RK4 x1", alpha=0.7)
plt.plot(t, rk4_x[1], label="RK4 x2", alpha=0.7)

plt.plot(t, scipy_x[0], label="SciPy x1", alpha=0.7)
plt.plot(t, scipy_x[1], label="SciPy x2", alpha=0.7)

plt.xlabel("Time")
plt.ylabel("Displacement")
plt.title("2-Mass 3-Spring System: ODE Solutions")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
