import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

from segway_control import Experiment

plt.style.use('seaborn-v0_8')

e = Experiment()
eigA = np.linalg.eigvals(e.A)
eigA = eigA.astype(complex)

t = (0, 20)
x0 = np.array([-0.1, -0.1, 0.1, 0.1])

pref_eigA = eigA.copy()
pref_eigA[0] = complex(-0.5, +1)
pref_eigA[1] = complex(-0.5, -1)

theta = e.ackermann_control(pref_eigA)
print(theta)

lin_sol = solve_ivp(e.lin_func, t, x0, args=(theta,), method="LSODA")
nonlin_sol = solve_ivp(e.nonlin_func, t, x0, args=(theta,), method="LSODA")

lin_u = lin_sol.y.T @ theta.T
nonlin_u = nonlin_sol.y.T @ theta.T

fig, axs = plt.subplots(5)
fig.set_figwidth(10)
fig.set_figheight(12)
ylabels = ['$\\theta$', '$x$', '$\\dot{\\theta}$', '$\\dot{x}$']
for i in range(4):
    axs[i].plot(lin_sol.t, lin_sol.y[i], color='red', linewidth=1)
    axs[i].plot(nonlin_sol.t, nonlin_sol.y[i], color='blue', linewidth=1)
    axs[i].set_xlabel('$t$')
    axs[i].set_ylabel(ylabels[i])
axs[4].plot(lin_sol.t, lin_u, color='red', linewidth=1)
axs[4].plot(nonlin_sol.t, nonlin_u, color='blue', linewidth=1)
axs[4].set_xlabel('$t$')
axs[4].set_ylabel('$V$')
fig.tight_layout()
plt.show()
