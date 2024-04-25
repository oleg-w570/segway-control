import numpy as np
from matplotlib import pyplot as plt

from discrete_observer_system import DiscreteObserverSystem

plt.style.use('seaborn-v0_8')

h = 0.1
obs = DiscreteObserverSystem(h)

end_time = 20.0
x0 = np.array([0.0, 0.01, -0.05, 0.01])

lin_t, lin_y, lin_u = obs.lin_sol(end_time, x0)
nonlin_t, nonlin_y, nonlin_u = obs.nonlin_sol(end_time, x0)

fig, axs = plt.subplots(5)
fig.set_figwidth(10)
fig.set_figheight(12)
ylabels = ['$\\theta$', '$x$', '$\\dot{\\theta}$', '$\\dot{x}$']
for i in range(4):
    axs[i].plot(lin_t, lin_y[i], color='red', linewidth=1)
    axs[i].plot(nonlin_t, nonlin_y[i], color='blue', linewidth=1)
    axs[i].set_xlabel('$t$')
    axs[i].set_ylabel(ylabels[i])
axs[4].plot(lin_t, lin_u, color='red', linewidth=1)
axs[4].plot(nonlin_t, nonlin_u, color='blue', linewidth=1)
axs[4].set_xlabel('$t$')
axs[4].set_ylabel('$V$')
fig.tight_layout()
plt.show()
