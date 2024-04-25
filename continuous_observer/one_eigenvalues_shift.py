import numpy as np
from matplotlib import pyplot as plt

from continuous_observer import ContinuousObserver

plt.style.use('seaborn-v0_8')

obs = ContinuousObserver()

end_time = 20.0
x0 = np.array([0.3, 0.5, 0.1, 0.2])

eigenvalues = obs.get_eigenvalues()
print(f"Eigenvalues of A {eigenvalues}")

eigenvalues[1] = -1.0
print(f"Preferred eigenvalues {eigenvalues}")

L = obs.gain_matrix(eigenvalues)
print(f"Gain matrix {L}")

lin_t, lin_y = obs.lin_sol(end_time, x0)
nonlin_t, nonlin_y = obs.nonlin_sol(end_time, x0)

fig, axs = plt.subplots(4)
fig.set_figwidth(10)
fig.set_figheight(12)
ylabels = ['$\\theta$', '$x$', '$\\dot{\\theta}$', '$\\dot{x}$']
for i in range(4):
    axs[i].plot(lin_t, lin_y[i] - lin_y[i+4], color='red', linewidth=1)
    axs[i].plot(nonlin_t, nonlin_y[i] - nonlin_y[i+4], color='blue', linewidth=1)
    axs[i].set_xlabel('$t$')
    axs[i].set_ylabel(ylabels[i])
fig.tight_layout()
plt.show()
