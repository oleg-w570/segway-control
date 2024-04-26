import numpy as np
from matplotlib import pyplot as plt

from discrete_observer import DiscreteObserver

plt.style.use('seaborn-v0_8')

h = 0.1
obs = DiscreteObserver(h)

end_time = 20.0
x0 = np.array([0.1, 0.1, 0.1, 0.1])

eigenvalues = obs.get_eigenvalues()
print(f"Eigenvalues of A {eigenvalues}")

eigenvalues[0] = np.exp(-0.5 * h)
eigenvalues[1] = np.exp(-1 * h)
print(f"Preferred eigenvalues {eigenvalues}")

L = obs.gain_matrix(eigenvalues)
print(f"Gain matrix {L.T}")

lin_t, lin_y = obs.lin_residual(end_time, x0)
nonlin_t, nonlin_y = obs.nonlin_residual(end_time, x0)

fig, axs = plt.subplots(4)
fig.set_figwidth(10)
fig.set_figheight(12)
ylabels = ['$\\theta$', '$x$', '$\\dot{\\theta}$', '$\\dot{x}$']
for i in range(4):
    axs[i].scatter(lin_t, lin_y[i], color='red', s=3)
    axs[i].scatter(nonlin_t, nonlin_y[i], color='blue', s=3)
    axs[i].set_xlabel('$t$')
    axs[i].set_ylabel(ylabels[i])
fig.tight_layout()
plt.show()
plt.scatter()
