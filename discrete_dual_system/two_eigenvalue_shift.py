import numpy as np
from matplotlib import pyplot as plt

from discrete_observer import DiscreteObserver

plt.style.use('seaborn-v0_8')

h = 0.1
r = DiscreteObserver(h)
eigA = np.linalg.eigvals(r.hatA)
print(f"Eigenvalues of A = {eigA}")

end_time = 10.0
x0 = np.array([0.1, 0.1, 0.1, 0.1])

pref_eigAd = eigA.copy()
pref_eigAd[0] = np.exp(-1 * h)
pref_eigAd[3] = np.exp(-0.5 * h)
print(f"Pref eigvals = {pref_eigAd}")

theta = r.ackermann_control(pref_eigAd)
print(f"Theta = {theta}")

t, y, u = r.sol(end_time, x0)

fig, axs = plt.subplots(4)
fig.set_figwidth(10)
fig.set_figheight(12)
ylabels = ['$\\theta$', '$x$', '$\\dot{\\theta}$', '$\\dot{x}$']
for i in range(4):
    axs[i].plot(t, y[i], color='red', linewidth=1)
    axs[i].set_xlabel('$t$')
    axs[i].set_ylabel(ylabels[i])
# axs[4].plot(t, u, color='red', linewidth=1)
# axs[4].set_xlabel('$t$')
# axs[4].set_ylabel('$V$')
fig.tight_layout()
plt.show()
