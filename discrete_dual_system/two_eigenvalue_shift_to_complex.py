import numpy as np
from matplotlib import pyplot as plt

from discrete_dual_system import DiscreteDualSystem

plt.style.use('seaborn-v0_8')

h = 0.1
s = DiscreteDualSystem(h)
eigA = np.linalg.eigvals(s.hatA)
eigA = eigA.astype(complex)
print(f"Eigenvalues of A = {eigA}")

end_time = 10.0
x0 = np.array([0.1, 0.1, 0.1, 0.1])

pref_eigAd = eigA.copy()
pref_eigAd[0] = np.exp(complex(-0.5, 1.5))
pref_eigAd[3] = np.exp(complex(-0.5, -1.5))
print(f"Pref eigvals = {pref_eigAd}")

theta = s.ackermann_control(pref_eigAd)
print(f"Theta = {theta}")

t, y, u = s.sol(end_time, x0)

fig, axs = plt.subplots(4)
fig.set_figwidth(10)
fig.set_figheight(12)
ylabels = ['$\\theta$', '$x$', '$\\dot{\\theta}$', '$\\dot{x}$']
for i in range(4):
    axs[i].plot(t, y[i], color='red', linewidth=1)
    axs[i].set_xlabel('$t$')
    axs[i].set_ylabel(ylabels[i])
fig.tight_layout()
plt.show()
