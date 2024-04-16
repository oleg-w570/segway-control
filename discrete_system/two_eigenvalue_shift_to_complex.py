import numpy as np
from matplotlib import pyplot as plt

from discrete_system import DiscreteSystem

plt.style.use('seaborn-v0_8')

h = 0.1
s = DiscreteSystem(h)
eigAd = np.linalg.eigvals(s.Ad)
eigAd = eigAd.astype(complex)

end_time = 10.0
x0 = np.array([-0.1, -0.1, 0.1, 0.1])

pref_eigAd = eigAd.copy()
pref_eigAd[0] = np.exp(complex(-0.5, 1.5) * h)
pref_eigAd[1] = np.exp(complex(-0.5, -1.5) * h)
print(f"Pref eigvals = {pref_eigAd}")

theta = s.ackermann_control(pref_eigAd)
print(f"Theta = {theta}")

lin_t, lin_y, lin_u = s.lin_sol(end_time, x0, theta)
nonlin_t, nonlin_y, nonlin_u = s.nonlin_sol(end_time, x0, theta)

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
