import numpy as np
from scipy.integrate import solve_ivp

from discrete_dual_system import DiscreteObserver
from discrete_system import DiscreteSystem


class DiscreteObserverSystem:
    def __init__(self, h: float):
        self.h = h
        self.system = DiscreteSystem(h)
        self.dual_system = DiscreteObserver(h)
        self.G = self.create_G()
        self.L = self.create_L()
        self.Acl = self.create_Acl()

    def create_G(self):
        pref_eigAd = np.linalg.eigvals(self.system.Ad)
        # pref_eigAd = pref_eigAd.astype(complex)
        pref_eigAd[0] = np.exp(-0.5 * self.h)
        pref_eigAd[1] = np.exp(-1 * self.h)
        # pref_eigAd[0] = np.exp(complex(-0.5, 1.5) * self.h)
        # pref_eigAd[1] = np.exp(complex(-0.5, -1.5) * self.h)
        return self.system.ackermann_control(pref_eigAd)

    def create_L(self):
        pref_eigHatA = np.linalg.eigvals(self.dual_system.hatA)
        # pref_eigHatA = pref_eigHatA.astype(complex)
        pref_eigHatA[0] = np.exp(-1 * self.h)
        pref_eigHatA[3] = np.exp(-0.5 * self.h)
        # pref_eigHatA[0] = np.exp(complex(-0.5, 1.5))
        # pref_eigHatA[3] = np.exp(complex(-0.5, -1.5))
        return -self.dual_system.ackermann_control(pref_eigHatA).transpose()

    def create_Acl(self):
        Acl = np.block(
            [
                [self.system.Ad, self.system.Bd @ self.G],
                [self.L @ self.dual_system.C.T,
                 self.system.Ad + self.system.Bd @ self.G - self.L @ self.dual_system.C.T],
            ]
        )
        return Acl

    def lin_sol(self, end_time: float, x0: np.ndarray):
        n_step = int(end_time / self.h)
        t = []
        u = []
        xk = x0
        obs_xk = np.array([0.0, 0.0, 0.0, 0.0])
        for k in range(n_step):
            uk = self.G @ obs_xk
            temp_sol = solve_ivp(lambda t, x: self.system.A @ x + self.system.B @ uk,
                                 (k * self.h, min((k + 1) * self.h, end_time)), xk, method="LSODA")
            obs_xk = self.system.Ad @ obs_xk + self.system.Bd @ uk + self.L @ self.dual_system.C.T @ (xk - obs_xk)
            t.extend(temp_sol.t)
            u.extend([uk] * len(temp_sol.t))
            y = temp_sol.y if k == 0 else np.append(y, temp_sol.y, axis=1)
            xk = temp_sol.y[:, -1]
        return t, y, u

    def nonlin_sol(self, end_time: float, x0: np.ndarray):
        n_step = int(end_time / self.h)
        t = []
        u = []
        xk = x0
        obs_xk = np.array([0.0, 0.0, 0.0, 0.0])
        for k in range(n_step):
            uk = self.G.reshape(4) @ obs_xk
            temp_sol = solve_ivp(self.system.gen_nonlin_func(uk), (k * self.h, min((k + 1) * self.h, end_time)), xk,
                                 method="LSODA")
            uk = self.G @ obs_xk
            obs_xk = self.system.Ad @ obs_xk + self.system.Bd @ uk + self.L @ self.dual_system.C.T @ (xk - obs_xk)
            t.extend(temp_sol.t)
            u.extend([uk] * len(temp_sol.t))
            y = temp_sol.y if k == 0 else np.append(y, temp_sol.y, axis=1)
            xk = temp_sol.y[:, -1]
        return t, y, u
