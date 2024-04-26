import control as ct
import numpy as np
from scipy.integrate import solve_ivp

from discrete_system import DiscreteSystem


class DiscreteObserver:
    def __init__(self, h: float):
        self.sys = DiscreteSystem(h)
        self.C = np.array([1.0, 1.0, 0.0, 0.0]).reshape(1, 4)
        self.L = None

    def get_eigenvalues(self):
        return np.linalg.eigvals(self.sys.Ad)

    def gain_matrix(self, pref_eigenvalues: np.ndarray):
        self.L = ct.acker(self.sys.Ad.T, self.C.T, pref_eigenvalues).transpose()
        return self.L

    def lin_residual1(self, end_time: float, x0: np.ndarray):
        n_step = int(end_time / self.sys.h)
        xk = x0
        obs_xk = x0
        # obs_xk = np.array([0., 0., 0., 0.])
        y = (xk - obs_xk).reshape(4, 1)
        t = [0.0]
        for k in range(n_step):
            temp_sol = solve_ivp(lambda t, x: self.sys.A @ x,
                                 (k * self.sys.h, (k + 1) * self.sys.h), xk, method="LSODA")
            # obs_xk = self.sys.Ad @ obs_xk + self.L @ self.C @ (xk - obs_xk)
            obs_xk = self.sys.Ad @ obs_xk
            xk = temp_sol.y[:, -1]
            y = np.append(y, (xk - obs_xk).reshape(4, 1), axis=1)
            t.append((k + 1) * self.sys.h)
        return t, y

    def lin_residual(self, end_time: float, x0: np.ndarray):
        n_step = int(end_time / self.sys.h)
        ek = x0
        t = [0.0]
        y = ek.reshape(4, 1)
        for k in range(n_step):
            ek = (self.sys.Ad - self.L @ self.C) @ ek
            y = np.append(y, ek.reshape(4, 1), axis=1)
            t.append((k + 1) * self.sys.h)
        return t, y

    def nonlin_residual(self, end_time: float, x0: np.ndarray):
        n_step = int(end_time / self.sys.h)
        xk = x0
        obs_xk = np.array([0., 0., 0., 0.])
        y = [xk - obs_xk]
        t = [0.0]
        for k in range(n_step):
            temp_sol = solve_ivp(self.clear_nonlin_func,
                                 (k * self.sys.h, (k + 1) * self.sys.h), xk, method="LSODA")
            obs_xk = self.sys.Ad @ obs_xk + self.L @ self.C @ (xk - obs_xk)
            xk = temp_sol.y[:, -1]
            y.append(xk - obs_xk)
            t.append((k + 1) * self.sys.h)
        y = np.array(y).transpose()
        return t, y

    def clear_nonlin_func(self, t: float, x: np.ndarray) -> np.ndarray:
        theta = np.array([0., 0., 0., 0.])
        u = theta.reshape(4) @ x
        denom = self.sys.m ** 2 * self.sys.l ** 2 * np.cos(x[0]) ** 2 - (
                self.sys.m + self.sys.M
        ) * (self.sys.I + self.sys.m * self.sys.l ** 2)
        temp_eq1 = (
                self.sys.Kt * (u - self.sys.Ks * x[3])
                - self.sys.Beq * x[3]
                - self.sys.m * self.sys.l * x[2] ** 2 * np.sin(x[0])
        )
        temp_eq2 = self.sys.m * self.sys.g * self.sys.l * np.sin(x[0]) - self.sys.Bp * x[2]
        y = np.array(
            [
                x[2],
                x[3],
                -(
                        self.sys.m * self.sys.l * temp_eq1 * np.cos(x[0])
                        + (self.sys.m + self.sys.M) * temp_eq2
                )
                / denom,
                -(
                        self.sys.m * self.sys.l * np.cos(x[0]) * temp_eq2
                        + (self.sys.I + self.sys.m * self.sys.l ** 2) * temp_eq1
                )
                / denom,
            ],
            dtype=float,
        )
        return y
