import control as ct
import numpy as np
from scipy.integrate import solve_ivp

from continuous_system import ContinuousSystem


class ContinuousObserver:
    def __init__(self):
        self.sys = ContinuousSystem()
        self.C = np.array([1., 1., 0., 0.]).reshape(1, 4)
        self.L = None
        self.clear_nonlin_func = self.sys.gen_nonlin_func(np.array([0., 0., 0., 0.]))

    def get_eigenvalues(self):
        return np.linalg.eigvals(self.sys.A)

    def gain_matrix(self, pref_eigenvalues):
        self.L = ct.acker(self.sys.A.T, self.C.T, pref_eigenvalues).transpose()
        return self.L

    def set_L(self, L):
        self.L = L

    def extended_lin_func(self, t: float, x: np.ndarray):
        y = np.empty(x.shape[0])
        y[:4] = self.sys.A @ x[:4]
        y[4:] = self.sys.A @ x[4:] + self.L @ self.C @ (x[:4] - x[4:])
        return y

    def lin_sol(self, end_time: float, x0: np.ndarray):
        sol = solve_ivp(self.extended_lin_func, (0, end_time), np.append(x0, [0., 0., 0., 0.]), method="LSODA")
        return sol.t, sol.y

    def extended_nonlin_func(self, t: float, x: np.ndarray):
        y = np.empty(x.shape[0])
        y[:4] = self.clear_nonlin_func(t, x[:4])
        # y[4:] = self.clear_nonlin_func(t, x[4:]) + self.L @ self.C @ (x[:4] - x[4:])
        y[4:] = self.sys.A @ x[4:] + self.L @ self.C @ (x[:4] - x[4:])
        return y

    def nonlin_sol(self, end_time: float, x0: np.ndarray):
        sol = solve_ivp(self.extended_nonlin_func, (0, end_time), np.append(x0, [0.0, 0.0, 0.0, 0.0]), method="LSODA")
        return sol.t, sol.y
