import control as ct
import numpy as np
from scipy.integrate import solve_ivp

from continuous_system import ContinuousSystem


class ContinuousObserverSystem:
    def __init__(self):
        self.sys = ContinuousSystem()
        self.C = np.array([1., 1., 0., 0.]).reshape(1, 4)
        self.L = self.calculate_gain_matrix()
        self.theta = self.calculate_control_matrix()

    def calculate_gain_matrix(self):
        eigenvalues = np.linalg.eigvals(self.sys.A)
        eigenvalues[0] = -1
        eigenvalues[1] = -1.5
        return ct.acker(self.sys.A.T, self.C.T, eigenvalues).transpose()

    def calculate_control_matrix(self):
        eigenvalues = np.linalg.eigvals(self.sys.A)
        eigenvalues[0] = -1
        eigenvalues[1] = -0.521
        return -ct.acker(self.sys.A, self.sys.B, eigenvalues)

    def extended_lin_func(self, t: float, x: np.ndarray):
        y = np.empty(x.shape[0])
        y[:4] = self.sys.A @ x[:4] + self.sys.B @ self.theta @ x[4:]
        y[4:] = (self.sys.A + self.sys.B @ self.theta) @ x[4:] + self.L @ self.C @ (x[:4] - x[4:])
        return y

    def lin_sol(self, end_time: float, x0: np.ndarray):
        sol = solve_ivp(self.extended_lin_func, (0, end_time), np.append(x0, [0., 0., 0., 0.]), method="LSODA")
        u = sol.y[:4].T @ self.theta.T
        return sol.t, sol.y, u

    def extended_nonlin_func(self, t: float, x: np.ndarray):
        y = np.empty(x.shape[0])
        y[:4] = self.nonlin_func(x)
        y[4:] = (self.sys.A + self.sys.B @ self.theta) @ x[4:] + self.L @ self.C @ (x[:4] - x[4:])
        return y

    def nonlin_sol(self, end_time: float, x0: np.ndarray):
        sol = solve_ivp(self.extended_nonlin_func, (0, end_time), np.append(x0, [0.0, 0.0, 0.0, 0.0]), method="LSODA")
        u = sol.y[:4].T @ self.theta.T
        return sol.t, sol.y, u

    def nonlin_func(self, x: np.ndarray):
        u = self.theta.reshape(4) @ x[4:]
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
