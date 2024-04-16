import control as ct
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from continuous_system import ContinuousSystem


class DiscreteSystem(ContinuousSystem):
    def __init__(self, h: float = 0.1) -> None:
        super().__init__()
        self.h = h
        self.Ad = self.create_Ad()
        self.Bd = self.create_Bd()

    def create_Ad(self) -> np.ndarray:
        Ad = expm(self.A * self.h)
        return Ad

    def create_Bd(self) -> np.ndarray:
        Bd = np.identity(self.A.shape[0]) * self.h
        temp_eq = np.identity(self.A.shape[0]) * self.h
        for k in range(1, 100):
            temp_eq @= self.A * self.h / (k + 1)
            Bd += temp_eq
        return Bd @ self.B

    def ackermann_control(self, pref_eigvals: np.ndarray) -> np.ndarray:
        theta = -ct.acker(self.Ad, self.Bd, pref_eigvals)
        theta = theta.reshape((1, 4))
        return theta

    def lin_sol(self, end_time: float, x0: np.ndarray, theta: np.ndarray):
        n_step = int(end_time / self.h)
        t = []
        u = []
        xk = x0
        for k in range(n_step):
            uk = theta @ xk
            temp_sol = solve_ivp(lambda t, x: self.A @ x + self.B @ uk, (k * self.h, min((k + 1) * self.h, end_time)),
                                 xk, method="LSODA")
            t.extend(temp_sol.t)
            u.extend([uk] * len(temp_sol.t))
            y = temp_sol.y if k == 0 else np.append(y, temp_sol.y, axis=1)
            xk = temp_sol.y[:, -1]
        return t, y, u

    def nonlin_sol(self, end_time: float, x0: np.ndarray, theta: np.ndarray):
        n_step = int(end_time / self.h)
        t = []
        u = []
        xk = x0
        for k in range(n_step):
            uk = theta.reshape(4) @ xk
            temp_sol = solve_ivp(self.gen_nonlin_func(uk), (k * self.h, min((k + 1) * self.h, end_time)), xk,
                                 method="LSODA")
            t.extend(temp_sol.t)
            u.extend([uk] * len(temp_sol.t))
            y = temp_sol.y if k == 0 else np.append(y, temp_sol.y, axis=1)
            xk = temp_sol.y[:, -1]
        return t, y, u

    def gen_nonlin_func(self, uk):
        def nonlin_func(t: float, x: np.ndarray):
            denom = self.m ** 2 * self.l ** 2 * np.cos(x[0]) ** 2 - (
                    self.m + self.M
            ) * (self.I + self.m * self.l ** 2)
            temp_eq1 = (
                    self.Kt * (uk - self.Ks * x[3])
                    - self.Beq * x[3]
                    - self.m * self.l * x[2] ** 2 * np.sin(x[0])
            )
            temp_eq2 = self.m * self.g * self.l * np.sin(x[0]) - self.Bp * x[2]
            y = np.array(
                [
                    x[2],
                    x[3],
                    -(
                            self.m * self.l * temp_eq1 * np.cos(x[0])
                            + (self.m + self.M) * temp_eq2
                    )
                    / denom,
                    -(
                            self.m * self.l * np.cos(x[0]) * temp_eq2
                            + (self.I + self.m * self.l ** 2) * temp_eq1
                    )
                    / denom,
                ],
                dtype=float,
            )
            return y

        return nonlin_func
