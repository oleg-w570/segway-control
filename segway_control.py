from pprint import pprint
import numpy as np
import control as ct


class Experiment:
    def __init__(self) -> None:
        self.m = 0.56
        self.M = 1.206  # Масса тележки
        self.I = 0.89  # Момент инерции относительно центра масс
        self.l = 0.1778  # Расстояние до центра масс
        self.Kt = 1.726  # Конструктивный параметр (Kt)
        self.Ks = 4.487  # Конструктивный параметр
        self.Beq = 5.4  # Коэффициент вязкого трения
        self.Bp = 1.4  # Коэффициент вязкого трения
        self.g = 9.81  # Ускорение свободного падения

        self.A = self.create_A()
        self.B = self.create_B()

    def create_A(self) -> np.ndarray:
        denom = (self.m + self.M) * (
            self.I + self.m * self.l**2
        ) - self.m**2 * self.l**2
        A = np.array(
            [
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [
                    self.m * self.g * self.l * (self.m + self.M) / denom,
                    0,
                    -(self.m + self.M) * self.Bp / denom,
                    -self.m * self.l * (self.Kt * self.Ks + self.Beq) / denom,
                ],
                [
                    self.m**2 * self.g * self.l**2 / denom,
                    0,
                    -self.m * self.l * self.Bp / denom,
                    -(self.Kt * self.Ks + self.Beq)
                    * (self.I + self.m * self.l**2)
                    / denom,
                ],
            ],
            dtype=float,
        )
        return A

    def create_B(self) -> np.ndarray:
        denom = (self.m + self.M) * (
            self.I + self.m * self.l**2
        ) - self.m**2 * self.l**2
        B = np.array(
            [
                [0],
                [0],
                [self.m * self.l * self.Kt / denom],
                [self.Kt * (self.I + self.m * self.l**2) / denom],
            ],
            dtype=float,
        )
        return B

    def lin_func(self, t: float, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        return (self.A + self.B @ theta) @ x

    def nonlin_func(self, t: float, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        u = theta.reshape(4) @ x
        denom = self.m**2 * self.l**2 * np.cos(x[0]) ** 2 - (
            self.m + self.M
        ) * (self.I + self.m * self.l**2)
        temp_eq1 = (
            self.Kt * (u - self.Ks * x[3])
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
                    + (self.I + self.m * self.l**2) * temp_eq1
                )
                / denom,
            ],
            dtype=float,
        )
        return y

    def ackermann_control(self, pref_eigvals: np.ndarray) -> np.ndarray:
        theta = -ct.acker(self.A, self.B, pref_eigvals)
        theta = theta.reshape((1, 4))
        return theta

