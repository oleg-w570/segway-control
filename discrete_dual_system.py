import control as ct
import numpy as np

from discrete_system import DiscreteSystem


class DiscreteDualSystem:
    def __init__(self, h: float):
        self.hatA = DiscreteSystem(h).Ad.transpose()
        self.C = np.array([1.0, 1.0, 0.0, 0.0]).reshape(4, 1)
        self.theta = None

    def ackermann_control(self, pref_eigvals: np.ndarray):
        self.theta = -ct.acker(self.hatA, self.C, pref_eigvals).reshape((1, 4))
        return self.theta

    def sol(self, end_time: float, x0: np.ndarray):
        t = np.linspace(0, end_time, 1000)
        y = []
        x = x0
        for _ in t:
            y.append(x)
            x = (self.hatA + self.C @ self.theta) @ x
        y = np.array(y).transpose()
        u = y.T @ self.theta.T
        return t, y, u
