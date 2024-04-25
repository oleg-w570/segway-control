import control as ct
import numpy as np

from discrete_system import DiscreteSystem


class DiscreteObserver:
    def __init__(self, h: float):
        self.sys = DiscreteSystem(h)
        self.C = np.array([1.0, 1.0, 0.0, 0.0]).reshape(1, 4)
        self.L = None

    def gain_matrix(self, pref_eigenvalues: np.ndarray):
        self.L = ct.acker(self.sys.Ad.T, self.C.T, pref_eigenvalues)
        return self.L





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
