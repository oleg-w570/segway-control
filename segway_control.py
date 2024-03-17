import numpy as np


class Parameters:
    def __init__(self):
        self.m = 0.56  # Масса маятника
        self.M = 1.206  # Масса тележки
        self.I = 0.89  # Момент инерции относительно центра масс
        self.l = 0.1778  # Расстояние до центра масс
        self.Kf = 1.726  # Конструктивный параметр (Kt)
        self.Ks = 4.487  # Конструктивный параметр
        self.Beq = 5.4  # Коэффициент вязкого трения
        self.Bp = 1.4  # Коэффициент вязкого трения
        self.g = 9.81  # Ускорение свободного падения


def continuous_system_coef(p: Parameters) -> tuple[np.ndarray, np.ndarray]:
    denum = (p.I + p.m * p.l**2) * (p.m + p.M) - p.m**2 * p.l**2
    
    A = np.empty((4, 4))
    A[0] = np.array([0, 0, 1, 0])
    A[1] = np.array([0, 0, 0, 1])
    A[2] = (
        np.array(
            [
                (p.m + p.M) * p.m * p.g * p.l,
                0,
                -(p.m + p.M) * p.Bp,
                -p.m * p.l * (p.Kf * p.Ks + p.Beq),
            ]
        )
        / denum
    )
    A[3] = (
        np.array(
            [
                p.m**2 * p.g * p.l**2,
                0,
                -p.m * p.l * p.Bp,
                -(p.I + p.m * p.l**2) * (p.Kf * p.Ks + p.Beq),
            ]
        )
        / denum
    )
    
    B = np.empty((4, 1))
    B[:, 0] = np.array([0, 0, p.m * p.l * p.Kf, (p.I + p.m * p.l**2) * p.Kf]).T / denum
    return A, B


def linear_odefun_no_obs(x, t, A, B, theta):
    return (A + B @ theta) @ x


def nonlinear_odefun_no_obs(x, t, p, theta):
    u = theta @ x
    denum = (p.I + p.m * p.l**2) * (p.m + p.M) - p.m**2 * p.l**2 * np.cos(x[0]) ** 2
    # ---
    dxdt = np.empty(x.shape[0])
    dxdt[0] = x[2]
    dxdt[1] = x[3]
    dxdt[2] = (
        p.m * (p.m + p.M) * p.g * p.l * np.sin(x[0])
        + p.m * p.l * np.cos(x[0]) * p.Kf * u
        - p.Bp * (p.m + p.M) * x[2]
        - (p.Beq + p.Kf * p.Ks) * p.m * p.l * np.cos(x[0]) * x[3]
        - p.m * p.l * np.cos(x[0]) * p.m * p.l**2 * x[2] ** 2 * np.sin(x[0])
    ) / denum
    dxdt[3] = (
        p.Kf * (p.I + p.m * p.l**2) * u
        - (p.Kf * p.Ks + p.Beq) * (p.I + p.m * p.l**2) * x[3]
        - (p.I + p.m * p.l**2) * p.m * p.l**2 * x[2] ** 2 * np.sin(x[0])
        - p.m * p.l * np.cos(x[0]) * p.Bp * x[2]
        + p.m**2 * p.l**2 * p.g * np.sin(x[0]) * np.cos(x[0])
    ) / denum
    return dxdt


def get_control_akkerman(A, B, desired_poly_coef):
    n = A.shape[0]
    # Матрица управляемости
    C = np.empty((n, n))
    C[:, 0] = B[:, 0]
    for i in range(1, n):
        C[:, i] = A @ C[:, i - 1]
    # Характеристический полином
    var = np.identity(n)
    ch_poly = np.zeros((n, n))
    for c in desired_poly_coef[::-1]:
        ch_poly += var * c
        var = var @ A
    # Единичный вектор-строка
    e_n = np.zeros((1, n))
    e_n[0, -1] = 1
    # Матрица обратной связи
    theta = np.empty((1, n))
    theta[0, :] = -e_n @ np.linalg.inv(C) @ ch_poly
    return theta
