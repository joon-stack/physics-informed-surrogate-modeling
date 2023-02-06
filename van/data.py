import numpy as np
import torch
from scipy.stats import qmc
import random

DIM = 11


def calc_misc(t, d, L1, L2, F1, F2, P, T, Sy, theta1, theta2):
    M = F1 * L1 * np.cos(theta1) + F2 * L2 * np.cos(theta2)
    A = np.pi * (d**2 - (d - 2 * t) ** 2) / 4
    I = np.pi * (d**4 - (d - 2 * t) ** 4) / 64
    J = 2 * I

    sigma_x = ((P + F1 * np.sin(theta1) + F2 * np.sin(theta2)) / A + M * d / 2 / I) * 1000
    tau_zx = T * d / 2 / J * 1000
    sigma_max = np.sqrt(sigma_x**2 + 3 * tau_zx**2)

    return M, A, I, J, sigma_x, tau_zx, sigma_max


def generate_x_y(y1: np.ndarray, u: np.ndarray) -> np.ndarray:
    x = y1
    # y = x**4 / 24 - 1 * x**3 / 6 + 1 * x**2 / 4
    y = -1 / 6 * x**3 + 1 / 2 * x**2
    return x, y


def generate_tasks(n: int, seed=None):
    sampler = qmc.LatinHypercube(d=DIM, seed=seed)
    sample_sup = sampler.random(n)
    sample_qry = sampler.random(n)
    l_bounds = (
        np.array([5.0, 42.0, 120.0, 60.0, 3.0, 3.0, 90.0, 175.0, 5.0, 10.0, 12.0], dtype=np.float32)
        * 0.8
    )
    u_bounds = (
        np.array([5.0, 42.0, 120.0, 60.0, 3.0, 3.0, 90.0, 175.0, 5.0, 10.0, 12.0], dtype=np.float32)
        * 1.2
    )

    sup = qmc.scale(sample_sup, l_bounds, u_bounds)
    qry = qmc.scale(sample_qry, l_bounds, u_bounds)
    return sup, qry


def generate_tasks_out(n: int, seed=None):
    sampler = qmc.LatinHypercube(d=DIM, seed=seed)
    sample_sup = sampler.random(n)
    sample_qry = sampler.random(n)
    l_bounds = (
        np.array([5.0, 42.0, 120.0, 60.0, 3.0, 3.0, 90.0, 175.0, 5.0, 10.0, 12.0], dtype=np.float32)
        * 0.6
    )
    u_bounds = (
        np.array([5.0, 42.0, 120.0, 60.0, 3.0, 3.0, 90.0, 175.0, 5.0, 10.0, 12.0], dtype=np.float32)
        * 1.4
    )

    sup = qmc.scale(sample_sup, l_bounds, u_bounds)
    qry = qmc.scale(sample_qry, l_bounds, u_bounds)
    return sup, qry


def generate_tasks_out2(n: int, seed=None):
    sampler = qmc.LatinHypercube(d=DIM, seed=seed)
    sample_sup = sampler.random(n)
    sample_qry = sampler.random(n)
    l_bounds = (
        np.array([5.0, 42.0, 120.0, 60.0, 3.0, 3.0, 90.0, 175.0, 5.0, 10.0, 12.0], dtype=np.float32)
        * 0.4
    )
    u_bounds = (
        np.array([5.0, 42.0, 120.0, 60.0, 3.0, 3.0, 90.0, 175.0, 5.0, 10.0, 12.0], dtype=np.float32)
        * 1.6
    )

    sup = qmc.scale(sample_sup, l_bounds, u_bounds)
    qry = qmc.scale(sample_qry, l_bounds, u_bounds)
    return sup, qry

def load_data():
    data = np.loadtxt('van/data/data.txt', usecols=[0,2], dtype=np.float32)
    x = data[:, 0].reshape(-1, 1)
    u = data[:, 1].reshape(-1, 1)
    return x, u


def generate_data(mode: str, n: int, task: np.ndarray):

    sampler = qmc.LatinHypercube(d=1, seed=None)
    sample = sampler.random(n)
    x = qmc.scale(sample, 0.0, 1.0)
    if mode == "data":
        # x, y = generate_x_y(x, task)
        x, y = load_data()
        idx = random.sample(range(len(x)), n)
        x = x[idx]
        y = y[idx]

        return x, y

    elif mode == "boundary":
        x = np.hstack([np.zeros(n), np.full(n, 1.0, dtype=np.float32)]).reshape(-1, 1)
        # x = np.hstack([np.zeros(n), np.full(n//2, 1.0, dtype=np.float32), np.zeros(n//2)]).reshape(-1, 1)
        y = np.zeros((n*4, 1))
        return x, y

    elif mode == "physics":
        # y_shape = (x.shape[0], x.shape[1] * 2)
        y = np.zeros(x.shape)
        return x, y


def generate_data_test(n: int):
    mu = np.array([5, 42, 120, 60, 3, 3, 90, 175, 5, 10], dtype=np.float32)
    sigma = np.array([0.1, 0.5, 1.2, 0.6, 0.3, 0.3, 9, 17.5, 0.25, 0.5], dtype=np.float32)
    x = np.random.normal(mu, sigma, (n, 10))
    t = x[:, 0]
    d = x[:, 1]
    L1 = x[:, 2]
    L2 = x[:, 3]
    F1 = x[:, 4]
    F2 = x[:, 5]
    T = x[:, 6]
    Sy = x[:, 7]
    theta1 = x[:, 8] * np.pi / 180
    theta2 = x[:, 9] * np.pi / 180
    P = np.random.gumbel(12, 1.2, n)

    M = F1 * L1 * np.cos(theta1) + F2 * L2 * np.cos(theta2)
    A = np.pi * (d**2 - (d - 2 * t) ** 2) / 4
    I = np.pi * (d**4 - (d - 2 * t) ** 4) / 64
    J = 2 * I
    # print(d, t, A, I)

    sigma_x = ((P + F1 * np.sin(theta1) + F2 * np.sin(theta2)) / A + M * d / 2 / I) * 1000
    tau_zx = T * d / 2 / J * 1000
    sigma_max = np.sqrt(sigma_x**2 + 3 * tau_zx**2)
    y = Sy - sigma_max

    return x, y


def to_tensor(x: np.array, requires_grad: bool = True) -> torch.Tensor:
    t = torch.from_numpy(x)
    t.requires_grad = requires_grad
    t = t.float()
    return t


def generate_task_data(sup: np.ndarray, qry: np.ndarray, mode: str, size_sup: int, size_qry: int):
    support_key = []
    query_key = []
    support_data = []
    query_data = []

    for s, q in zip(sup, qry):
        data = generate_data(mode=mode, n=size_sup, task=s)
        support_key.append(s)
        support_data.append(data)
        data = generate_data(mode=mode, n=size_qry, task=q)
        query_key.append(q)
        query_data.append(data)

    support = (support_key, support_data)
    query = (query_key, query_data)

    return (support, query)


if __name__ == "__main__":
    print(load_data())