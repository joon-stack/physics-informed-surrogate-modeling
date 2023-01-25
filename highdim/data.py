import numpy as np
import torch
from scipy.stats import qmc

DIM = 100


def generate_y(y1: np.ndarray, u: np.ndarray) -> np.ndarray:
    x = y1 * u
    n = y1.shape[1]
    sigma = 0.2
    y = (n + 3 * sigma * np.sqrt(n)) - np.sum(x, 1)
    return y


def generate_tasks(n: int):
    sampler = qmc.LatinHypercube(d=DIM)
    sample_sup = sampler.random(n)
    sample_qry = sampler.random(n)
    l_bounds = np.full(DIM, 0.8)
    u_bounds = np.full(DIM, 1.2)
    sup = qmc.scale(sample_sup, l_bounds, u_bounds)
    qry = qmc.scale(sample_qry, l_bounds, u_bounds)
    return sup, qry


def generate_data(mode: str, n: int, task: np.ndarray):

    ones = np.ones(DIM, dtype=np.float32)
    sigma = np.ones(DIM, dtype=np.float32)

    x = np.random.normal(ones, sigma, (n, DIM))
    if mode == "data":

        y = generate_y(x, task)

        return x, y

    elif mode == "physics":
        # f_yf1**2 = 4 * uf1 ** 2
        # y = 4 * uf1**2
        # f_yr = 3
        y = np.full(yf1.shape, 3.0)
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
    print(generate_y(1, 1, 1, 1, 1, 1, 1.0, 0.1, 1.0, 1.0, 1.0, 0.5))
