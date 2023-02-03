import numpy as np
import torch
from scipy.stats import qmc
import pandas as pd

DIM = 100


def generate_x_y(y1: np.ndarray, u: np.ndarray) -> np.ndarray:
    x = y1 + u
    n = y1.shape[1]
    sigma = 0.2
    y = (n + 3 * sigma * np.sqrt(n)) - np.sum(x, 1)
    return x, y


def generate_tasks(n: int, seed=None):
    sampler = qmc.LatinHypercube(d=DIM, seed=seed)
    sample_sup = sampler.random(n)
    sample_qry = sampler.random(n)
    l_bounds = np.full(DIM, -0.2)
    u_bounds = np.full(DIM, 0.2)
    sup = qmc.scale(sample_sup, l_bounds, u_bounds)
    qry = qmc.scale(sample_qry, l_bounds, u_bounds)
    return sup, qry


def generate_tasks_out(n: int, seed=None):
    sampler = qmc.LatinHypercube(d=DIM, seed=seed)
    sample_sup = sampler.random(n)
    sample_qry = sampler.random(n)
    l_bounds = np.full(DIM, -0.4)
    u_bounds = np.full(DIM, 0.4)
    sup = qmc.scale(sample_sup, l_bounds, u_bounds)
    qry = qmc.scale(sample_qry, l_bounds, u_bounds)
    return sup, qry


def generate_tasks_out2(n: int, seed=None):
    sampler = qmc.LatinHypercube(d=DIM, seed=seed)
    sample_sup = sampler.random(n)
    sample_qry = sampler.random(n)
    l_bounds = np.full(DIM, -1.0)
    u_bounds = np.full(DIM, 1.0)
    sup = qmc.scale(sample_sup, l_bounds, u_bounds)
    qry = qmc.scale(sample_qry, l_bounds, u_bounds)
    return sup, qry


def generate_data(mode: str, n: int, task: np.ndarray):

    mu = np.zeros(DIM, dtype=np.float32)
    sigma = np.full(DIM, 0.2, dtype=np.float32)

    x = np.random.lognormal(mu, sigma, (n, DIM))
    if mode == "data":
        x, y = generate_x_y(x, task)

        return x, y

    elif mode == "physics":
        y = np.full((n, DIM), -1.0, dtype=np.float32)
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


def output_data(n: int):
    x, y = generate_data(mode="data", n=n, task=np.zeros(DIM, dtype=np.float32))
    data = np.hstack([x, y.reshape(-1, 1)])
    pd.DataFrame(data).to_csv("sample.csv", header=False, index=False)


if __name__ == "__main__":
    output_data(100)
