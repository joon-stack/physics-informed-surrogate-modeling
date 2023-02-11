import numpy as np
import torch
from scipy.stats import qmc


def generate_y(x) -> np.ndarray:
    y = 1/2 * x**2 - 1/6 * x**3
    y = y.reshape(x.shape)
    return y


def generate_tasks(n: int):
    sampler = qmc.LatinHypercube(d=2)
    sample_sup = sampler.random(n)
    sample_qry = sampler.random(n)
    l_bounds = [3.5, 0.05]
    u_bounds = [35, 0.5]
    sup = qmc.scale(sample_sup, l_bounds, u_bounds)
    qry = qmc.scale(sample_qry, l_bounds, u_bounds)
    return sup, qry


def generate_data(mode: str, n: int, task: np.ndarray):
    sampler = qmc.LatinHypercube(d=1)
    sample = sampler.random(n)
    l_bounds = 0.0
    u_bounds = 1.0
    x = qmc.scale(sample, l_bounds, u_bounds)
    if mode == "data":
        
        y = generate_y(x)

        return x, y

    elif mode == "physics":
        y = np.zeros(x.shape)
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
    data = generate_data(mode="data", n=1, task=np.array([7, 0.1]))
    print(data)
