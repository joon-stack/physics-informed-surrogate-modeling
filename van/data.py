import numpy as np
import torch
from scipy.stats import qmc
import random
from fem import get_fem_data
import matplotlib.pyplot as plt

DESIGN_SIZE = 4

def normalize(y_min: np.ndarray, y_max: np.ndarray, y: np.ndarray):
    y_max = np.broadcast_to(y_max.reshape(1, -1), y.shape)
    y_min = np.broadcast_to(y_min.reshape(1, -1), y.shape)
    res = (y - y_min) / (y_max - y_min)
    # plt.plot(y[:, 0])
    # plt.plot(y[:, 1])
    # plt.show()
    return res

def denormalize(y_min: np.ndarray, y_max: np.ndarray, y: np.ndarray):
    y_max = np.broadcast_to(y_max.reshape(1, -1), y.shape)
    y_min = np.broadcast_to(y_min.reshape(1, -1), y.shape)
    res = y * (y_max - y_min) + y_min
    return res

def generate_tasks(n: int, seed=None):
    sampler = qmc.LatinHypercube(d=DESIGN_SIZE * 2, seed=seed)
    sample_sup = sampler.random(n)
    sample_qry = sampler.random(n)
    # l_bounds = [0.01, 0.01, 0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 0.05, 0.05]
    # u_bounds = [0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2]
    l_bounds = [0.01, 0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 0.05]
    u_bounds = [0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2]

    sup = qmc.scale(sample_sup, l_bounds, u_bounds)
    qry = qmc.scale(sample_qry, l_bounds, u_bounds)
    return sup, qry

def load_data():
    data = np.loadtxt('van/data/data.txt', usecols=[0,2], dtype=np.float32)
    x = data[:, 0].reshape(-1, 1)
    u = data[:, 1].reshape(-1, 1)
    return x, u


def fem_data(task: np.ndarray, design_size: int, element_size: int):
    x, stress, disp = get_fem_data(task, design_size, element_size)
    x = np.array(x)
    stress = np.array(stress)
    disp = np.array(disp)
    
    return x, stress, disp


def generate_data(mode: str, n: int, task: np.ndarray):
    # sampler = qmc.LatinHypercube(d=1, seed=None)
    # sample = sampler.random(n)
    # x = qmc.scale(sample, 0.0, 1.0)
    x, stress, disp = fem_data(task=task, design_size=4, element_size=100)
    if mode == "data":
        # x, y = generate_x_y(x, task)
        # x, y = load_data()
        sampler = qmc.LatinHypercube(d=1, seed=None)
        sample = sampler.random(n)
        idx = qmc.scale(sample, 0, len(x))
        idx = idx.astype(np.int8)
        # else:
        #     idx = np.random.choice(100, n, replace=False)
        #     idx = idx.reshape(-1, 1)
        # print(idx.shape)
        x_sample = x[idx]
        stress_sample = stress[idx]
        disp_sample = disp[idx]
        y = np.hstack((stress_sample, disp_sample))
        return x_sample, y

    elif mode == "boundary":
        # x = np.hstack([np.zeros(n), np.full(n, 1.0, dtype=np.float32)]).reshape(-1, 1)
        # x = np.hstack([np.zeros(n), np.full(n//2, 1.0, dtype=np.float32), np.zeros(n//2)]).reshape(-1, 1)
        # y = np.zeros((n*4, 1))
        x = np.zeros(n, dtype=np.float32)
        y = np.zeros(x.shape, dtype=np.float32)
        return x, y

    elif mode == "physics":
        # y_shape = (x.shape[0], x.shape[1] * 2)
        x = np.random.uniform(low=0.0, high=1.0, size=n)
        y = np.zeros(x.shape, dtype=np.float32)
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
