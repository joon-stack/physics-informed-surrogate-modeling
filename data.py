import numpy as np
import torch
from burgers import *

LB_X = -1.0
RB_X = 1.0
LB_T = 0.0
RB_T = 1.0

LB_X_OOD = 1.0
RB_X_OOD = 1.5
LB_T_OOD = 0.0
RB_T_OOD = 1.0

RANDOM = False

NU = 0.01 / np.pi

NU_LOW = 0.001 / np.pi
NU_HIGH = 0.1 / np.pi


def generate_y(x: np.array, t: np.array, xn: int, tn: int, alpha: NU) -> np.array:
    y = burgers_viscous_time_exact1(alpha, xn, x, tn, t).T.reshape(-1, 1)
    return y


def generate_y_boundary(shape: tuple) -> np.array:
    return np.zeros(shape)


def generate_y_initial(x: np.array) -> np.array:
    y = -np.sin(np.pi * x)
    return y


def generate_data(
    mode: str,
    num_x: int,
    num_t: int,
    num_b: int,
    num_i: int,
    lb_x: float,
    rb_x: float,
    lb_t: float,
    rb_t: float,
    random: bool,
    alpha: float = NU,
) -> tuple[np.array, np.array]:
    interval_x = (rb_x - lb_x) / num_x
    interval_t = (rb_t - lb_t) / num_t
    x = (
        np.random.uniform(lb_x, rb_x, num_x)
        if random
        else np.arange(lb_x, rb_x + interval_x, interval_x)
    )
    t = (
        np.random.uniform(lb_t, rb_t, num_t)
        if random
        else np.arange(lb_t, rb_t + interval_t, interval_t)
    )

    if mode == "data":
        y = generate_y(x, t, num_x + 1, num_t + 1, alpha)
        x, t = np.meshgrid(x, t)
        x = x.reshape(-1, 1)
        t = t.reshape(-1, 1)

    elif mode == "physics":
        x, t = np.meshgrid(x, t)
        x = x.reshape(-1, 1)
        t = t.reshape(-1, 1)
        y = np.zeros(x.shape[0])

    elif mode == "boundary":
        interval_t = (rb_t - lb_t) / (num_b // 2)
        t = (
            np.random.uniform(lb_t, rb_t, num_b // 2)
            if random
            else np.arange(lb_t, rb_t + interval_t, interval_t)
        )
        t = np.hstack([t, t])
        x = np.vstack([np.full(t.shape[0] // 2, lb_x), np.full(t.shape[0] - t.shape[0] // 2, rb_x)])
        y = generate_y_boundary(t.shape[0])

    elif mode == "initial":
        x = (
            np.random.uniform(lb_x, rb_x, num_i)
            if random
            else np.arange(lb_x, rb_x + interval_x, interval_x)
        )
        t = np.zeros(x.shape[0])
        y = generate_y_initial(x)

    return x, t, y


def to_tensor(x: np.array, requires_grad: bool = True) -> torch.Tensor:
    t = torch.from_numpy(x)
    t.requires_grad = requires_grad
    t = t.float()
    t = t.reshape(-1, 1)
    return t


def generate_tasks(size: int, low: float, high: float):
    """Generate PINN tasks. Sample alpha of tasks.

    Args:
         size (int): number of tasks
         ood (boolean): whether out-of-distribution tasks be sampled or not

    Returns:
         tasks (list): tasks (support)
    """

    tasks = []
    for _ in range(size):
        alpha_lb, alpha_rb = low, high

        alpha_sup = np.random.uniform(low=low, high=high, size=1)
        alpha_qry = alpha_sup
        task = (alpha_sup, alpha_qry)
        tasks.append(task)

    return tasks
