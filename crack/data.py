import numpy as np
import torch
from scipy.stats import qmc


def generate_y(x: np.ndarray, task: np.ndarray, scenario: np.ndarray) -> np.ndarray:
    C_rep, C_f, a_f = task
    t, a0, C, sigma_r, N_avg = x
    G, m, a_mu, C_ins, t_end = scenario
    a = (
        a0 * np.exp(C * G**2 * sigma_r**2 * N_avg * t)
        if m == 2
        else np.power(
            (
                (1 - 0.5 * m) * C * G**m * sigma_r**m * N_avg * t * np.power(np.pi, 0.5 * m)
                + np.power(a0, (1 - 0.5 * m))
            ),
            (1 / 1 - 0.5 * m),
        )
    )

    pod = 1 - np.exp(-a / a_mu)

    cost = 2 * C_ins + n_rep * C_rep + n_f * C_f

    return a


def generate_tasks(n: int, seed=None):
    # C_rep, C_f, a_f
    sampler = qmc.LatinHypercube(d=3, seed=seed)
    sample_sup = sampler.random(n)
    sample_qry = sampler.random(n)
    l_bounds = [25, 101, 30]
    u_bounds = [100, 400, 70]
    sup = qmc.scale(sample_sup, l_bounds, u_bounds)
    qry = qmc.scale(sample_qry, l_bounds, u_bounds)
    return sup, qry


def generate_data(mode: str, n: int, task: np.ndarray):

    a0 = np.random.lognormal(0.5, 0.05, 1)
    C = np.random.lognormal(2.3 * 10 ** (-12), 0.3 * 2.3 * 10 ** (-12))
    sigma_r = np.random.lognormal(20, 0, 2.0, 1)
    N_avg = np.random.lognormal(5 * 10**5, 5 * 10**4, 1)
    G = 1.12
    m = 3.0
    a_mu = 1.8
    C_ins = 5
    t_end = 25

    C_rep, C_f, a_f = task

    scenario = (a0, C, sigma_r, N_avg, G, m, a_mu, C_ins, t_end)

    t = np.random.randint(1, 20, n)

    if mode == "data":

        y = generate_y(t, task, scenario)

        return t, y

    elif mode == "physics":
        y = np.full((n, DIM), -1.0, dtype=np.float32)
        return t, y


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
    print(generate_y(np.ones((1, DIM), np.float32), np.ones((1, DIM), np.float32)))
