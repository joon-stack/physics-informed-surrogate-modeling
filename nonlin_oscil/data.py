import numpy as np
import torch
from scipy.stats import qmc


def generate_y(
    yk1: np.ndarray,
    yk2: np.ndarray,
    yf1: np.ndarray,
    yt1: np.ndarray,
    ym: np.ndarray,
    yr: np.ndarray,
    uk1: float,
    uk2: float,
    uf1: float,
    ut1: float,
    um: float,
    ur: float,
) -> np.ndarray:
    xk1 = yk1 * uk1
    xk2 = yk2 * uk2
    xf1 = yf1 * uf1
    xt1 = yt1 * ut1
    xm = ym * um
    xr = yr * ur
    # print(xk1, xk2, xf1, xt1, xm, xr)
    y = 3 * xr - np.abs((2 * xf1 / (xk1 + xk2)) * np.sin(xt1 * 0.5 * (xk1 + xk2) / xm))
    # print(y)
    # y = y.reshape(x1.shape)
    return y


def generate_tasks(n: int):
    sampler = qmc.LatinHypercube(d=6)
    sample_sup = sampler.random(n)
    sample_qry = sampler.random(n)
    l_bounds = [0.8, 0.08, 0.8, 0.8, 0.8, 0.4]
    u_bounds = [1.2, 0.12, 1.2, 1.2, 1.2, 0.6]
    sup = qmc.scale(sample_sup, l_bounds, u_bounds)
    qry = qmc.scale(sample_qry, l_bounds, u_bounds)
    return sup, qry


def generate_data(mode: str, n: int, task: np.ndarray):

    uk1, uk2, uf1, ut1, um, ur = task

    # sk1 = 0.1 * uk1
    # sk2 = 0.1 * uk2
    # sf1 = 0.2 * uf1
    # st1 = 0.2 * ut1
    # sm = 0.05 * um
    # sr = 0.1 * ur
    sk1 = 0.1
    sk2 = 0.01
    sf1 = 0.2
    st1 = 0.2
    sm = 0.05
    sr = 0.05

    x = np.random.normal([1, 1, 1, 1, 1, 1], [sk1, sk2, sf1, st1, sm, sr], (n, 6))
    if mode == "data":
        yk1 = x[:, 0]
        yk2 = x[:, 1]
        yf1 = x[:, 2]
        yt1 = x[:, 3]
        ym = x[:, 4]
        yr = x[:, 5]

        y = generate_y(yk1, yk2, yf1, yt1, ym, yr, uk1, uk2, uf1, ut1, um, ur)

        return x, y

    elif mode == "physics":
        # f_yf1**2 = 4 * uf1 ** 2
        y = 4 * uf1**2
        y = y.reshape(uf1.shape)
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
