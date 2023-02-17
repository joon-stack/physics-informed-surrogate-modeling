import torch
import torch.nn as nn
import numpy as np

import wandb
import os

from tqdm import trange, tqdm

from models import hybrid_model

import matplotlib.pyplot as plt

from metrics import compute_nrmse, compute_mse
from data import (
    generate_data,
    to_tensor,
    generate_tasks,
    generate_tasks_out,
    generate_tasks_out2,
    normalize,
    denormalize
)

# k1, k2, f1, t1, m, r
# TASK = np.full(DIM, 0.2, dtype=np.float32)

DIM = 1
# TASK, _ = generate_tasks(1)
# print(TASK)
# TASK = np.array([5, 42, 120, 60, 3, 3, 90, 175, 5, 10, 12], dtype=np.float32)
# TASK = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5])

VAL_INTERVAL = 1
LOG_INTERVAL = 10
SAVE_INTERVAL = 100


def save(ep, model, optim, loss, fname):
    torch.save(
        {
            "epoch": ep,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "loss": loss,
        },
        fname,
    )


def train(
    epochs: int,
    lr: float,
    size: int,
    f_size: int,
    b_size: int,
    fpath: str,
    mode: str,
    device_no: int,
    seed: int,
    task_out: int,
) -> dict:
    if task_out == 0:
        task, _ = generate_tasks(1, seed)
    elif task_out == 1:
        task, _ = generate_tasks_out(1, seed)
    elif task_out == 2:
        task, _ = generate_tasks_out2(1, seed)

    # task = task[0]
    task = np.array([0.02, 0.018, 0.016, 0.014, 0.012, 0.4, 0.36, 0.32, 0.28, 0.24])
    # task = TASK[0]
    # task = task.reshape(DIM)

    print(f"Current Mode: {mode}")
    print(f"Current Task: {task}")
    # DEVICE = torch.device(f"cuda:{device_no}" if torch.cuda.is_available() else "cpu")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current Device: {DEVICE}")
    model = hybrid_model(neuron_size=64, layer_size=6, dim=DIM)

    if fpath:
        model.load_state_dict(torch.load(fpath)["model_state_dict"])
        print(f"Model loaded from {fpath}")

    optim = torch.optim.Adam(model.parameters(), lr=lr)

    print(model)

    if mode == "data":
        x_train, y_train = generate_data(mode=mode, n=size, task=task)
        # y_train = y_train[1]
        y_min = np.min(y_train, axis=0)
        y_max = np.max(y_train, axis=0)
        
        y_train = normalize(y_min, y_max, y_train)

        x_val, y_val = generate_data(mode=mode, n=100, task=task)
        # y_val = y_val.reshape(-1, 1)
        # y_val = y_val[1]
        y_val = normalize(y_min, y_max, y_val)


        # print(x_train, y_train)
        x_train = to_tensor(x_train).reshape(-1, 1)
        y_train = to_tensor(y_train)
        # y_train = to_tensor(y_train).reshape(-1, 1)

        x_val = to_tensor(x_val).reshape(-1, 1)
        y_val = to_tensor(y_val)
        # y_val = to_tensor(y_val).reshape(-1, 1)
        

        x_train = x_train.to(DEVICE)
        y_train = y_train.to(DEVICE)
        x_val = x_val.to(DEVICE)
        y_val = y_val.to(DEVICE)


    elif mode == "hybrid":
        x_train, y_train = generate_data(mode="data", n=size, task=task)
        y_min = np.min(y_train, axis=0)
        y_max = np.max(y_train, axis=0)
        y_train = normalize(y_min, y_max, y_train)
        x_train = to_tensor(x_train).reshape(-1, 1)
        y_train = to_tensor(y_train)
        # y_train = y_train[1].reshape(-1, 1)
        

        x_f_train, y_f_train = generate_data(mode="physics", n=f_size, task=task)
        x_f_train = to_tensor(x_f_train).reshape(-1, 1)
        y_f_train = to_tensor(y_f_train).reshape(-1, 1)

        # x_b_train, y_b_train = generate_data(mode="boundary", n=b_size, task=task)
        # x_b_train = to_tensor(x_b_train)
        # y_b_train = to_tensor(y_b_train)

        x_val, y_val = generate_data(mode="data", n=100, task=task)
        y_val = normalize(y_min, y_max, y_val)
        x_val = to_tensor(x_val).reshape(-1, 1)
        y_val = to_tensor(y_val)
        # y_val = y_val[1].reshape(-1, 1)

        x_train = x_train.to(DEVICE)
        y_train = y_train.to(DEVICE)
        x_f_train = x_f_train.to(DEVICE)
        y_f_train = y_f_train.to(DEVICE)
        # x_b_train = x_b_train.to(DEVICE)
        # y_b_train = y_b_train.to(DEVICE)
        x_val = x_val.to(DEVICE)
        y_val = y_val.to(DEVICE)


    model = model.to(DEVICE)

    loss_func = nn.MSELoss()

    if mode == "data":

        for epoch in trange(1, epochs + 1):
            model.train()
            optim.zero_grad()
            loss_train = loss_func(y_train, model(x_train))
            # print(x_train)

            loss_train.to(DEVICE)

            loss_train.backward()

            optim.step()

            if epoch % VAL_INTERVAL == 0:
                model.eval()
                with torch.no_grad():
                    loss_val = loss_func(y_val, model(x_val))
                    nrmse = compute_nrmse(model(x_val).cpu().detach().numpy(), y_val.cpu().detach().numpy())
                    wandb.log(
                        {
                            "loss_val": loss_val.item(),
                            "nrmse": nrmse,
                        },
                        # commit=False,
                    )

            # wandb.log({"ep": epoch, "loss_train": loss_train.item()})

            if epoch % SAVE_INTERVAL == 0:
                pass

    elif mode == "hybrid":

        for epoch in trange(1, epochs + 1):
            model.train()

            optim.zero_grad()


            # print(x_train)
            # print(x_f_train)

            loss_d_train = loss_func(y_train, model(x_train))

            # loss_b_train = model.calc_loss_b(x_b_train, y_b_train)
            loss_f_train = model.calc_loss_f(x_f_train, y_f_train)

            loss_d_train.to(DEVICE)
            # loss_b_train.to(DEVICE)
            loss_f_train.to(DEVICE)

            loss_train = loss_d_train + 1* loss_f_train

            loss_train.backward()

            optim.step()

            if epoch % VAL_INTERVAL == 0:
                model.eval()
                with torch.no_grad():
                    loss_val = loss_func(y_val, model(x_val))
                    nrmse = compute_nrmse(model(x_val).cpu().detach().numpy(), y_val.cpu().detach().numpy())
                    wandb.log(
                        {
                            "loss_val": loss_val.item(),
                            "nrmse": nrmse,
                        },
                        # commit=False,
                    )

            if epoch % SAVE_INTERVAL == 0:
                pass

            # wandb.log(
            #     {
            #         "ep": epoch,
            #         "loss_d_train": loss_d_train.item(),
            #         # "loss_b_train": loss_b_train.item(),
            #         "loss_f_train": loss_f_train.item(),
            #         "loss_train": loss_train.item(),
            #     }
            # )
    x_plot = x_val.detach().cpu().numpy().squeeze()
    y_plot = model(x_val).detach().cpu().numpy()
    y_val = y_val.detach().cpu().numpy()
    y_plot = denormalize(y_min, y_max, y_plot)
    y_val = denormalize(y_min, y_max, y_val)

    s = x_plot.argsort()
    x_plot = x_plot[s]
    y_plot = y_plot[s].squeeze()
    y_val = y_val[s].squeeze()

    plt.plot(x_plot, y_plot[:, 0], 'r--', label="model")
    plt.plot(x_plot, y_val[:, 0], 'b-', label="truth")
    plt.legend()
    plt.show()
    # plt.cla()
    # plt.plot(x_plot, y_plot[:, 1], 'r--', label="model")
    # plt.plot(x_plot, y_val[:, 1], 'b-', label="truth")
    # plt.legend()
    # plt.show()

    # stress_plot = y_plot[:, 0]
    # disp_plot = y_plot[:, 1]
    # stress_data = [[x, y] for (x, y) in zip(x_plot, stress_plot)]
    # disp_data = [[x, y] for (x, y) in zip(x_plot, disp_plot)]
    # stress_table = wandb.Table(data=stress_data, columns=["x", "stress"])
    # disp_table = wandb.Table(data=disp_data, columns=["x", "disp"])
    # wandb.log({"stress": wandb.plot.scatter(stress_table, "x", "stress")})
    # wandb.log({"disp": wandb.plot.scatter(disp_table, "x", "disp")})


if __name__ == "__main__":
    pass
