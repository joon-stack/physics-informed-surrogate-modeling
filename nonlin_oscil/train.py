import torch
import torch.nn as nn
import numpy as np

import wandb
import os

from tqdm import trange, tqdm

from models import hybrid_model

# from maml import MAML, MAML_hybrid

from metrics import compute_nrmse, compute_mse
from data import generate_data, to_tensor, generate_tasks

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TASK = (1.0, 0.1, 1.0, 1.5, 2.0, 0.5)

VAL_INTERVAL = 10
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
    fpath: str,
    mode: str,
    task: np.ndarray,
) -> dict:
    task = np.array(task, dtype=np.float32)
    print(f"Current Mode: {mode}")
    print(f"Current Task: {task}")
    # model = hybrid_model(neuron_size=64, layer_size=6, dim=2, log_dir=log_dir)
    model = hybrid_model(neuron_size=64, layer_size=6, dim=6)

    # if mode == "data":
    #     model = nn.DataParallel(model)
    # print(torch.load(fpath)["model_state_dict"])
    # model = hybrid_model(neuron_size=5, layer_size=6, dim=2, log_dir=log_dir)
    # modelpath = "logs/maml/state1000.model"
    if fpath:
        model.load_state_dict(torch.load(fpath)["model_state_dict"])
        print(f"Model loaded from {fpath}")

    optim = torch.optim.Adam(model.parameters(), lr=lr)

    print(model)

    if mode == "data":
        x_train, y_train = generate_data(mode=mode, n=size, task=task)
        x_train = to_tensor(x_train)
        y_train = to_tensor(y_train).reshape(-1, 1)

        x_val, y_val = generate_data(mode=mode, n=size, task=task)
        x_val = to_tensor(x_val)
        y_val = to_tensor(y_val).reshape(-1, 1)

        x_train = x_train.to(DEVICE)
        y_train = y_train.to(DEVICE)
        x_val = x_val.to(DEVICE)
        y_val = y_val.to(DEVICE)

    elif mode == "hybrid":
        x_train, y_train = generate_data(mode="data", n=size, task=task)
        x_train = to_tensor(x_train)
        y_train = to_tensor(y_train).reshape(-1, 1)

        x_f_train, y_f_train = generate_data(mode="physics", n=10000, task=task)
        x_f_train = to_tensor(x_f_train)
        y_f_train = to_tensor(y_f_train).reshape(-1, 1)

        x_val, y_val = generate_data(mode="data", n=size, task=task)
        x_val = to_tensor(x_val)
        y_val = to_tensor(y_val).reshape(-1, 1)

        x_train = x_train.to(DEVICE)
        y_train = y_train.to(DEVICE)
        x_f_train = x_f_train.to(DEVICE)
        y_f_train = y_f_train.to(DEVICE)
        x_val = x_val.to(DEVICE)
        y_val = y_val.to(DEVICE)

    model = model.to(DEVICE)

    loss_func = nn.MSELoss()

    # generate x, y to calculate pf
    x, y = generate_data(mode="data", n=70000, task=np.array(task))
    prob = calc_actual_prob(y)
    wandb.log({"actual_pf": prob})

    if mode == "data":

        for epoch in trange(1, epochs + 1):
            model.train()

            optim.zero_grad()
            loss_train = loss_func(y_train, model(x_train))

            loss_train.to(DEVICE)

            loss_train.backward()

            optim.step()

            if epoch % VAL_INTERVAL == 0:
                model.eval()
                loss_val = loss_func(y_val, model(x_val))
                mse = compute_mse(model(x_val).cpu().detach().numpy(), y_val.cpu().detach().numpy())
                prob_hat = calc_modeled_prob(model, x)
                wandb.log({"loss_val": loss_val.item(), "mse": mse, "pf": prob_hat}, commit=False)

            wandb.log({"ep": epoch, "loss_train": loss_train.item()})

            if epoch % SAVE_INTERVAL == 0:
                pass

    elif mode == "hybrid":

        for epoch in trange(1, epochs + 1):
            model.train()

            optim.zero_grad()

            loss_d_train = loss_func(y_train, model(x_train))

            loss_f_train = model.calc_loss_f(x_f_train, y_f_train)

            loss_d_train.to(DEVICE)
            loss_f_train.to(DEVICE)

            loss_train = loss_d_train + loss_f_train

            loss_train.backward()

            optim.step()

            if epoch % VAL_INTERVAL == 0:
                model.eval()

                loss_val = loss_func(y_val, model(x_val))
                mse = compute_mse(model(x_val).cpu().detach().numpy(), y_val.cpu().detach().numpy())
                prob_hat = calc_modeled_prob(model, x)
                wandb.log({"loss_val": loss_val.item(), "mse": mse, "pf": prob_hat}, commit=False)

            if epoch % SAVE_INTERVAL == 0:
                pass

            wandb.log(
                {
                    "ep": epoch,
                    "loss_d_train": loss_d_train.item(),
                    "loss_f_train": loss_f_train.item(),
                    "loss_train": loss_train.item(),
                }
            )

    # fname = os.path.join(wandb.run.dir, "model.h5")
    # save(epoch, model, optim, loss_train.item(), fname)

    # prob_hat, prob = calc_act_mod_prob(model, 1000000)
    # print(f"Actual Pf: {prob:.5f}")
    # print(f"Modeled Pf: {prob_hat:.5f}")


def calc_modeled_prob(model, x):
    x = to_tensor(x).to(DEVICE)
    model = model.to(DEVICE)
    y_hat = model(x)

    prob_hat = calc_prob(y_hat)

    return prob_hat


def calc_actual_prob(y):
    prob = calc_prob(y)
    return prob


def calc_prob(y):
    res = torch.zeros(y.shape)
    res[y < 0] = 1
    y_count = torch.count_nonzero(res)
    prob = y_count / len(y)
    return prob


if __name__ == "__main__":
    _, y = generate_data(mode="data", n=70000, task=np.array(TASK))
    prob = calc_prob(y)
    print(f"Actual Pf: {prob:.5f}")
