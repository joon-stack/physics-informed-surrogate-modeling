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

TASK = (1.0, 0.1, 1.1, 1.1, 1.0, 0.7)

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
    fpath: str,
    mode: str = "physics",
) -> dict:

    print(f"Current Mode: {mode}")
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
        x_train, y_train = generate_data(mode=mode, n=size, task=TASK)
        x_train = to_tensor(x_train)
        y_train = to_tensor(y_train).reshape(-1, 1)

        x_val, y_val = generate_data(mode=mode, n=size, task=TASK)
        x_val = to_tensor(x_val)
        y_val = to_tensor(y_val).reshape(-1, 1)

        x_train = x_train.to(DEVICE)
        y_train = y_train.to(DEVICE)
        x_val = x_val.to(DEVICE)
        y_val = y_val.to(DEVICE)

    elif mode == "hybrid":
        x_train, y_train = generate_data(mode="data", n=size, task=TASK)
        x_train = to_tensor(x_train)
        y_train = to_tensor(y_train).reshape(-1, 1)

        x_f_train, y_f_train = generate_data(mode="physics", n=10000, task=TASK)
        x_f_train = to_tensor(x_f_train)
        y_f_train = to_tensor(y_f_train).reshape(-1, 1)

        x_val, y_val = generate_data(mode="data", n=size, task=TASK)
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

    losses_train = []
    losses_val = []

    nrmse = {"id": 0.0, "ood": 0.0}

    if mode == "data":

        for epoch in trange(1, epochs + 1):
            model.train()

            optim.zero_grad()
            loss_train = loss_func(y_train, model(x_train))

            loss_train.to(DEVICE)

            loss_train.backward()

            optim.step()

            losses_train += [loss_train.item()]

            if epoch % VAL_INTERVAL == 0:
                model.eval()
                loss_val = loss_func(y_val, model(x_val))
                losses_val += [loss_val.item()]
                mse = compute_mse(model(x_val).cpu().detach().numpy(), y_val.cpu().detach().numpy())
                # wandb.log({"loss_val": loss_val.item(), "mse": mse}, commit=False)

            # wandb.log({"ep": epoch, "loss_train": loss_train.item()})

            if epoch % SAVE_INTERVAL == 0:
                pass

    elif mode == "physics":
        losses_b_train = []
        losses_i_train = []
        losses_f_train = []

        for epoch in trange(1, epochs + 1):
            model.train()

            optim.zero_grad()

            loss_f_train = model.calc_loss_f(x_f_train, y_f_train)

            loss_f_train.to(DEVICE)

            loss_train = loss_b_train + loss_f_train + loss_i_train

            loss_train.backward()

            optim.step()

            losses_b_train += [loss_b_train.item()]
            losses_i_train += [loss_i_train.item()]
            losses_f_train += [loss_f_train.item()]
            losses_train += [loss_train.item()]

            if epoch % VAL_INTERVAL == 0:
                model.eval()

                loss_val = loss_func(y_val, model(in_val))
                # print(f"Validation loss {loss_val.item(): .3f}")
                losses_val += [loss_val.item()]
                nrmse = compute_nrmse(
                    model(in_val).cpu().detach().numpy(), y_val.cpu().detach().numpy()
                )
                wandb.log({"loss_val": loss_train.item(), "nrmse": nrmse}, commit=False)

            if epoch % SAVE_INTERVAL == 0:
                pass

            wandb.log(
                {
                    "ep": epoch,
                    "loss_b_train": loss_b_train.item(),
                    "loss_i_train": loss_i_train.item(),
                    "loss_f_train": loss_f_train.item(),
                    "loss_train": loss_train.item(),
                }
            )

    elif mode == "hybrid":
        fname = f"./logs/{mode}"
        if not os.path.exists(fname):
            os.makedirs(fname)
        losses_d_train = []
        losses_b_train = []
        losses_i_train = []
        losses_f_train = []

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

            losses_d_train += [loss_d_train.item()]
            losses_f_train += [loss_f_train.item()]
            losses_train += [loss_train.item()]

            if epoch % VAL_INTERVAL == 0:
                model.eval()

                loss_val = loss_func(y_val, model(x_val))
                losses_val += [loss_val.item()]

                mse = compute_mse(model(x_val).cpu().detach().numpy(), y_val.cpu().detach().numpy())
                wandb.log({"loss_val": loss_train.item(), "mse": mse}, commit=False)

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

    prob_hat, prob = calc_act_mod_prob(model, 1000000)
    print(f"Actual Pf: {prob:.5f}")
    print(f"Modeled Pf: {prob_hat:.5f}")
    return nrmse


def calc_act_mod_prob(model, n):
    x, y = generate_data(mode="data", n=n, task=np.array(TASK))
    x = to_tensor(x).to(DEVICE)
    model = model.to(DEVICE)
    y_hat = model(x)

    prob_hat = calc_prob(y_hat)
    prob = calc_prob(y)

    return prob_hat, prob


def calc_prob(y):
    res = torch.zeros(y.shape)
    res[y < 0] = 1
    y_count = torch.count_nonzero(res)
    prob = y_count / len(y)
    return prob


if __name__ == "__main__":
    _, y = generate_data(mode="data", n=1000000, task=np.array(TASK))
    prob = calc_prob(y)
    print(f"Actual Pf: {prob:.5f}")
