import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import tensorboard

import argparse
import os


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL_INTERVAL = 100
LOG_INTERVAL = 100
SAVE_INTERVAL = 5000
LB = -10.0
RB = 10.0
RANDOM = False


def generate_y(x: np.array) -> np.array:
    y = np.sin(1 * x)
    return y


def generate_data(
    mode: str, num: int, lb: float, rb: float, random: bool
) -> tuple[np.array, np.array]:
    interval = (rb - lb) / num
    x = np.random.uniform(lb, rb, num) if random else np.arange(lb, rb + interval, interval)

    if mode == "data":
        y = generate_y(x)
    elif mode == "physics":
        y = np.zeros(x.shape)
    elif mode == "boundary":
        x = np.vstack([np.full(1, lb), np.full(1, rb)])
        y = generate_y(x)

    return x, y


def to_tensor(x: np.array, requires_grad: bool = True) -> torch.Tensor:
    t = torch.from_numpy(x)
    t.requires_grad = requires_grad
    t = t.float()
    t = t.reshape(-1, 1)
    return t


class hybrid_model(nn.Module):
    def __init__(self, neuron_size: int, layer_size: int, dim: int, log_dir: str) -> None:

        super(hybrid_model, self).__init__()

        layers = []

        for i in range(layer_size):
            if i == 0:
                layer = nn.Linear(dim, neuron_size)
            elif i == layer_size - 1:
                layer = nn.Linear(neuron_size, 1)
            else:
                layer = nn.Linear(neuron_size, neuron_size)

            layers.append(layer)

        self.module1 = nn.Sequential(*layers)

        self.device = DEVICE
        self.log_dir = log_dir

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        act_func = nn.Tanh()

        tmp = data

        for n, layer in enumerate(self.module1):
            if n == len(self.module1) - 1:
                tmp = layer(tmp)
                break
            tmp = act_func(layer(tmp))

        return tmp

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return self.forward(data)

    def calc_loss_f(self, data: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if data == None:
            return 0

        u_hat = self(data)
        x = data.reshape(-1, 1)

        deriv_1 = autograd.grad(u_hat.sum(), data, create_graph=True)
        u_hat_x = deriv_1[0].reshape(-1, 1)
        deriv_2 = autograd.grad(u_hat_x.sum(), data, create_graph=True)
        u_hat_x_x = deriv_2[0].reshape(-1, 1)

        # modify here
        f = u_hat_x_x + 1 * u_hat
        func = nn.MSELoss()

        return func(f, target)

    def save(self, epoch):
        torch.save(self.state_dict(), f"{os.path.join(self.log_dir, 'state')}{epoch}.model")


def plot_progress(epochs: int, losses: dict, train_mode: str, mode: str) -> None:
    interval = 1 if train_mode == "train" else VAL_INTERVAL
    plt.cla()
    if mode == "data" or train_mode == "validation":
        plt.plot(np.arange(1, epochs + 1, interval), losses["total"], label="Supervised loss")
    elif mode == "physics":
        plt.plot(np.arange(1, epochs + 1, interval), losses["boundary"], label="Boundary loss")
        plt.plot(np.arange(1, epochs + 1, interval), losses["pde"], label="PDE loss")
        plt.plot(np.arange(1, epochs + 1, interval), losses["total"], label="Total loss")
    elif mode == "hybrid":
        plt.plot(np.arange(1, epochs + 1, interval), losses["data"], label="Supervised loss")
        plt.plot(np.arange(1, epochs + 1, interval), losses["boundary"], label="Boundary loss")
        plt.plot(np.arange(1, epochs + 1, interval), losses["pde"], label="PDE loss")
        plt.plot(np.arange(1, epochs + 1, interval), losses["total"], label="Total loss")

    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.yscale("log")

    plt.title(f"{train_mode} progress")
    plt.savefig(f"./fig/test_{train_mode}_{mode}.jpg")


def plot_comparison(model: torch.nn.Module, mode: str) -> None:
    plt.cla()
    x_plot = np.arange(LB, RB, 0.01)
    x_plot_tensor = to_tensor(x_plot, requires_grad=False).to(DEVICE)
    pred = model(x_plot_tensor).cpu().detach().numpy()
    truth = generate_y(x_plot)

    plt.plot(x_plot, truth, "b", linewidth=3, label="Ground truth")
    plt.plot(x_plot, pred, "r--", label="Prediction")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("u")
    plt.savefig(f"./fig/test_comparison_{mode}")
    nrmse = compute_nrmse(pred, truth)
    print(nrmse)


def compute_nrmse(pred: np.array, truth: np.array) -> float:
    pred = pred.reshape(truth.shape)
    nrmse = np.sum((pred - truth) ** 2) / np.sum(pred**2)
    return nrmse


def train(
    writer: tensorboard.SummaryWriter,
    log_dir: str,
    epochs: int = 10000,
    lr: float = 0.1,
    i_size: int = 500,
    b_size: int = 500,
    f_size: int = 1000,
    mode: str = "physics",
):

    print(f"Current Mode: {mode}")
    model = hybrid_model(neuron_size=5, layer_size=2, dim=1, log_dir=log_dir)

    optim = torch.optim.Adam(model.parameters(), lr=lr)

    print(model)

    if mode == "data":
        x_train, y_train = generate_data(mode=mode, num=10, lb=LB, rb=RB, random=RANDOM)
        x_train = to_tensor(x_train)
        y_train = to_tensor(y_train)

        x_val, y_val = generate_data(mode=mode, num=100, lb=LB, rb=RB, random=RANDOM)
        x_val = to_tensor(x_val)
        y_val = to_tensor(y_val)

        x_train = x_train.to(DEVICE)
        y_train = y_train.to(DEVICE)
        x_val = x_val.to(DEVICE)
        y_val = y_val.to(DEVICE)

    elif mode == "physics":
        x_b_train, y_b_train = generate_data(mode="boundary", num=1, lb=LB, rb=RB, random=RANDOM)
        x_b_train = to_tensor(x_b_train)
        y_b_train = to_tensor(y_b_train)

        x_f_train, y_f_train = generate_data(mode=mode, num=f_size, lb=LB, rb=RB, random=RANDOM)
        x_f_train = to_tensor(x_f_train)
        y_f_train = to_tensor(y_f_train)

        x_val, y_val = generate_data(mode="data", num=100, lb=LB, rb=RB, random=RANDOM)
        x_val = to_tensor(x_val)
        y_val = to_tensor(y_val)

        x_b_train = x_b_train.to(DEVICE)
        y_b_train = y_b_train.to(DEVICE)
        x_f_train = x_f_train.to(DEVICE)
        y_f_train = y_f_train.to(DEVICE)
        x_val = x_val.to(DEVICE)
        y_val = y_val.to(DEVICE)

    elif mode == "hybrid":
        x_train, y_train = generate_data(mode="data", num=10, lb=LB, rb=RB, random=RANDOM)
        x_train = to_tensor(x_train)
        y_train = to_tensor(y_train)

        x_b_train, y_b_train = generate_data(
            mode="boundary", num=b_size, lb=LB, rb=RB, random=RANDOM
        )
        x_b_train = to_tensor(x_b_train)
        y_b_train = to_tensor(y_b_train)

        x_f_train, y_f_train = generate_data(
            mode="physics", num=f_size, lb=LB, rb=RB, random=RANDOM
        )
        x_f_train = to_tensor(x_f_train)
        y_f_train = to_tensor(y_f_train)

        x_val, y_val = generate_data(mode="data", num=100, lb=LB, rb=RB, random=RANDOM)
        x_val = to_tensor(x_val)
        y_val = to_tensor(y_val)

        x_train = x_train.to(DEVICE)
        y_train = y_train.to(DEVICE)
        x_b_train = x_b_train.to(DEVICE)
        y_b_train = y_b_train.to(DEVICE)
        x_f_train = x_f_train.to(DEVICE)
        y_f_train = y_f_train.to(DEVICE)
        x_val = x_val.to(DEVICE)
        y_val = y_val.to(DEVICE)

    model = model.to(DEVICE)

    loss_func = nn.MSELoss()

    losses_train = []
    losses_val = []

    if mode == "data":
        for epoch in range(1, epochs + 1):
            model.train()

            optim.zero_grad()

            loss_train = loss_func(y_train, model(x_train))

            loss_train.to(DEVICE)

            loss_train.backward()

            optim.step()

            losses_train += [loss_train.item()]

            if epoch % LOG_INTERVAL == 0:
                print(f"Epoch {epoch}: Loss {loss_train.item(): .3f}")
                writer.add_scalar("loss/train", loss_train.item(), epoch)

            if epoch % VAL_INTERVAL == 0:
                model.eval()

                loss_val = loss_func(y_val, model(x_val))
                print(f"Validation loss {loss_val.item(): .3f}")
                losses_val += [loss_val.item()]
                writer.add_scalar("loss/validation", loss_val.item(), epoch)

            if epoch % SAVE_INTERVAL == 0:
                model.save(epoch)

    elif mode == "physics":
        losses_b_train = []
        losses_f_train = []

        for epoch in range(1, epochs + 1):
            model.train()

            optim.zero_grad()

            loss_b_train = loss_func(y_b_train, model(x_b_train))
            loss_f_train = model.calc_loss_f(x_f_train, y_f_train)

            loss_b_train.to(DEVICE)
            loss_f_train.to(DEVICE)

            loss_train = loss_b_train + loss_f_train

            loss_train.backward()

            optim.step()

            losses_b_train += [loss_b_train.item()]
            losses_f_train += [loss_f_train.item()]
            losses_train += [loss_train.item()]

            if epoch % LOG_INTERVAL == 0:
                print(
                    f"Epoch {epoch}: Boundary loss {loss_b_train.item(): .3f}, PDE loss {loss_f_train.item(): .3f}, Total loss {loss_train.item(): .3f}"
                )
                writer.add_scalar("loss/boundary", loss_b_train.item(), epoch)
                writer.add_scalar("loss/PDE", loss_f_train.item(), epoch)
                writer.add_scalar("loss/total", loss_train.item(), epoch)

            if epoch % VAL_INTERVAL == 0:
                model.eval()

                loss_val = loss_func(y_val, model(x_val))
                print(f"Validation loss {loss_val.item(): .3f}")
                losses_val += [loss_val.item()]
                writer.add_scalar("loss/validation", loss_val.item(), epoch)

            if epoch % SAVE_INTERVAL == 0:
                model.save(epoch)

    elif mode == "hybrid":
        losses_d_train = []
        losses_b_train = []
        losses_f_train = []

        for epoch in range(1, epochs + 1):
            model.train()

            optim.zero_grad()

            loss_d_train = loss_func(y_train, model(x_train))
            loss_b_train = loss_func(y_b_train, model(x_b_train))
            loss_f_train = model.calc_loss_f(x_f_train, y_f_train)

            loss_d_train.to(DEVICE)
            loss_b_train.to(DEVICE)
            loss_f_train.to(DEVICE)

            loss_train = loss_d_train + loss_b_train + loss_f_train

            loss_train.backward()

            optim.step()

            losses_d_train += [loss_d_train.item()]
            losses_b_train += [loss_b_train.item()]
            losses_f_train += [loss_f_train.item()]
            losses_train += [loss_train.item()]

            if epoch % LOG_INTERVAL == 0:
                print(
                    f"Epoch {epoch}: Supervised loss {loss_d_train.item(): .3f}, Boundary loss {loss_b_train.item(): .3f}, PDE loss {loss_f_train.item(): .3f}, Total loss {loss_train.item(): .3f}"
                )
                writer.add_scalar("loss/train", loss_d_train.item(), epoch)
                writer.add_scalar("loss/boundary", loss_b_train.item(), epoch)
                writer.add_scalar("loss/PDE", loss_f_train.item(), epoch)
                writer.add_scalar("loss/total", loss_train.item(), epoch)

            if epoch % VAL_INTERVAL == 0:
                model.eval()

                loss_val = loss_func(y_val, model(x_val))
                print(f"Validation loss {loss_val.item(): .3f}")
                losses_val += [loss_val.item()]
                writer.add_scalar("loss/validation", loss_val.item(), epoch)

            if epoch % SAVE_INTERVAL == 0:
                model.save(epoch)

    if mode == "data":
        losses_train_dict = {"total": losses_train}
    elif mode == "physics":
        losses_train_dict = {
            "boundary": losses_b_train,
            "pde": losses_f_train,
            "total": losses_train,
        }
    elif mode == "hybrid":
        losses_train_dict = {
            "data": losses_d_train,
            "boundary": losses_b_train,
            "pde": losses_f_train,
            "total": losses_train,
        }

    losses_val_dict = {"total": losses_val}

    plot_progress(epochs, losses_train_dict, train_mode="train", mode=mode)
    plot_progress(epochs, losses_val_dict, train_mode="validation", mode=mode)
    plot_comparison(model, mode=mode)


def main(args: dict) -> None:
    log_dir = args.log_dir
    if log_dir is None:
        log_dir = f"./logs/{args.mode}/{args.epochs}epochs_{args.learning_rate}lr_{args.num_data}data_{args.num_boundary_data}data_b_{args.num_pde_data}data_f"
    print(f"log_dir: {log_dir}")
    writer = tensorboard.SummaryWriter(log_dir=log_dir)
    print(f"device: {DEVICE}")
    train(
        epochs=args.epochs,
        lr=args.learning_rate,
        i_size=0,
        b_size=args.num_boundary_data,
        f_size=args.num_pde_data,
        mode=args.mode,
        log_dir=log_dir,
        writer=writer,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a model!")
    parser.add_argument("--mode", type=str, default="data", help="training mode")
    parser.add_argument(
        "--num_data", type=int, default=10, help="number of labeled data (supervised learning)"
    )
    parser.add_argument(
        "--num_boundary_data",
        type=int,
        default=2,
        help="number of boundary data (physics-informed learning)",
    )
    parser.add_argument(
        "--num_pde_data",
        type=int,
        default=10000,
        help="number of pde data (physics-informed learning)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20000,
        help="number of epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=int,
        default=0.01,
        help="learning rate",
    )
    parser.add_argument(
        "--log_dir", type=str, default=None, help="directory to save to or load from"
    )

    main_args = parser.parse_args()
    main(main_args)
