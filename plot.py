import matplotlib.pyplot as plt
import numpy as np
import torch

from data import generate_data, generate_y, to_tensor
from metrics import compute_nrmse

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VAL_INTERVAL = 100
LOG_INTERVAL = 10
SAVE_INTERVAL = 100

LB_X = -1.0
RB_X = 1.0
LB_T = 0.0
RB_T = 1.0

LB_X_OOD = 1.0
RB_X_OOD = 1.5
LB_T_OOD = 0.0
RB_T_OOD = 1.0

NU = 0.01 / np.pi


def plot_progress(epochs: int, losses: dict, train_mode: str, mode: str, maml: bool) -> None:
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

    # plt.title(f"{train_mode} progress")
    plt.savefig(f"./fig_burgers/test_{train_mode}_{mode}_maml{maml}.jpg")


def plot_comparison(
    model: torch.nn.Module, mode: str, ood: bool = False, maml: bool = True
) -> float:
    plt.cla()

    if not ood:
        x_plot = np.arange(LB_X, RB_X, 0.02)
        t_plot = np.arange(LB_T, RB_T, 0.01)
    else:
        x_plot = np.arange(LB_X_OOD, RB_X_OOD, 0.005)
        t_plot = np.arange(LB_T_OOD, RB_T_OOD, 0.01)

    truth = generate_y(x_plot, t_plot, 100, 100, alpha=NU)
    x_plot, t_plot = np.meshgrid(x_plot, t_plot)
    x_plot = x_plot.reshape(-1, 1)
    t_plot = t_plot.reshape(-1, 1)
    x_plot_tensor = to_tensor(x_plot, requires_grad=False).to(DEVICE)
    t_plot_tensor = to_tensor(t_plot, requires_grad=False).to(DEVICE)
    in_plot_tensor = torch.hstack([x_plot_tensor, t_plot_tensor])
    pred = model(in_plot_tensor).cpu().detach().numpy()

    plt.figure(figsize=(10, 8))
    # plt.subplot(2, 1, 1)
    plt.scatter(x_plot, t_plot, c=pred, cmap="seismic")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.colorbar()
    # plt.title(f"Prediction OOD: {ood}")

    # plt.subplot(2, 1, 2)
    # plt.scatter(x_plot, t_plot, c=truth, cmap="seismic")
    # plt.xlabel("x")
    # plt.ylabel("t")
    # plt.colorbar()
    # plt.title("Ground truth")

    plt.savefig(f"./fig_burgers/test_comparison_{mode}_maml{maml}_ood_{str(ood)}")
    nrmse = compute_nrmse(pred, truth)
    return nrmse


def plot_progress_maml(train_loss, epochs):
    plt.cla()
    plt.plot(np.arange(1, epochs + 1, 1), train_loss["inner_loss_pre_adapt"], label="pre-adapt")
    plt.plot(np.arange(1, epochs + 1, 1), train_loss["inner_loss_post_adapt"], label="post-adapt")
    plt.legend()
    plt.savefig("fig_burgers/maml_train_progress.png")


def plot_validation_maml(val_loss, epochs):
    plt.cla()
    plt.plot(
        np.arange(0, epochs + VAL_INTERVAL, VAL_INTERVAL),
        val_loss["inner_loss_pre_adapt"],
        label="pre-adapt",
    )
    plt.plot(
        np.arange(0, epochs + VAL_INTERVAL, VAL_INTERVAL),
        val_loss["inner_loss_post_adapt"],
        label="post-adapt",
    )
    plt.legend()
    plt.savefig("fig_burgers/maml_val_progress.png")


def plot_nrmse_maml(nrmse, epochs):
    plt.cla()
    plt.plot(
        np.arange(0, epochs + VAL_INTERVAL, VAL_INTERVAL),
        nrmse["nrmse_val_pre_adapt"],
        label="pre-adapt",
    )
    plt.plot(
        np.arange(0, epochs + VAL_INTERVAL, VAL_INTERVAL),
        nrmse["nrmse_val_post_adapt"],
        label="post-adapt",
    )
    plt.yscale("log")
    plt.legend()
    plt.savefig("fig_burgers/maml_nrmse.png")
