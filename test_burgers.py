import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import tensorboard

import argparse
import os

from burgers import *

from copy import deepcopy


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

    # print(f"x_shape{x.shape}, t_shape{t.shape}, y_shape{y.shape}")
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
        # alpha_qry = np.random.uniform(low=low, high=high, size=size)
        task = (alpha_sup, alpha_qry)
        tasks.append(task)

    return tasks


class MAML:
    def __init__(
        self,
        num_inner_steps,
        inner_lr,
        outer_lr,
        log_dir,
        x_d_size,
        t_d_size,
        b_size,
        i_size,
    ):

        """Initializes First-Order Model-Agnostic Meta-Learning to Train Data-driven Surrogate Models.

        Args:
            num_inner_steps (int): number of inner-loop optimization steps
            inner_lr (float): learning rate for inner-loop optimization
            outer_lr (float): learning rate for outer-loop optimization
            num_data_b (int): number of boundary data
            num_data_f (int): number of PDE data
        """

        print("Initializing MAML surrogate model")

        self.model = hybrid_model(neuron_size=5, layer_size=3, dim=2, log_dir=log_dir)
        print("Current device: ", DEVICE)
        print(self.model)
        self.model.to(DEVICE)
        self.device = DEVICE

        self._num_inner_steps = num_inner_steps

        self._inner_lr = inner_lr
        self._outer_lr = outer_lr
        self._optimizer = torch.optim.Adam(self.model.parameters(), lr=self._outer_lr)

        self.x_d_size = x_d_size
        self.t_d_size = t_d_size
        self.b_size = b_size
        self.i_size = i_size

        self._train_step = 0

        self.log_dir = log_dir

        print("Finished initialization of MAML-PINN model")

    def _inner_loop(self, theta, support, train=True):

        """Computes the adapted network parameters via the MAML inner loop.

        Args:
            theta (List[Tensor]): current model parameters
            support (Tensor): support task. (alpha)
            train (Boolean): whether the model is trained or not,
                             if true, it returns gradient

        Returns:
            parameters (phi) (List[Tensor]): adapted network parameters
            inner_loss (list[float]): support set loss over the course of
                the inner loop, length num_inner_steps + 1
            grad (list[Tensor]): gradient of loss w.r.t. phi
        """

        inner_loss = []

        nrmse_batch = []

        model_phi = deepcopy(self.model)
        model_phi.load_state_dict(theta)
        model_phi.to(self.device)

        if train:
            model_phi.train()
        else:
            model_phi.eval()

        loss_fn = nn.MSELoss()
        opt_fn = torch.optim.Adam(model_phi.parameters(), lr=self._inner_lr)

        alpha = support

        if train:
            x_train, t_train, y_train = generate_data(
                mode="data",
                num_x=self.x_d_size,
                num_t=self.t_d_size,
                num_b=self.b_size,
                num_i=self.i_size,
                lb_x=LB_X,
                rb_x=RB_X,
                lb_t=LB_T,
                rb_t=RB_T,
                random=RANDOM,
                alpha=alpha,
            )

        else:
            x_train, t_train, y_train = generate_data(
                mode="data",
                num_x=self.x_d_size,
                num_t=self.t_d_size,
                num_b=self.b_size,
                num_i=self.i_size,
                lb_x=LB_X,
                rb_x=RB_X,
                lb_t=LB_T,
                rb_t=RB_T,
                random=RANDOM,
                alpha=alpha,
            )

        x_train = to_tensor(x_train)
        t_train = to_tensor(t_train)
        y_train = to_tensor(y_train)

        x_train = x_train.to(DEVICE)
        t_train = t_train.to(DEVICE)
        y_train = y_train.to(DEVICE)

        in_train = torch.hstack([x_train, t_train])

        num_inner_steps = self._num_inner_steps

        for _ in range(1, num_inner_steps + 1):
            nrmse = (
                compute_nrmse(
                    model_phi(in_train).cpu().detach().numpy(), y_train.cpu().detach().numpy()
                )
                if not train
                else None
            )
            if not train:
                nrmse_batch += [nrmse]

            opt_fn.zero_grad()
            loss = loss_fn(y_train, model_phi(in_train))
            loss.to(DEVICE)
            loss.backward()
            opt_fn.step()
            inner_loss += [loss.item()]

        loss = loss_fn(y_train, model_phi(in_train))
        inner_loss += [loss.item()]
        grad = torch.autograd.grad(loss, model_phi.parameters()) if train else None
        phi = model_phi.state_dict()
        nrmse = (
            compute_nrmse(
                model_phi(in_train).cpu().detach().numpy(), y_train.cpu().detach().numpy()
            )
            if not train
            else None
        )
        if not train:
            nrmse_batch += [nrmse]

        assert phi != None
        assert len(inner_loss) == num_inner_steps + 1

        return phi, grad, inner_loss, nrmse_batch

    def _outer_loop(self, task_batch, train=None):
        """Computes the MAML loss and metrics on a batch of tasks.

        Args:
            task_batch (tuple): batch of tasks from PDE
            each task consists with (support, query)
            each support and query consists with (alpha, beta)
            train (bool): whether we are training or evaluating

        Returns:
            outer_loss (Tensor): mean MAML loss over the batchk
            accuracies_support (ndarray): support set accuracy over the
                course of the inner loop, averaged over the task batch
                shape (num_inner_steps + 1,)
            accuracy_query (float): query set accuracy of the adapted
                parameters, averaged over the task batch
        """

        theta = self.model.state_dict()

        inner_loss = []

        grad_sum = [torch.zeros(w.shape).to(self.device) for w in list(self.model.parameters())]

        nrmse_batch = []

        model_outer = deepcopy(self.model)
        model_outer.load_state_dict(theta)
        model_outer.to(self.device)

        loss_fn = nn.MSELoss()
        for task in task_batch:
            support, query = task
            alpha = query
            phi, grad, loss_sup, nrmse = self._inner_loop(theta, support, train)
            inner_loss.append(loss_sup)

            model_outer.load_state_dict(phi)
            if train:
                x_train, t_train, y_train = generate_data(
                    mode="data",
                    num_x=self.x_d_size,
                    num_t=self.t_d_size,
                    num_b=self.b_size,
                    num_i=self.i_size,
                    lb_x=LB_X,
                    rb_x=RB_X,
                    lb_t=LB_T,
                    rb_t=RB_T,
                    random=RANDOM,
                    alpha=alpha,
                )
            else:
                x_train, t_train, y_train = generate_data(
                    mode="data",
                    num_x=self.x_d_size,
                    num_t=self.t_d_size,
                    num_b=self.b_size,
                    num_i=self.i_size,
                    lb_x=LB_X,
                    rb_x=RB_X,
                    lb_t=LB_T,
                    rb_t=RB_T,
                    random=RANDOM,
                    alpha=alpha,
                )

            x_train = to_tensor(x_train)
            t_train = to_tensor(t_train)
            y_train = to_tensor(y_train)

            x_train = x_train.to(DEVICE)
            t_train = t_train.to(DEVICE)
            y_train = y_train.to(DEVICE)

            in_train = torch.hstack([x_train, t_train])

            loss = loss_fn(y_train, model_outer(in_train))

            grad = torch.autograd.grad(loss, model_outer.parameters()) if train else None

            if train:
                for g_sum, g in zip(grad_sum, grad):
                    g_sum += g
            else:
                nrmse_batch += [nrmse]

        if train:
            for g in grad_sum:
                g /= len(task_batch)
            for w, g in zip(list(self.model.parameters()), grad_sum):
                w.grad = g
            self._optimizer.step()

        return np.mean(inner_loss, axis=0), np.mean(nrmse_batch, axis=0)

    def train(self, train_steps, num_train_tasks, num_val_tasks):
        """Train the MAML.

        Optimizes MAML meta-parameters

        Args:
            train_steps (int): the number of steps this model should train for
        """
        print("Start MAML training at iteration {}".format(self._train_step))

        train_loss = {"inner_loss_pre_adapt": [], "inner_loss_post_adapt": []}

        val_loss = {"inner_loss_pre_adapt": [], "inner_loss_post_adapt": []}

        nrmse = {"nrmse_val_post_adapt": [], "nrmse_val_pre_adapt": []}

        val_tasks = generate_tasks(num_val_tasks, low=0.001 / np.pi, high=0.1 / np.pi)
        inner_loss_val, nrmse_val = self._outer_loop(val_tasks, train=False)
        print(
            f"Validation before training pre-adaptation | Inner_loss: {inner_loss_val[0]:.4f} | NRMSE: {nrmse_val[0]:.4f}"
        )

        val_loss["inner_loss_pre_adapt"].append(inner_loss_val[0])
        nrmse["nrmse_val_pre_adapt"].append(nrmse_val[0])

        print(
            f"Validation before training post-adaptation | Inner_loss: {inner_loss_val[-1]:.4f} | NRMSE: {nrmse_val[-1]:.4f}"
        )

        val_loss["inner_loss_post_adapt"].append(inner_loss_val[-1])
        nrmse["nrmse_val_post_adapt"].append(nrmse_val[-1])

        for i in range(1, train_steps + 1):
            self._train_step += 1
            train_tasks = generate_tasks(num_train_tasks, low=NU_LOW, high=NU_HIGH)
            inner_loss, _ = self._outer_loop(train_tasks, train=True)

            train_loss["inner_loss_pre_adapt"].append(inner_loss[0])
            train_loss["inner_loss_post_adapt"].append(inner_loss[-1])

            if i % SAVE_INTERVAL == 0:
                print(f"Step {i} Model saved")
                self.model.save(i)

            if i % LOG_INTERVAL == 0:
                print(f"Step {self._train_step} Pre-Adapt | Inner_loss: {inner_loss[0]:.4f}")
                print(f"Step {self._train_step} Post-Adapt | Inner_loss: {inner_loss[-1]:.4f}")

            if i % VAL_INTERVAL == 0:
                inner_loss_val, nrmse_val = self._outer_loop(val_tasks, train=False)
                val_loss["inner_loss_pre_adapt"].append(inner_loss_val[0])
                nrmse["nrmse_val_pre_adapt"].append(nrmse_val[0])
                val_loss["inner_loss_post_adapt"].append(inner_loss_val[-1])
                nrmse["nrmse_val_post_adapt"].append(nrmse_val[-1])
                print(
                    f"Validation pre-adapt | Inner_loss: {inner_loss_val[0]:.4f} | NRMSE: {nrmse_val[0]:.4f}"
                )
                print(
                    f"Validation post-adapt | Inner_loss: {inner_loss_val[-1]:.4f} | NRMSE: {nrmse_val[-1]:.4f}"
                )

        return train_loss, val_loss, nrmse, self.model


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
        x = data[:, 0].reshape(-1, 1)
        t = data[:, 1].reshape(-1, 1)

        deriv_1 = autograd.grad(u_hat.sum(), data, create_graph=True)
        u_hat_x = deriv_1[0][:, 0].reshape(-1, 1)
        u_hat_t = deriv_1[0][:, 1].reshape(-1, 1)
        deriv_2 = autograd.grad(u_hat_x.sum(), data, create_graph=True)
        u_hat_x_x = deriv_2[0][:, 0].reshape(-1, 1)

        # modify here
        f = u_hat_t + u_hat * u_hat_x - NU * u_hat_x_x
        func = nn.MSELoss()

        return func(f, target)

    def save(self, epoch):
        torch.save(self.state_dict(), f"{os.path.join(self.log_dir, 'state')}{epoch}.model")


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


def plot_comparison(model: torch.nn.Module, mode: str, ood: bool = False, maml: bool = True) -> float:
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


def compute_nrmse(pred: np.array, truth: np.array) -> float:
    pred = pred.reshape(truth.shape)
    nrmse = np.sum((pred - truth) ** 2) / np.sum(pred**2)
    return nrmse

def plot_progress_maml(train_loss, epochs):
    plt.cla()
    plt.plot(np.arange(1, epochs + 1, 1), train_loss['inner_loss_pre_adapt'], label="pre-adapt")
    plt.plot(np.arange(1, epochs + 1, 1), train_loss['inner_loss_post_adapt'], label="post-adapt")
    plt.legend()
    plt.savefig("fig_burgers/maml_train_progress.png")

def plot_validation_maml(val_loss, epochs):
    plt.cla()
    plt.plot(np.arange(0, epochs + VAL_INTERVAL, VAL_INTERVAL), val_loss['inner_loss_pre_adapt'], label="pre-adapt")
    plt.plot(np.arange(0, epochs + VAL_INTERVAL, VAL_INTERVAL), val_loss['inner_loss_post_adapt'], label="post-adapt")
    plt.legend()
    plt.savefig("fig_burgers/maml_val_progress.png")

def plot_nrmse_maml(nrmse, epochs):
    plt.cla()
    plt.plot(np.arange(0, epochs + VAL_INTERVAL, VAL_INTERVAL), nrmse['nrmse_val_pre_adapt'], label="pre-adapt")
    plt.plot(np.arange(0, epochs + VAL_INTERVAL, VAL_INTERVAL), nrmse['nrmse_val_post_adapt'], label="post-adapt")
    plt.yscale("log")
    plt.legend()
    plt.savefig("fig_burgers/maml_nrmse.png")


def train(
    writer: tensorboard.SummaryWriter,
    log_dir: str,
    epochs: int,
    lr: float,
    x_d_size: int,
    t_d_size: int,
    i_size: int,
    b_size: int,
    x_size: int,
    t_size: int,
    load: bool,
    mode: str = "physics",
) -> dict:

    print(f"Current Mode: {mode}")
    model = hybrid_model(neuron_size=5, layer_size=3, dim=2, log_dir=log_dir)
    modelpath = "logs/maml/state1000.model"
    print(load)
    if load:
        model.load_state_dict(torch.load(modelpath))
        print("Model loaded")

    optim = torch.optim.Adam(model.parameters(), lr=lr)

    print(model)

    if mode == "data":
        x_train, t_train, y_train = generate_data(
            mode=mode,
            num_x=x_d_size,
            num_t=t_d_size,
            num_b=b_size,
            num_i=i_size,
            lb_x=LB_X,
            rb_x=RB_X,
            lb_t=LB_T,
            rb_t=RB_T,
            random=RANDOM,
        )
        x_train = to_tensor(x_train)
        t_train = to_tensor(t_train)
        y_train = to_tensor(y_train)

        x_val, t_val, y_val = generate_data(
            mode=mode,
            num_x=100,
            num_t=100,
            num_b=b_size,
            num_i=i_size,
            lb_x=LB_X,
            rb_x=RB_X,
            lb_t=LB_T,
            rb_t=RB_T,
            random=RANDOM,
        )
        x_val = to_tensor(x_val)
        t_val = to_tensor(t_val)
        y_val = to_tensor(y_val)

        x_train = x_train.to(DEVICE)
        t_train = t_train.to(DEVICE)
        y_train = y_train.to(DEVICE)
        x_val = x_val.to(DEVICE)
        t_val = t_val.to(DEVICE)
        y_val = y_val.to(DEVICE)

        in_train = torch.hstack([x_train, t_train])
        in_val = torch.hstack([x_val, t_val])

    elif mode == "physics":
        x_b_train, t_b_train, y_b_train = generate_data(
            mode="boundary",
            num_x=x_size,
            num_t=t_size,
            num_b=b_size,
            num_i=i_size,
            lb_x=LB_X,
            rb_x=RB_X,
            lb_t=LB_T,
            rb_t=RB_T,
            random=RANDOM,
        )
        x_b_train = to_tensor(x_b_train)
        t_b_train = to_tensor(t_b_train)
        y_b_train = to_tensor(y_b_train)

        print(
            f"x_b_train {x_b_train.shape}, t_b_train {t_b_train.shape}, y_b_train {y_b_train.shape}"
        )

        x_i_train, t_i_train, y_i_train = generate_data(
            mode="initial",
            num_x=x_size,
            num_t=t_size,
            num_b=b_size,
            num_i=i_size,
            lb_x=LB_X,
            rb_x=RB_X,
            lb_t=LB_T,
            rb_t=RB_T,
            random=RANDOM,
        )
        x_i_train = to_tensor(x_i_train)
        t_i_train = to_tensor(t_i_train)
        y_i_train = to_tensor(y_i_train)

        print(
            f"x_i_train {x_i_train.shape}, t_i_train {t_i_train.shape}, y_i_train {y_i_train.shape}"
        )

        x_f_train, t_f_train, y_f_train = generate_data(
            mode=mode,
            num_x=x_size,
            num_t=t_size,
            num_b=b_size,
            num_i=i_size,
            lb_x=LB_X,
            rb_x=RB_X,
            lb_t=LB_T,
            rb_t=RB_T,
            random=RANDOM,
        )
        x_f_train = to_tensor(x_f_train)
        t_f_train = to_tensor(t_f_train)
        y_f_train = to_tensor(y_f_train)

        print(
            f"x_f_train {x_f_train.shape}, t_f_train {t_f_train.shape}, y_f_train {y_f_train.shape}"
        )

        x_val, t_val, y_val = generate_data(
            mode="data",
            num_x=100,
            num_t=100,
            num_b=b_size,
            num_i=i_size,
            lb_x=LB_X,
            rb_x=RB_X,
            lb_t=LB_T,
            rb_t=RB_T,
            random=RANDOM,
        )
        x_val = to_tensor(x_val)
        t_val = to_tensor(t_val)
        y_val = to_tensor(y_val)

        x_b_train = x_b_train.to(DEVICE)
        t_b_train = t_b_train.to(DEVICE)
        y_b_train = y_b_train.to(DEVICE)
        x_i_train = x_i_train.to(DEVICE)
        t_i_train = t_i_train.to(DEVICE)
        y_i_train = y_i_train.to(DEVICE)
        x_f_train = x_f_train.to(DEVICE)
        t_f_train = t_f_train.to(DEVICE)
        y_f_train = y_f_train.to(DEVICE)
        x_val = x_val.to(DEVICE)
        t_val = t_val.to(DEVICE)
        y_val = y_val.to(DEVICE)

        in_b_train = torch.hstack([x_b_train, t_b_train])
        in_i_train = torch.hstack([x_i_train, t_i_train])
        in_f_train = torch.hstack([x_f_train, t_f_train])
        in_val = torch.hstack([x_val, t_val])

    elif mode == "hybrid":
        x_train, t_train, y_train = generate_data(
            mode="data",
            num_x=x_d_size,
            num_t=t_d_size,
            num_b=b_size,
            num_i=i_size,
            lb_x=LB_X,
            rb_x=RB_X,
            lb_t=LB_T,
            rb_t=RB_T,
            random=RANDOM,
        )
        x_train = to_tensor(x_train)
        t_train = to_tensor(t_train)
        y_train = to_tensor(y_train)

        x_b_train, t_b_train, y_b_train = generate_data(
            mode="boundary",
            num_x=x_size,
            num_t=t_size,
            num_b=b_size,
            num_i=i_size,
            lb_x=LB_X,
            rb_x=RB_X,
            lb_t=LB_T,
            rb_t=RB_T,
            random=RANDOM,
        )
        x_b_train = to_tensor(x_b_train)
        t_b_train = to_tensor(t_b_train)
        y_b_train = to_tensor(y_b_train)

        x_i_train, t_i_train, y_i_train = generate_data(
            mode="initial",
            num_x=x_size,
            num_t=t_size,
            num_b=b_size,
            num_i=i_size,
            lb_x=LB_X,
            rb_x=RB_X,
            lb_t=LB_T,
            rb_t=RB_T,
            random=RANDOM,
        )
        x_i_train = to_tensor(x_i_train)
        t_i_train = to_tensor(t_i_train)
        y_i_train = to_tensor(y_i_train)

        x_f_train, t_f_train, y_f_train = generate_data(
            mode="physics",
            num_x=x_size,
            num_t=t_size,
            num_b=b_size,
            num_i=i_size,
            lb_x=LB_X,
            rb_x=RB_X,
            lb_t=LB_T,
            rb_t=RB_T,
            random=RANDOM,
        )
        x_f_train = to_tensor(x_f_train)
        t_f_train = to_tensor(t_f_train)
        y_f_train = to_tensor(y_f_train)

        x_val, t_val, y_val = generate_data(
            mode="data",
            num_x=100,
            num_t=100,
            num_b=b_size,
            num_i=i_size,
            lb_x=LB_X,
            rb_x=RB_X,
            lb_t=LB_T,
            rb_t=RB_T,
            random=RANDOM,
        )
        x_val = to_tensor(x_val)
        t_val = to_tensor(t_val)
        y_val = to_tensor(y_val)

        x_train = x_train.to(DEVICE)
        t_train = t_train.to(DEVICE)
        y_train = y_train.to(DEVICE)
        x_b_train = x_b_train.to(DEVICE)
        t_b_train = t_b_train.to(DEVICE)
        y_b_train = y_b_train.to(DEVICE)
        x_i_train = x_i_train.to(DEVICE)
        t_i_train = t_i_train.to(DEVICE)
        y_i_train = y_i_train.to(DEVICE)
        x_f_train = x_f_train.to(DEVICE)
        t_f_train = t_f_train.to(DEVICE)
        y_f_train = y_f_train.to(DEVICE)
        x_val = x_val.to(DEVICE)
        t_val = t_val.to(DEVICE)
        y_val = y_val.to(DEVICE)

        in_train = torch.hstack([x_train, t_train])
        in_b_train = torch.hstack([x_b_train, t_b_train])
        in_i_train = torch.hstack([x_i_train, t_i_train])
        in_f_train = torch.hstack([x_f_train, t_f_train])
        in_val = torch.hstack([x_val, t_val])

    model = model.to(DEVICE)

    loss_func = nn.MSELoss()

    losses_train = []
    losses_val = []

    nrmse = {"id": 0.0, "ood": 0.0}

    if mode == "data":
        for epoch in range(1, epochs + 1):
            model.train()

            optim.zero_grad()
            loss_train = loss_func(y_train, model(in_train))

            loss_train.to(DEVICE)

            loss_train.backward()

            optim.step()

            losses_train += [loss_train.item()]

            if epoch % LOG_INTERVAL == 0:
                print(f"Epoch {epoch}: Loss {loss_train.item(): .3f}")
                writer.add_scalar("loss/train", loss_train.item(), epoch)

            if epoch % VAL_INTERVAL == 0:
                model.eval()

                loss_val = loss_func(y_val, model(in_val))
                print(f"Validation loss {loss_val.item(): .3f}")
                losses_val += [loss_val.item()]
                writer.add_scalar("loss/validation", loss_val.item(), epoch)

            if epoch % SAVE_INTERVAL == 0:
                model.save(epoch)

    elif mode == "physics":
        losses_b_train = []
        losses_i_train = []
        losses_f_train = []

        for epoch in range(1, epochs + 1):
            model.train()

            optim.zero_grad()

            loss_b_train = loss_func(y_b_train, model(in_b_train))
            loss_i_train = loss_func(y_i_train, model(in_i_train))
            loss_f_train = model.calc_loss_f(in_f_train, y_f_train)

            loss_b_train.to(DEVICE)
            loss_i_train.to(DEVICE)
            loss_f_train.to(DEVICE)

            loss_train = loss_b_train + loss_f_train + loss_i_train

            loss_train.backward()

            optim.step()

            losses_b_train += [loss_b_train.item()]
            losses_i_train += [loss_i_train.item()]
            losses_f_train += [loss_f_train.item()]
            losses_train += [loss_train.item()]

            if epoch % LOG_INTERVAL == 0:
                print(
                    f"Epoch {epoch}: Boundary loss {loss_b_train.item(): .3f}, Initial loss {loss_i_train.item(): .3f}, PDE loss {loss_f_train.item(): .3f}, Total loss {loss_train.item(): .3f}"
                )
                writer.add_scalar("loss/boundary", loss_b_train.item(), epoch)
                writer.add_scalar("loss/initial", loss_i_train.item(), epoch)
                writer.add_scalar("loss/PDE", loss_f_train.item(), epoch)
                writer.add_scalar("loss/total", loss_train.item(), epoch)

            if epoch % VAL_INTERVAL == 0:
                model.eval()

                loss_val = loss_func(y_val, model(in_val))
                print(f"Validation loss {loss_val.item(): .3f}")
                losses_val += [loss_val.item()]
                writer.add_scalar("loss/validation", loss_val.item(), epoch)

            if epoch % SAVE_INTERVAL == 0:
                model.save(epoch)

    elif mode == "hybrid":
        losses_d_train = []
        losses_b_train = []
        losses_i_train = []
        losses_f_train = []

        for epoch in range(1, epochs + 1):
            model.train()

            optim.zero_grad()

            loss_d_train = loss_func(y_train, model(in_train))
            loss_b_train = 0 * loss_func(y_b_train, model(in_b_train))
            loss_i_train = loss_func(y_i_train, model(in_i_train))

            loss_f_train = model.calc_loss_f(in_f_train, y_f_train)

            loss_d_train.to(DEVICE)
            loss_b_train.to(DEVICE)
            loss_i_train.to(DEVICE)
            loss_f_train.to(DEVICE)

            loss_train = loss_d_train + loss_b_train + loss_f_train + loss_i_train

            loss_train.backward()

            optim.step()

            losses_d_train += [loss_d_train.item()]
            losses_b_train += [loss_b_train.item()]
            losses_i_train += [loss_i_train.item()]
            losses_f_train += [loss_f_train.item()]
            losses_train += [loss_train.item()]

            if epoch % LOG_INTERVAL == 0:
                print(
                    f"Epoch {epoch}: Supervised loss {loss_d_train.item(): .3f}, Boundary loss {loss_b_train.item(): .3f}, Initial loss {loss_i_train.item(): .3f}, PDE loss {loss_f_train.item(): .3f}, Total loss {loss_train.item(): .3f}"
                )
                writer.add_scalar("loss/train", loss_d_train.item(), epoch)
                writer.add_scalar("loss/boundary", loss_b_train.item(), epoch)
                writer.add_scalar("loss/initial", loss_i_train.item(), epoch)
                writer.add_scalar("loss/PDE", loss_f_train.item(), epoch)
                writer.add_scalar("loss/total", loss_train.item(), epoch)

            if epoch % VAL_INTERVAL == 0:
                model.eval()

                loss_val = loss_func(y_val, model(in_val))
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
    print(losses_train_dict)

    plot_progress(epochs, losses_train_dict, train_mode="train", mode=mode, maml=load)
    plot_progress(epochs, losses_val_dict, train_mode="validation", mode=mode, maml=load)
    nrmse_id = plot_comparison(model, mode=mode, maml=load)
    nrmse_ood = plot_comparison(model, mode=mode, maml=load, ood=True)
    nrmse["id"] = nrmse_id
    nrmse["ood"] = nrmse_ood
    return nrmse


def train_maml() -> None:
    log_dir = "./logs/maml/"
    maml = MAML(10, 0.01, 0.01, log_dir, 5, 5, 0, 0)
    epochs = 1000
    train_loss, val_loss, nrmse, model = maml.train(epochs, 20, 5)
    plot_progress_maml(train_loss, epochs)
    plot_validation_maml(val_loss, epochs)
    plot_nrmse_maml(nrmse, epochs)


def main(args: dict) -> None:
    log_dir = args.log_dir
    if log_dir is None:
        log_dir = f"./logs/{args.mode}/{args.epochs}epochs_{args.learning_rate}lr_{args.num_supervised_x_data}x{args.num_supervised_t_data}data_{args.num_initial_data}data_i_{args.num_boundary_data}data_b_{args.num_x_data}x{args.num_t_data}data_f"
    print(f"log_dir: {log_dir}")
    writer = tensorboard.SummaryWriter(log_dir=log_dir)
    print(f"device: {DEVICE}")
    nrmse = {"id": [], "ood": [], "id_maml": [], "ood_maml": []}
    for _ in range(args.num_iter):
        nrmse_dict = train(
            epochs=args.epochs,
            lr=args.learning_rate,
            x_d_size=args.num_supervised_x_data,
            t_d_size=args.num_supervised_t_data,
            i_size=args.num_initial_data,
            b_size=args.num_boundary_data,
            x_size=args.num_x_data,
            t_size=args.num_t_data,
            mode=args.mode,
            log_dir=log_dir,
            writer=writer,
            load=False,
        )
        nrmse["id"].append(nrmse_dict["id"])
        nrmse["ood"].append(nrmse_dict["ood"])

        nrmse_dict = train(
            epochs=args.epochs,
            lr=args.learning_rate,
            x_d_size=args.num_supervised_x_data,
            t_d_size=args.num_supervised_t_data,
            i_size=args.num_initial_data,
            b_size=args.num_boundary_data,
            x_size=args.num_x_data,
            t_size=args.num_t_data,
            mode=args.mode,
            log_dir=log_dir,
            writer=writer,
            load=True,
        )
        nrmse["id_maml"].append(nrmse_dict["id"])
        nrmse["ood_maml"].append(nrmse_dict["ood"])


    std_id = np.std(nrmse["id"])
    std_ood = np.std(nrmse["ood"])
    avg_id = np.mean(nrmse["id"])
    avg_ood = np.mean(nrmse["ood"])

    std_id_maml = np.std(nrmse["id_maml"])
    std_ood_maml = np.std(nrmse["ood_maml"])
    avg_id_maml = np.mean(nrmse["id_maml"])
    avg_ood_maml = np.mean(nrmse["ood_maml"])

    print(
        f"ID | mean: {avg_id :.3f}, 95% CI: {avg_id - 1.96 * std_id / np.sqrt(args.num_iter): .3f} ~ {avg_id + 1.96 * std_id / np.sqrt(args.num_iter): .3f}"
    )
    print(
        f"OOD | mean: {avg_ood :.3f}, 95% CI: {avg_ood - 1.96 * std_ood / np.sqrt(args.num_iter): .3f} ~ {avg_ood + 1.96 * std_ood / np.sqrt(args.num_iter): .3f}"
    )

    print(
        f"ID_MAML | mean: {avg_id_maml :.3f}, 95% CI: {avg_id_maml - 1.96 * std_id_maml / np.sqrt(args.num_iter): .3f} ~ {avg_id_maml + 1.96 * std_id_maml / np.sqrt(args.num_iter): .3f}"
    )
    print(
        f"OOD_MAML | mean: {avg_ood_maml :.3f}, 95% CI: {avg_ood_maml - 1.96 * std_ood_maml / np.sqrt(args.num_iter): .3f} ~ {avg_ood_maml + 1.96 * std_ood_maml / np.sqrt(args.num_iter): .3f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a model!")
    parser.add_argument("--mode", type=str, default="data", help="training mode")
    parser.add_argument(
        "--num_supervised_x_data",
        type=int,
        default=5,
        help="number of labeled x data (supervised learning)",
    )
    parser.add_argument(
        "--num_supervised_t_data",
        type=int,
        default=5,
        help="number of labeled t data (supervised learning)",
    )
    parser.add_argument(
        "--num_boundary_data",
        type=int,
        default=10,
        help="number of boundary data (physics-informed learning)",
    )
    parser.add_argument(
        "--num_initial_data",
        type=int,
        default=10,
        help="number of initial data (physics-informed learning)",
    )
    parser.add_argument(
        "--num_x_data",
        type=int,
        default=100,
        help="number of x (pde) data (physics-informed learning)",
    )
    parser.add_argument(
        "--num_t_data",
        type=int,
        default=100,
        help="number of t (pde) data (physics-informed learning)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
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
    parser.add_argument(
        "--num_iter", type=int, default=5, help="how many times to iterate training"
    )
    parser.add_argument(
        "--load", type=bool, default=False, help="whether to load pretrained model or not"
    )

    main_args = parser.parse_args()
    
    train_maml()
    main(main_args)
