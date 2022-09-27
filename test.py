import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL_INTERVAL = 100
LOG_INTERVAL = 100


def generate_y(x: np.array) -> np.array:
    y = np.sin(20 * x)
    return y


def generate_data(mode: str, num: int, lb: float, rb: float) -> tuple[np.array, np.array]:
    x = np.random.uniform(lb, rb, num)
    y = generate_y(x)
    return x, y


def to_tensor(x: np.array, requires_grad: bool = True) -> torch.Tensor:
    t = torch.from_numpy(x)
    t.requires_grad = requires_grad
    t = t.float()
    t = t.reshape(-1, 1)
    return t


class hybrid_model(nn.Module):
    def __init__(self, neuron_size: int, layer_size: int, dim: int) -> None:

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

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        act_func = nn.Sigmoid()

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
        u_hat = self(data)
        x = data.reshape(-1, 1)

        deriv_1 = autograd.grad(u_hat.sum(), data, create_graph=True)
        u_hat_x = deriv_1[0].reshape(-1, 1)
        deriv_2 = autograd.grad(u_hat_x.sum(), data, create_graph=True)
        u_hat_x_x = deriv_2[0].reshape(-1, 1)

        # modify here
        f = u_hat_x_x + x
        func = nn.MSELoss()

        return func(f, target)


def plot_progress(epochs: int, losses: list, mode: str) -> None:
    interval = 1 if mode == "train" else VAL_INTERVAL
    plt.cla()
    plt.plot(np.arange(1, epochs + 1, interval), losses)
    plt.title(f"{mode} progress")
    plt.savefig(f"./fig/test_{mode}.jpg")


def plot_comparison(model: torch.nn.Module) -> None:
    plt.cla()
    x_plot = np.arange(-1, 1, 0.01)
    x_plot_tensor = to_tensor(x_plot, requires_grad=False).to(DEVICE)
    pred = model(x_plot_tensor).cpu().detach().numpy()
    truth = generate_y(x_plot)

    plt.scatter(x_plot, pred, label="Prediction")
    plt.scatter(x_plot, truth, label="Ground truth")
    plt.legend()
    plt.savefig("./fig/test_comparison")
    nrmse = compute_nrmse(pred, truth)
    print(nrmse)


def compute_nrmse(pred: np.array, truth: np.array) -> float:
    pred = pred.reshape(truth.shape)
    nrmse = np.sum((pred - truth) ** 2) / np.sum(pred**2)
    return nrmse


def train(
    epochs: int = 1000,
    lr: float = 0.1,
    i_size: int = 500,
    b_size: int = 500,
    f_size: int = 1000,
    mode: str = "data",
):

    model = hybrid_model(neuron_size=5, layer_size=2, dim=1)

    optim = torch.optim.Adam(model.parameters(), lr=lr)

    print(model)

    if mode == "data":
        x_train, y_train = generate_data(mode="data", num=10, lb=-1, rb=1)
        x_train = to_tensor(x_train)
        y_train = to_tensor(y_train)

        x_val, y_val = generate_data(mode="data", num=2, lb=-1, rb=1)
        x_val = to_tensor(x_val)
        y_val = to_tensor(y_val)

        x_train = x_train.to(DEVICE)
        y_train = y_train.to(DEVICE)
        x_val = x_val.to(DEVICE)
        y_val = y_val.to(DEVICE)

    model = model.to(DEVICE)

    loss_func = nn.MSELoss()

    losses_train = []
    losses_val = []

    loss_save = np.inf

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

        if epoch % VAL_INTERVAL == 0:
            model.eval()

            loss_val = loss_func(y_val, model(x_val))
            print(f"Validation loss {loss_val.item(): .3f}")
            losses_val += [loss_val.item()]

    plot_progress(epochs, losses_train, mode="train")
    plot_progress(epochs, losses_val, mode="validation")
    plot_comparison(model)


if __name__ == "__main__":
    train()
