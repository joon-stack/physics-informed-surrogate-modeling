import torch
import torch.nn as nn
import torch.autograd as autograd
import os
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        act_func = nn.ReLU()

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
        x1 = data[:, 0].reshape(-1, 1)
        x2 = data[:, 1].reshape(-1, 1)
        x3 = data[:, 2].reshape(-1, 1)

        deriv_1 = autograd.grad(u_hat.sum(), data, create_graph=True)
        u_hat_x1 = deriv_1[0][:, 0].reshape(-1, 1)
        deriv_2 = autograd.grad(u_hat_x1.sum(), data, create_graph=True)
        u_hat_x1_x1 = deriv_2[0][:, 0].reshape(-1, 1)

        # modify here
        f = u_hat + u_hat_x1_x1
        func = nn.MSELoss()

        return func(f, target)
