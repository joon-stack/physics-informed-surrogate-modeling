import torch
import torch.nn as nn
import numpy as np

from models import hybrid_model
from maml import MAML

from data import generate_data, to_tensor
from plot import (
    plot_progress,
    plot_progress_maml,
    plot_nrmse_maml,
    plot_validation_maml,
    plot_comparison,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LB_X = -1.0
RB_X = 1.0
LB_T = 0.0
RB_T = 1.0

LB_X_OOD = 1.0
RB_X_OOD = 1.5
LB_T_OOD = 0.0
RB_T_OOD = 1.0

NU = 0.01 / np.pi
RANDOM = True

VAL_INTERVAL = 100
LOG_INTERVAL = 10
SAVE_INTERVAL = 100


def train(
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

            if epoch % VAL_INTERVAL == 0:
                model.eval()

                loss_val = loss_func(y_val, model(in_val))
                print(f"Validation loss {loss_val.item(): .3f}")
                losses_val += [loss_val.item()]

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

            if epoch % VAL_INTERVAL == 0:
                model.eval()

                loss_val = loss_func(y_val, model(in_val))
                print(f"Validation loss {loss_val.item(): .3f}")
                losses_val += [loss_val.item()]

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

            if epoch % VAL_INTERVAL == 0:
                model.eval()

                loss_val = loss_func(y_val, model(in_val))
                print(f"Validation loss {loss_val.item(): .3f}")
                losses_val += [loss_val.item()]

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
