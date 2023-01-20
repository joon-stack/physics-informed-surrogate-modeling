import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import trange, tqdm
import wandb
import argparse
import os

from burgers_util import *

from copy import deepcopy

from train import train

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


NU_LOW = 0.001 / np.pi
NU_HIGH = 0.1 / np.pi

NU = np.random.uniform(NU_LOW, NU_HIGH, 1)


def main(args: dict) -> None:
    log_dir = args.log_dir
    if log_dir is None:
        log_dir = f"./logs/{args.mode}/{args.epochs}epochs_{args.learning_rate}lr_{args.num_supervised_x_data}x{args.num_supervised_t_data}data_{args.num_initial_data}data_i_{args.num_boundary_data}data_b_{args.num_x_data}x{args.num_t_data}data_f"
    print(f"log_dir: {log_dir}")
    print(f"device: {DEVICE}")
    nrmse = {"id": [], "ood": [], "id_maml": [], "ood_maml": []}
    for _ in range(args.num_iter):
        train(
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
            fpath=args.fpath,
            task=args.task / np.pi,
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
        default=2,
        help="number of labeled t data (supervised learning)",
    )
    parser.add_argument(
        "--num_boundary_data",
        type=int,
        default=100,
        help="number of boundary data (physics-informed learning)",
    )
    parser.add_argument(
        "--num_initial_data",
        type=int,
        default=100,
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
        type=float,
        default=0.01,
        help="learning rate",
    )
    parser.add_argument(
        "--log_dir", type=str, default=None, help="directory to save to or load from"
    )
    parser.add_argument(
        "--num_iter", type=int, default=1, help="how many times to iterate training"
    )
    parser.add_argument("--fpath", type=str, default=None, help="pre-trained model path")
    parser.add_argument("--project", type=str, default="burgers_small", help="wandb project name")
    parser.add_argument("--run_name", type=str, default=None, help="wandb run name")
    parser.add_argument("--task", type=float, default=0.01, help="task, (value)/np.pi = task")
    cfg = parser.parse_args()
    wandb.init(project=cfg.project, config=cfg)
    if cfg.run_name != None:
        wandb.run.name = cfg.run_name
    main(cfg)
