import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import trange, tqdm
import wandb
import argparse
import os


from copy import deepcopy

from train import train


def main(args: dict) -> None:
    train(
        epochs=args.epochs,
        lr=args.learning_rate,
        size=args.d_size,
        f_size=args.f_size,
        fpath=args.fpath,
        mode=args.mode,
        device_no=args.device_no,
        seed=args.seed,
        task_out=args.task_out,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a model!")
    parser.add_argument("--mode", type=str, default="data", help="training mode")
    parser.add_argument(
        "--d_size",
        type=int,
        default=100,
        help="number of labeled data (supervised learning)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10000,
        help="number of epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0001,
        help="learning rate",
    )
    parser.add_argument(
        "--f_size", type=int, default=1000, help="number of physics data (hybrid learning)"
    )

    parser.add_argument("--fpath", type=str, default=None, help="pre-trained model path")
    parser.add_argument("--project", type=str, default="temp", help="wandb project name")
    parser.add_argument("--run_name", type=str, default=None, help="wandb run name")
    parser.add_argument("--device_no", type=int, default=0, help="cuda device number")
    parser.add_argument("--seed", type=int, default=None, help="seed for task generation")
    parser.add_argument(
        "--task_out", type=int, default=0, help="OOD tasks index, 0=ID, 1=OOD_1, 2=OOD_2"
    )
    cfg = parser.parse_args()
    wandb.init(project=cfg.project, config=cfg)
    if cfg.run_name != None:
        wandb.run.name = cfg.run_name
    main(cfg)
