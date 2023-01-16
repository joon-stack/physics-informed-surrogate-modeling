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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL_INTERVAL = 100
LOG_INTERVAL = 10
SAVE_INTERVAL = 100

RANDOM = False


def main(args: dict) -> None:
    print(f"device: {DEVICE}")
    train(
        epochs=args.epochs,
        lr=args.learning_rate,
        size=args.d_size,
        fpath=args.fpath,
        mode=args.mode,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a model!")
    parser.add_argument("--mode", type=str, default="data", help="training mode")
    parser.add_argument(
        "--d_size",
        type=int,
        default=10,
        help="number of labeled data (supervised learning)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="number of epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="learning rate",
    )

    parser.add_argument("--fpath", type=str, default=None, help="pre-trained model path")
    parser.add_argument("--project", type=str, default="ishigami", help="wandb project name")
    parser.add_argument("--run_name", type=str, default=None, help="wandb run name")
    cfg = parser.parse_args()
    wandb.init(project=cfg.project, config=cfg)
    if cfg.run_name != None:
        wandb.run.name = cfg.run_name
    main(cfg)
