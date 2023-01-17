from maml import *
import wandb
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument(
        "--project",
        type=str,
        default="maml_ishigami",
        help="project name",
    )
    parser.add_argument(
        "--num_inner_steps",
        type=int,
        default=1,
        help="number of inner steps",
    )
    parser.add_argument(
        "--num_outer_steps",
        type=int,
        default=1000,
        help="number of outer steps",
    )
    parser.add_argument(
        "--num_train_tasks",
        type=int,
        default=500,
        help="number of train tasks",
    )
    parser.add_argument(
        "--num_val_tasks",
        type=int,
        default=100,
        help="number of val tasks",
    )
    parser.add_argument(
        "--x_size",
        type=int,
        default=200,
        help="number of labeled data",
    )
    parser.add_argument("--f_size", type=int, default=1000, help="number of physics data (physics)")
    parser.add_argument(
        "--inner_lr",
        type=float,
        default=0.01,
        help="learning rate of inner steps",
    )
    parser.add_argument(
        "--outer_lr",
        type=float,
        default=0.001,
        help="learning rate of outer steps",
    )
    parser.add_argument(
        "--mode", type=str, default="hybrid", help="training mode (data, physics, hybrid)"
    )
    parser.add_argument(
        "--num_sample_tasks",
        type=int,
        default=25,
        help="number of sampled tasks size when training a single loop",
    )
    parser.add_argument(
        "--num_sample_data",
        type=int,
        default=10,
        help="number of sampled data size when training a single loop",
    )

    cfg = parser.parse_args()

    wandb.init(project=cfg.project, config=cfg)
    if cfg.mode == "data":
        maml = MAML(
            cfg.num_inner_steps,
            cfg.inner_lr,
            cfg.outer_lr,
            cfg.x_size,
            cfg.num_sample_tasks,
            cfg.num_sample_data,
        )
    elif cfg.mode == "hybrid" or "physics":
        maml = MAML_hybrid(
            cfg.num_inner_steps,
            cfg.inner_lr,
            cfg.outer_lr,
            cfg.x_size,
            cfg.f_size,
            cfg.num_sample_tasks,
            cfg.num_sample_data,
            cfg.mode,
        )

    maml.train(cfg.num_outer_steps, cfg.num_train_tasks, cfg.num_val_tasks)
