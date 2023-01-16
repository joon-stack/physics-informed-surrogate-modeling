import torch
import torch.nn as nn
import random

import wandb

from tqdm import trange, tqdm
from copy import deepcopy

import os

from models import hybrid_model
from data import *
from metrics import *

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

VAL_INTERVAL = 10
LOG_INTERVAL = 10
SAVE_INTERVAL = 100

# NU_LOW = 0.001 / np.pi
# NU_HIGH = 0.02 / np.pi

NU_LOW = -3
NU_HIGH = -1

RANDOM = False


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
        sampled_tasks_size,
        sampled_data_size,
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

        self.model = hybrid_model(neuron_size=64, layer_size=6, dim=2, log_dir=log_dir)
        self.model = nn.DataParallel(self.model)
        print("Current device: ", DEVICE)
        print(self.model)
        self.model.to(DEVICE)
        self.device = DEVICE

        print(f"Random: {RANDOM}")

        self._num_inner_steps = num_inner_steps

        self._inner_lr = inner_lr
        self._outer_lr = outer_lr
        self._optimizer = torch.optim.Adam(self.model.parameters(), lr=self._outer_lr)

        self.x_d_size = x_d_size
        self.t_d_size = t_d_size
        self.b_size = b_size
        self.i_size = i_size

        self.sampled_tasks_size = sampled_tasks_size
        self.sampled_data_size = sampled_data_size

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

        # alpha = support

        x_train, t_train, y_train = support

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

    def _outer_loop(self, task_data, train=None):
        """Computes the MAML loss and metrics on a batch of tasks.

        Args:
            task_batch (tuple): batch of tasks from PDE
            each task consists with (support, query)
            each support and query consists with (alpha, beta)
            train (bool): whether we are training or evaluating

        Returns:
            outer_loss (Tensor): mean MAML loss over the batch
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

        sup, qry = task_data
        sup_key, sup_data = sup
        qry_key, qry_data = qry

        sampled_sup = random.sample(sup_data, self.sampled_tasks_size)

        sampled_qry = random.sample(qry_data, self.sampled_tasks_size)

        for idx in tqdm(range(len(sampled_sup))):
            sup = sampled_sup[idx]
            qry = sampled_qry[idx]

            data_idx = random.sample(range(len(sup[0])), self.sampled_data_size)
            x_sup = sup[0][data_idx]
            t_sup = sup[1][data_idx]
            y_sup = sup[2][data_idx]

            sup = (x_sup, t_sup, y_sup)

            phi, grad, loss_sup, nrmse = self._inner_loop(theta, sup, train)
            inner_loss.append(loss_sup)

            model_outer.load_state_dict(phi)
            data_idx = random.sample(range(len(qry[0])), self.sampled_data_size)
            x_train = qry[0][data_idx]
            t_train = qry[1][data_idx]
            y_train = qry[2][data_idx]

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
                g /= len(sup)
            for w, g in zip(list(self.model.parameters()), grad_sum):
                w.grad = g
            self._optimizer.step()

        mean_inner_loss = np.mean(inner_loss, axis=0)
        mean_nrmse_batch = np.mean(nrmse_batch, axis=0) if len(nrmse_batch) > 0 else 0
        return mean_inner_loss, mean_nrmse_batch

    def save(self, ep, loss):

        fname = os.path.join(wandb.run.dir, "model.h5")
        torch.save(
            {
                "epoch": ep,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "loss": loss,
            },
            fname,
        )

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

        val_tasks = generate_tasks(num_val_tasks, low=NU_LOW, high=NU_HIGH)
        val_data = generate_task_data(
            val_tasks,
            self.x_d_size,
            self.t_d_size,
            self.b_size,
            self.i_size,
            mode="data",
            lb_x=LB_X,
            rb_x=RB_X,
            lb_t=LB_T,
            rb_t=RB_T,
        )

        inner_loss_val, nrmse_val = self._outer_loop(val_data, train=False)
        wandb.log(
            {
                "inner_loss_pre_adapt_val": inner_loss_val[0],
                "inner_loss_post_adapt_val": inner_loss_val[-1],
                "nrmse_pre_adapt": nrmse_val[0],
                "nrmse_post_adapt": nrmse_val[-1],
                "ep": 0,
            },
        )
        train_tasks = generate_tasks(num_train_tasks, low=NU_LOW, high=NU_HIGH)
        train_data = generate_task_data(
            train_tasks,
            self.x_d_size,
            self.t_d_size,
            self.b_size,
            self.i_size,
            mode="data",
            lb_x=LB_X,
            rb_x=RB_X,
            lb_t=LB_T,
            rb_t=RB_T,
        )
        for i in trange(1, train_steps + 1):
            self._train_step += 1

            inner_loss, _ = self._outer_loop(train_data, train=True)

            train_loss["inner_loss_pre_adapt"].append(inner_loss[0])
            train_loss["inner_loss_post_adapt"].append(inner_loss[-1])

            if i % SAVE_INTERVAL == 0:
                print(f"Step {i} Model saved")
                self.save(i, inner_loss)

            if i % VAL_INTERVAL == 0:
                inner_loss_val, nrmse_val = self._outer_loop(val_data, train=False)
                wandb.log(
                    {
                        "inner_loss_pre_adapt_val": inner_loss_val[0],
                        "inner_loss_post_adapt_val": inner_loss_val[-1],
                        "nrmse_pre_adapt": nrmse_val[0],
                        "nrmse_post_adapt": nrmse_val[-1],
                    },
                    commit=False,
                )

            wandb.log(
                {
                    "ep": self._train_step,
                    "inner_loss_pre_adapt": inner_loss[0],
                    "inner_loss_post_adapt": inner_loss[-1],
                }
            )

        return train_loss, val_loss, nrmse, self.model


class MAML_hybrid:
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
        sampled_tasks_size,
        sampled_data_size,
        mode="hybrid",
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
        self.model = hybrid_model(neuron_size=64, layer_size=6, dim=2, log_dir=log_dir)
        # self.model = nn.DataParallel(self.model)
        # self.model = hybrid_model(neuron_size=5, layer_size=3, dim=2, log_dir=log_dir)
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

        self.sampled_tasks_size = sampled_tasks_size
        self.sampled_data_size = sampled_data_size

        self.mode = mode

        self._train_step = 0

        self.log_dir = log_dir

        print("Finished initialization of MAML-PINN model")
        print(f"Current mode: {mode}")

    def _inner_loop(self, theta, support, support_key, train=True):

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

        x_train, t_train, y_train = support

        if train:
            x_b_train, t_b_train, y_b_train = generate_data(
                mode="boundary",
                num_x=self.x_d_size,
                num_t=self.t_d_size,
                num_b=self.b_size,
                num_i=self.i_size,
                lb_x=LB_X,
                rb_x=RB_X,
                lb_t=LB_T,
                rb_t=RB_T,
                random=RANDOM,
                alpha=support_key,
            )

            x_i_train, t_i_train, y_i_train = generate_data(
                mode="initial",
                num_x=self.x_d_size,
                num_t=self.t_d_size,
                num_b=self.b_size,
                num_i=self.i_size,
                lb_x=LB_X,
                rb_x=RB_X,
                lb_t=LB_T,
                rb_t=RB_T,
                random=RANDOM,
                alpha=support_key,
            )

            x_f_train, t_f_train, y_f_train = generate_data(
                mode="physics",
                num_x=self.x_d_size,
                num_t=self.t_d_size,
                num_b=self.b_size,
                num_i=self.i_size,
                lb_x=LB_X,
                rb_x=RB_X,
                lb_t=LB_T,
                rb_t=RB_T,
                random=RANDOM,
                alpha=support_key,
            )

        x_train = to_tensor(x_train)
        t_train = to_tensor(t_train)
        y_train = to_tensor(y_train)

        x_train = x_train.to(DEVICE)
        t_train = t_train.to(DEVICE)
        y_train = y_train.to(DEVICE)

        in_train = torch.hstack([x_train, t_train])

        if train:
            x_b_train = to_tensor(x_b_train)
            t_b_train = to_tensor(t_b_train)
            y_b_train = to_tensor(y_b_train)

            x_i_train = to_tensor(x_i_train)
            t_i_train = to_tensor(t_i_train)
            y_i_train = to_tensor(y_i_train)

            x_f_train = to_tensor(x_f_train)
            t_f_train = to_tensor(t_f_train)
            y_f_train = to_tensor(y_f_train)

            x_b_train = x_b_train.to(DEVICE)
            t_b_train = t_b_train.to(DEVICE)
            y_b_train = y_b_train.to(DEVICE)

            x_i_train = x_i_train.to(DEVICE)
            t_i_train = t_i_train.to(DEVICE)
            y_i_train = y_i_train.to(DEVICE)

            x_f_train = x_f_train.to(DEVICE)
            t_f_train = t_f_train.to(DEVICE)
            y_f_train = y_f_train.to(DEVICE)

            in_b_train = torch.hstack([x_b_train, t_b_train])
            in_i_train = torch.hstack([x_i_train, t_i_train])
            in_f_train = torch.hstack([x_f_train, t_f_train])

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
            loss_d = (
                loss_fn(y_train, model_phi(in_train))
                if (self.mode == "hybrid") or (self.mode == "physics" and not train)
                else 0
            )
            loss_b = loss_fn(y_b_train, model_phi(in_b_train)) if train else 0
            loss_i = loss_fn(y_i_train, model_phi(in_i_train)) if train else 0
            loss_f = model_phi.calc_loss_f(in_f_train, y_f_train) if train else 0

            loss = loss_d + loss_b + loss_i + loss_f
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

    def _outer_loop(self, task_data, train=None):
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

        sup, qry = task_data
        sup_key, sup_data = sup
        qry_key, qry_data = qry

        sampled_sup = random.sample(sup_data, self.sampled_tasks_size)
        sampled_qry = random.sample(qry_data, self.sampled_tasks_size)

        for idx in tqdm(range(len(sampled_sup))):
            sup = sampled_sup[idx]
            qry = sampled_qry[idx]

            data_idx = random.sample(range(len(sup[0])), self.sampled_data_size)
            x_sup = sup[0][data_idx]
            t_sup = sup[1][data_idx]
            y_sup = sup[2][data_idx]

            sup = (x_sup, t_sup, y_sup)

            phi, grad, loss_sup, nrmse = self._inner_loop(theta, sup, sup_key, train)
            inner_loss.append(loss_sup)

            model_outer.load_state_dict(phi)

            data_idx = random.sample(range(len(qry[0])), self.sampled_data_size)
            x_train = qry[0][data_idx]
            t_train = qry[1][data_idx]
            y_train = qry[2][data_idx]

            if train:

                x_b_train, t_b_train, y_b_train = generate_data(
                    mode="boundary",
                    num_x=self.x_d_size,
                    num_t=self.t_d_size,
                    num_b=self.b_size,
                    num_i=self.i_size,
                    lb_x=LB_X,
                    rb_x=RB_X,
                    lb_t=LB_T,
                    rb_t=RB_T,
                    random=RANDOM,
                    alpha=qry_key,
                )

                x_i_train, t_i_train, y_i_train = generate_data(
                    mode="initial",
                    num_x=self.x_d_size,
                    num_t=self.t_d_size,
                    num_b=self.b_size,
                    num_i=self.i_size,
                    lb_x=LB_X,
                    rb_x=RB_X,
                    lb_t=LB_T,
                    rb_t=RB_T,
                    random=RANDOM,
                    alpha=qry_key,
                )

                x_f_train, t_f_train, y_f_train = generate_data(
                    mode="physics",
                    num_x=self.x_d_size,
                    num_t=self.t_d_size,
                    num_b=self.b_size,
                    num_i=self.i_size,
                    lb_x=LB_X,
                    rb_x=RB_X,
                    lb_t=LB_T,
                    rb_t=RB_T,
                    random=RANDOM,
                    alpha=qry_key,
                )

            x_train = to_tensor(x_train)
            t_train = to_tensor(t_train)
            y_train = to_tensor(y_train)

            x_train = x_train.to(DEVICE)
            t_train = t_train.to(DEVICE)
            y_train = y_train.to(DEVICE)

            in_train = torch.hstack([x_train, t_train])

            if train:

                x_b_train = to_tensor(x_b_train)
                t_b_train = to_tensor(t_b_train)
                y_b_train = to_tensor(y_b_train)

                x_i_train = to_tensor(x_i_train)
                t_i_train = to_tensor(t_i_train)
                y_i_train = to_tensor(y_i_train)

                x_f_train = to_tensor(x_f_train)
                t_f_train = to_tensor(t_f_train)
                y_f_train = to_tensor(y_f_train)

                x_b_train = x_b_train.to(DEVICE)
                t_b_train = t_b_train.to(DEVICE)
                y_b_train = y_b_train.to(DEVICE)

                x_i_train = x_i_train.to(DEVICE)
                t_i_train = t_i_train.to(DEVICE)
                y_i_train = y_i_train.to(DEVICE)

                x_f_train = x_f_train.to(DEVICE)
                t_f_train = t_f_train.to(DEVICE)
                y_f_train = y_f_train.to(DEVICE)

                in_b_train = torch.hstack([x_b_train, t_b_train])
                in_i_train = torch.hstack([x_i_train, t_i_train])
                in_f_train = torch.hstack([x_f_train, t_f_train])

            loss_d = loss_fn(y_train, model_outer(in_train)) if self.mode == "hybrid" else 0
            loss_b = loss_fn(y_b_train, model_outer(in_b_train)) if train else 0
            loss_i = loss_fn(y_i_train, model_outer(in_i_train)) if train else 0
            loss_f = model_outer.calc_loss_f(in_f_train, y_f_train) if train else 0

            loss = loss_d + loss_b + loss_i + loss_f

            grad = torch.autograd.grad(loss, model_outer.parameters()) if train else None

            if train:
                for g_sum, g in zip(grad_sum, grad):
                    g_sum += g
            else:
                nrmse_batch += [nrmse]

        if train:
            for g in grad_sum:
                g /= len(sup)
            for w, g in zip(list(self.model.parameters()), grad_sum):
                w.grad = g
            self._optimizer.step()

        mean_inner_loss = np.mean(inner_loss, axis=0)
        mean_nrmse_batch = np.mean(nrmse_batch, axis=0) if len(nrmse_batch) > 0 else 0
        return mean_inner_loss, mean_nrmse_batch

    def save(self, ep, loss):

        fname = os.path.join(wandb.run.dir, "model.h5")
        torch.save(
            {
                "epoch": ep,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "loss": loss,
            },
            fname,
        )

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

        val_tasks = generate_tasks(num_val_tasks, low=NU_LOW, high=NU_HIGH)
        val_data = generate_task_data(
            val_tasks,
            self.x_d_size,
            self.t_d_size,
            self.b_size,
            self.i_size,
            mode="data",
            lb_x=LB_X,
            rb_x=RB_X,
            lb_t=LB_T,
            rb_t=RB_T,
        )
        inner_loss_val, nrmse_val = self._outer_loop(val_data, train=False)
        wandb.log(
            {
                "inner_loss_pre_adapt_val": inner_loss_val[0],
                "inner_loss_post_adapt_val": inner_loss_val[-1],
                "nrmse_pre_adapt": nrmse_val[0],
                "nrmse_post_adapt": nrmse_val[-1],
                "ep": 0,
            },
        )
        train_tasks = generate_tasks(num_train_tasks, low=NU_LOW, high=NU_HIGH)
        train_data = generate_task_data(
            train_tasks,
            self.x_d_size,
            self.t_d_size,
            self.b_size,
            self.i_size,
            mode="data",
            lb_x=LB_X,
            rb_x=RB_X,
            lb_t=LB_T,
            rb_t=RB_T,
        )
        for i in trange(1, train_steps + 1):
            self._train_step += 1

            inner_loss, _ = self._outer_loop(train_data, train=True)

            train_loss["inner_loss_pre_adapt"].append(inner_loss[0])
            train_loss["inner_loss_post_adapt"].append(inner_loss[-1])

            if i % SAVE_INTERVAL == 0:
                # print(f"Step {i} Model saved")
                # self.save(i, inner_loss)
                pass

            if i % VAL_INTERVAL == 0:
                inner_loss_val, nrmse_val = self._outer_loop(val_data, train=False)
                wandb.log(
                    {
                        "inner_loss_pre_adapt_val": inner_loss_val[0],
                        "inner_loss_post_adapt_val": inner_loss_val[-1],
                        "nrmse_pre_adapt": nrmse_val[0],
                        "nrmse_post_adapt": nrmse_val[-1],
                    },
                    commit=False,
                )

            wandb.log(
                {
                    "ep": self._train_step,
                    "inner_loss_pre_adapt": inner_loss[0],
                    "inner_loss_post_adapt": inner_loss[-1],
                }
            )

        self.save(i, inner_loss)

        return train_loss, val_loss, nrmse, self.model
