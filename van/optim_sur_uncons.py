import torch 
import torch.nn as nn
import numpy as np
from scipy.optimize import minimize
from anastruct import SystemElements
from copy import copy

from models import hybrid_model
from data import *
from tqdm import trange
from metrics import *

import wandb

import argparse

# Problem configurations
ELEMENT_SIZE = 4
BEAM_LENGTH = 1.0
MAX_STRESS = 14e8 / 1e9
MAX_DISP = 0.01
PENALTY = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FEM():
    def __init__(self):
        self.ncall = 0
        self.history = []
        self.stress_history = []
        self.disp_history = []
        self.error_history = []
        self.model_update_call = 0

    def __call__(self, task):
        b = task[:ELEMENT_SIZE]
        h = task[ELEMENT_SIZE:]
        I = b * h**3 / 12
        P = 400000
        E = 2e11
        if np.sum(task <= 0) > 0:
            return PENALTY, PENALTY
        nodes = np.linspace(0.0, BEAM_LENGTH, ELEMENT_SIZE+1)
        ss = SystemElements()
        for i in range(ELEMENT_SIZE):
            ss.add_element(location=[[nodes[i], 0], [nodes[i+1], 0]], EI=E*I[i])
        ss.add_support_fixed(node_id=1)
        ss.point_load(node_id=ELEMENT_SIZE+1, Fy=-0.5*P)
        ss.point_load(node_id=ELEMENT_SIZE // 2 + 1, Fy=-P)

        ss.solve()
        moment = np.zeros(ELEMENT_SIZE)
        disp = np.zeros(ELEMENT_SIZE + 1)
        el_res = ss.get_element_result_range('moment')
        no_res = ss.get_node_result_range('uy')
        
        stress = np.array(el_res) * h / I / 2
        stress = np.append(stress, 0.0)
        disp = -np.array(no_res)
        stress = stress / 1e9

        self.disp = np.max(disp)
        self.stress = np.max(stress)

        self.disp_history.append(self.disp)
        self.stress_history.append(self.stress)

        # print(f"stress: {self.stress:.0f}, disp: {self.disp:.4f}")

        return self.disp, self.stress

    def target(self, task: np.ndarray):
        # print(task)
        self.ncall += 1
        # self(task)
        b = task[:ELEMENT_SIZE]
        h = task[ELEMENT_SIZE:]
        res = np.sum(b*h*BEAM_LENGTH/ELEMENT_SIZE)
        disp, stress = self(task)
        cons_d = max(0, disp/MAX_DISP - 1)
        cons_s = max(0, stress/MAX_STRESS - 1)
        cons_bh = np.array([h[i] / (20 * b[i]) - 1 for i in range(ELEMENT_SIZE)])
        cons_bh = np.where(cons_bh > 0, cons_bh, 0)
        bnds_b = np.array([0.01 / b[i] - 1 for i in range(ELEMENT_SIZE)])
        bnds_h = np.array([0.05 / h[i] - 1 for i in range(ELEMENT_SIZE)])
        bnds_b = np.where(bnds_b > 0, bnds_b, 0)
        bnds_h = np.where(bnds_h > 0, bnds_h, 0)

        res = res + PENALTY * (cons_d + cons_s + np.sum(cons_bh) + np.sum(bnds_b) + np.sum(bnds_h))
        self.history.append(res)
        wandb.log({
            "cost": res
        })
        return res

    def cons_disp(self, task):
        disp, _ = self(task)
        wandb.log({
            "displacement": disp
        })
        res = 1 - disp/ MAX_DISP
        # print(res)
        return res

    def cons_stress(self, task):
        _, stress = self(task)
        wandb.log({
            "stress": stress
        })
        # print(task, stress)
        
        res = 1 - stress/ MAX_STRESS
        return res

    def plot(self, x):
        b = x[:ELEMENT_SIZE]
        h = x[ELEMENT_SIZE:]
        I = b * h**3 / 12
        P = 400000
        E = 2e11
        nodes = np.linspace(0.0, BEAM_LENGTH, ELEMENT_SIZE+1)
        ss = SystemElements()
        for i in range(ELEMENT_SIZE):
            ss.add_element(location=[[nodes[i], 0], [nodes[i+1], 0]], EI=E*I[i])
        ss.add_support_fixed(node_id=1)
        ss.point_load(node_id=ELEMENT_SIZE+1, Fy=-0.5*P)
        ss.point_load(node_id=3, Fy=-P)

        ss.solve()
        ss.show_structure()
        ss.show_displacement()
        ss.show_bending_moment()

        
def optimize_beam_with_FEM(mode: str, fpath: str, method: str, maxiter: int):
    # b0 = np.full(ELEMENT_SIZE, 0.02, dtype=np.float32)
    # h0 = np.full(ELEMENT_SIZE, 0.4, dtype=np.float32)
    b0 = np.array([0.02, 0.02, 0.02, 0.02], dtype=np.float32)
    h0 = np.array([0.4, 0.4, 0.4, 0.4], dtype=np.float32)
    x0 = np.hstack([b0, h0])
    
    sim = FEM()
    sim(x0)

    # cons = [{'type': 'ineq', 'fun': sim.cons_stress},
    #         {'type': 'ineq', 'fun': sim.cons_disp},
    #         {'type': 'ineq', 'fun': lambda x: 20 * x[0] - x[0 + ELEMENT_SIZE]},
    #         {'type': 'ineq', 'fun': lambda x: 20 * x[1] - x[1 + ELEMENT_SIZE]},
    #         {'type': 'ineq', 'fun': lambda x: 20 * x[2] - x[2 + ELEMENT_SIZE]},
    #         {'type': 'ineq', 'fun': lambda x: 20 * x[3] - x[3 + ELEMENT_SIZE]},
    #         {'type': 'ineq', 'fun': lambda x: 20 * x[4] - x[4 + ELEMENT_SIZE]},
    #         {'type': 'ineq', 'fun': lambda x: x[0 + ELEMENT_SIZE] - x[0]},
    #         {'type': 'ineq', 'fun': lambda x: x[1 + ELEMENT_SIZE] - x[1]},
    #         {'type': 'ineq', 'fun': lambda x: x[2 + ELEMENT_SIZE] - x[2]},
    #         {'type': 'ineq', 'fun': lambda x: x[3 + ELEMENT_SIZE] - x[3]},
    #         {'type': 'ineq', 'fun': lambda x: x[4 + ELEMENT_SIZE] - x[4]},
    # ]

    # for i in range(ELEMENT_SIZE):
    #     cons.append({'type': 'ineq', 'fun': lambda x: 20 * x[i] - x[i + ELEMENT_SIZE]})
    bnds = [(0.01, None) for _ in range(ELEMENT_SIZE)]
    bnds_2 = [(0.05, None) for _ in range(ELEMENT_SIZE)]
    bnds = bnds + bnds_2
    # res = minimize(sim.target, x0, method='SLSQP', constraints=cons, bounds=bnds, options={"maxiter": 10000, "disp": True})
    res = minimize(sim.target, x0, method=method, options={"maxiter": maxiter, "disp": True})
    # res = basinhopping(sim.target, x0)
    my_table = wandb.Table(columns=["b1", "b2", "b3", "b4", "h1", "h2", "h3", "h4"], data=[res.x])
    wandb.log({
        "optim_x": my_table,
        "optim_cost": res.fun,
        "model_update_call": sim.model_update_call
    })
    # print(sim.ncall)

    # fig, ax = plt.subplots(2, 2)
    # ax[0][0].plot(sim.history)
    # ax[0][1].plot(sim.disp_history)
    # ax[1][0].plot(sim.stress_history)
    # plt.savefig(f"van/figures/fem_{method}.png")

    # plt.show()
    sim.plot(res.x)
    return res


class Sim():
    def __init__(self, fpath, mode, d_size, epochs, dist_bound):
        self.fpath = fpath
        self.mode = mode
        self.d_size = d_size 
        self.epochs = epochs
        self.tmp = np.zeros(ELEMENT_SIZE*2)
        self.y_max = 0
        self.y_min = 0
        self.history = []
        self.disp_history = []
        self.stress_history = []
        self.error_history = []
        self.ncall = 0
        self.model_update_call = 0
        self.dist_bound = dist_bound

    def __call__(self, task):
        # print(task)
        mode = self.mode
        d_size = self.d_size
        epochs = self.epochs
        model = hybrid_model(neuron_size=64, layer_size=6, dim=1)

        dist = np.sqrt(np.mean( (task - self.tmp)**2 ))


        if dist > self.dist_bound:
        # if self.ncall % 100 == 0:
            # update saved task to calculate distance
            self.tmp = copy(task)
            # print("task saved")

            
            fpath = self.fpath
            if fpath:
                model.load_state_dict(torch.load(fpath)['model_state_dict'])
                # print(f"Model loaded from {fpath}")

            model = model.to(DEVICE)
            optim = torch.optim.Adam(model.parameters(), lr=0.01)

            x_train, y_train = generate_data(mode="data", n=d_size, task=task)
            y_min = np.min(y_train, axis=0)
            y_max = np.max(y_train, axis=0)
            self.y_min = y_min
            self.y_max = y_max

            y_train = normalize(y_min, y_max, y_train)

            x_train = to_tensor(x_train).reshape(-1, 1)
            y_train = to_tensor(y_train)

            x_train = x_train.to(DEVICE)
            y_train = y_train.to(DEVICE)

            if mode == "hybrid":
                x_f_train, y_f_train = generate_data(mode="physics", n=10000, task=task)
                x_f_train = to_tensor(x_f_train).reshape(-1, 1)
                y_f_train = to_tensor(y_f_train).reshape(-1, 1)

                x_f_train_2, y_f_train_2 = generate_data(mode="boundary", n=100, task=task)
                x_f_train_2 = to_tensor(x_f_train_2).reshape(-1, 1)
                y_f_train_2 = to_tensor(y_f_train_2).reshape(-1, 1)
                
                x_f_train = x_f_train.to(DEVICE)
                y_f_train = y_f_train.to(DEVICE)
                x_f_train_2 = x_f_train_2.to(DEVICE)
                y_f_train_2 = y_f_train_2.to(DEVICE)

            loss_func = nn.MSELoss()

            if mode == "data":
                for _ in range(1, epochs + 1):
                    model.train()
                    optim.zero_grad()
                    loss_train = loss_func(y_train, model(x_train))
                    loss_train.to(DEVICE)
                    loss_train.backward()
                    optim.step()
                wandb.log({
                    "loss_d_train": loss_train,
                },
                commit=False,
                )

            elif mode == "hybrid":
                for _ in range(1, epochs + 1):
                    model.train()
                    optim.zero_grad()
                    loss_d_train = loss_func(y_train, model(x_train))
                    loss_f_train = model.calc_loss_f(x_f_train, y_f_train) + model.calc_loss_f_2(x_f_train_2, y_f_train_2)
                    loss_d_train.to(DEVICE)
                    loss_f_train.to(DEVICE)
                    loss_train = loss_d_train + 100 * loss_f_train
                    loss_train.backward()
                    optim.step()
                wandb.log(
                    {
                    "loss_d_train": loss_d_train,
                    "loss_f_train": loss_f_train,
                    },
                commit=False,
                )

                    
            
            self.model = model.state_dict()
            self.model_update_call += 1
            # print("MODEL UPDATED")

            x_plot, y_ans = generate_data(mode="data", n=100, task=task)
            x_plot = x_plot.squeeze()
            s = x_plot.argsort()
            x_plot = x_plot[s]
            y_ans = y_ans[s]
            x_torch = torch.tensor(x_plot, dtype=torch.float32).to(DEVICE).reshape(-1, 1)
            y_plot = model(x_torch)
            y_plot = denormalize(y_min, y_max, y_plot.detach().cpu().numpy())
            nrmse = compute_nrmse(y_plot, y_ans)
            self.error_history.append(nrmse)
            wandb.log({
                "nrmse": nrmse
            })
        

        else:
            model = hybrid_model(neuron_size=64, layer_size=6, dim=1)
            model.load_state_dict(self.model)
            model.to(DEVICE)
            # print("MODEL LOADED")


        
        x_plot, y_ans = generate_data(mode="data", n=100, task=task)
        x_plot = x_plot.squeeze()
        s = x_plot.argsort()
        x_plot = x_plot[s]
        y_ans = y_ans[s]
        x_torch = torch.tensor(x_plot, dtype=torch.float32).to(DEVICE).reshape(-1, 1)
        y_plot = model(x_torch)
        y_plot = denormalize(self.y_min, self.y_max, y_plot.detach().cpu().numpy())
        nrmse = compute_nrmse(y_plot, y_ans)
        self.error_history.append(nrmse)
        wandb.log({
            "nrmse": nrmse
        })

        x = torch.linspace(0.0, 1.0, 100).reshape(-1, 1).to(DEVICE)
        y = denormalize(self.y_min, self.y_max, model(x).detach().cpu())
        stress, disp = torch.max(y, axis=0).values.numpy()


        return disp, stress


    def cons_stress(self, task):
        # print("C1")
        if np.count_nonzero(task) < 10:
            stress = 1000.0
        # if np.product(task) == 0.0:
        #     stress = 1000.0
        else:
            _, stress = self(task)
        # print(task, stress)
        res = 1 - stress / MAX_STRESS
        self.stress_history.append(stress)
        wandb.log({
            "stress": stress
        })
        return res

    def cons_disp(self, task):
        # print("C2")
        if  np.count_nonzero(task) < 10:
            disp = 1000.0
        else:
            disp, _ = self(task)
        res = 1 - disp / MAX_DISP
        self.disp_history.append(disp)
        wandb.log({
            "displacement": disp
        })
        return res

    def target(self, task: np.ndarray):
        self.ncall += 1
        b = task[:ELEMENT_SIZE]
        h = task[ELEMENT_SIZE:]
        res = np.sum(b*h*BEAM_LENGTH/ELEMENT_SIZE)
        disp, stress = self(task)
        cons_d = max(0, disp/MAX_DISP - 1)
        cons_s = max(0, stress/MAX_STRESS - 1)
        cons_bh = np.array([h[i] / (20 * b[i]) - 1 for i in range(ELEMENT_SIZE)])
        cons_bh = np.where(cons_bh > 0, cons_bh, 0)
        bnds_b = np.array([0.01 / b[i] - 1 for i in range(ELEMENT_SIZE)])
        bnds_h = np.array([0.05 / h[i] - 1 for i in range(ELEMENT_SIZE)])
        bnds_b = np.where(bnds_b > 0, bnds_b, 0)
        bnds_h = np.where(bnds_h > 0, bnds_h, 0)

        res = res + PENALTY * (cons_d + cons_s + np.sum(cons_bh) + np.sum(bnds_b) + np.sum(bnds_h))
        self.history.append(res)
        wandb.log({
            "cost": res
        })
        return res
        


def optimize_beam(mode: str, fpath: str, method: str, d_size: int, steps: int, maxiter: int, dist_bound: int):
    # b0 = np.full(ELEMENT_SIZE, 0.02, dtype=np.float32)
    # b0 = np.array([0.12, 0.11, 0.10, 0.09, 0.08], dtype=np.float32)
    # h0 = np.full(ELEMENT_SIZE, 0.4, dtype=np.float32)
    # h0 = np.array([0.24, 0.22, 0.2, 0.18, 0.16], dtype=np.float32)
    # b0 = np.array([0.02, 0.018, 0.016, 0.014, 0.012], dtype=np.float32)
    # h0 = np.array([0.4, 0.36, 0.32, 0.28, 0.24], dtype=np.float32)
    b0 = np.array([0.02, 0.02, 0.02, 0.02], dtype=np.float32)
    h0 = np.array([0.4, 0.4, 0.4, 0.4], dtype=np.float32)
    x0 = np.hstack([b0, h0])
    
    sim = Sim(fpath, mode, d_size, steps, dist_bound)

    sim(x0)

    # cons = [{'type': 'ineq', 'fun': sim.cons_stress},
    #         {'type': 'ineq', 'fun': sim.cons_disp},
    #         {'type': 'ineq', 'fun': lambda x: 20 * x[0] - x[0 + ELEMENT_SIZE]},
    #         {'type': 'ineq', 'fun': lambda x: 20 * x[1] - x[1 + ELEMENT_SIZE]},
    #         {'type': 'ineq', 'fun': lambda x: 20 * x[2] - x[2 + ELEMENT_SIZE]},
    #         {'type': 'ineq', 'fun': lambda x: 20 * x[3] - x[3 + ELEMENT_SIZE]},
    #         {'type': 'ineq', 'fun': lambda x: 20 * x[4] - x[4 + ELEMENT_SIZE]},
    #         {'type': 'ineq', 'fun': lambda x: x[0 + ELEMENT_SIZE] - x[0]},
    #         {'type': 'ineq', 'fun': lambda x: x[1 + ELEMENT_SIZE] - x[1]},
    #         {'type': 'ineq', 'fun': lambda x: x[2 + ELEMENT_SIZE] - x[2]},
    #         {'type': 'ineq', 'fun': lambda x: x[3 + ELEMENT_SIZE] - x[3]},
    #         {'type': 'ineq', 'fun': lambda x: x[4 + ELEMENT_SIZE] - x[4]},
    # ]

    # for i in range(ELEMENT_SIZE):
    #     cons.append({'type': 'ineq', 'fun': lambda x: 20 * x[i] - x[i + ELEMENT_SIZE]})
    # bnds = [(0.01, 0,1) for _ in range(ELEMENT_SIZE)]
    # bnds_2 = [(0.05, 0.2) for _ in range(ELEMENT_SIZE)]
    # bnds = bnds + bnds_2
    res = minimize(sim.target, x0, method=method, options={"maxiter": maxiter, "disp": True})

    # fig, ax = plt.subplots(2, 2)
    # ax[0][0].plot(sim.history)
    # ax[0][1].plot(sim.disp_history)
    # ax[1][0].plot(sim.stress_history)
    # ax[1][1].plot(sim.error_history)

    # my_table = wandb.Table(columns=["b1", "b2", "b3", "b4", "b5", "h1", "h2", "h3", "h4", "h5"], data=[res.x])
    my_table = wandb.Table(columns=["b1", "b2", "b3", "b4", "h1", "h2", "h3", "h4"], data=[res.x])
    wandb.log({
        "optim_x": my_table,
        "optim_cost": res.fun,
        "model_update_call": sim.model_update_call
    })

    # is_loaded = "scratch" if fpath == None else "load"
    # plt.savefig(f"van/figures/{mode}_{is_loaded}.png")
    # plt.show()
    # plt.savefig(f"van/figures/{mode}_{fpath}.png")
    return res




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("Train a model!")
    parser.add_argument("--mode", type=str, default="data", help="training mode")
    parser.add_argument("--method", type=str, default="trust-constr", help="optimization method: turst-constr or SLSQP")
    parser.add_argument("--model", type=str, default="fem", help="optimize with FEM or DL model")
    parser.add_argument("--d_size", type=int, default=3, help="data size to train DL model")
    parser.add_argument("--steps", type=int, default=10, help="training steps to train DL model")
    parser.add_argument("--fpath", type=str, default=None, help="load meta-trained path or not")
    parser.add_argument("--run_name", type=str, default=None, help="wandb run name")
    parser.add_argument("--project", type=str, default="optimize", help="wandb project name")
    parser.add_argument("--maxiter", type=int, default=20, help="max iteration in optimization")
    parser.add_argument("--dist_bound", type=float, default=0.05, help="minimum distance to update the surrogate model")
    cfg = parser.parse_args()
    
    wandb.init(project=cfg.project, config=cfg)
    if cfg.run_name != None:
        wandb.run.name = cfg.run_name

    if cfg.model == 'fem':
        res = optimize_beam_with_FEM(mode=cfg.mode, fpath=None, method=cfg.method, maxiter=cfg.maxiter)
    elif cfg.model == 'dl':
        res = optimize_beam(mode=cfg.mode, fpath=cfg.fpath, method=cfg.method, d_size=cfg.d_size, steps=cfg.steps, maxiter=cfg.maxiter, dist_bound=cfg.dist_bound)
    
    
    print(res.x)
    print(res.fun)
