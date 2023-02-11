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
ELEMENT_SIZE = 5
BEAM_LENGTH = 1.0
MAX_STRESS = 14e8
MAX_DISP = 0.01

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FEM():
    def __init__(self):
        self.ncall = 0
        self.history = []
        self.stress_history = []
        self.disp_history = []
        self.error_history = []

    def __call__(self, task):
        b = task[:ELEMENT_SIZE]
        h = task[ELEMENT_SIZE:]
        I = b * h**3 / 12
        P = 400000
        E = 2e11
        nodes = np.linspace(0.0, BEAM_LENGTH, ELEMENT_SIZE+1)
        ss = SystemElements()
        for i in range(ELEMENT_SIZE):
            ss.add_element(location=[[nodes[i], 0], [nodes[i+1], 0]], EI=E*I[i])
        ss.add_support_fixed(node_id=1)
        ss.point_load(node_id=ELEMENT_SIZE+1, Fy=-P)

        ss.solve()
        moment = np.zeros(ELEMENT_SIZE)
        disp = np.zeros(ELEMENT_SIZE + 1)
        el_res = ss.get_element_results()
        no_res = ss.get_node_displacements()
        for i, x in enumerate(el_res):
            moment[i] = x['Mmax']
        for i, x in enumerate(no_res):
            disp[i] = x[2]
        
        self.disp = np.max(-disp)
        self.stress = np.max(moment*h/2/I)

        self.disp_history.append(self.disp)
        self.stress_history.append(self.stress)

        # print(f"stress: {self.stress:.0f}, disp: {self.disp:.4f}")

        return self.disp, self.stress

    def target(self, task: np.ndarray):
        self.ncall += 1
        # self(task)
        b = task[:ELEMENT_SIZE]
        h = task[ELEMENT_SIZE:]
        res = np.sum(b*h*BEAM_LENGTH/ELEMENT_SIZE)
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
        ss.point_load(node_id=ELEMENT_SIZE+1, Fy=-P)

        ss.solve()
        ss.show_structure()
        ss.show_displacement()
        ss.show_bending_moment()

        


class Sim():
    def __init__(self, model, mode, n, epochs):
        self.model = model
        self.mode = mode
        self.n = n 
        self.epochs = epochs
        self.tmp = None
        self.history = []
        self.disp_history = []
        self.stress_history = []
        self.error_history = []

    def __call__(self, task):
        mode = self.mode
        n = self.n
        epochs = self.epochs
        model = copy(self.model)

        optim = torch.optim.Adam(model.parameters(), lr=0.01)

        x_train, y_train = generate_data(mode="data", n=n, task=task)
        y_min = np.min(y_train, axis=0)
        y_max = np.max(y_train, axis=0)

        y_train = normalize(y_min, y_max, y_train)

        x_train = to_tensor(x_train).reshape(-1, 1)
        y_train = to_tensor(y_train)

        x_train = x_train.to(DEVICE)
        y_train = y_train.to(DEVICE)

        if mode == "hybrid":
            x_f_train, y_f_train = generate_data(mode="physics", n=100, task=task)
            x_f_train = to_tensor(x_f_train).reshape(-1, 1)
            y_f_train = to_tensor(y_f_train).reshape(-1, 1)
            
            x_f_train = x_f_train.to(DEVICE)
            y_f_train = y_f_train.to(DEVICE)

        loss_func = nn.MSELoss()

        if mode == "data":
            for _ in range(1, epochs + 1):
                model.train()
                optim.zero_grad()
                loss_train = loss_func(y_train, model(x_train))
                loss_train.to(DEVICE)
                loss_train.backward()
                optim.step()

        elif mode == "hybrid":
            for _ in range(1, epochs + 1):
                model.train()
                optim.zero_grad()
                loss_d_train = loss_func(y_train, model(x_train))
                loss_f_train = model.calc_loss_f(x_f_train, y_f_train)
                loss_d_train.to(DEVICE)
                loss_f_train.to(DEVICE)
                loss_train = loss_d_train + loss_f_train
                loss_train.backward()
                optim.step()
        
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
        # print(f"NRMSE: {compute_nrmse(y_plot, y_ans):.4f}")

        # plt.plot(x_plot, y_plot[:, 0], 'r--', label='model')
        # plt.plot(x_plot, y_ans[:, 0], 'b-', label='answer')
        # plt.legend()
        # plt.show()
        # plt.cla()
        # plt.plot(x_plot, y_plot[:, 1], 'r--', label='model')
        # plt.plot(x_plot, y_ans[:, 1], 'b-', label='answer')
        # plt.legend()
        # plt.show()

        x = torch.linspace(0.0, 1.0, 100).reshape(-1, 1).to(DEVICE)
        y = denormalize(y_min, y_max, model(x).detach().cpu())
        stress, disp = torch.max(y, axis=0).values.numpy()

        self.stress = stress
        self.disp = disp
        # self.disp_history.append(disp)
        # self.stress_history.append(stress)

        return self.disp, self.stress
        print(f"stress: {stress:4f}, disp: {disp:.4f}")


    def cons_stress(self, task):
        # print("C1")
        _, stress = self(task)
        stress = self.stress
        res = 1 - stress / MAX_STRESS
        self.stress_history.append(stress)
        wandb.log({
            "stress": stress
        })
        return res

    def cons_disp(self, task):
        # print("C2")
        disp, _ = self(task)
        disp = self.disp
        res = 1 - disp / MAX_DISP
        self.disp_history.append(disp)
        wandb.log({
            "displacement": disp
        })
        return res

    def target(self, task: np.ndarray):
        # print("TARGET")
        # if np.array_equal(self.tmp, task) == False:
        #     self.tmp = task
        #     self(task)
        b = task[:ELEMENT_SIZE]
        h = task[ELEMENT_SIZE:]
        res = np.sum(b*h*BEAM_LENGTH/ELEMENT_SIZE)
        self.history.append(res)
        wandb.log({
            "cost": res
        })
        return res
        


def optimize_beam(mode: str, fpath: str, method: str, d_size: int, steps: int):
    b0 = np.full(ELEMENT_SIZE, 0.01, dtype=np.float32)
    h0 = np.full(ELEMENT_SIZE, 0.2, dtype=np.float32)
    x0 = np.hstack([b0, h0])
    model = hybrid_model(neuron_size=64, layer_size=6, dim=1)
    if fpath:
        model.load_state_dict(torch.load(fpath)['model_state_dict'])
        print(f"Model loaded from {fpath}")
    
    model = model.to(DEVICE)
    
    sim = Sim(model, mode, d_size, steps)
    sim(x0)

    cons = [{'type': 'ineq', 'fun': sim.cons_stress},
            {'type': 'ineq', 'fun': sim.cons_disp},
            {'type': 'ineq', 'fun': lambda x: 20 * x[0] - x[0 + ELEMENT_SIZE]},
            {'type': 'ineq', 'fun': lambda x: 20 * x[1] - x[1 + ELEMENT_SIZE]},
            {'type': 'ineq', 'fun': lambda x: 20 * x[2] - x[2 + ELEMENT_SIZE]},
            {'type': 'ineq', 'fun': lambda x: 20 * x[3] - x[3 + ELEMENT_SIZE]},
            {'type': 'ineq', 'fun': lambda x: 20 * x[4] - x[4 + ELEMENT_SIZE]},
    ]

    # for i in range(ELEMENT_SIZE):
    #     cons.append({'type': 'ineq', 'fun': lambda x: 20 * x[i] - x[i + ELEMENT_SIZE]})
    bnds = [(0.01, 0.1) for _ in range(ELEMENT_SIZE)]
    bnds_2 = [(0.05, 0.4) for _ in range(ELEMENT_SIZE)]
    bnds = bnds + bnds_2
    # res = minimize(sim.target, x0, method='SLSQP', constraints=cons, bounds=bnds, options={"maxiter": 1000, "disp": True})
    res = minimize(sim.target, x0, method='trust-constr', constraints=cons, bounds=bnds, options={"maxiter": 1000, "disp": True})

    # fig, ax = plt.subplots(2, 2)
    # ax[0][0].plot(sim.history)
    # ax[0][1].plot(sim.disp_history)
    # ax[1][0].plot(sim.stress_history)
    # ax[1][1].plot(sim.error_history)

    wandb.log({
        "optim_x": res.x,
        "optim_cost": res.fun
    })

    # is_loaded = "scratch" if fpath == None else "load"
    # plt.savefig(f"van/figures/{mode}_{is_loaded}.png")
    # plt.show()
    # plt.savefig(f"van/figures/{mode}_{fpath}.png")
    return res


def optimize_beam_with_FEM(mode: str, fpath: str, method: str):
    b0 = np.full(ELEMENT_SIZE, 0.01, dtype=np.float32)
    h0 = np.full(ELEMENT_SIZE, 0.2, dtype=np.float32)
    x0 = np.hstack([b0, h0])
    
    sim = FEM()
    sim(x0)

    cons = [{'type': 'ineq', 'fun': sim.cons_stress},
            {'type': 'ineq', 'fun': sim.cons_disp},
            {'type': 'ineq', 'fun': lambda x: 20 * x[0] - x[0 + ELEMENT_SIZE]},
            {'type': 'ineq', 'fun': lambda x: 20 * x[1] - x[1 + ELEMENT_SIZE]},
            {'type': 'ineq', 'fun': lambda x: 20 * x[2] - x[2 + ELEMENT_SIZE]},
            {'type': 'ineq', 'fun': lambda x: 20 * x[3] - x[3 + ELEMENT_SIZE]},
            {'type': 'ineq', 'fun': lambda x: 20 * x[4] - x[4 + ELEMENT_SIZE]},
    ]

    # for i in range(ELEMENT_SIZE):
    #     cons.append({'type': 'ineq', 'fun': lambda x: 20 * x[i] - x[i + ELEMENT_SIZE]})
    bnds = [(0.01, None) for _ in range(ELEMENT_SIZE)]
    bnds_2 = [(0.05, None) for _ in range(ELEMENT_SIZE)]
    bnds = bnds + bnds_2
    # res = minimize(sim.target, x0, method='SLSQP', constraints=cons, bounds=bnds, options={"maxiter": 10000, "disp": True})
    res = minimize(sim.target, x0, method=method, constraints=cons, bounds=bnds, options={"maxiter": 10000, "disp": True})
    wandb.log({
        "optim_x": res.x,
        "optim_cost": res.fun
    })
    # print(sim.ncall)

    # fig, ax = plt.subplots(2, 2)
    # ax[0][0].plot(sim.history)
    # ax[0][1].plot(sim.disp_history)
    # ax[1][0].plot(sim.stress_history)
    # plt.savefig(f"van/figures/fem_{method}.png")

    # plt.show()
    # sim.plot(res.x)
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
    cfg = parser.parse_args()
    
    wandb.init(project="optimize", config=cfg)
    if cfg.run_name != None:
        wandb.run.name = cfg.run_name

    if cfg.model == 'fem':
        res = optimize_beam_with_FEM(mode=cfg.mode, fpath=None, method=cfg.method)
    elif cfg.model == 'dl':
        res = optimize_beam(mode=cfg.mode, fpath=cfg.fpath, method=cfg.method, d_size=cfg.d_size, steps=cfg.steps)
    
    
    print(res.x)
    print(res.fun)
