import numpy as np
from scipy.optimize import minimize
from anastruct import SystemElements
import matplotlib.pyplot as plt

ELEMENT_SIZE = 5
BEAM_LENGTH = 1.0

MAX_STRESS = 14e8 / 1e9
MAX_DISP = 0.01

HISTORY = []
DISP_HISTORY = []
STRESS_HISTORY = []

def FEM_simulation(b: np.ndarray, h: np.ndarray):
    I = b * h**3 / 12
    P = 400000
    E = 2e11
    nodes = np.linspace(0.0, BEAM_LENGTH, ELEMENT_SIZE+1)
    ss = SystemElements()
    for i in range(ELEMENT_SIZE):
        ss.add_element(location=[[nodes[i], 0], [nodes[i+1], 0]], EI=E*I[i])
    ss.add_support_fixed(node_id=1)
    ss.point_load(node_id=ELEMENT_SIZE+1, Fy=-2 * P)
    ss.point_load(node_id=ELEMENT_SIZE // 2 +1 , Fy = -P)
    print(ss.element_map[1].EI)
    # ss.point_load(node_id=ELEMENT_SIZE//2, Fy=-P)
    # print(b, h)
    ss.solve()
    moment = np.zeros(ELEMENT_SIZE)
    disp = np.zeros(ELEMENT_SIZE + 1)
    el_res = ss.get_element_results()
    no_res = ss.get_node_displacements()
    for i, x in enumerate(el_res):
        moment[i] = x['Mmax']
    for i, x in enumerate(no_res):
        disp[i] = x[2]
    
    # ss.show_displacement()
    disp = -disp
    stress = moment * h / 2 / I
    # print(moment[0], b[0], h[0], I[0], stress[0])

    # print(f"stress: {np.max(stress):4f}, disp: {np.max(disp):.4f}")
    DISP_HISTORY.append(np.max(disp))
    STRESS_HISTORY.append(np.max(stress))

    return np.max(disp), np.max(stress)

def plot_FEM(b: np.ndarray, h:np.ndarray):
    I = b * h**3 / 12
    P = 400000
    E = 2e11
    nodes = np.linspace(0.0, BEAM_LENGTH, ELEMENT_SIZE+1)
    ss = SystemElements()
    for i in range(ELEMENT_SIZE):
        ss.add_element(location=[[nodes[i], 0], [nodes[i+1], 0]], EI=E*I[i])
    ss.add_support_fixed(node_id=1)
    ss.point_load(node_id=ELEMENT_SIZE+1, Fy=-P)
    
    # ss.point_load(node_id=ELEMENT_SIZE//2, Fy=-P)
    ss.solve()
    ss.show_structure()
    ss.show_displacement()
    # ss.show_shear_force()
    ss.show_bending_moment()

def target(x: np.ndarray):
    # print(x)
    b = x[:ELEMENT_SIZE]
    h = x[ELEMENT_SIZE:]
    res = np.sum(b*h*BEAM_LENGTH/ELEMENT_SIZE)
    HISTORY.append(res)
    return res

def constraint_disp(x):
    b = x[:ELEMENT_SIZE]
    h = x[ELEMENT_SIZE:]
    disp, _ = FEM_simulation(b, h)
    res = 1 - disp/ MAX_DISP

    # print(res)
    return res

def constraint_stress(x):
    b = x[:ELEMENT_SIZE]
    h = x[ELEMENT_SIZE:]
    _, stress = FEM_simulation(b, h)
    # print(stress)
    res = 1 - stress / MAX_STRESS
    # print(res)

    return res
    
def optimize_beam():
    
    b0 = np.full(ELEMENT_SIZE, 0.05, dtype=np.float32)
    h0 = np.full(ELEMENT_SIZE, 0.4, dtype=np.float32)
    x0 = np.hstack([b0, h0])
    cons = [{'type': 'ineq', 'fun': constraint_disp},
            {'type': 'ineq', 'fun': constraint_stress},
            {'type': 'ineq', 'fun': lambda x: 20 * x[0] - x[0 + ELEMENT_SIZE]},
            {'type': 'ineq', 'fun': lambda x: 20 * x[1] - x[1 + ELEMENT_SIZE]},
            {'type': 'ineq', 'fun': lambda x: 20 * x[2] - x[2 + ELEMENT_SIZE]},
            {'type': 'ineq', 'fun': lambda x: 20 * x[3] - x[3 + ELEMENT_SIZE]},
            {'type': 'ineq', 'fun': lambda x: 20 * x[4] - x[4 + ELEMENT_SIZE]},
    ]
    # for i in range(ELEMENT_SIZE):
    #     cons.append({'type': 'ineq', 'fun': lambda x: 20 * x[i] - x[i + ELEMENT_SIZE]})
    bnds = [(0.01, None) for _ in range(ELEMENT_SIZE)]
    bnds_2 = [(0.2, None) for _ in range(ELEMENT_SIZE)]
    bnds = bnds + bnds_2
    res = minimize(target, x0, method='Nelder-Mead', constraints=cons, bounds=bnds, options={"maxiter": 10000, "disp": True})
    print(res.x)
    fig, ax = plt.subplots(1, 3)
    ax[0].plot(HISTORY)
    ax[1].plot(DISP_HISTORY)
    ax[2].plot(STRESS_HISTORY)
    plt.show()
    # print(res.fun * 1000000)
    # print(res.x)
    # print(res)
    b_opt = res.x[:ELEMENT_SIZE]
    h_opt = res.x[ELEMENT_SIZE:]
    plot_FEM(b_opt, h_opt)


if __name__ == "__main__":
    optimize_beam()
