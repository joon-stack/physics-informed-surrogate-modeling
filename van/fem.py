from anastruct import SystemElements
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.autograd as autograd

def get_fem_data(task, design_size, element_size, verbose=False):
    b = task[:design_size]
    h = task[design_size:]
    beam_length = 1.0

    E = 2e11
    P = 400000
    I = b * h**3 / 12
    # P = 1000
    # I = 1
    # E = 1000

    nodes = np.linspace(0.0, beam_length, element_size+1)
    nodes_for_query = nodes[:element_size] + beam_length / element_size / 2
    nodes_for_element = np.linspace(0.0, beam_length, design_size+1)

    b_all = np.zeros(element_size)
    h_all = np.zeros(element_size)

    for i in range(design_size):
        b_all = np.where((nodes_for_query >= nodes_for_element[i]) & (nodes_for_query < nodes_for_element[i+1]), b[i], b_all)
        h_all = np.where((nodes_for_query >= nodes_for_element[i]) & (nodes_for_query < nodes_for_element[i+1]), h[i], h_all)
    b_all = np.where(nodes_for_query == 0.0, b[0], b_all)
    h_all = np.where(nodes_for_query == 0.0, h[0], h_all)


    I_all = b_all * h_all**3 / 12
    # I_all = np.ones(b_all.shape, dtype=np.float32)

    ss = SystemElements()
    for i in range(element_size):
        ss.add_element(location=[[nodes[i], 0], [nodes[i+1], 0]], EI=E*I_all[i])
    ss.add_support_fixed(node_id=1)
    ss.point_load(node_id=element_size+1, Fy=- 0.5 * P)
    # ss.q_load(element_id=np.arange(1, element_size+1), q=-1000) 
    ss.point_load(node_id=element_size//2 + 1, Fy=-P)

    ss.solve()
    if verbose:
        print(b, h)
        print(b_all, h_all)
        ss.show_structure()
        ss.show_displacement()
        ss.show_bending_moment()
    el_res = ss.get_element_result_range('moment')
    no_res = ss.get_node_result_range('uy')

    
    stress = np.array(el_res) * h_all / I_all / 2
    # stress = np.array(el_res) * h / I / 2
    stress = np.append(stress, 0.0)
    disp = -np.array(no_res)
    stress = stress / 1e9

    if verbose:
        print(np.max(stress), np.max(disp))
        # plt.plot(nodes, stress)
        plt.show()
        
    # print(np.max(stress))
    # print(np.max(disp))

    # plt.plot(nodes, stress)
    # plt.show()
    # print(disp)


    # stress /= 1e15
    # disp /= 10000

    # ss.show_structure()
    # 
    # ss.show_displacement()

    return nodes, stress, disp

if __name__ == "__main__":
    get_fem_data(np.array([0.02050586, 0.01724277, 0.01250413, 0.01837767, 0.40745353, 0.27931903, 0.25007818, 0.27561148]), 4, 10, verbose=True)
