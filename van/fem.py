from anastruct import SystemElements
import numpy as np
import matplotlib.pyplot as plt

def get_fem_data(task, design_size, element_size):
    b = task[:design_size]
    h = task[design_size:]
    beam_length = 1.0

    E = 2e11
    P = 400000
    I = b * h**3 / 12
    
    nodes = np.linspace(0.0, beam_length, element_size+1)
    nodes_for_query = nodes[:element_size]
    nodes_for_element = np.linspace(0.0, beam_length, design_size+1)

    b_all = np.zeros(element_size)
    h_all = np.zeros(element_size)

    for i in range(design_size):
        b_all = np.where((nodes_for_query > nodes_for_element[i]) & (nodes_for_query <= nodes_for_element[i+1]), b[i], b_all)
        h_all = np.where((nodes_for_query > nodes_for_element[i]) & (nodes_for_query <= nodes_for_element[i+1]), b[i], h_all)
    b_all = np.where(nodes_for_query == 0.0, b[0], b_all)
    h_all = np.where(nodes_for_query == 0.0, h[0], h_all)
    I_all = b_all * h_all**3 / 12

    ss = SystemElements()
    for i in range(element_size):
        ss.add_element(location=[[nodes[i], 0], [nodes[i+1], 0]], EI=E*I_all[i])
    ss.add_support_fixed(node_id=1)
    ss.point_load(node_id=element_size+1, Fy=-P)
    # ss.point_load(node_id=element_size//2, Fy=-P)

    ss.solve()
    el_res = ss.get_element_result_range('moment')
    no_res = ss.get_node_result_range('uy')

    stress = np.array(el_res) * h_all / I_all
    stress = np.append(stress, 0.0)
    disp = -np.array(no_res)


    # stress /= 1e15
    # disp /= 10000


    # ss.show_structure()
    # ss.show_displacement()

    return nodes, stress, disp
