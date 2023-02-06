import numpy as np
from scipy.optimizer import minimize

def FEM_simulation(b: np.ndarray, h: np.ndarray):
    I = b * h**3 / 12
    P = 50000
    E = 2000000
    