import numpy as np


def compute_nrmse(pred: np.array, truth: np.array) -> float:
    pred = pred.reshape(truth.shape)
    nrmse = np.sum((pred - truth) ** 2) / np.sum(pred**2)
    return nrmse
