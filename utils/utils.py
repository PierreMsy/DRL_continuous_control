import numpy as np


def to_np(tensor) -> np.ndarray:
    return tensor.cpu().detach().numpy()