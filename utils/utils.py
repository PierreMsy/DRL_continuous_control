import numpy as np

import torch.optim as optim
from torch import nn

def to_np(tensor) -> np.ndarray:
    return tensor.cpu().detach().numpy()

class OptimizerCreator():

    def __init__(self):
        self.builders = {
            'Adam': lambda parameters, kwargs : optim.Adam(parameters, **kwargs)
        }

    def create(self, optimizer, parameters, kwargs):
        return self.builders[optimizer](parameters, kwargs)

class CriterionCreator():

    def __init__(self):
        self.builders = {
            'MSE': nn.MSELoss
        }

    def create(self, criterion):
        return self.builders[criterion]()