import torch
from torch import nn
import torch.nn.functional as F

from ccontrol.utils import OptimizerCreator

#TODO : Docstring
#TODO : Modularize the size of the network
#TODO : Modularize the last activation function
#TODO : Add tests

class Actor_network(nn.Module):
    """[summary]
    TODO
    Args:
        nn ([type]): [description]
    """
    
    def __init__(self, context, config):
        super(Actor_network, self).__init__()
        
        self.seed = torch.manual_seed(config.seed)

        self.fc_from_state = nn.Linear(context.state_size, config.fc_hl[0])
        self.fc_hl_1 = nn.Linear(config.fc_hl[0], config.fc_hl[1])
        self.fc_to_action = nn.Linear(config.fc_hl[1], context.action_size)

        self.optimizer = OptimizerCreator().create(
            config.optimizer, self.parameters(), config.optim_kwargs)

    def forward(self, state):

        features = F.relu(self.fc_from_state(state))
        features = F.relu(self.fc_hl_1(features))
        actions = torch.tanh(self.fc_to_action(features)) 

        return actions 
