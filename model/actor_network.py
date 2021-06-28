from torch import nn, manual_seed
import torch.nn.functional as F

#TODO : Docstring
#TODO : Modularize the size of the network
#TODO : Modularize the last activation function
#TODO : Introduce a RL context with the action and state size ?
#TODO : Add tests

class Actor_network(nn.Module):
    """[summary]

    Args:
        nn ([type]): [description]
    """
    
    def __init__(self, state_size, action_size, config):
        
        self.seed = manual_seed(config.seed)

        self.fc_from_state = nn.Linear(state_size, config.fc_from_state_size)
        self.fc_hl_1 = nn.Linear(config.fc_from_state_size, config.fc_hl_1_size)
        self.fc_to_action = nn.Linear(config.fc_hl_1_size, action_size)

    def forward(self, state):

        features = F.relu(self.fc_from_state(state))
        features = F.relu(self.fc_hl_1(features))
        actions = F.tanh(features) 

        return actions 
