from torch import nn, manual_seed
import torch.nn.functional as F

#TODO : Docstring
#TODO : Modularize the size of the network
#TODO : Modularize the last activation function
#TODO : Add tests

class Actor_network(nn.Module):
    """[summary]

    Args:
        nn ([type]): [description]
    """
    
    def __init__(self, context, config):
        super(Actor_network, self).__init__()
        
        self.seed = manual_seed(config.seed)

        self.fc_from_state = nn.Linear(context.state_size, config.fc_hl[0])
        self.fc_hl_1 = nn.Linear(config.fc_hl[0], config.fc_hl[1])
        self.fc_to_action = nn.Linear(config.fc_hl[1], context.action_size)

    def forward(self, state):

        features = F.relu(self.fc_from_state(state))
        features = F.relu(self.fc_hl_1(features))
        actions = F.tanh(features) 

        return actions 
