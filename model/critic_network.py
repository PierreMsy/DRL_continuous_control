import numpy as np

import torch
from torch import  nn
import torch.nn.functional as F

from ccontrol.utils import OptimizerCreator

class Critic_network(nn.Module):
    """
    The critic is a neural network updated thanks to TD estimates that
    given an action and a state, approximates the Q-value, the expected return
    from the input state if the agent takes the input action.
    The purpose of the critic is to limit the variance of the actor by giving
    it a baseline enabling it to focus on the advantages of the actions.
    """

    def __init__(self, context, config):
        super(Critic_network, self).__init__()

        self.seed = torch.manual_seed(config.seed)
        self.config = config

        self.fc1 = nn.Linear(context.state_size, config.fc_hl[0])
        self.bn1 = nn.LayerNorm(config.fc_hl[0])
        self.fc2 = nn.Linear(config.fc_hl[0] + context.action_size, config.fc_hl[1])
        self.bn2 = nn.LayerNorm(config.fc_hl[1])
        self.fc_to_Q = nn.Linear(config.fc_hl[1], 1)
        self.initialize_parameters()

        self.criterion = config.criterion
        self.optimizer = OptimizerCreator().create(
            config.optimizer, self.parameters(), config.optim_kwargs)
    
    def initialize_parameters(self):
        '''
        Initialize the weight and biais of all layers with a unifom distribution 
        of spread 1 / sqrt(layer_size)
        '''
        spread = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc1.weight.data, -spread, spread)
        torch.nn.init.uniform_(self.fc1.bias.data, -spread, spread)

        spread = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc2.weight.data, -spread, spread)
        torch.nn.init.uniform_(self.fc2.bias.data, -spread, spread)

        spread = self.config.last_layer_init
        torch.nn.init.uniform_(self.fc_to_Q.weight.data, -spread, spread)
        torch.nn.init.uniform_(self.fc_to_Q.bias.data, -spread, spread)

    def forward(self, state, action):

        state_features = self.bn1(self.fc1(state))
        state_features = F.relu(state_features)

        state_features_actions =  torch.cat((state_features, action), dim=1)
        state_action_features =  self.bn2(self.fc2(state_features_actions))
        state_action_features = F.relu(state_action_features)
        
        Q_hat = self.fc_to_Q(state_action_features)

        return Q_hat        