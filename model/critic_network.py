import torch
from torch import  nn
import torch.nn.functional as F


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
        self.optmizer = config.optimizer
        self.criterion = config.criterion

        self.fc_from_state = nn.Linear(context.state_size, config.fc_hl[0])
        self.fc_state_action = nn.Linear(config.fc_hl[0] + context.action_size,
                                         config.fc_hl[1])
        self.fc_to_Q = nn.Linear(config.fc_hl[1], 1)

    def forward(self, state, action):

        state_features = F.relu(self.fc_from_state(state))
        state_features_actions =  torch.cat(state_features, action)
        state_action_features = F.relu(self.fc_state_action(state_features_actions))
        Q_hat = self.fc_to_Q(state_action_features)

        return Q_hat        