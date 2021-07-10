import torch
from torch.nn.functional import relu

from ccontrol.agent import Base_agent
from ccontrol.model import Actor_network, Critic_network
from ccontrol.utils import to_np, BufferCreator

#TODO : Docstring
class DDPG_agent(Base_agent):

    """
    Deep deterministic policy gradient agent.
    An agent will interact with the environnment to maximize the expected reward.
    This agent use a model-free, off-policy actor-critic algorithm using deep function approximators
    that can learn policies in high-dimensional, continuous action spaces.
    """

    def __init__(self, context, config) -> None:
        
        self.config = config
        self.buffer = BufferCreator().create_buffer(config)

        self.actor_network = Actor_network(context, config)
        self.actor_traget_network = Actor_network(context, config)
        self.actor_traget_network.load_state_dict(self.actor_network.state_dict())

        self.critic_network = Critic_network(context, config)
        self.critic_target_network = Critic_network(context, config)
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

    def act(self, state):

        state = torch.from_numpy(state).float().to(self.config.device)

        self.actor_network.eval()
        with torch.no_grad():
            action = self.actor_network.forward(state)
        self.actor_network.train()

        return to_np(action)

    def step(self, state, action, next_state, reward, done):

        #add to buffer
        self.buffer.add(state, action, next_state, reward, done)

        # if enough experiences learn
        if len(self.buffer) >= self.config.batch_size:
            self.learn()

    def learn(self):
        raise NotImplementedError