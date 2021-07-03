
from deep_reinforcement_learning.model import Actor_network, Critic_network
from deep_reinforcement_learning.agent import Base_agent

#TODO : Docstring
#TODO : Reorganize to have from models import critic_net, actor_net
class DDPG_agent(Base_agent):
    """
    Deep deterministic policy gradient agent.
    An agent will interact with the environnment to maximize the expected reward.
    This agent use a model-free, off-policy actor-critic algorithm using deep function approximators
    that can learn policies in high-dimensional, continuous action spaces.
    """

    def __init__(self, context, config) -> None:
        
        self.actor_network = Actor_network(context, config)
        self.actor_traget_network = Actor_network(context, config)
        self.actor_traget_network.load_state_dict(self.actor_network.state_dict())

        self.critic_network = Critic_network()
        self.critic_target_network = Critic_network()
        self.critic_traget_network.load_state_dict(self.critic_network.state_dict())