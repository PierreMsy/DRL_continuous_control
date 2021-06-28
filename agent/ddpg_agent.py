
from agent import Base_agent
from model.actor_network import Actor_network

#TODO : Docstring
class DDPG_agent(Base_agent):
    """[summary]

    Args:
        Base_agent ([type]): [description]
    """

    def __init__(self, config) -> None:
        
        self.actor_network = Actor_network()
        self.actor_traget_network = Actor_network()
        self.actor_traget_network.load_state_dict(self.actor_network.state_dict())

    