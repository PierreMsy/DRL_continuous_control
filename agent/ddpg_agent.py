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
        '''
        Sample experiences from the replay buffer and updates the critic and the actor.

        - The critic is updates based uppon a temporal difference error of the state-action value function
          using the actor to compute the actionsfrom the next state.
          error to minimize w.r.t w : r + γ * Q'_w'(s(t+1), µ'_θ'(s(t+1))) - Q_w(s(t),a(t))

        - The actor is updated using direct approximates of the state-action values from the critic.
          value to maximize w.r.t θ : Q_w(s(t), µ_θ(s(t+1)))  
        '''

        states, actions, rewards, next_states, dones = self.buffer.sample()
        next_actions = self.actor_traget_network(next_states)

        TD_target = rewards + self.config.gamma * self.critic_target_network(next_states, next_actions)
        TD_error = self.critic_network(states, actions) - TD_target

        self.critic_network.zero_grad()
        self.critic_network.criterion(TD_error)
        self.critic_network.step()

        actions_pred = self.actor_network(states)
        Q_t_hat = self.critic_network(states, actions_pred)

        self.actor_network.zero_grad()
        self.actor_network.criterion(-(Q_t_hat).mean()) # - because torch add a minus sign to compute the loss
        self.actor_network.step()

        self.soft_update(self.actor_network, self.actor_traget_network, self.config.tau)
        self.soft_update(self.critic_network, self.critic_traget_network, self.config.tau)
        
def soft_update(netwok, target_network, tau):
    '''
    net_weights = (1-τ) * net_weights + τ * target_net_weights 
    ''' 
    target_network.data.copy_(
        (1 - tau) * netwok.parameters().copy() + tau * target_network.parameters().copy())