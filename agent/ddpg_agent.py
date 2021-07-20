import numpy as np
import torch

from ccontrol.agent import Base_agent
from ccontrol.model import Actor_network, Critic_network
from ccontrol.utils import to_np, BufferCreator, CriterionCreator, NoiseCreator


class DDPG_agent(Base_agent):

    """
    Deep deterministic policy gradient agent.
    An agent will interact with the environnment to maximize the expected reward.
    This agent use a model-free, off-policy actor-critic algorithm using deep function approximators
    that can learn policies in high-dimensional, continuous action spaces.
    """

    def __init__(self, context, config) -> None:
        
        self.context = context
        self.config = config
        self.buffer = BufferCreator().create(config)
        self.t_step = 0

        self.noise = NoiseCreator().create(config.noise.method,
            context.action_size, config.noise.kwargs)

        self.actor_network = Actor_network(context, config.actor)
        self.actor_target_network = Actor_network(context, config.actor)
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())

        self.critic_network = Critic_network(context, config.critic)
        self.critic_target_network = Critic_network(context, config.critic)
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        self.critic_criterion = CriterionCreator().create(config.critic.criterion)

    def act(self, state, noise=False):

        state = torch.from_numpy(state).float().to(self.config.device)

        self.actor_network.eval()
        with torch.no_grad():
            action = self.actor_network.forward(state)
        self.actor_network.train()
        action = to_np(action)
        
        if noise:
            action = np.clip(action + self.noise.sample(),
                self.context.action_min, self.context.action_max)

        return action

    def step(self, state, action, reward, next_state, done):
        
        self.t_step +=1
        self.buffer.add(state, action, reward, next_state, done)

        if (len(self.buffer) >= self.config.batch_size) & (self.t_step % self.config.update_every == 0):
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
        next_actions = self.actor_target_network(next_states)

        TD_targets = rewards + (self.config.gamma * 
                                self.critic_target_network(next_states, next_actions) * (1 - dones))
        Q_values = self.critic_network(states, actions)
        loss = self.critic_criterion(TD_targets, Q_values)

        self.critic_network.optimizer.zero_grad()
        loss.backward()
        self.critic_network.optimizer.step()

        actions_pred = self.actor_network(states)
        Q_values = self.critic_network(states, actions_pred)
        
        loss = -(Q_values).mean()
        self.actor_network.optimizer.zero_grad()
        loss.backward()
        self.actor_network.optimizer.step()

        soft_update(self.actor_target_network, self.actor_network, self.config.tau)
        soft_update(self.critic_target_network, self.critic_network, self.config.tau)
        
def soft_update(target_network, netwok, tau):
    '''
    net_weights = (1-τ) * net_weights + τ * target_net_weights 
    ''' 
    for target_param, local_param in zip(target_network.parameters(), netwok.parameters()):
        target_param.data.copy_(
            (1.0 - tau) * target_param.data + tau * local_param.data)
    