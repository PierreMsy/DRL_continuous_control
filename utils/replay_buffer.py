import numpy as np
import torch
import random
from collections import deque, namedtuple
from abc import abstractmethod

class BufferCreator:

    def __init__(self):
        self.builders = {
            'uniform': lambda config: UniformReplayBuffer(config)
        }
    
    def create(self, config):
        return self.builders[config.buffer_type](config)

class ReplayBuffer:
    
    def __init__(self, config):
        
        self.Experience = namedtuple('Experience',
         ['state', 'action', 'reward', 'next_state', 'done'])
        self.memory = deque(maxlen=config.buffer_size)
        self.config = config

    @abstractmethod
    def add(self, state, action, reward, next_state, done):
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def sample(self):
        raise NotImplementedError("Please Implement this method")    

    def __len__(self):
        return len(self.memory)

    def _convert_to_torch(self, args):
        #TODO Utilize *kwargs / dict / getattr and config for types. np.asarray?
        return (torch.from_numpy(np.vstack(arg)).float().to(self.config.device) for arg in args) 

class UniformReplayBuffer(ReplayBuffer):

    def __init__(self, config):
        super().__init__(config)

    def add(self, state, action, reward, next_state, done):
        '''
        Create an experience tuple from one interaction and add it to the memory
        '''
        experience = self.Experience(
            state=state, action=action, reward=reward, next_state=next_state, done=done)
        self.memory.append(experience)

    def sample(self, batch_size=None):
        '''
        Random sample as much experiences as requested by the batch_size 
        '''
        if batch_size is None:
            batch_size = self.config.batch_size

        experiences = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = self._convert_to_torch(zip(*experiences))
        return states, actions, rewards, next_states, dones

