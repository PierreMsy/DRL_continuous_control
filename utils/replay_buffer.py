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
    
    def create_buffer(self, config):
        return self.builders[config.buffer_type](config)

class ReplayBuffer:
    
    def __init__(self, config):
        
        self.Experience = namedtuple('Experience',
         ['state', 'action', 'reward', 'next_state', 'done'])
        self.buffer = deque(maxlen=config.buffer_size)
        self.config = config

    @abstractmethod
    def add(self, state, action, reward, next_state, done):
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def sample(self):
        raise NotImplementedError("Please Implement this method")    

    def __len__(self):
        return len(self.buffer)

    def _convert_to_torch(self, *args):
        #TODO Utilize *kwargs / dict / getattr and config for types. np.asarray?
        return (torch.from_numpy(np.array(arg)).float().to(self.config.device) for arg in args) 

class UniformReplayBuffer(ReplayBuffer):

    def __init__(self, config):
        super().__init__(config)

    def add(self, state, action, reward, next_state, done):
        '''
        Create an experience tuple from one interaction and add it to the buffer
        '''
        experience = self.Experience(
            state=state, action=action, reward=reward, next_state=next_state, done=done)
        self.buffer.append(experience)

    def sample(self, sample_size):
        '''
        Random sample as much experiences as requested by the sample_size 
        '''
        experiences = random.sample(self.buffer, sample_size)
        states, actions, rewards, next_states, dones = self._convert_to_torch(zip(*experiences))
        return states, actions, rewards, next_states, dones

