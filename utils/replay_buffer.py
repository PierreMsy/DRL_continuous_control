import random
from collections import deque, namedtuple

class ReplayBuffer:
    
    def __init__(self, config):
        
        self.Experience = namedtuple('Experience',
         ['state', 'action', 'reward', 'next_state', 'done'])
        self.buffer = deque(maxlen=config.buffer_size)

    def add(self, state, action, reward, next_state, done):
        pass

    def sample(self):
        pass    

    def __len__(self):
        return len(self.buffer)

class UniformReplayBuffer(ReplayBuffer):

    def __init__(self):
        super().__init__()

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
        states, actions, rewards, next_states, dones = _convert_to_torch(zip(*experiences))
        return states, actions, rewards, next_states, dones

def _convert_to_torch(states, actions, rewards, next_states, dones):
    # Utilize *kwargs / dict / getattr and config for types. np.asarray?

    return states, actions, rewards, next_states, dones