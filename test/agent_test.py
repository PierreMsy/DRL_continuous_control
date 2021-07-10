import pytest
import numpy as np

from unityagents import UnityEnvironment

from ccontrol.agent import DDPG_agent
from ccontrol.utils import Config, Context

from ccontrol.test.mock import env

PATH_ENV = ('C:/Users/pierr/Desktop/DataScience/Formation/2020_Udacity_DeepRL/'+
            'deep-reinforcement-learning/Unity_environments/'+
            'Reacher_Windows_x86_64_1_Agent/Reacher_Windows_x86_64/Reacher.exe')


def test_that_when_acting_agent_must_return_an_action():
    
    ctx = Context(env(33, 4))
    cfg = Config(seed=1)
    agent = DDPG_agent(ctx, cfg)

    state = np.array([.5] *33)

    action = agent.act(state)

    assert type(action) == np.ndarray

def test_that_when_agent_step_an_experience_is_added_to_the_buffer():

    env = UnityEnvironment(file_name=PATH_ENV)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]

    ctx = Context(brain)
    cfg = Config(seed=1)
    agent = DDPG_agent(ctx, cfg)

    state = env_info.vector_observations[0]

    action = agent.act(state)

    env_info = env.step(action)[brain_name]
    next_state = env_info.vector_observations[0]
    reward = env_info.rewards[0]
    done = env_info.local_done[0]
    
    initial_buffer_length = len(agent.buffer)
    agent.step(state, action, next_state, reward, done)
    updated_buffer_length = len(agent.buffer)

    assert initial_buffer_length == 0
    assert updated_buffer_length == 1

