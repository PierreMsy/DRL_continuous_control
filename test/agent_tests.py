import pytest
import numpy as np

from unityagents import UnityEnvironment

from ccontrol.agent import DDPG_agent

PATH_ENV = ('C:/Users/pierr/Desktop/DataScience/Formation/2020_Udacity_DeepRL/'+
            'deep-reinforcement-learning/Unity_environments/'+
            'Reacher_Windows_x86_64_1_Agent/Reacher_Windows_x86_64/Reacher.exe')

def test_that_when_acting_agent_must_return_an_action():

    env = UnityEnvironment(file_name=PATH_ENV)

    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=True)[brain_name]
    state = env_info.vector_observations[0]

    agent = DDPG_agent()

    action = agent.act(state)

    assert type(action) == np.ndarray

