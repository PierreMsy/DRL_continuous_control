import pytest
import numpy as np

from unityagents import UnityEnvironment

from ccontrol.agent import DDPG_agent
from ccontrol.utils import Config, Context, to_np

from ccontrol.test.mock import env

PATH_ENV = ('C:/Users/pierr/Desktop/DataScience/Formation/2020_Udacity_DeepRL/'+
            'deep-reinforcement-learning/Unity_environments/'+
            'Reacher_Windows_x86_64_1_Agent/Reacher_Windows_x86_64/Reacher.exe')


def test_that_when_acting_agent_must_return_an_action():
    
    ctx = Context(env(33, 4))
    agent = DDPG_agent(ctx, Config())

    state = np.array([.5] *33)

    action = agent.act(state)

    assert type(action) == np.ndarray

def test_that_when_agent_step_an_experience_is_added_to_the_buffer():

    env = UnityEnvironment(file_name=PATH_ENV)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]

    ctx = Context(brain)
    agent = DDPG_agent(ctx, Config())

    state = env_info.vector_observations[0]

    action = agent.act(state)
    next_state, reward, done = _env_step(action, env, brain_name)

    initial_buffer_length = len(agent.buffer)
    agent.step(state, action, next_state, reward, done)
    updated_buffer_length = len(agent.buffer)

    assert initial_buffer_length == 0
    assert updated_buffer_length == 1

def test_that_when_agent_learn_it_updates_its_net_weights():

    env = UnityEnvironment(file_name=PATH_ENV)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]

    # 3 steps needed to learn
    cfg = Config(batch_size=3)
    agent = DDPG_agent(Context(brain), cfg)
    state = env_info.vector_observations[0]
    initial_critic_weights = to_np(next(agent.critic_network.parameters()))
    initial_actor_weights = to_np(next(agent.actor_network.parameters()))

    for _ in range(2):
    
        action = agent.act(state)
        next_state, reward, done = _env_step(env, brain_name, action)
        agent.step(state, action, next_state, reward, done)
        state = next_state

    before_learning_critic_weights = to_np(next(agent.critic_network.parameters()))
    before_learning_actor_weights = to_np(next(agent.actor_network.parameters()))

    action = agent.act(state)
    next_state, reward, done = _env_step(env, brain_name, action)
    agent.step(state, action, next_state, reward, done)

    after_learning_critic_weights = to_np(next(agent.critic_network.parameters()))
    after_learning_actor_weights = to_np(next(agent.actor_network.parameters()))

    # check that before learning, weights are still the same
    assert (initial_critic_weights == before_learning_critic_weights).all()
    assert (initial_actor_weights == before_learning_actor_weights).all()

    # check that after learning, some weights are different
    assert (initial_critic_weights != after_learning_critic_weights).any()
    assert (initial_actor_weights != after_learning_actor_weights).any()

def _env_step(env, brain_name, action):

    env_info = env.step(action)[brain_name]
    next_state = env_info.vector_observations[0]
    reward = env_info.rewards[0]
    done = env_info.local_done[0]

    return next_state, reward, done