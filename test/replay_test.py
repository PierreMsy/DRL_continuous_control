import pytest
import numpy as np

from ccontrol.utils import Config
from ccontrol.utils.replay_buffer import UniformReplayBuffer


def test_that_tensor_sampled_from_uniform_buffer_have_correct_size():

    state_size = 33
    action_size = 4
    nb_experiences = 3

    buffer = UniformReplayBuffer(Config())

    for _ in range(nb_experiences):
        state = np.random.random(state_size)
        action = np.random.random(action_size)
        reward = np.random.random()
        next_state = np.random.random(state_size)
        done = False

        buffer.add(state, action, reward, next_state, done)

    states, actions, rewards, next_states, dones = buffer.sample(batch_size=nb_experiences)

    assert list(states.size()) == [nb_experiences, state_size]
    assert list(actions.size()) == [nb_experiences, action_size]
    assert list(rewards.size()) == [nb_experiences, 1]
    assert list(next_states.size()) == [nb_experiences, state_size]
    assert list(dones.size()) == [nb_experiences, 1]
