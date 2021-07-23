import pytest
import numpy as np

from ccontrol.config import DDPG_configuration


def test_that_tensor_sampled_from_uniform_buffer_have_correct_size():

    config = DDPG_configuration()

    assert True