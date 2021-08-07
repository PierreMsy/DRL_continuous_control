import pytest
import numpy as np

from ccontrol.config import DDPG_configuration


def test_that_overriding_parrameter_updates_dict_but_not_base_dict():

    config = DDPG_configuration(gamma=.97)
    
    assert config.base_dict['gamma'] == .99
    assert config.dict['gamma'] == .97

def test_that_overriding_model_parrameter_updates_dict_but_not_base_dict():

    config = DDPG_configuration(actor_config={'architecture': 'BN'})
    
    assert config.base_dict['ddpg']['actor']['architecture'] == 'vanilla'
    assert config.dict['ddpg']['actor']['architecture'] == 'BN'