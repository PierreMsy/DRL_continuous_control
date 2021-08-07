import json
import os
from copy import deepcopy


PATH_JSON = os.path.join(os.path.dirname(__file__),
    r'./config.json')

class Configuration:

    def set_attr(self, attr, value):
        if value:
            setattr(self, attr, value)
            self.dict[attr] = value
        else:
            setattr(self, attr, self.base_dict[attr])

    def __init__(self,
                 seed=None,
                 device=None,
                 gamma=None,
                 tau=None,
                 batch_size=None,
                 update_every=None):

        with open(PATH_JSON, 'r') as json_config:
            self.base_dict = json.load(json_config)
            self.dict = deepcopy(self.base_dict)

        self.set_attr('seed', seed)
        self.set_attr('device', device)
        self.set_attr('gamma', gamma) # discount factor
        self.set_attr('tau', tau) # target net soft update rate
        self.set_attr('batch_size', batch_size)
        self.set_attr('update_every', update_every)

    def __str__(self):
        representation = f""" *** BASE ***
 device : {self.device}
 gamma : {self.gamma}
 tau : {self.tau}
 batch_size : {self.batch_size}
 update every : {self.update_every}"""
        return representation

class DDPG_configuration(Configuration):

    def set_ddpg_attr(self, attr, value):
        if value:
            setattr(self, attr, value)
            self.dict['ddpg'][attr] = value
        else:
            setattr(self, attr, self.base_dict['ddpg'][attr])

    def update_dict(self, d_ref, d_ovr):
        for k,v in d_ref.items():
            if k in d_ovr:
                if type(v) == dict:
                    self.update_dict(d_ref[k], d_ovr[k])
                else:
                    d_ref[k] = d_ovr[k]

    def __init__(self,
                 seed=None,
                 device=None,
                 gamma=None,
                 tau=None,
                 batch_size=None,
                 update_every=None,
                 buffer_size=None,
                 buffer_type=None,
                 critic_config={},
                 actor_config={},
                 noise_config={}):
        super().__init__(seed=seed, device=device, gamma=gamma, tau=tau,
            batch_size=batch_size, update_every=update_every)

        self.set_ddpg_attr('buffer_size', buffer_size)
        self.set_ddpg_attr('buffer_type', buffer_type)

        self.update_dict(self.dict['ddpg']['critic'], critic_config)
        self.critic = Critic_configuration(self.dict['ddpg']['critic'])

        self.update_dict(self.dict['ddpg']['actor'], actor_config)
        self.actor = Actor_configuration(self.dict['ddpg']['actor'])

        self.update_dict(self.dict['ddpg']['noise'], noise_config)
        self.noise = Noise_configuration(self.dict['ddpg']['noise'])
    
    def __str__(self):
        representation = f"""{super().__str__()}
 *** BUFFER ***
buffer_size : {self.buffer_size}
buffer_type : {self.buffer_type}
 *** ACTOR ***
{self.actor.__str__()}
  *** CRITIC  ***
{self.critic.__str__()}
  *** NOISE  ***
{self.noise.__str__()}
 """
        return representation

class Critic_configuration:

    def __init__(self, dict_config):
        
        self.seed = dict_config['seed']
        self.hidden_layers = dict_config['hidden_layers']
        self.last_layer_init = dict_config['last_layer_init']
        self.architecture = dict_config['architecture']
        self.criterion = dict_config['criterion']
        self.optimizer = dict_config['optimizer']
        self.optim_kwargs = {
            'lr' : dict_config['learning_rate']
        }
    
    def __str__(self):
        return f"""learning rate : {self.optim_kwargs['lr']}        
architecture : {self.architecture}"""

class Actor_configuration:

    def __init__(self, dict_config):
        
        self.seed = dict_config['seed']
        self.hidden_layers = dict_config['hidden_layers']
        self.last_layer_init = dict_config['last_layer_init']
        self.architecture = dict_config['architecture']
        self.optimizer = dict_config['optimizer']
        self.optim_kwargs = {
            'lr' : dict_config['learning_rate']
        }
    
    def __str__(self):
        return f"""learning rate : {self.optim_kwargs['lr']}
architecture : {self.architecture}"""

class Noise_configuration:

    def __init__(self, dict_config):

        self.method = dict_config['method']
        self.mu = dict_config['mu']
        self.sigma = dict_config['sigma']
        self.theta = dict_config['theta']

        self.kwargs = {
            'mu' : self.mu,
            'sigma' : self.sigma,
            'theta' : self.theta
        }

    def __str__(self):
        return f"""method : {self.method}"""
        