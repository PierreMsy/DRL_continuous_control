class Configuration:

# TODO Use json
# TODO use dictionary/polymorphism
# TODO add attribute "to log"
# TODO create a run object with score, confi and plot functions

    def __init__(self,
                 seed=1,
                 gamma= .99,
                 tau=1e-3,
                 update_every=5,
                 batch_size=32,
                 buffer_size=int(1e5)):
        
        self.seed = seed
        self.device = 'cpu'

        self.gamma = gamma # discount factor
        self.tau = tau # target net soft update rate
        self.update_every = update_every
        
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.buffer_type = 'uniform'

    def __str__(self):
        representation = f"""
 *** GENERAL ***
 seed : {self.seed}
 device : {self.device}
 update every : {self.update_every}
  *** BUFFER  ***
 type : {self.buffer_type}
 size : {self.buffer_size}
 batch size : {self.batch_size}
 """
        return representation

class DDPG_configuration(Configuration):

    def __init__(self,
                 seed=1,
                 gamma= .99,
                 tau=5e-3,
                 batch_size=64,
                 buffer_size=int(1e4),
                 critic_configuration=None,
                 actor_configuration=None,
                 noise_configuration=None):
        super().__init__(seed=seed, gamma=gamma, tau=tau,
                         batch_size=batch_size, buffer_size=buffer_size)
        
        if critic_configuration is None:
            critic_configuration = Critic_configration()
        self.critic = critic_configuration

        if actor_configuration is None:
            actor_configuration = Actor_configration()
        self.actor = actor_configuration

        if noise_configuration is None:
            noise_configuration = Noise_configuration()
        self.noise = noise_configuration
    
    def __str__(self):
        representation = f"""
{super().__str__()}
 *** ACTOR ***
{self.actor.__str__()}
  *** CRITIC  ***
{self.critic.__str__()}
  *** NOISE  ***
{self.noise.__str__()}
 """
        return representation

class Critic_configration:

    def __init__(self,
                fc_hidden_layers=[64, 64],
                last_layer_init= 3e-3,
                learning_rate=1e-3,
                optimizer='Adam',
                criterion='MSE',
                seed=1):

        self.fc_hl = fc_hidden_layers
        self.last_layer_init = last_layer_init 
        self.criterion = criterion
        self.optimizer = optimizer
        self.optim_kwargs = {
            'lr' : learning_rate
        }
        self.seed=seed
    
    def __str__(self):
        return "This is the critic"

class Actor_configration:

    def __init__(self,
                fc_hidden_layers=[64, 64],
                last_layer_init= 3e-3,
                learning_rate=1e-4,
                optimizer='Adam',
                seed=1):

        self.fc_hl = fc_hidden_layers
        self.last_layer_init = last_layer_init
        self.optimizer = optimizer
        self.optim_kwargs = {
            'lr' : learning_rate
        }
        self.seed=seed
    
    def __str__(self):
        return "This is the actor"

class Noise_configuration:

    def __init__(self,
                 method='OU',
                 mu=.0,
                 sigma=.2,
                 theta=.15):

        self.method = method
        self.mu = mu
        self.sigma = sigma
        self.theta = theta

        self.kwargs = {
            'mu' : mu,
            'sigma' : sigma,
            'theta' : theta
        }
        
        