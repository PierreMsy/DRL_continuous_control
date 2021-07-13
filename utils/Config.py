class Config:

# TODO Compartmentalize config
# TODO Use json
# TODO use dictionary/polymorphism
# TODO add attribute "to log"
# TODO create a run object with score, confi and plot functions

    def __init__(self,
                 seed=1,
                 gamma= .99,
                 tau=5e-3,
                 batch_size=32,
                 buffer_size=500,
                 criterion='MSE',
                 optimizer='Adam',
                 learning_rate=1e-2):
        
        self.seed = seed
        self.device = 'cpu'

        self.gamma = gamma #discount factor
        self.tau = tau #target net soft update rate
        
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.buffer_type = 'uniform'

        self.fc_hl = [64, 64]
        self.optimizer = 'Adam'
        self.critic_criterion = 'MSE'

        self.criterion = criterion
        self.optimizer = optimizer
        self.optim_kwargs = {
            'lr' : learning_rate
        }

    def __str__(self):
        representation = f"""
 *** GENERAL ***
 seed : {self.seed}
 device : {self.device}

 *** LEARNING ***
batch size : {self.batch_size}

 *** NETWORK ***
 fully connected hidden layers : {self.fc_hl}

  *** Buffer ***
 type : {self.buffer_type}
 size : {self.buffer_size}
 """
        return representation