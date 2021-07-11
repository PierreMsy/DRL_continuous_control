class Config:

# TODO Compartmentalize config
# TODO Use json
# TODO use dictionary/polymorphism
# TODO add attribute "to log"
# TODO create a run object with score, confi and plot functions

    def __init__(self,
                 seed=1,
                 batch_size=32,
                 buffer_size=500):
        
        self.seed = seed
        self.fc_hl = [64, 64]
        self.device = 'cpu'
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.buffer_type = 'uniform'

        self.optimizer = 'Adam'
        self.criterion = 'MSE'

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