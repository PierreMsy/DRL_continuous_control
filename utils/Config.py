class Config:

## TODO Compartmentalize config

    def __init__(self, seed) -> None:
        
        self.seed = seed
        self.fc_hl = [64, 64]
        self.device = 'cpu'
        self.batch_size = 64
        self.buffer_size = 500
        self.buffer_type = 'uniform'

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