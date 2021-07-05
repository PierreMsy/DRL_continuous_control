class Config:
    
    def __init__(self, seed) -> None:
        
        self.seed = seed
        self.fc_hl = [64, 64]
        self.device = 'cpu'

    def __str__(self):
        representation = f"""
 *** GENERAL ***
 seed : {self.seed}
 device : {self.device}
 
 *** NETWORK ***
 fully connected hidden layers : {self.fc_hl}
 """
        return representation