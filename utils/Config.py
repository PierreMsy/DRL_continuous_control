class Config:
    
    def __init__(self, seed) -> None:
        self.seed = seed
        self.fc_hl = [64, 64]

    def __str__(self):
        representation = f"""
        seed : {self.seed}
        """
        return representation